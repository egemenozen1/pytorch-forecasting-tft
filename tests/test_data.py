from dataclasses import replace

import numpy as np

from air_quality_tft.config import ExperimentConfig
from air_quality_tft.data import (
    KNOWN_REALS,
    SENSOR_FEATURES,
    build_datasets,
    create_dataloaders,
    load_raw_data,
    prepare_data,
)


def test_raw_dataset_shape_and_missing_target() -> None:
    data = load_raw_data(ExperimentConfig().data_path)
    assert len(data) == 9357
    assert data["datetime"].is_monotonic_increasing
    assert data["time_idx"].tolist() == list(range(9357))
    assert int(data["co_gt"].isna().sum()) == 1683


def test_chronological_splits_and_train_fitted_imputation() -> None:
    config = replace(ExperimentConfig(), max_encoder_length=24, max_prediction_length=6)
    prepared = prepare_data(config)
    assert prepared.train["datetime"].max() < prepared.validation["datetime"].min()
    assert prepared.validation["datetime"].max() < prepared.test["datetime"].min()
    assert len(prepared.train) + len(prepared.validation) + len(prepared.test) == 9357
    raw = load_raw_data(config.data_path)
    for column in SENSOR_FEATURES:
        expected = float(raw.iloc[: len(prepared.train)][column].median())
        assert np.isclose(prepared.imputation_values[column], expected)
        assert prepared.frame[column].notna().all()


def test_future_known_features_are_calendar_only() -> None:
    assert set(KNOWN_REALS) == {
        "time_idx",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "year_sin",
        "year_cos",
    }
    assert not set(SENSOR_FEATURES).intersection(KNOWN_REALS)


def test_forecasting_datasets_share_train_fitted_parameters() -> None:
    config = replace(
        ExperimentConfig(),
        max_encoder_length=24,
        max_prediction_length=6,
        batch_size=16,
    )
    prepared = prepare_data(config)
    datasets = build_datasets(prepared, config)
    assert datasets.training.get_parameters()["scalers"].keys() == (
        datasets.validation.get_parameters()["scalers"].keys()
    )
    assert len(datasets.training) > 0
    _, target = next(iter(create_dataloaders(datasets, config)[0]))
    values, weights = target
    assert values.shape == weights.shape == (16, 6)
    assert set(weights.unique().tolist()).issubset({0.0, 1.0})
