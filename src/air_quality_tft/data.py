from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

from .config import ExperimentConfig

# PyTorch Forecasting 1.5 passes NumPy arrays back to scikit-learn scalers that
# were fitted with DataFrame column names. Scikit-learn emits the same harmless
# feature-name warning once per sample, obscuring real training diagnostics.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)

if TYPE_CHECKING:
    from pytorch_forecasting import TimeSeriesDataSet


COLUMN_NAMES = {
    "CO(GT)": "co_gt",
    "PT08.S1(CO)": "pt08_s1_co",
    "NMHC(GT)": "nmhc_gt",
    "C6H6(GT)": "c6h6_gt",
    "PT08.S2(NMHC)": "pt08_s2_nmhc",
    "NOx(GT)": "nox_gt",
    "PT08.S3(NOx)": "pt08_s3_nox",
    "NO2(GT)": "no2_gt",
    "PT08.S4(NO2)": "pt08_s4_no2",
    "PT08.S5(O3)": "pt08_s5_o3",
    "T": "temperature_c",
    "RH": "relative_humidity",
    "AH": "absolute_humidity",
}

# NMHC(GT) is intentionally excluded: 8,443 of the 9,357 valid timestamped rows
# use the -200 missing-value sentinel, so it provides little reliable coverage.
SENSOR_FEATURES = [
    "pt08_s1_co",
    "c6h6_gt",
    "pt08_s2_nmhc",
    "nox_gt",
    "pt08_s3_nox",
    "no2_gt",
    "pt08_s4_no2",
    "pt08_s5_o3",
    "temperature_c",
    "relative_humidity",
    "absolute_humidity",
]

KNOWN_REALS = [
    "time_idx",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
]


@dataclass
class PreparedData:
    frame: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_idx: int
    validation_end_idx: int
    imputation_values: dict[str, float]
    summary: dict[str, object]


@dataclass
class ForecastDatasets:
    training: "TimeSeriesDataSet"
    validation: "TimeSeriesDataSet"
    test: "TimeSeriesDataSet"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the UCI CSV, validate its hourly index, and mark missing values.

    The source file has 114 trailing empty rows. Those rows have no timestamp or
    measurements and are removed; timestamped observations are never removed as
    an outlier-cleaning shortcut.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"UCI Air Quality CSV not found: {path}")

    raw = pd.read_csv(path, sep=";", decimal=",").dropna(axis=1, how="all")
    required = {"Date", "Time", *COLUMN_NAMES}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    timestamps = pd.to_datetime(
        raw["Date"].astype("string") + " " + raw["Time"].astype("string"),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    invalid = timestamps.isna()
    if raw.loc[invalid, list(COLUMN_NAMES)].notna().any(axis=None):
        raise ValueError("Found rows with measurements but an invalid Date/Time value.")

    data = raw.loc[~invalid, ["Date", "Time", *COLUMN_NAMES]].copy()
    data["datetime"] = timestamps.loc[~invalid]
    data = data.drop(columns=["Date", "Time"]).rename(columns=COLUMN_NAMES)
    data = data.sort_values("datetime", kind="stable").reset_index(drop=True)

    if data["datetime"].duplicated().any():
        duplicate_values = data.loc[data["datetime"].duplicated(), "datetime"].head().tolist()
        raise ValueError(f"Duplicate timestamps found: {duplicate_values}")

    numeric_columns = list(COLUMN_NAMES.values())
    data[numeric_columns] = data[numeric_columns].replace(-200, np.nan)
    elapsed_hours = (
        (data["datetime"] - data["datetime"].iloc[0]).dt.total_seconds() / 3600
    ).astype(int)
    if not np.array_equal(elapsed_hours.to_numpy(), np.arange(len(data))):
        raise ValueError("Timestamped observations are not a complete, consecutive hourly series.")

    data["time_idx"] = elapsed_hours
    data["series_id"] = "uci_air_quality"
    data["co_observed"] = data["co_gt"].notna().astype("float32")
    data["co_actual"] = data["co_gt"]
    return data


def _add_calendar_features(data: pd.DataFrame) -> None:
    hour = data["datetime"].dt.hour
    weekday = data["datetime"].dt.dayofweek
    day_of_year = data["datetime"].dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)


def prepare_data(config: ExperimentConfig) -> PreparedData:
    """Create chronological splits and causal, train-fitted imputations."""

    config.validate()
    data = load_raw_data(config.data_path)
    n_rows = len(data)
    train_stop = int(n_rows * config.train_fraction)
    validation_stop = int(n_rows * (config.train_fraction + config.validation_fraction))
    minimum_rows = config.max_encoder_length + config.max_prediction_length
    if min(train_stop, validation_stop - train_stop, n_rows - validation_stop) < minimum_rows:
        raise ValueError(
            "Each chronological split must contain at least encoder_length + prediction_length rows."
        )

    train_raw = data.iloc[:train_stop]
    imputation_values: dict[str, float] = {}
    for column in ["co_gt", *SENSOR_FEATURES]:
        median = float(train_raw[column].median(skipna=True))
        if not np.isfinite(median):
            raise ValueError(f"Training data cannot provide an imputation median for {column}.")
        imputation_values[column] = median

    for column in SENSOR_FEATURES:
        data[f"{column}_observed"] = data[column].notna().astype("float32")
        # Forward filling is causal. The train median is used only for leading gaps.
        data[column] = data[column].ffill().fillna(imputation_values[column])

    # PyTorch Forecasting requires a finite target tensor. The filled value is used
    # as model context only; target_weight=0 removes originally missing CO labels
    # from the quantile loss and all reported metrics.
    data["co_target"] = data["co_gt"].ffill().fillna(imputation_values["co_gt"])
    data["target_weight"] = data["co_observed"]
    _add_calendar_features(data)

    data["split"] = "test"
    data.loc[: train_stop - 1, "split"] = "train"
    data.loc[train_stop : validation_stop - 1, "split"] = "validation"

    train = data.iloc[:train_stop].copy()
    validation = data.iloc[train_stop:validation_stop].copy()
    test = data.iloc[validation_stop:].copy()
    summary: dict[str, object] = {
        "source_rows": n_rows,
        "timestamp_start": data["datetime"].min().isoformat(),
        "timestamp_end": data["datetime"].max().isoformat(),
        "target": "CO(GT), hourly average concentration in mg/m^3",
        "missing_target_rows": int((data["co_observed"] == 0).sum()),
        "observed_target_rows": int(data["co_observed"].sum()),
        "excluded_feature": "NMHC(GT) (8,443/9,357 timestamped rows missing)",
        "splits": {
            "train": _split_summary(train),
            "validation": _split_summary(validation),
            "test": _split_summary(test),
        },
        "train_fitted_imputation_medians": imputation_values,
    }
    return PreparedData(
        frame=data,
        train=train,
        validation=validation,
        test=test,
        train_end_idx=int(train["time_idx"].iloc[-1]),
        validation_end_idx=int(validation["time_idx"].iloc[-1]),
        imputation_values=imputation_values,
        summary=summary,
    )


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "start": frame["datetime"].min().isoformat(),
        "end": frame["datetime"].max().isoformat(),
        "observed_target_rows": int(frame["co_observed"].sum()),
    }


def _valid_forecast_starts(frame: pd.DataFrame, horizon: int) -> set[int]:
    weights = frame.set_index("time_idx")["target_weight"]
    starts: set[int] = set()
    first_idx = int(weights.index.min())
    last_idx = int(weights.index.max())
    for start in range(first_idx, last_idx - horizon + 2):
        if float(weights.reindex(range(start, start + horizon), fill_value=0).sum()) > 0:
            starts.add(start)
    return starts


def build_datasets(prepared: PreparedData, config: ExperimentConfig) -> ForecastDatasets:
    """Fit encoders/scalers on train only and reuse them for validation/test."""

    from pytorch_forecasting import TimeSeriesDataSet

    observed_flags = [f"{column}_observed" for column in SENSOR_FEATURES]
    unknown_reals = ["co_target", "co_observed", *SENSOR_FEATURES, *observed_flags]
    training = TimeSeriesDataSet(
        prepared.train,
        time_idx="time_idx",
        target="co_target",
        weight="target_weight",
        group_ids=["series_id"],
        max_encoder_length=config.max_encoder_length,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=["series_id"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        randomize_length=False,
    )

    validation_context = prepared.frame.loc[
        prepared.frame["time_idx"] <= prepared.validation_end_idx
    ]
    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_context,
        min_prediction_idx=prepared.train_end_idx + 1,
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training,
        prepared.frame,
        min_prediction_idx=prepared.validation_end_idx + 1,
        stop_randomization=True,
    )

    training = training.filter(
        lambda index: index.time_idx_first_prediction.isin(
            _valid_forecast_starts(prepared.train, config.max_prediction_length)
        )
    )
    validation = validation.filter(
        lambda index: index.time_idx_first_prediction.isin(
            _valid_forecast_starts(validation_context, config.max_prediction_length)
        )
    )
    test = test.filter(
        lambda index: index.time_idx_first_prediction.isin(
            _valid_forecast_starts(prepared.frame, config.max_prediction_length)
        )
    )
    return ForecastDatasets(training=training, validation=validation, test=test)


def create_dataloaders(datasets: ForecastDatasets, config: ExperimentConfig):
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "persistent_workers": config.num_workers > 0,
    }
    return (
        datasets.training.to_dataloader(train=True, **loader_kwargs),
        datasets.validation.to_dataloader(train=False, **loader_kwargs),
        datasets.test.to_dataloader(train=False, **loader_kwargs),
    )
