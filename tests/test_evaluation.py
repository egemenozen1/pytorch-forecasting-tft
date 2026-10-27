import pandas as pd
import pytest

from air_quality_tft.evaluation import compute_baseline_metrics, compute_metrics


def test_metrics_ignore_missing_target_weights() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 999.0],
            "prediction_p50": [1.0, 3.0, -999.0],
            "prediction_p10": [0.5, 1.5, -1000.0],
            "prediction_p90": [1.5, 3.5, 1000.0],
            "target_weight": [1.0, 1.0, 0.0],
            "target_time": pd.date_range("2024-01-01", periods=3, freq="h"),
        }
    )
    metrics = compute_metrics(frame)
    assert metrics["evaluated_forecast_points"] == 2
    assert metrics["excluded_missing_target_points"] == 1
    assert metrics["mae"] == pytest.approx(0.5)


def test_metrics_fail_loudly_without_valid_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1.0],
            "prediction_p50": [1.0],
            "target_weight": [0.0],
            "target_time": [pd.Timestamp("2024-01-01")],
        }
    )
    with pytest.raises(ValueError, match="No observed"):
        compute_metrics(frame)


def test_baseline_metrics_do_not_collide_with_tft_median() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1.0, 2.0],
            "prediction_p50": [100.0, 100.0],
            "persistence_prediction": [1.0, 3.0],
            "target_weight": [1.0, 1.0],
            "target_time": pd.date_range("2024-01-01", periods=2, freq="h"),
        }
    )
    assert compute_baseline_metrics(frame)["mae"] == pytest.approx(0.5)
