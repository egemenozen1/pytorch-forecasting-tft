from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Compute metrics only for finite, originally observed CO labels."""

    required = {"actual", "prediction_p50", "target_weight"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    valid = (
        predictions["target_weight"].gt(0)
        & np.isfinite(predictions["actual"])
        & np.isfinite(predictions["prediction_p50"])
    )
    evaluated = predictions.loc[valid]
    if evaluated.empty:
        raise ValueError("No observed, finite test targets are available for evaluation.")
    if evaluated["actual"].nunique() < 2:
        raise ValueError("R² requires at least two distinct observed target values.")

    metrics: dict[str, float | int] = {
        "mae": float(mean_absolute_error(evaluated["actual"], evaluated["prediction_p50"])),
        "rmse": float(mean_squared_error(evaluated["actual"], evaluated["prediction_p50"]) ** 0.5),
        "r2": float(r2_score(evaluated["actual"], evaluated["prediction_p50"])),
        "evaluated_forecast_points": int(len(evaluated)),
        "unique_test_timestamps": int(evaluated["target_time"].nunique()),
        "excluded_missing_target_points": int((~valid).sum()),
    }
    if {"prediction_p10", "prediction_p90"}.issubset(evaluated.columns):
        within = evaluated["actual"].between(
            evaluated["prediction_p10"], evaluated["prediction_p90"], inclusive="both"
        )
        metrics["p10_p90_coverage"] = float(within.mean())
        metrics["mean_p10_p90_width"] = float(
            (evaluated["prediction_p90"] - evaluated["prediction_p10"]).mean()
        )
    return metrics


def prediction_frame(
    prediction_result: Any,
    source_frame: pd.DataFrame,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    """Convert PyTorch Forecasting's window output to an auditable long table."""

    output = prediction_result.output.detach().cpu().numpy()
    target, weight = prediction_result.y
    actual = target.detach().cpu().numpy()
    if weight is None:
        weights = np.ones_like(actual, dtype=float)
    else:
        weights = weight.detach().cpu().numpy()
    index = prediction_result.index.reset_index(drop=True)
    if output.ndim != 3 or output.shape[2] != len(quantiles):
        raise ValueError(
            f"Unexpected prediction shape {output.shape}; expected [windows, horizon, quantiles]."
        )

    timestamp_by_idx = source_frame.set_index("time_idx")["datetime"]
    rows: list[dict[str, object]] = []
    for window_idx in range(output.shape[0]):
        first_time_idx = int(index.loc[window_idx, "time_idx"])
        for horizon_idx in range(output.shape[1]):
            target_idx = first_time_idx + horizon_idx
            row: dict[str, object] = {
                "forecast_origin_idx": first_time_idx - 1,
                "target_time_idx": target_idx,
                "target_time": timestamp_by_idx.loc[target_idx],
                "horizon_hour": horizon_idx + 1,
                "actual": float(actual[window_idx, horizon_idx]),
                "target_weight": float(weights[window_idx, horizon_idx]),
            }
            for quantile_idx, quantile in enumerate(quantiles):
                row[f"prediction_p{int(round(quantile * 100)):02d}"] = float(
                    output[window_idx, horizon_idx, quantile_idx]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def add_persistence_baseline(predictions: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    target_by_idx = source_frame.set_index("time_idx")["co_target"]
    result = predictions.copy()
    result["persistence_prediction"] = result["forecast_origin_idx"].map(target_by_idx)
    if result["persistence_prediction"].isna().any():
        raise ValueError("Could not align persistence baseline with every forecast origin.")
    return result


def compute_baseline_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    baseline = predictions.copy()
    baseline["prediction_p50"] = baseline["persistence_prediction"]
    # Interval diagnostics describe TFT quantiles and do not apply to a point
    # persistence baseline.
    baseline = baseline.drop(columns=["prediction_p10", "prediction_p90"], errors="ignore")
    return compute_metrics(baseline)


def metrics_by_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for horizon, group in predictions.groupby("horizon_hour", sort=True):
        row = {"horizon_hour": int(horizon), **compute_metrics(group)}
        rows.append(row)
    if not rows:
        raise ValueError("No horizon groups were available for evaluation.")
    return pd.DataFrame(rows)


def plot_forecast(predictions: pd.DataFrame, output_path: str | Path, hours: int = 168) -> None:
    """Plot a readable week of maximum-horizon forecasts and uncertainty."""

    max_horizon = int(predictions["horizon_hour"].max())
    valid = predictions.loc[
        predictions["target_weight"].gt(0) & predictions["horizon_hour"].eq(max_horizon)
    ].copy()
    if valid.empty:
        raise ValueError("Cannot plot forecasts without observed test targets.")
    first_forecasts = valid.sort_values("target_time").head(hours)
    if first_forecasts.empty:
        raise ValueError("No forecast rows remained after selecting the plotting window.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(
        first_forecasts["target_time"],
        first_forecasts["prediction_p10"],
        first_forecasts["prediction_p90"],
        color="#9ecae1",
        alpha=0.45,
        label="TFT 10th–90th percentile",
    )
    ax.plot(
        first_forecasts["target_time"],
        first_forecasts["actual"],
        color="#202124",
        linewidth=1.8,
        label="Observed CO",
    )
    ax.plot(
        first_forecasts["target_time"],
        first_forecasts["prediction_p50"],
        color="#1f77b4",
        linewidth=1.6,
        label="TFT median forecast",
    )
    ax.set_title(f"UCI Air Quality test period: {max_horizon}-hour-ahead TFT forecast")
    ax.set_xlabel("Target timestamp")
    ax.set_ylabel("CO concentration (mg/m³)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_json(path: str | Path, values: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
