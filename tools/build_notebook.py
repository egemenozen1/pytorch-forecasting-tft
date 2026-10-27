"""Build the compact, reproducible analysis notebook.

The notebook is generated with nbformat so its structure stays valid and JSON
churn is deliberate. Run from the repository root after installing dev deps.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook(
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
)

notebook.cells = [
    markdown(
        """
# Multi-horizon CO forecasting with a Temporal Fusion Transformer

## tl;dr

This notebook is the readable companion to the tested pipeline in `src/air_quality_tft/`.
It validates the UCI Air Quality data, shows the leakage controls, describes the TFT
experiment, and reads the versioned held-out-test artifacts. Training is opt-in below
because a CPU run takes several minutes.
"""
    ),
    markdown(
        """
## Context & Methods

**Question.** Given the previous 168 hourly observations, forecast the next 24 hourly
values of `CO(GT)` (mg/m³).

### Key assumptions

- Rows are split chronologically into 70% train, 15% validation, and 15% test.
- `-200` means missing, not a physical measurement.
- Sensor/analyzer readings are observed-only covariates; only calendar signals are
  future-known.
- Missing CO values are available as causal context after filling, but receive zero
  loss weight and are excluded from metrics.
- The test period is evaluated once after validation-based checkpoint selection.
"""
    ),
    code(
        """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from air_quality_tft.config import ExperimentConfig
from air_quality_tft.data import SENSOR_FEATURES, prepare_data

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if not (ROOT / "pyproject.toml").is_file():
    raise RuntimeError("Run this notebook from the repository root or notebooks/ directory.")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"train": "#4c78a8", "validation": "#f2cf5b", "test": "#e45756"}
config = ExperimentConfig()
prepared = prepare_data(config)
"""
    ),
    markdown("## Data"),
    code(
        """
split_table = pd.DataFrame(prepared.summary["splits"]).T
split_table.index.name = "split"
display(split_table)
print(
    f"Observed CO rows: {prepared.summary['observed_target_rows']:,}; "
    f"missing CO rows: {prepared.summary['missing_target_rows']:,}."
)
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(12, 4.5))
for split, frame in [("train", prepared.train), ("validation", prepared.validation), ("test", prepared.test)]:
    ax.plot(frame["datetime"], frame["co_actual"], linewidth=0.7, color=COLORS[split], label=split)
ax.set_title("CO observations and chronological experiment split")
ax.set_xlabel("Timestamp")
ax.set_ylabel("CO concentration (mg/m³)")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
plt.show()
"""
    ),
    code(
        """
missing_share = (
    prepared.frame[["co_actual", *SENSOR_FEATURES]]
    .isna()
    .mean()
    .sort_values(ascending=True)
    .mul(100)
)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(missing_share.index, missing_share.values, color="#4c78a8")
ax.set_title("Missing values in source measurements before causal filling")
ax.set_xlabel("Missing rows (%)")
ax.set_ylabel("")
fig.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## Model

The model is PyTorch Forecasting's implementation of the Temporal Fusion Transformer,
not a from-scratch TFT implementation. The training dataset fits encoders and scalers;
validation and test datasets are created with `TimeSeriesDataSet.from_dataset(...)` so
those fitted transformations are reused. Quantile loss targets the 0.1, 0.5, and 0.9
quantiles. Early stopping and checkpoint selection monitor validation loss only.
"""
    ),
    code(
        """
model_spec = pd.Series({
    "history_hours": config.max_encoder_length,
    "forecast_hours": config.max_prediction_length,
    "quantiles": list(config.quantiles),
    "hidden_size": config.hidden_size,
    "attention_heads": config.attention_head_size,
    "dropout": config.dropout,
    "learning_rate": config.learning_rate,
    "seed": config.seed,
})
display(model_spec.to_frame("value"))
"""
    ),
    markdown(
        """
### Optional training

Set `RUN_TRAINING = True` to reproduce `artifacts/current/`. Leaving it false makes
the notebook quick to execute while retaining the committed, verified experiment.
"""
    ),
    code(
        """
RUN_TRAINING = False
if RUN_TRAINING:
    from air_quality_tft.pipeline import run_training
    run_training(config)
else:
    print("Using verified artifacts/current outputs. Toggle RUN_TRAINING to retrain.")
"""
    ),
    markdown("## Results"),
    code(
        """
metrics_path = ROOT / "artifacts" / "current" / "metrics.json"
if not metrics_path.is_file():
    raise FileNotFoundError("Run `python -m air_quality_tft.cli train` before this section.")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

comparison = pd.DataFrame({
    "TFT (all horizons)": metrics["tft"],
    "Persistence baseline": metrics["persistence_baseline"],
}).loc[["mae", "rmse", "r2", "evaluated_forecast_points"]]
display(comparison)
horizon_comparison = pd.DataFrame(metrics["selected_horizons"]).T[
    ["mae", "rmse", "r2", "p10_p90_coverage"]
]
display(horizon_comparison)
"""
    ),
    code(
        """
horizon_metrics = pd.read_csv(ROOT / "artifacts" / "current" / "metrics_by_horizon.csv")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(horizon_metrics["horizon_hour"], horizon_metrics["mae"], marker="o", label="MAE", color="#4c78a8")
ax.plot(horizon_metrics["horizon_hour"], horizon_metrics["rmse"], marker="s", label="RMSE", color="#e45756")
ax.set_title("Held-out test error by forecast horizon")
ax.set_xlabel("Forecast horizon (hours)")
ax.set_ylabel("Error (mg/m³)")
ax.set_xticks([1, 6, 12, 18, 24])
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
"""
    ),
    code(
        """
display(Image(filename=str(ROOT / "artifacts" / "current" / "forecast_vs_actual.png")))
"""
    ),
    markdown("## Checks"),
    code(
        """
predictions = pd.read_csv(ROOT / "artifacts" / "current" / "test_predictions.csv")
assert predictions["target_time_idx"].min() > prepared.validation_end_idx
assert predictions.loc[predictions["target_weight"].gt(0), ["actual", "prediction_p50"]].notna().all(axis=None)
assert not predictions.loc[predictions["target_weight"].eq(0), "target_time_idx"].empty
assert np.isfinite([metrics["tft"]["mae"], metrics["tft"]["rmse"], metrics["tft"]["r2"]]).all()
print("Checks passed: test boundary, target masking, and finite reported metrics.")
"""
    ),
    markdown(
        """
## Takeaways

- The verified TFT improves MAE and RMSE over the same-origin persistence baseline,
  but its aggregate test R² is slightly negative; this is not a strong universal fit.
- The 1-hour result is materially stronger than the 24-hour result, as expected for
  a multi-step forecast.
- The nominal 80% (`p10`–`p90`) interval is under-covered, so these quantiles should
  not be presented as calibrated uncertainty without further work.
- The dataset covers one site and roughly thirteen months; this is a portfolio
  experiment, not evidence of deployment readiness.
"""
    ),
]

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
