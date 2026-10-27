from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import sklearn
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from .config import ExperimentConfig, _portable_path
from .data import build_datasets, create_dataloaders, prepare_data
from .evaluation import (
    add_persistence_baseline,
    compute_baseline_metrics,
    compute_metrics,
    metrics_by_horizon,
    plot_forecast,
    prediction_frame,
    write_json,
)


def runtime_versions() -> dict[str, str]:
    import lightning
    import pytorch_forecasting

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "lightning": lightning.__version__,
        "pytorch_forecasting": pytorch_forecasting.__version__,
    }


def _make_trainer(config: ExperimentConfig, checkpoint_callback: ModelCheckpoint | None = None):
    callbacks: list[Any] = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=config.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    return pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=1,
        deterministic="warn",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=config.output_dir / "logs", name="tft"),
        enable_model_summary=True,
        enable_progress_bar=False,
    )


def evaluate_model(model, test_loader, prepared, config: ExperimentConfig) -> dict[str, object]:
    result = model.predict(
        test_loader,
        mode="quantiles",
        return_index=True,
        return_y=True,
        trainer_kwargs={"accelerator": config.accelerator, "devices": 1},
    )
    predictions = prediction_frame(result, prepared.frame, config.quantiles)
    predictions = add_persistence_baseline(predictions, prepared.frame)
    metrics = compute_metrics(predictions)
    baseline_metrics = compute_baseline_metrics(predictions)
    horizon_metrics = metrics_by_horizon(predictions)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.output_dir / "test_predictions.csv", index=False)
    horizon_metrics.to_csv(config.output_dir / "metrics_by_horizon.csv", index=False)
    plot_forecast(predictions, config.output_dir / "forecast_vs_actual.png")
    evaluation = {
        "scope": (
            "Rolling 24-hour forecasts whose first target is in the held-out test period; "
            "metrics use only finite CO observations with target_weight=1."
        ),
        "point_forecast": "0.5 quantile (median)",
        "tft": metrics,
        "persistence_baseline": baseline_metrics,
        "selected_horizons": {
            "1_hour": compute_metrics(predictions.loc[predictions["horizon_hour"].eq(1)]),
            "24_hour": compute_metrics(
                predictions.loc[
                    predictions["horizon_hour"].eq(config.max_prediction_length)
                ]
            ),
        },
        "per_horizon_file": _portable_path(config.output_dir / "metrics_by_horizon.csv"),
    }
    write_json(config.output_dir / "metrics.json", evaluation)
    return evaluation


def run_training(config: ExperimentConfig) -> dict[str, object]:
    config.validate()
    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    prepared = prepare_data(config)
    datasets = build_datasets(prepared, config)
    train_loader, validation_loader, test_loader = create_dataloaders(datasets, config)

    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    model = TemporalFusionTransformer.from_dataset(
        datasets.training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=len(config.quantiles),
        loss=QuantileLoss(quantiles=list(config.quantiles)),
        optimizer="adam",
        reduce_on_plateau_patience=2,
        log_interval=-1,
    )
    trainer = _make_trainer(config, checkpoint_callback)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    if not checkpoint_callback.best_model_path:
        raise RuntimeError("Training finished without producing a best validation checkpoint.")

    best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)
    evaluation = evaluate_model(best_model, test_loader, prepared, config)
    log_metrics = pd.read_csv(Path(trainer.logger.log_dir) / "metrics.csv")
    validation_history = log_metrics.dropna(subset=["val_loss"])
    if validation_history.empty:
        raise RuntimeError("Training log does not contain validation loss history.")
    best_validation_row = validation_history.loc[validation_history["val_loss"].idxmin()]
    record: dict[str, object] = {
        **config.to_dict(),
        "versions": runtime_versions(),
        "best_checkpoint": _portable_path(Path(checkpoint_callback.best_model_path)),
        "best_validation_loss": float(checkpoint_callback.best_model_score),
        "training": {
            "epochs_completed": int(len(validation_history)),
            "best_epoch_zero_based": int(best_validation_row["epoch"]),
            "stopped_early": bool(len(validation_history) < config.max_epochs),
        },
    }
    write_json(config.output_dir / "run_config.json", record)
    write_json(config.output_dir / "data_summary.json", prepared.summary)
    return evaluation


def evaluate_checkpoint(config: ExperimentConfig, checkpoint_path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".ckpt":
        raise ValueError(
            "Expected a PyTorch Lightning .ckpt file. Legacy .pth state_dict files do not "
            "contain enough preprocessing/model metadata for safe automatic loading."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    pl.seed_everything(config.seed, workers=True)
    prepared = prepare_data(config)
    datasets = build_datasets(prepared, config)
    _, _, test_loader = create_dataloaders(datasets, config)
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    if int(model.hparams.max_encoder_length) != config.max_encoder_length:
        raise ValueError("Checkpoint encoder length does not match run configuration.")
    if int(model.hparams.output_size) != len(config.quantiles):
        raise ValueError("Checkpoint quantile output size does not match run configuration.")
    return evaluate_model(model, test_loader, prepared, config)
