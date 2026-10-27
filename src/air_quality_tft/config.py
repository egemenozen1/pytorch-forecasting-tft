from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by training, evaluation, and the notebook."""

    data_path: Path = PROJECT_ROOT / "data" / "AirQualityUCI.csv"
    output_dir: Path = PROJECT_ROOT / "artifacts" / "current"
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    max_encoder_length: int = 168
    max_prediction_length: int = 24
    batch_size: int = 128
    num_workers: int = 0
    max_epochs: int = 10
    learning_rate: float = 1e-3
    hidden_size: int = 16
    attention_head_size: int = 4
    hidden_continuous_size: int = 8
    dropout: float = 0.10
    gradient_clip_val: float = 0.10
    early_stopping_patience: int = 3
    quantiles: tuple[float, ...] = field(default_factory=lambda: (0.1, 0.5, 0.9))
    accelerator: str = "auto"

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction

    def validate(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. Run commands from the repository "
                "root or pass --data-path explicitly."
            )
        if self.train_fraction <= 0 or self.validation_fraction <= 0 or self.test_fraction <= 0:
            raise ValueError("Train, validation, and test fractions must all be positive.")
        if self.max_encoder_length < 1 or self.max_prediction_length < 1:
            raise ValueError("Encoder and prediction lengths must be positive.")
        if 0.5 not in self.quantiles:
            raise ValueError("Quantiles must include 0.5 so point metrics use the median forecast.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_path"] = _portable_path(self.data_path)
        values["output_dir"] = _portable_path(self.output_dir)
        values["quantiles"] = list(self.quantiles)
        values["test_fraction"] = self.test_fraction
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        accepted = {
            key: value
            for key, value in values.items()
            if key in cls.__dataclass_fields__  # type: ignore[attr-defined]
        }
        data_path = Path(accepted["data_path"])
        output_dir = Path(accepted["output_dir"])
        accepted["data_path"] = data_path if data_path.is_absolute() else PROJECT_ROOT / data_path
        accepted["output_dir"] = (
            output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
        )
        accepted["quantiles"] = tuple(float(q) for q in accepted["quantiles"])
        return cls(**accepted)
