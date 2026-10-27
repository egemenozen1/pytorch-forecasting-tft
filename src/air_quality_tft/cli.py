from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .data import prepare_data
from .evaluation import write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-aware multi-horizon TFT forecasting for UCI Air Quality."
    )
    parser.add_argument("command", choices=["data-summary", "train", "evaluate"])
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accelerator", default=None, choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(values)
    else:
        config = ExperimentConfig()
    overrides = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
    }
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


def main() -> None:
    args = _parser().parse_args()
    config = _config_from_args(args)
    if args.command == "data-summary":
        prepared = prepare_data(config)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.output_dir / "data_summary.json", prepared.summary)
        print(json.dumps(prepared.summary, indent=2))
        return

    # Heavy imports stay out of data-summary, making data validation quick.
    from .pipeline import evaluate_checkpoint, run_training

    if args.command == "train":
        result = run_training(config)
    else:
        if args.checkpoint is None:
            raise SystemExit("evaluate requires --checkpoint path/to/best.ckpt")
        result = evaluate_checkpoint(config, args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
