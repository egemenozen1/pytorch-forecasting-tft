
"""Deprecated compatibility helper for the original filename."""

from air_quality_tft.config import ExperimentConfig
from air_quality_tft.data import prepare_data


if __name__ == "__main__":
    prepared = prepare_data(ExperimentConfig())
    print(prepared.summary)
