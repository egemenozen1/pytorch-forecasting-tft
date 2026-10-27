"""Backward-compatible entry point for the repository's original script name.

Prefer ``python -m air_quality_tft.cli ...`` after ``pip install -e .``.
"""

from air_quality_tft.cli import main


if __name__ == "__main__":
    main()
