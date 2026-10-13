"""Visualize diffeomorphic PSMAReg registration results."""

from __future__ import annotations

import argparse
import logging
import sys

from psmareg.evaluation.registration_visualization import run_visualization
from psmareg.evaluation.visualization_cli import (
    add_visualization_arguments,
    visualization_settings,
)


def main() -> None:
    """Parse visualization options and render the selected pair."""
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        stream=sys.stdout)
    parser = argparse.ArgumentParser()
    add_visualization_arguments(parser)
    run_visualization(visualization_settings(parser.parse_args()))


if __name__ == "__main__":
    main()
