"""Command line: ``python -m painpred --data data --results results [--steps figure1 figure2] [--overwrite]``.

Use ``--input <raw study folder>`` instead of ``--data`` to rerun the steps that need the participants' images.
"""

from __future__ import annotations

import argparse
import sys

from .config import Project
from .pipeline import STEPS, run_steps
from .utils import PipelineError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="painpred", description=__doc__)
    parser.add_argument("--data", help="shared data folder (BIDS layout); no subject images needed")
    parser.add_argument("--input", help="raw study folder (default: $PAINPRED_INPUT_ROOT)")
    parser.add_argument("--results", help="output folder (default: $PAINPRED_RESULTS_ROOT or <input>/Results)")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), help="steps to run (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing results")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(STEPS))
        return 0
    try:
        run_steps(Project(args.input, args.results, args.data), args.steps, args.overwrite)
    except PipelineError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
