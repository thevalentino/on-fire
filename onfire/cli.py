"""Command-line interface for onfire."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from onfire.pipeline.runner import FailFastError, reduce
from onfire.setup import generate_frame_table


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``onfire`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="onfire",
        description="Reduction pipeline for Magellan/FIRE echelle spectroscopy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Scan a raw data directory and write the fixed-width Frame Table.",
    )
    setup_parser.add_argument("raw_dir", help="Directory of raw *.fits exposures.")
    setup_parser.add_argument(
        "-o",
        "--output",
        default="frames.ascii",
        help="Destination Frame Table path (default: frames.ascii).",
    )
    setup_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Frame Table.",
    )

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="Run the full reduction pipeline from a YAML Parameter Config.",
    )
    reduce_parser.add_argument(
        "config",
        help="Path to the YAML Parameter Config.",
    )
    reduce_parser.add_argument(
        "--frames",
        default=None,
        help="Frame Table path (default: the config 'frames' value).",
    )
    reduce_parser.add_argument(
        "--until",
        default=None,
        help="Stop after completing this reduction step (name or ordinal).",
    )
    reduce_parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        help="Resume from the saved checkpoint of the preceding step.",
    )
    reduce_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Halt on the first failure and print reproduction code.",
    )
    reduce_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an IPython session bound to the state when the run stops.",
    )
    reduce_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for FITS checkpoints (default: config 'output_dir').",
    )
    reduce_parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip writing Plotly QA HTML artifacts.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``onfire`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        try:
            table = generate_frame_table(
                args.raw_dir, output=args.output, overwrite=args.overwrite
            )
        except (FileNotFoundError, FileExistsError) as exc:
            print(f"onfire setup: error: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote {args.output} ({len(table.frames)} frames)")
        return 0

    if args.command == "reduce":
        try:
            state = reduce(
                args.frames,
                args.config,
                until=args.until,
                from_step=args.from_step,
                fail_fast=args.fail_fast,
                output_dir=args.output_dir,
                qa=not args.no_qa,
                interactive=args.interactive,
            )
        except FailFastError as exc:
            print(f"onfire reduce: {exc}", file=sys.stderr)
            print(exc.reproduction_code, file=sys.stderr)
            return 1
        except (FileNotFoundError, ValueError) as exc:
            print(f"onfire reduce: error: {exc}", file=sys.stderr)
            return 2
        print(f"Completed through step {state.step} ({state.step_name})")
        return 0

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
