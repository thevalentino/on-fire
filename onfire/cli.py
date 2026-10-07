"""Command-line interface for onfire."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

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

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
