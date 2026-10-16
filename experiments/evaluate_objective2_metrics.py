"""CLI for Objective 2 AUROC and adaptive ACE evaluation."""

import argparse
import json
from pathlib import Path

from experiments.objective2_metrics import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_COMPARISON_PATH,
    DEFAULT_OUTPUT_PATH,
    MetricsError,
    evaluate_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-bins", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_path.exists():
        print(f"METRICS ERROR: refusing to overwrite existing file: {args.output_path}")
        return 1
    try:
        result = evaluate_metrics(
            args.annotations_path, args.comparison_path, max_bins=args.max_bins
        )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (MetricsError, OSError, ValueError) as exc:
        print(f"METRICS ERROR: {exc}")
        return 1
    print(f"Saved Objective 2 metrics to {args.output_path}")
    print(f"Status: {result['status']}")
    print(f"Counts: {json.dumps(result['counts'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
