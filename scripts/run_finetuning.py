#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.training.train_qlora import train_adapter
from nlq_to_es.training.make_mixed_jsonl import apply_soft_balancing

VALID_ADAPTERS = {"basic", "agg", "knn", "mixed"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a QLoRA adapter."
    )

    parser.add_argument(
        "--adapter",
        required=True,
        choices=sorted(VALID_ADAPTERS),
        help="Adapter type to train.",
    )

    parser.add_argument(
        "--skip-balancing",
        action="store_true",
        help="Skip soft balancing step (only relevant for 'mixed').",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    adapter = args.adapter

    # Only apply balancing when needed
    if adapter == "mixed" and not args.skip_balancing:
        print("Applying soft balancing for mixed adapter...")
        apply_soft_balancing()

    print(f"Training adapter: {adapter}")
    train_adapter(adapter)


if __name__ == "__main__":
    main()