#!/usr/bin/env python3
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.training.train_qlora import train_adapter


def main():
    adapter = "basic"   # "basic" | "agg" | "knn" | "mixed"
    train_adapter(adapter)


if __name__ == "__main__":
    main()