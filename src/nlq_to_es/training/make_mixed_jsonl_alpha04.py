import json
import random
import argparse
from pathlib import Path
from math import pow

def read_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic", required=True)
    parser.add_argument("--agg", required=True)
    parser.add_argument("--knn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=0,
                        help="Total size of merged dataset (0 = sum of originals)")
    args = parser.parse_args()

    random.seed(args.seed)

    basic = read_jsonl(Path(args.basic))
    agg   = read_jsonl(Path(args.agg))
    knn   = read_jsonl(Path(args.knn))

    nb, na, nk = len(basic), len(agg), len(knn)

    # sampling weights
    wb = pow(nb, args.alpha)
    wa = pow(na, args.alpha)
    wk = pow(nk, args.alpha)
    total_w = wb + wa + wk

    pb, pa, pk = wb / total_w, wa / total_w, wk / total_w

    total_size = args.size if args.size > 0 else (nb + na + nk)

    mixed = []
    for _ in range(total_size):
        r = random.random()
        if r < pb:
            mixed.append(random.choice(basic))
        elif r < pb + pa:
            mixed.append(random.choice(agg))
        else:
            mixed.append(random.choice(knn))

    random.shuffle(mixed)

    write_jsonl(Path(args.out), mixed)

    print("Merged dataset written to:", args.out)
    print("Original sizes:", {"basic": nb, "agg": na, "knn": nk})
    print("Alpha:", args.alpha)
    print("Sampling probabilities:",
          {"basic": round(pb, 3), "agg": round(pa, 3), "knn": round(pk, 3)})
    print("Total merged size:", len(mixed))

if __name__ == "__main__":
    main()
