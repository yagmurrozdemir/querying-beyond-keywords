from pathlib import Path
import random
from math import pow
import json
import argparse

from nlq_to_es.config import FT_PROMPT_CONFIG, SAMPLING_CONFIG




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


def create_mixed_dataset(split):
    alpha = SAMPLING_CONFIG["alpha"]
    seed = SAMPLING_CONFIG["seed"]
    size = SAMPLING_CONFIG["size"]

    random.seed(seed)

    split_cfg = FT_PROMPT_CONFIG[split]

    basic = read_jsonl(Path(split_cfg["basic"]))
    agg = read_jsonl(Path(split_cfg["agg"]))
    knn = read_jsonl(Path(split_cfg["knn"]))

    nb, na, nk = len(basic), len(agg), len(knn)

    # sampling weights
    wb = pow(nb, alpha)
    wa = pow(na, alpha)
    wk = pow(nk, alpha)
    total_w = wb + wa + wk

    pb, pa, pk = wb / total_w, wa / total_w, wk / total_w

    total_size = size if size > 0 else (nb + na + nk)

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

    out_path = Path(split_cfg["mixed"])
    write_jsonl(out_path, mixed)

    print(f"[{split}] mixed dataset written to:", out_path)
    print("Sampling probabilities:",
          {"basic": round(pb, 3), "agg": round(pa, 3), "knn": round(pk, 3)})

def apply_soft_balancing():
    create_mixed_dataset("validation")
    create_mixed_dataset("train")
    create_mixed_dataset("test")