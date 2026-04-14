import json
from pathlib import Path

def read_text_file(input_file_path):
    with open(input_file_path, 'r') as f:
        result = f.read().strip()
    return result

def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()

def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items




