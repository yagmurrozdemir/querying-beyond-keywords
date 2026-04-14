from pathlib import Path


def write_text(path: Path, text) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(str(text))

def write_text_file(output_file_path, content):
    with open(output_file_path, 'w') as f:
        f.write(str(content))


def write_empty_file(output_file_path):
    with open(output_file_path, 'w') as f:
        f.write('')


def write_nlq_file(nlq, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(str(nlq))


def write_mapping_file(mapping, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(str(mapping))
