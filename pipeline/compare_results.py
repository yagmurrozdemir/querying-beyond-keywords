from pathlib import Path

def compare_files(file1_path, file2_path):
    try:
        content1 = Path(file1_path).read_text(encoding='utf-8').strip()
        content2 = Path(file2_path).read_text(encoding='utf-8').strip()
    except FileNotFoundError as e:
        print(f"[warning] File not found: {e}")
        return 0

    return int(content1 == content2)
