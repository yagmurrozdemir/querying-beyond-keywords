import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.config import HF_DATASET_CONFIG


def copy_dataset_dir(src: Path, dst: Path, force: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {src}")

    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(
                f"Target directory already exists: {dst}\n"
                f"Use --force to overwrite it."
            )

    shutil.copytree(src, dst)


def main(force: bool = False) -> None:
    repo_id = HF_DATASET_CONFIG["dataset_name"]
    cache_dir = HF_DATASET_CONFIG.get("cache_dir")
    local_output_dir = PROJECT_ROOT / HF_DATASET_CONFIG.get("local_output_dir", "data/dataset")

    print(f"[INFO] Downloading dataset repo: {repo_id}")
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )

    print(f"[INFO] Snapshot downloaded to: {snapshot_path}")

    source_dataset_dir = snapshot_path 

    print(f"[INFO] Copying dataset folder to: {local_output_dir}")
    copy_dataset_dir(source_dataset_dir, local_output_dir, force=force)

    print("[DONE] Dataset is ready.")
    print(f"[DONE] Local dataset path: {local_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset from Hugging Face into data/dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local dataset directory if it exists.",
    )
    args = parser.parse_args()

    main(force=args.force)