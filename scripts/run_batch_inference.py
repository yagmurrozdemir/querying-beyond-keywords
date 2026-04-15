#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.config import DATA_CONFIG, PREDICTIONS_DIR
from nlq_to_es.execution.mapping import get_index_mapping
from nlq_to_es.execution.mapping import create_es_client
from nlq_to_es.generation.inference_pipeline import (
    build_prompt,
    extract_index_name,
    extract_nlq,
    generate_raw_output,
    get_model_tag,
    load_model_bundle,
    postprocess_output,
)
from nlq_to_es.io.readers import load_jsonl
from nlq_to_es.io.writers import write_text


TEST_SPLIT = "test"
VALID_QUERY_TYPES = {"basic", "agg", "knn"}
VALID_SETTINGS = {"zero_shot", "few_shot", "finetuned"}
VALID_ADAPTERS = {"basic", "agg", "knn", "mixed"}


def get_output_dir(setting: str, query_type: str, run_id: int, adapter=None) -> Path:
    model_tag = get_model_tag(setting=setting, adapter=adapter)
    return PREDICTIONS_DIR / setting / model_tag / TEST_SPLIT / query_type / f"run_{run_id}"


def run_batch_inference(
    query_type: str,
    setting: str,
    run_id: int,
    adapter=None,
) -> None:
    input_path = DATA_CONFIG[TEST_SPLIT][query_type]
    output_dir = get_output_dir(
        setting=setting,
        query_type=query_type,
        run_id=run_id,
        adapter=adapter,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(input_path)
    model_bundle = load_model_bundle(setting=setting, adapter=adapter)
    es_client = create_es_client()
    for i, example in enumerate(examples, start=1):
        nlq = extract_nlq(example)
        index_name = extract_index_name(example)
        index_mapping = get_index_mapping(es_client, index_name)

        prompt_text = build_prompt(
            index_mapping=index_mapping,
            nlq=nlq,
            setting=setting,
            project_root=PROJECT_ROOT,
            model_bundle=model_bundle,
        )

        raw_output = generate_raw_output(
            prompt_text=prompt_text,
            setting=setting,
            adapter=adapter,
            model_bundle=model_bundle,
        )

        final_output = postprocess_output(
            raw_output=raw_output,
            query_type=query_type,
        )

        out_path = output_dir / f"{i}.out"
        write_text(out_path, final_output)

        print(f"[{i}/{len(examples)}] saved -> {out_path}")

    print()
    print("Batch inference completed.")
    print(f"Input file : {input_path}")
    print(f"Output dir : {output_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference for test examples."
    )

    parser.add_argument(
        "--query-type",
        required=True,
        choices=sorted(VALID_QUERY_TYPES),
        help="Query type to run.",
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=sorted(VALID_SETTINGS),
        help="Inference setting to use.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run identifier, for example 1, 2, 3, 4, or 5.",
    )
    parser.add_argument(
        "--adapter",
        choices=sorted(VALID_ADAPTERS),
        default=None,
        help="Adapter name. Required for finetuned setting; should be omitted otherwise.",
    )

    args = parser.parse_args()

    if args.run_id < 1:
        parser.error("--run-id must be a positive integer.")

    if args.setting == "finetuned" and args.adapter is None:
        parser.error("--adapter is required when --setting=finetuned.")

    if args.setting != "finetuned" and args.adapter is not None:
        parser.error("--adapter should only be used when --setting=finetuned.")

    return args

def main() -> None:
    args = parse_args()

    run_batch_inference(
        query_type=args.query_type,
        setting=args.setting,
        run_id=args.run_id,
        adapter=args.adapter,
    )


if __name__ == "__main__":
    main()