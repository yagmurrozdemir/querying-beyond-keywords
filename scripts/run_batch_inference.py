#!/usr/bin/env python3
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.config import DATA_CONFIG
from nlq_to_es.execution.mapping import get_index_mapping
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


def get_output_dir(setting: str, query_type: str, split: str, run_id: int, adapter=None) -> Path:
    model_tag = get_model_tag(setting=setting, adapter=adapter)
    return PROJECT_ROOT / "data" / "predictions" / setting / model_tag / split / query_type / f"run_{run_id}"


def main():
    # ------------------------------------------------------------------
    # Hardcoded batch inputs
    # ------------------------------------------------------------------
    split = "test"              # "train" | "validation" | "test"
    query_type = "basic"        # "basic" | "agg" | "knn"
    setting = "zero_shot"       # "zero_shot" | "few_shot" | "finetuned"
    adapter = None              # None | "basic" | "agg" | "knn" | "mixed"
    run_id = 1

    input_path = PROJECT_ROOT / DATA_CONFIG[split][query_type]
    output_dir = get_output_dir(
        setting=setting,
        query_type=query_type,
        split=split,
        run_id=run_id,
        adapter=adapter,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(input_path)
    model_bundle = load_model_bundle(setting=setting, adapter=adapter)

    for i, example in enumerate(examples, start=1):
        nlq = extract_nlq(example)
        index_name = extract_index_name(example)
        index_mapping = get_index_mapping(index_name)

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


if __name__ == "__main__":
    main()