#!/usr/bin/env python3
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.execution.mapping import get_index_mapping
from nlq_to_es.execution.execute import create_es_client, execute_query
from nlq_to_es.generation.inference_pipeline import (
    build_prompt,
    generate_raw_output,
    load_model_bundle,
    postprocess_output,
)


def main():
    # ------------------------------------------------------------------
    # Hardcoded single-query inputs
    # ------------------------------------------------------------------
    query_type = "basic"        # "basic" | "agg" | "knn"
    adapter = None              # None | "basic" | "agg" | "knn" | "mixed"
    setting = "few_shot"       # "zero_shot" | "few_shot" | "finetuned"
    nlq = "what is the fuel propulsion where the fleet series (quantity) is 310-329 (20)?"
    gold_query = ""             # provided input, not used in this script
    index_name = "table_10007452_3"

    index_mapping = get_index_mapping(index_name)
    model_bundle = load_model_bundle(setting=setting, adapter=adapter)

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

    final_query = postprocess_output(
        raw_output=raw_output,
        query_type=query_type,
    )

    es_client = create_es_client()
    execution_result = execute_query(
        es_client=es_client,
        index_name=index_name,
        query_input=final_query,
    )

    print("=== Generated Query ===")
    print(final_query)
    print()

    print("=== Execution Result ===")
    print(execution_result)


if __name__ == "__main__":
    main()