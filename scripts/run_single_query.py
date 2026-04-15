#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

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

VALID_QUERY_TYPES = {"basic", "agg", "knn"}
VALID_SETTINGS = {"zero_shot", "few_shot", "finetuned"}
VALID_ADAPTERS = {"basic", "agg", "knn", "mixed"}

def run_single_query(
    query_type: str,
    setting: str,
    nlq: str,
    index_name: str,
    adapter=None,
) -> None:
    es_client = create_es_client()
    index_mapping = get_index_mapping(es_client, index_name)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and execute a query for a single natural language question."
    )

    parser.add_argument(
        "--query-type",
        required=True,
        choices=sorted(VALID_QUERY_TYPES),
        help="Query type to generate.",
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=sorted(VALID_SETTINGS),
        help="Inference setting to use.",
    )
    parser.add_argument(
        "--nlq",
        required=True,
        help="Natural language query.",
    )
    parser.add_argument(
        "--index-name",
        required=True,
        help="Elasticsearch index name.",
    )
    parser.add_argument(
        "--adapter",
        choices=sorted(VALID_ADAPTERS),
        default=None,
        help="Adapter name (only for finetuned setting).",
    )

    args = parser.parse_args()

    if args.setting == "finetuned" and args.adapter is None:
        parser.error("--adapter is required when --setting=finetuned.")

    if args.setting != "finetuned" and args.adapter is not None:
        parser.error("--adapter should only be used when --setting=finetuned.")

    return args


def main() -> None:
    args = parse_args()

    run_single_query(
        query_type=args.query_type,
        setting=args.setting,
        nlq=args.nlq,
        index_name=args.index_name,
        adapter=args.adapter,
    )


if __name__ == "__main__":
    main()