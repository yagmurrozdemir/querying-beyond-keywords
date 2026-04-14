#!/usr/bin/env python3
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.config import DATA_CONFIG
from nlq_to_es.execution.execute import create_es_client
from nlq_to_es.evaluation.execution_evaluator import (
    evaluate_prediction_to_files,
)
from nlq_to_es.generation.inference_pipeline import (
    extract_index_name,
    get_model_tag,
)
from nlq_to_es.io.readers import load_jsonl, read_text


def get_prediction_dir(setting: str, query_type: str, split: str, run_id: int, adapter=None) -> Path:
    model_tag = get_model_tag(setting=setting, adapter=adapter)
    return PROJECT_ROOT / "data" / "predictions" / setting / model_tag / split / query_type / f"run_{run_id}"


def get_result_dirs(setting: str, split: str, query_type: str, run_id: int, adapter=None):
    model_tag = get_model_tag(setting=setting, adapter=adapter)

    pred_dir = PROJECT_ROOT / "data" / "results" / setting / model_tag / split / query_type / f"run_{run_id}"
    gold_dir = PROJECT_ROOT / "data" / "results" / "gold" / split / query_type

    pred_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    return pred_dir, gold_dir


def run_batch_evaluation(
    split: str,
    query_type: str,
    setting: str,
    adapter,
    run_id: int,
):
    input_path = PROJECT_ROOT / DATA_CONFIG[split][query_type]
    prediction_dir = get_prediction_dir(
        setting=setting,
        query_type=query_type,
        split=split,
        run_id=run_id,
        adapter=adapter,
    )
    pred_result_dir, gold_result_dir = get_result_dirs(
        setting=setting,
        split=split,
        query_type=query_type,
        run_id=run_id,
        adapter=adapter,
    )

    examples = load_jsonl(input_path)
    es_client = create_es_client()

    correct = 0
    total = 0

    for i, example in enumerate(examples, start=1):
        pred_path = prediction_dir / f"{i}.out"
        pred_result_path = pred_result_dir / f"{i}.txt"
        gold_result_path = gold_result_dir / f"{i}.txt"
        temp_dir = PROJECT_ROOT / "data" / "intermediate" / "gold_eval_tmp" / split / query_type / str(i)

        if not pred_path.exists():
            print(f"[{i}/{len(examples)}] missing prediction: {pred_path}")
            continue

        predicted_query = read_text(pred_path)
        index_name = extract_index_name(example)

        result = evaluate_prediction_to_files(
            es_client=es_client,
            predicted_query=predicted_query,
            gold_example=example,
            index_name=index_name,
            pred_result_path=pred_result_path,
            gold_result_path=gold_result_path,
        )

        correct += result["score"]
        total += 1

        print(f"[{i}/{len(examples)}] score={result['score']}")

    accuracy = (correct / total) if total > 0 else 0.0

    print()
    print("Batch evaluation completed.")
    print(f"Input file      : {input_path}")
    print(f"Prediction dir  : {prediction_dir}")
    print(f"Pred result dir : {pred_result_dir}")
    print(f"Gold result dir : {gold_result_dir}")
    print(f"Correct         : {correct}")
    print(f"Total           : {total}")
    print(f"Execution Acc.  : {accuracy:.4f}")


def run_single_evaluation(
    split: str,
    query_type: str,
    example_id: int,
    predicted_query: str,
):
    input_path = PROJECT_ROOT / DATA_CONFIG[split][query_type]
    examples = load_jsonl(input_path)

    if example_id < 1 or example_id > len(examples):
        raise IndexError(f"example_id must be between 1 and {len(examples)}")

    example = examples[example_id - 1]
    index_name = extract_index_name(example)
    es_client = create_es_client()

    pred_result_path = PROJECT_ROOT / "data" / "results" / "_single" / query_type / f"{example_id}_pred.txt"
    gold_result_path = PROJECT_ROOT / "data" / "results" / "_single" / query_type / f"{example_id}_gold.txt"
    temp_dir = PROJECT_ROOT / "data" / "intermediate" / "gold_eval_tmp" / "_single" / query_type / str(example_id)

    result = evaluate_prediction_to_files(
        es_client=es_client,
        predicted_query=predicted_query,
        gold_example=example,
        index_name=index_name,
        pred_result_path=pred_result_path,
        gold_result_path=gold_result_path,
    )

    print("Single evaluation completed.")
    print(f"Score        : {result['score']}")
    print()
    print("=== Gold Query ===")
    print(result["gold_query"])
    print()
    print("=== Predicted Result ===")
    print(result["pred_result"])
    print()
    print("=== Gold Result ===")
    print(result["gold_result"])


def main():
    mode = "batch"              # "single" | "batch"

    if mode == "batch":
        split = "test"          # "train" | "validation" | "test"
        query_type = "basic"    # "basic" | "agg" | "knn"
        setting = "zero_shot"   # "zero_shot" | "few_shot" | "finetuned"
        adapter = None          # None | "basic" | "agg" | "knn" | "mixed"
        run_id = 1

        run_batch_evaluation(
            split=split,
            query_type=query_type,
            setting=setting,
            adapter=adapter,
            run_id=run_id,
        )
        return

    if mode == "single":
        split = "test"
        query_type = "basic"
        example_id = 1

        predicted_query = """
{
  "query": {
    "match_all": {}
  }
}
""".strip()

        run_single_evaluation(
            split=split,
            query_type=query_type,
            example_id=example_id,
            predicted_query=predicted_query,
        )
        return

    raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()