#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.config import DATA_CONFIG, PREDICTIONS_DIR, RESULTS_DIR
from nlq_to_es.execution.execute import create_es_client
from nlq_to_es.evaluation.execution_evaluator import evaluate_prediction
from nlq_to_es.generation.inference_pipeline import extract_index_name, get_model_tag
from nlq_to_es.io.readers import load_jsonl, read_text
from nlq_to_es.io.writers import write_text


TEST_SPLIT = "test"
VALID_QUERY_TYPES = {"basic", "agg", "knn"}
VALID_SETTINGS = {"zero_shot", "few_shot", "finetuned"}
VALID_ADAPTERS = {"basic", "agg", "knn", "mixed"}

def get_prediction_dir(query_type: str, setting: str, run_id: int, adapter=None) -> Path:
    model_tag = get_model_tag(setting=setting, adapter=adapter)
    return PREDICTIONS_DIR / setting / model_tag / TEST_SPLIT / query_type / f"run_{run_id}"


def get_result_dirs(query_type: str, setting: str, run_id: int, adapter=None):
    model_tag = get_model_tag(setting=setting, adapter=adapter)

    pred_dir = RESULTS_DIR / setting / model_tag / TEST_SPLIT / query_type / f"run_{run_id}"
    gold_dir = RESULTS_DIR / "gold" / TEST_SPLIT / query_type

    pred_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    return pred_dir, gold_dir


def run_batch_evaluation(
    query_type: str,
    setting: str,
    run_id: int,
    adapter=None,
) -> None:
    input_path = DATA_CONFIG[TEST_SPLIT][query_type]
    prediction_dir = get_prediction_dir(
        query_type=query_type,
        setting=setting,
        run_id=run_id,
        adapter=adapter,
    )
    pred_result_dir, gold_result_dir = get_result_dirs(
        query_type=query_type,
        setting=setting,
        run_id=run_id,
        adapter=adapter,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    examples = load_jsonl(input_path)
    es_client = create_es_client()

    correct = 0
    total = 0
    missing = 0

    for i, example in enumerate(examples, start=1):
        pred_path = prediction_dir / f"{i}.out"
        pred_result_path = pred_result_dir / f"{i}.txt"
        gold_result_path = gold_result_dir / f"{i}.txt"

        if not pred_path.exists():
            missing += 1
            print(f"[{i}/{len(examples)}] missing prediction: {pred_path}")
            continue

        predicted_query = read_text(pred_path)
        index_name = extract_index_name(example)

        result = evaluate_prediction(
            es_client=es_client,
            predicted_query=predicted_query,
            gold_example=example,
            index_name=index_name,
        )

        write_text(pred_result_path, result["pred_result"])
        write_text(gold_result_path, result["gold_result"])

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
    print(f"Missing         : {missing}")
    print(f"Execution Acc.  : {accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation for predictions."
    )

    parser.add_argument(
        "--query-type",
        required=True,
        choices=sorted(VALID_QUERY_TYPES),
        help="Query type to evaluate.",
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=sorted(VALID_SETTINGS),
        help="Evaluation setting.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run identifier (e.g., 1–5).",
    )
    parser.add_argument(
        "--adapter",
        choices=sorted(VALID_ADAPTERS),
        default=None,
        help="Adapter name (only for finetuned setting).",
    )

    args = parser.parse_args()

    if args.run_id < 1:
        parser.error("--run-id must be a positive integer.")

    if args.setting == "finetuned" and args.adapter is None:
        parser.error("--adapter is required when --setting=finetuned.")

    if args.setting != "finetuned" and args.adapter is not None:
        parser.error("--adapter should only be used with --setting=finetuned.")

    return args


def main() -> None:
    args = parse_args()

    run_batch_evaluation(
        query_type=args.query_type,
        setting=args.setting,
        run_id=args.run_id,
        adapter=args.adapter,
    )

if __name__ == "__main__":
    main()