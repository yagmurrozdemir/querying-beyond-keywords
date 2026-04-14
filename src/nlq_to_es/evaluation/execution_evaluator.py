import ast
import json
from pathlib import Path

from nlq_to_es.evaluation.compare_results import compare_files, compare
from nlq_to_es.io.readers import read_text
from nlq_to_es.io.writers import write_text
from nlq_to_es.postprocessing.gold_query_materializer import convert_to_elasticsearch_dsl
from nlq_to_es.postprocessing.knn_pipeline import process_knn_output


def safe_execute(es_client, index_name: str, query_text: str):
    try:
        from nlq_to_es.execution.execute import execute_query

        return execute_query(es_client=es_client, index_name=index_name, query_input=query_text)

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def evaluate_prediction(
    es_client,
    predicted_query: str,
    gold_example: dict,
    index_name: str,
):
    pred_result = safe_execute(es_client, index_name, predicted_query)

    try:
        #gold_query = materialize_gold_query(gold_example)
        gold_query = convert_to_elasticsearch_dsl(gold_example)
        gold_result = safe_execute(es_client, index_name, gold_query)
    except Exception as e:
        gold_query = None
        gold_result = f"ERROR: {type(e).__name__}: {e}"



    score = compare(pred_result, gold_result)

    return {
        "score": score,
        "pred_result": pred_result,
        "gold_result": gold_result,
        "gold_query": gold_query,
    }
