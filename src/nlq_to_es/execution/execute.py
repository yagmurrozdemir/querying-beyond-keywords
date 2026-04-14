import ast
import json
import os
from typing import Any, Dict, Union

from elasticsearch import Elasticsearch
from nlq_to_es.config import ELASTIC_CONFIG


def create_es_client() -> Elasticsearch:
    return Elasticsearch(
        [ELASTIC_CONFIG["host"]],
        basic_auth=(
            ELASTIC_CONFIG["username"],
            ELASTIC_CONFIG["password"]
        ),
        verify_certs=ELASTIC_CONFIG["verify_certs"],
        ssl_show_warn=False,
    )


def strip_markdown_fences(raw_text: str) -> str:
    """
    Remove markdown code fences like ```json ... ``` if present.
    """
    lines = raw_text.strip().splitlines()

    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]

    while lines and not lines[-1].strip():
        lines.pop()

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def parse_query_payload(query_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Accept either:
      - a Python dict
      - a raw JSON string (possibly wrapped in markdown fences)

    Return a parsed Elasticsearch DSL dict.
    """
    if isinstance(query_input, dict):
        return query_input

    if not isinstance(query_input, str):
        raise TypeError("query_input must be either a dict or a JSON string.")

    cleaned = strip_markdown_fences(query_input)

    if not cleaned:
        raise ValueError("Query input is empty after cleaning.")

    return json.loads(cleaned)


def normalize_es_response(response: Dict[str, Any]) -> Any:
    """
    Convert Elasticsearch response into the simplified format
    used by your evaluation pipeline.
    """
    es_result = []

    if "aggregations" not in response:
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            if isinstance(source, dict):
                es_result.extend(list(source.values()))
            else:
                parsed = ast.literal_eval(f"{source}")
                es_result.extend(list(parsed.values()))
    else:
        for _, agg_data in response["aggregations"].items():
            if isinstance(agg_data, dict):
                if "value" in agg_data:
                    es_result.append(agg_data["value"])
                elif "buckets" in agg_data:
                    es_result.append(agg_data["buckets"])

    if len(es_result) == 1:
        return es_result[0]

    return es_result


def execute_query(
    es_client: Elasticsearch,
    index_name: str,
    query_input: Union[str, Dict[str, Any]],
    check_index_exists: bool = True,
) -> Any:
    """
    Parse, execute, and normalize an Elasticsearch query.
    Works for both standard and k-NN queries, as long as the
    input is already fully materialized valid DSL.
    """
    payload = parse_query_payload(query_input)

    if check_index_exists and not es_client.indices.exists(index=index_name):
        raise ValueError(f"Index not found: {index_name}")

    response = es_client.search(index=index_name, **payload)
    return normalize_es_response(response)


def safe_execute_query(
    es_client: Elasticsearch,
    index_name: str,
    query_input: Union[str, Dict[str, Any]],
    check_index_exists: bool = True,
) -> Any:
    """
    Same as execute_query, but returns readable error strings
    instead of raising exceptions. Useful for batch evaluation.
    """
    try:
        return execute_query(
            es_client=es_client,
            index_name=index_name,
            query_input=query_input,
            check_index_exists=check_index_exists,
        )
    except json.JSONDecodeError as e:
        return f"Invalid JSON query: {e}"
    except exceptions.ApiError as e:
        status = getattr(e, "status_code", "unknown")
        error = getattr(e, "error", "unknown")
        message = getattr(e, "message", str(e))
        return f"Elasticsearch ApiError: {status} {error}: {message}"
    except Exception as e:
        return f"Unexpected error: {type(e).__name__}: {e}"
