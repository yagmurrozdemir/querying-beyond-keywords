from pathlib import Path
import json
import ast

SYSTEM_MESSAGE = """You are an expert assistant for generating Elasticsearch queries.

Given a natural language query and an Elasticsearch index mapping, your task is to generate a valid Elasticsearch DSL query that correctly answers the request.
""".strip()


def normalize_dsl_to_json_text(raw: str, pretty: bool = True) -> str:
    raw = raw.strip()

    try:
        obj = json.loads(raw)
        return json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)
    except Exception:
        pass

    try:
        obj = ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"Invalid format: {e}")

    return json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)

def generate_prompt(index_mapping: str, nl_query: str, dsl_query: str):

    dsl_json_text = normalize_dsl_to_json_text(dsl_query)

    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"{nl_query}\n\n{index_mapping}"},
            {"role": "assistant", "content": dsl_json_text},
        ]
    }

    return example
def generate_inference_prompt(index_mapping: str, nl_query: str):

    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"{nl_query}\n\n{index_mapping}"}
        ]
    }

    return example