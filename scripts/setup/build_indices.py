#!/usr/bin/env python3
import json
from pathlib import Path

from elasticsearch import helpers
import sys

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlq_to_es.config import TABLE_UPLOAD_CONFIG
from nlq_to_es.execution.execute import create_es_client


def create_index_with_mapping(es_client, index_name, headers, types):
    properties = {}

    for header, field_type in zip(headers, types):
        if not header:
            continue

        header = header.replace("No.", "No")

        if field_type == "text":
            field_mapping = {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            }
        elif field_type == "dense_vector":
            field_mapping = {
                "type": "dense_vector",
                "dims": 512
            }
        else:
            field_mapping = {
                "type": "double"
            }

        properties[header] = field_mapping

    mapping = {
        "settings": {
            "index.mapping.ignore_malformed": True
        },
        "mappings": {
            "dynamic": "strict",
            "properties": properties
        }
    }

    if es_client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists. Skipping creation.")
        return

    es_client.indices.create(index=index_name, body=mapping)


def upload_table_to_index(es_client, index_name, headers, rows, types):
    actions = []

    for row in rows:
        doc = {}

        for header, field_type, value in zip(headers, types, row):
            header = header.replace("No.", "No")

            if value == "" or value is None:
                value = None
            elif field_type == "real" and not isinstance(value, (int, float)):
                value = value.replace(",", ".")

            doc[header] = value

        actions.append({
            "_index": index_name,
            "_source": doc
        })

    try:
        success, failed = helpers.bulk(es_client, actions, raise_on_error=False)
        if failed:
            print(f"Failed documents for {index_name}: {failed}")
    except Exception as e:
        print(f"Bulk upload error for {index_name}: {e}")


def iter_tables(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())

            if "header" not in data:
                continue

            yield data


def build_index_name(table_id):
    return f'table{table_id.replace("-", "_")[1:]}'


def main():
    es_client = create_es_client()
    jsonl_file = PROJECT_ROOT / TABLE_UPLOAD_CONFIG["jsonl_file"]
    
    for data in iter_tables(jsonl_file):
        index_name = build_index_name(data["id"])
        headers = data["header"]
        types = data["types"]
        rows = data["rows"]

        create_index_with_mapping(es_client, index_name, headers, types)
        upload_table_to_index(es_client, index_name, headers, rows, types)


if __name__ == "__main__":
    main()