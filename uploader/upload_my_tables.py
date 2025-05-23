import os
import json
import time
import requests
import re
from elasticsearch import Elasticsearch, helpers

# --- Environment Config ---
ES_HOST = os.getenv("ELASTIC_HOST", "http://localhost:9200")
JSONL_FILE = "tables.jsonl"  # Ensure this file is present in the same directory

# --- Wait for Elasticsearch to be ready ---
for _ in range(30):
    try:
        res = requests.get(ES_HOST)
        if res.status_code == 200:
            print("✅ Elasticsearch is up.")
            break
    except:
        pass
    print("⏳ Waiting for Elasticsearch...")
    time.sleep(2)
else:
    raise Exception("❌ Elasticsearch not reachable.")

# --- Set cluster setting (no auth required in unsecured mode) ---
print("⚙️ Setting max_shards_per_node...")
requests.put(
    f"{ES_HOST}/_cluster/settings",
    headers={"Content-Type": "application/json"},
    json={"persistent": {"cluster.max_shards_per_node": 10000}}
)

# --- Initialize Elasticsearch client ---
es = Elasticsearch(ES_HOST)

# --- Define helper functions ---

def create_index_with_mapping(es_client, index_name, headers, types):
    properties = {}

    for header, t in zip(headers, types):
        if not header:
            continue
        header = header.replace('No.', 'No')
        if t == "text":
            field_mapping = {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            }
        elif t == "dense_vector":
            field_mapping = {"type": "dense_vector", "dims": 512}
        else:  # Numeric or real
            field_mapping = {"type": "double"}

        properties[header] = field_mapping

    mapping = {
        "settings": {"index.mapping.ignore_malformed": True},
        "mappings": {
            "dynamic": "strict",
            "properties": properties
        }
    }

    if es_client.indices.exists(index=index_name):
        print(f"ℹ️ Index {index_name} already exists.")
    else:
        es_client.indices.create(index=index_name, body=mapping)
        print(f"✅ Created index: {index_name}")

def upload_table_to_index(es_client, index_name, headers, rows, types):
    actions = []
    for row in rows:
        doc = {}
        for i, (header, t) in enumerate(zip(headers, types)):
            value = row[i]
            if value == "" or value is None:
                value = None
            elif t == "real" and isinstance(value, str):
                value = value.replace(',', '.')
            doc[header] = value

        actions.append({
            "_index": index_name,
            "_source": doc
        })

    try:
        success, failed = helpers.bulk(es_client, actions, raise_on_error=False)
        print(f"📦 Uploaded {success} docs to {index_name}. Failed: {len(failed)}")
        if failed:
            print("❗ Some documents failed:", failed)
    except Exception as e:
        print(f"❌ Bulk upload error: {e}")

# --- Main upload loop ---
with open(JSONL_FILE, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        if 'header' in data and 'id' in data:
            table_id = f"table{data['id'].replace('-', '_')[1:]}"
            headers = data["header"]
            types = data["types"]
            rows = data["rows"]

            create_index_with_mapping(es, table_id, headers, types)
            upload_table_to_index(es, table_id, headers, rows, types)
