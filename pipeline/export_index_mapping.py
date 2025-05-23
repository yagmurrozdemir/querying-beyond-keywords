from elasticsearch import Elasticsearch
import os
import sys
import time
import requests


def wait_for_elasticsearch():
    host = os.getenv("ELASTIC_HOST", "http://localhost:9200")
    print(f"⏳ Waiting for Elasticsearch at {host}...")
    for _ in range(30):
        try:
            r = requests.get(host)
            if r.status_code == 200:
                print("✅ Elasticsearch is ready.")
                return
        except:
            pass
        time.sleep(2)
    raise Exception("❌ Elasticsearch is not reachable after 60 seconds.")


def create_es_connection() -> Elasticsearch:
    ES_HOST = os.getenv("ELASTIC_HOST", "http://localhost:9200")
    return Elasticsearch(ES_HOST)

def main(index_name, output_file_path):
    # Get index mapping
    wait_for_elasticsearch()
    es_client = create_es_connection()
    mapping = es_client.indices.get_mapping(index=index_name)
    # Save the output to a file
    with open(output_file_path, 'w') as f:
        f.write(str(mapping))

if __name__ == "__main__":

    index_name = sys.argv[1]
    output_file_path = sys.argv[2]
    main(index_name, output_file_path)