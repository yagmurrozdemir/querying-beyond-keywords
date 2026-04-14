from elasticsearch import Elasticsearch
from nlq_to_es.config import ELASTIC_CONFIG


def create_es_client():
    return Elasticsearch(
        [ELASTIC_CONFIG["host"]],
        basic_auth=(
            ELASTIC_CONFIG["username"],
            ELASTIC_CONFIG["password"]
        ),
        verify_certs=ELASTIC_CONFIG["verify_certs"],
    )


def get_index_mapping(index_name):
    es_client = create_es_client()
    mapping = es_client.indices.get_mapping(index=index_name)
    return str(mapping)
