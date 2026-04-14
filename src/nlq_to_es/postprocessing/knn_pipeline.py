from .knn_embedding import extract_text_from_output, get_embedding
from .knn_materializer import insert_embedding


def process_knn_output(raw_output: str, embedding_model: str = "clip"):
    text = extract_text_from_output(raw_output)

    # no k-NN case
    if text is None:
        return raw_output.strip()

    embedding = get_embedding(embedding_model, text)

    final_query = insert_embedding(raw_output, str(embedding))

    return final_query
