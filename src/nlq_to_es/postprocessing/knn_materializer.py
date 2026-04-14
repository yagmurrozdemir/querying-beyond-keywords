def insert_embedding(raw_output: str, embedding: str):
    if "~" not in raw_output:
        return raw_output.strip()

    # clean markdown if present
    result = raw_output.replace("```json", "").replace("```", "").strip()

    query_template, _ = result.split("~", 1)

    query = query_template.replace('"$vector$"', embedding)
    query = query.replace('$vector$', embedding)

    # optional: adjust similarity
    query = query.replace('"similarity": 0.98', '"similarity": 0.90')

    return query.strip()
