import sys
import json
from pathlib import Path
import ast
from elasticsearch import Elasticsearch, exceptions

# --- Use ELASTIC_HOST from environment (default to localhost:9200)
import os
ES_HOST = os.getenv("ELASTIC_HOST", "http://localhost:9200")
es = Elasticsearch(ES_HOST)

def main(index_name, input_file_path, output_file_path):
    # Ensure the output directory exists
    out_path = Path(output_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        lines = Path(input_file_path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        out_path.touch()  # create an empty file
        print(f"[warn] {input_file_path} not found → created empty {output_file_path}")
        return

    # Remove Markdown fences (``` or ```json)
    if lines and lines[0].lstrip().startswith("```"):
        print(f"[info] Stripping Markdown fence: {lines[0].strip()}")
        lines = lines[1:]

    while lines and not lines[-1].strip():
        lines.pop()

    if lines and lines[-1].strip() == "```":
        print(f"[info] Stripping closing fence: {lines[-1].strip()}")
        lines = lines[:-1]

    raw = "\n".join(lines).strip()
    if not raw:
        print("⚠️ No JSON found in input file.")
        out_path.write_text("No JSON found between fences.")
        return

    try:
        query_template = json.loads(raw)
    except json.JSONDecodeError as e:
        print("❌ Invalid JSON:", e)
        out_path.write_text("Invalid JSON format in input query file.")
        return

    try:
        response = es.search(index=index_name, body=query_template)

        results = []
        if "aggregations" not in response:
            for hit in response.get("hits", {}).get("hits", []):
                parsed = ast.literal_eval(f"{hit['_source']}")
                results.extend(list(parsed.values()))
        else:
            for agg_data in response.get("aggregations", {}).values():
                if "value" in agg_data:
                    results.append(agg_data["value"])
                elif "buckets" in agg_data:
                    results.append(agg_data["buckets"])

        if results:
            results = results if len(results) > 1 else results[0]
            out_path.write_text(json.dumps(results, indent=2))
        else:
            out_path.write_text("[]")

    except exceptions.BadRequestError as e:
        out_path.write_text("BadRequestError: " + str(e.info))
    except exceptions.ApiError as e:
        out_path.write_text("ApiError: " + str(e.info))
    except Exception as e:
        out_path.write_text("Unexpected error: " + str(e))

# --- Script mode: accept args from subprocess.run
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python elastic.py <index_name> <input_file> <output_file>")
        sys.exit(1)

    index_name = sys.argv[1]
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    main(index_name, input_file_path, output_file_path)
