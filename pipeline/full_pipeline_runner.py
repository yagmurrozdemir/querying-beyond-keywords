#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path
from compare_results import compare_files  # Updated name

def run(cmd_list):
    subprocess.run(cmd_list, check=True)

def main(index_name, nlq, json_data):
    # Ensure outputs directory exists
    Path("outputs").mkdir(exist_ok=True)

    # 1. index mapping prep
    run(["python3", "export_index_mapping.py", index_name, "outputs/mapping.txt"])

    # 2. LLM DSL generation
    run(["python3", "generate_query_from_nlq.py", nlq, "outputs/mapping.txt", "outputs/llm_response.txt"])

    # 3. Embedding
    run(["python3", "generate_embedding.py", "outputs/llm_response.txt", "outputs/embedding.txt"])

    # 4. Inject embedding into query
    run(["python3", "inject_embedding_into_query.py", "outputs/llm_response.txt", "outputs/embedding.txt", "outputs/query.txt"])

    # 5. Execute query
    run(["python3", "execute_query.py", index_name, "outputs/query.txt", "outputs/final_result.txt"])

    # 6. Execute ground truth query
    run(["python3", "run_correct_query.py", json_data, "outputs/correct_result.txt"])

    print("✅ Query run complete")

if __name__ == "__main__":
    total = 0
    groups = ["agg", "other", "knn"]
    for group in groups:
        correct_count = 0
        input_file = f"data/TEST_SET/{group}_query.jsonl"

        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                data = json.loads(line)
                table_id = data["table_id"]
                index_name = f"table{table_id.replace('-', '_')[1:]}"
                nlq = data["question"]
                json_data = json.dumps(data)

                try:
                    total += 1
                    main(index_name, nlq, json_data)
                    correct_count += compare_files("outputs/correct_result.txt", "outputs/final_result.txt")
                except subprocess.CalledProcessError as e:
                    print(f"Pipeline failed at step: {e.cmd}", file=sys.stderr)
                    sys.exit(1)

                print(f"{correct_count} / {total}")
