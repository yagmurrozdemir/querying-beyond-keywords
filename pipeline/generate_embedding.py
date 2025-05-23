import os
import requests
import sys

CLIP_HOST = os.getenv("CLIP_HOST", "http://localhost:8000")

def get_embedding(text):
    response = requests.post(
        f"{CLIP_HOST}/text-embedding/",
        json={"text": text},
        headers={"Content-Type": "application/json"}
    )
    return response.json()["embedding"][0]

def main(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        result = f.read().strip()

    if "~" not in result:
        with open(output_file_path, 'w') as f:
            f.write('')
        return

    query_template, text = result.split('~', 1)
    output = str(get_embedding(text))

    with open(output_file_path, 'w') as f:
        f.write(output)

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    main(input_file_path, output_file_path)
