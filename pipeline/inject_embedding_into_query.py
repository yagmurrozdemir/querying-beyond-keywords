import sys


def main(input_file1_path, input_file2_path, output_file_path):
    # Read input from task_1
    with open(input_file1_path, 'r') as f:
        result = f.read().strip()

    result = result.replace("```json", "").replace("```", "").strip()


    if "~" not in result:
        with open(output_file_path, 'w') as f:
            f.write(str(result))  # Write an empty file
        return

    query_template, text = result.split('~', 1)

    with open(input_file2_path, 'r') as f:
        embedding_str = f.read().strip()

    query_template_with_vector = query_template.replace('"$vector$"', embedding_str)
    query_template_with_vector = query_template_with_vector.replace('$vector$', embedding_str)

    # Save the result to a file
    with open(output_file_path, 'w') as f:
        f.write(str(query_template_with_vector))

if __name__ == "__main__":
    input_file1_path = sys.argv[1]
    input_file2_path = sys.argv[2]
    output_file_path = sys.argv[3]
    main(input_file1_path, input_file2_path, output_file_path)    

