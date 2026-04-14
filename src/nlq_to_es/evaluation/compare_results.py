from nlq_to_es.io.readers import read_text_file

def compare(content1, content2):
    if content1 == content2:
        return 1
    else:
        return 0

def compare_files(file1_path, file2_path):
    content1 = read_text_file(file1_path)
    content2 = read_text_file(file2_path)

    if content1 == content2:
        return 1
    else:
        return 0
