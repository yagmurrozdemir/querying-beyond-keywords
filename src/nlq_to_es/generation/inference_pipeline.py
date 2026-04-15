import re
from pathlib import Path

from nlq_to_es.config import MODEL_CONFIG
from nlq_to_es.generation.zero_shot_prompt_generator import generate_prompt as generate_zero_shot_prompt
from nlq_to_es.generation.few_shot_prompt_generator import generate_prompt as generate_few_shot_prompt
from nlq_to_es.generation.ollama_response_generator import get_ollama_response
from nlq_to_es.io.readers import read_text
from nlq_to_es.postprocessing.knn_pipeline import process_knn_output


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)



def extract_nlq(example: dict) -> str:
    return example['question']



def extract_index_name(example: dict) -> str:
    table_id = example['table_id']
    index_name = f"table{table_id.replace('-', '_')[1:]}"
    return index_name

def get_model_tag(setting: str, adapter=None) -> str:
    if setting == "finetuned":
        if adapter is None:
            raise ValueError("adapter must be provided for finetuned inference.")
        return f"{adapter}_adapter"

    model_name = MODEL_CONFIG[setting]["model_name"]
    return sanitize_name(model_name)


def load_model_bundle(setting: str, adapter=None):
    if setting != "finetuned":
        return None

    if adapter is None:
        raise ValueError("adapter must be provided when setting='finetuned'.")

    from nlq_to_es.generation.finetuned_response_generator import load_finetuned_model

    return load_finetuned_model(setting=setting, adapter=adapter)


def build_prompt(
    index_mapping: str,
    nlq: str,
    setting: str,
    project_root: Path,
    model_bundle=None,
) -> str:
    if setting == "zero_shot":
        template = read_text(project_root / "prompts" / "templates" / "zero_shot_prompt_template.txt")
        return generate_zero_shot_prompt(
            index_mapping=index_mapping,
            nl_query=nlq,
            template=template,
        )

    if setting == "few_shot":
        template = read_text(project_root / "prompts" / "templates" / "few_shot_prompt_template.txt")
        return generate_few_shot_prompt(
            index_mapping=index_mapping,
            nl_query=nlq,
            template=template,
        )

    if setting == "finetuned":
        if model_bundle is None:
            raise ValueError("model_bundle is required for finetuned prompt generation.")

        from nlq_to_es.generation.finetuning_prompt_generator import SYSTEM_MESSAGE

        tokenizer = model_bundle["tokenizer"]
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"{nlq}\n\n{index_mapping}"},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError(f"Unsupported setting: {setting}")


def generate_raw_output(
    prompt_text: str,
    setting: str,
    adapter=None,
    model_bundle=None,
) -> str:
    if setting in {"zero_shot", "few_shot"}:
        return get_ollama_response(prompt_text=prompt_text, setting=setting)

    if setting == "finetuned":
        from nlq_to_es.generation.finetuned_response_generator import get_finetuned_response

        return get_finetuned_response(
            prompt_text=prompt_text,
            setting=setting,
            adapter=adapter,
            model_bundle=model_bundle,
        )

    raise ValueError(f"Unsupported setting: {setting}")


def postprocess_output(raw_output: str, query_type: str) -> str:
    if query_type == "knn":
        return process_knn_output(raw_output)
    return raw_output.strip()