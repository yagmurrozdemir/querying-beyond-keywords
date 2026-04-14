import importlib.util

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from nlq_to_es.config import MODEL_CONFIG, BACKEND_CONFIG


def load_finetuned_model(
    setting: str = "finetuned",
    adapter: str = "basic"
):
    model_cfg = MODEL_CONFIG[setting]

    if model_cfg["backend"] != "huggingface":
        raise ValueError(f"Setting '{setting}' is not configured for Hugging Face.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_name"],
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None

    if has_bitsandbytes:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model_name"],
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        base = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model_name"],
            torch_dtype=torch_dtype,
        )

        if torch.backends.mps.is_available():
            base = base.to("mps")
        elif torch.cuda.is_available():
            base = base.to("cuda")
        else:
            base = base.to("cpu")

    model = PeftModel.from_pretrained(base, model_cfg["adapters"][adapter])
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


@torch.inference_mode()
def get_finetuned_response(
    prompt_text: str,
    setting: str = "finetuned",
    adapter: str = "basic",
    model_bundle=None,
):
    model_cfg = MODEL_CONFIG[setting]

    if model_cfg["backend"] != "huggingface":
        raise ValueError(f"Setting '{setting}' is not configured for Hugging Face.")

    if model_bundle is None:
        model_bundle = load_finetuned_model(setting, adapter)

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = model_cfg["temperature"] > 0.0

    generate_kwargs = {
        "max_new_tokens": model_cfg["max_new_tokens"],
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = model_cfg["temperature"]
        generate_kwargs["top_p"] = model_cfg["top_p"]

    gen = model.generate(
        **inputs,
        **generate_kwargs,
    )

    full = tokenizer.decode(gen[0], skip_special_tokens=True)

    response = full
    if full.startswith(prompt_text):
        response = full[len(prompt_text):]

    return response.strip()