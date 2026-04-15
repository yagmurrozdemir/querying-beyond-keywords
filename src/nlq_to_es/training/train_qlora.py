import os
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from nlq_to_es.config import TRAINING_CONFIG, FT_PROMPT_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_text_and_labels(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
):
    """
    Expects each JSONL line to contain:
    {"messages": [{"role":"system","content":"..."},
                  {"role":"user","content":"..."},
                  {"role":"assistant","content":"..."}]}

    Only assistant completion tokens are used for loss.
    System + user tokens are masked with -100.
    """
    messages: List[Dict[str, str]] = example["messages"]

    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Each example must end with an assistant message.")

    prompt_messages = messages[:-1]
    assistant_msg = messages[-1]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_messages = prompt_messages + [assistant_msg]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))

    input_ids = torch.tensor(
        [pad(x["input_ids"], 0) for x in batch],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [pad(x["attention_mask"], 0) for x in batch],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [pad(x["labels"], -100) for x in batch],
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_adapter_data_paths(adapter: str):
    return {
        "train_file": str(Path(FT_PROMPT_CONFIG["train"][adapter])),
        "validation_file": str(Path(FT_PROMPT_CONFIG["validation"][adapter]))
    }



def train_adapter(adapter: str):
    train_cfg = TRAINING_CONFIG
    data_cfg = get_adapter_data_paths(adapter)
    output_dir = train_cfg["output_path"][adapter]

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg["model_name_or_path"],
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        train_cfg["model_name_or_path"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {"train": data_cfg["train_file"]}
    if Path(data_cfg["validation_file"]).exists():
        data_files["validation"] = data_cfg["validation_file"]

    ds = load_dataset("json", data_files=data_files)

    def map_fn(ex):
        return build_text_and_labels(ex, tokenizer, train_cfg["max_length"])

    train_ds = ds["train"].map(
        map_fn,
        remove_columns=ds["train"].column_names,
    )

    eval_ds = None
    if "validation" in ds:
        eval_ds = ds["validation"].map(
            map_fn,
            remove_columns=ds["validation"].column_names,
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=True,
        fp16=False,
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=train_cfg["eval_steps"] if eval_ds is not None else None,
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        choices=["basic", "agg", "knn", "mixed"],
    )
    args = parser.parse_args()

    train_adapter(args.adapter)


if __name__ == "__main__":
    main()