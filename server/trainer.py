"""LoRA training module: fine-tune Qwen 2.5 7B with PEFT on generated JSONL."""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

from server.database import update_adapter

ADAPTERS_DIR = Path("/app/data/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_chatml_dataset(jsonl_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    """Load ChatML JSONL and tokenize for training."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            samples.append(entry)

    def tokenize(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)

        # Build labels: mask system + user tokens, only train on assistant response
        labels = encoded["input_ids"].copy()

        # Find where assistant response starts by encoding everything before it
        non_assistant = messages[:-1]  # system + user
        prefix_text = tokenizer.apply_chat_template(
            non_assistant, tokenize=False, add_generation_prompt=True
        )
        prefix_len = len(tokenizer(prefix_text, truncation=True, max_length=max_length)["input_ids"])

        # Mask prefix tokens with -100
        labels[:prefix_len] = [-100] * prefix_len
        encoded["labels"] = labels

        return encoded

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(tokenize, remove_columns=["messages"])
    return dataset


def train_lora(
    adapter_id: str,
    dataset_path: str,
    base_model: str = None,
    resume_from: str = None,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    max_length: int = 2048,
    gradient_accumulation: int = 4,
):
    """Fine-tune base model with LoRA adapter on the given dataset.

    If resume_from is set (path to existing LoRA adapter), loads those weights
    and continues training on the new dataset instead of starting fresh.
    """
    base_model = base_model or BASE_MODEL
    adapter_dir = ADAPTERS_DIR / adapter_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    update_adapter(adapter_id, status="training")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset = load_chatml_dataset(dataset_path, tokenizer, max_length)
        total_samples = len(dataset)

        # Split 90/10
        split = dataset.train_test_split(test_size=0.1, seed=42)

        # Load model in 4-bit for T4 compatibility
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,
        )

        if resume_from and Path(resume_from).exists():
            # Continue training: load existing LoRA weights
            model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        else:
            # Fresh training: create new LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config)

        # Training args
        training_args = TrainingArguments(
            output_dir=str(adapter_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        result = trainer.train()

        # Save adapter
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # Update registry
        final_loss = result.training_loss
        update_adapter(
            adapter_id,
            status="ready",
            lora_path=str(adapter_dir),
            train_samples=total_samples,
            train_epochs=epochs,
            train_loss=final_loss,
        )

        return {
            "status": "ready",
            "samples": total_samples,
            "loss": final_loss,
            "adapter_path": str(adapter_dir),
        }

    except Exception as e:
        update_adapter(adapter_id, status="failed", metadata=json.dumps({"error": str(e)}))
        raise
