"""Inference module: serve Qwen base model with LoRA hot-swap."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTERS_DIR = Path("/app/data/adapters")


class InferenceEngine:
    """Manages base model + swappable LoRA adapters."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_adapter_id = None
        self._base_loaded = False

    def load_base(self):
        """Load the base Qwen model (once)."""
        if self._base_loaded:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,
        )
        self._base_loaded = True
        self.current_adapter_id = None

    def load_adapter(self, adapter_id: str, adapter_path: str):
        """Load or swap a LoRA adapter on top of the base model."""
        if not self._base_loaded:
            self.load_base()

        if self.current_adapter_id == adapter_id:
            return  # Already loaded

        # If an adapter is already attached, unload it first
        if self.current_adapter_id is not None:
            self.model = self.model.unload()

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.current_adapter_id = adapter_id

    def unload_adapter(self):
        """Go back to base model only."""
        if self.current_adapter_id is not None:
            self.model = self.model.unload()
            self.current_adapter_id = None

    def generate(
        self,
        instruction: str,
        input_text: str,
        adapter_id: str = None,
        adapter_path: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Run inference with optional LoRA adapter."""
        if not self._base_loaded:
            self.load_base()

        if adapter_id and adapter_path:
            self.load_adapter(adapter_id, adapter_path)
        elif adapter_id is None and self.current_adapter_id is not None:
            self.unload_adapter()

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def batch_generate(
        self,
        pairs: list[dict],
        adapter_id: str = None,
        adapter_path: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> list[str]:
        """Run inference on a batch of instruction/input pairs."""
        results = []
        for pair in pairs:
            result = self.generate(
                instruction=pair["instruction"],
                input_text=pair["input"],
                adapter_id=adapter_id,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(result)
        return results


# Singleton
engine = InferenceEngine()
