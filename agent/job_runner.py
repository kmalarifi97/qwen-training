"""Job runner — executes training and inference jobs on the GPU."""

import asyncio
import json
import logging
import os
import zipfile
from pathlib import Path

import httpx
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

logger = logging.getLogger("agent.job_runner")


class JobRunner:
    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    async def run_job(self, job_data: dict, progress_callback) -> dict:
        """Execute a job. Returns {"status": ..., "result": ...}."""
        self._cancelled = False
        job_id = job_data["job_id"]
        job_type = job_data.get("job_type", "train")

        try:
            config = json.loads(job_data.get("config", "{}"))
        except (json.JSONDecodeError, TypeError):
            config = {}

        # Clear GPU memory before each job
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        try:
            if job_type == "train":
                return await self._run_training(job_id, config, progress_callback)
            elif job_type == "inference":
                return await self._run_inference(job_id, config, progress_callback)
            else:
                return {"status": "failed", "error": f"Unknown job type: {job_type}"}
        except Exception as e:
            logger.error(f"Job {job_id[:8]} failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    async def _run_training(self, job_id: str, config: dict, progress_callback) -> dict:
        """Download dataset, train LoRA, upload adapter."""
        server_base = config["server_base_url"]
        dataset_url = config["dataset_url"]
        adapter_upload_url = config["adapter_upload_url"]
        base_model = config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        epochs = config.get("epochs", 3)
        batch_size = config.get("batch_size", 1)
        lr = config.get("learning_rate", 2e-4)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 16)
        max_length = config.get("max_length", 512)
        gradient_accumulation = config.get("gradient_accumulation", 8)
        resume_from_url = config.get("resume_from_url")

        # 1. Download dataset
        await progress_callback(job_id, "Downloading dataset...")
        dataset_path = self.model_cache_dir / f"dataset_{job_id}.jsonl"
        await self._download_file(f"{server_base}{dataset_url}", str(dataset_path))
        logger.info(f"Dataset downloaded: {dataset_path}")

        # 2. Download existing adapter if continuing training
        resume_path = None
        if resume_from_url:
            await progress_callback(job_id, "Downloading existing adapter...")
            adapter_zip = self.model_cache_dir / f"resume_{job_id}.zip"
            await self._download_file(f"{server_base}{resume_from_url}", str(adapter_zip))
            resume_path = str(self.model_cache_dir / f"resume_{job_id}")
            with zipfile.ZipFile(str(adapter_zip), "r") as zf:
                zf.extractall(resume_path)
            logger.info(f"Existing adapter downloaded to {resume_path}")

        # 3. Train
        await progress_callback(job_id, "Loading model...")
        output_dir = self.model_cache_dir / f"adapter_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = await self._train_lora(
            job_id=job_id,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            base_model=base_model,
            resume_from=resume_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_length=max_length,
            gradient_accumulation=gradient_accumulation,
            progress_callback=progress_callback,
        )

        if self._cancelled:
            return {"status": "cancelled", "reason": "Job cancelled"}

        # 4. Upload adapter
        await progress_callback(job_id, "Uploading trained adapter...")
        zip_path = self.model_cache_dir / f"adapter_{job_id}.zip"
        self._zip_directory(str(output_dir), str(zip_path))
        await self._upload_file(f"{server_base}{adapter_upload_url}", str(zip_path))

        logger.info(f"Adapter uploaded for job {job_id[:8]}")

        return {
            "status": "completed",
            "result": {
                "final_loss": result["loss"],
                "total_steps": result["steps"],
                "samples": result["samples"],
            },
        }

    async def _train_lora(self, job_id, dataset_path, output_dir, base_model,
                          resume_from, epochs, batch_size, lr, lora_rank, lora_alpha,
                          max_length, gradient_accumulation, progress_callback):
        """Blocking LoRA fine-tuning — runs in the current thread."""

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load and tokenize dataset
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        def tokenize(example):
            messages = example["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)

            labels = encoded["input_ids"].copy()
            non_assistant = messages[:-1]
            prefix_text = tokenizer.apply_chat_template(
                non_assistant, tokenize=False, add_generation_prompt=True
            )
            prefix_len = len(tokenizer(prefix_text, truncation=True, max_length=max_length)["input_ids"])
            labels[:prefix_len] = [-100] * prefix_len
            encoded["labels"] = labels
            return encoded

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(tokenize, remove_columns=["messages"])
        total_samples = len(dataset)

        split = dataset.train_test_split(test_size=0.1, seed=42)

        await progress_callback(job_id, f"Loading {base_model}...")

        # Load model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Enable gradient checkpointing to save VRAM
        model.gradient_checkpointing_enable()

        if resume_from and Path(resume_from).exists():
            await progress_callback(job_id, "Loading existing LoRA adapter...")
            model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model = get_peft_model(model, lora_config)

        await progress_callback(job_id, "Training started...",
                                train_progress={"samples": total_samples, "epochs": epochs})

        # Custom callback to report progress
        class ProgressCallback:
            def __init__(self, job_id, cb, loop):
                self.job_id = job_id
                self.cb = cb
                self.loop = loop
                self.log = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    entry = {
                        "step": state.global_step,
                        "loss": round(logs["loss"], 4),
                        "epoch": round(state.epoch, 2) if state.epoch else 0,
                    }
                    self.log.append(entry)
                    asyncio.run_coroutine_threadsafe(
                        self.cb(self.job_id, f"Step {state.global_step}, loss: {logs['loss']:.4f}",
                                train_progress={
                                    "step": state.global_step,
                                    "total_steps": state.max_steps,
                                    "loss": round(logs["loss"], 4),
                                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                                    "log": self.log[-20:],
                                }),
                        self.loop,
                    )

        from transformers import TrainerCallback

        class WsProgressCallback(TrainerCallback):
            def __init__(self, progress_tracker):
                self.tracker = progress_tracker

            def on_log(self, args, state, control, logs=None, **kwargs):
                self.tracker.on_log(args, state, control, logs, **kwargs)

        loop = asyncio.get_event_loop()
        tracker = ProgressCallback(job_id, progress_callback, loop)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=lr,
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            callbacks=[WsProgressCallback(tracker)],
        )

        result = trainer.train()

        # Save adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return {
            "loss": result.training_loss,
            "steps": result.global_step,
            "samples": total_samples,
        }

    async def _run_inference(self, job_id: str, config: dict, progress_callback) -> dict:
        """Load base model + LoRA adapter and run inference."""
        server_base = config["server_base_url"]
        adapter_url = config.get("adapter_url")
        base_model = config.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        instruction = config.get("instruction", "")
        input_text = config.get("input_text", "")
        max_tokens = config.get("max_tokens", 1024)
        temperature = config.get("temperature", 0.3)

        # Download adapter if provided
        adapter_path = None
        if adapter_url:
            await progress_callback(job_id, "Downloading adapter...")
            adapter_zip = self.model_cache_dir / f"inf_adapter_{job_id}.zip"
            adapter_path = str(self.model_cache_dir / f"inf_adapter_{job_id}")
            await self._download_file(f"{server_base}{adapter_url}", str(adapter_zip))
            with zipfile.ZipFile(str(adapter_zip), "r") as zf:
                zf.extractall(adapter_path)

        await progress_callback(job_id, "Loading model...")

        # Load base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        # Load LoRA adapter
        if adapter_path:
            await progress_callback(job_id, "Loading adapter...")
            model = PeftModel.from_pretrained(model, adapter_path)

        await progress_callback(job_id, "Generating...")

        # Build chat messages
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Cleanup GPU memory
        del model, inputs, output_ids
        torch.cuda.empty_cache()

        return {
            "status": "completed",
            "result": {"response": response},
        }

    async def _download_file(self, url: str, dest: str):
        """Download a file from the server."""
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)

    async def _upload_file(self, url: str, file_path: str):
        """Upload a file to the server."""
        async with httpx.AsyncClient(timeout=600) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(url, files={"file": ("adapter.zip", f, "application/zip")})
                resp.raise_for_status()

    def _zip_directory(self, dir_path: str, zip_path: str):
        """Zip a directory."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.startswith("checkpoint"):
                        continue  # skip checkpoint dirs
                    full = os.path.join(root, file)
                    arcname = os.path.relpath(full, dir_path)
                    zf.write(full, arcname)
