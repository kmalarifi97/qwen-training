"""Qwen Training Engine — FastAPI app with model registry, LoRA training, and inference."""

import asyncio
import json
import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.database import (
    init_db,
    create_interface, list_interfaces, get_interface, delete_interface,
    create_adapter, update_adapter, list_adapters, get_adapter, delete_adapter,
    create_job, update_job, list_jobs, get_job,
)
from server.dataprep import (
    parse_csv, generate_pairs_from_csv, generate_pairs_from_text,
    save_pairs_jsonl, save_chatml_jsonl,
    generate_pairs_gemini_csv, generate_pairs_gemini_text,
    GEMINI_API_KEY,
)
from server.trainer import train_lora
from server.inference import engine

UPLOAD_DIR = Path("/app/data/uploads")
DATASET_DIR = Path("/app/data/datasets")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

app = FastAPI(title="Qwen Training Engine", version="1.0.0")


@app.on_event("startup")
def startup():
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
#  Interfaces
# ──────────────────────────────────────────────

@app.get("/api/interfaces")
def api_list_interfaces():
    return list_interfaces()


@app.post("/api/interfaces")
def api_create_interface(
    name: str = Form(...),
    description: str = Form(""),
    file_type: str = Form(...),
    input_template: str = Form(...),
    output_schema: str = Form(...),  # JSON string
    instruction: str = Form(...),
):
    try:
        schema = json.loads(output_schema)
    except json.JSONDecodeError:
        raise HTTPException(400, "output_schema must be valid JSON")
    iface = create_interface(name, description, file_type, input_template, schema, instruction)
    return iface


@app.delete("/api/interfaces/{interface_id}")
def api_delete_interface(interface_id: str):
    delete_interface(interface_id)
    return {"ok": True}


# ──────────────────────────────────────────────
#  Adapters
# ──────────────────────────────────────────────

@app.get("/api/adapters")
def api_list_adapters():
    return list_adapters()


@app.post("/api/adapters")
def api_create_adapter(
    name: str = Form(...),
    interface_id: str = Form(...),
):
    iface = get_interface(interface_id)
    if not iface:
        raise HTTPException(404, "Interface not found")
    adapter = create_adapter(name, interface_id, BASE_MODEL)
    return adapter


@app.delete("/api/adapters/{adapter_id}")
def api_delete_adapter(adapter_id: str):
    adapter = get_adapter(adapter_id)
    if adapter and adapter["lora_path"]:
        lora_dir = Path(adapter["lora_path"])
        if lora_dir.exists():
            shutil.rmtree(lora_dir)
    delete_adapter(adapter_id)
    return {"ok": True}


# ──────────────────────────────────────────────
#  Data Preparation
# ──────────────────────────────────────────────

@app.post("/api/dataprep")
async def api_dataprep(
    background_tasks: BackgroundTasks,
    interface_id: str = Form(...),
    file: UploadFile = File(...),
    output_field_mapping: str = Form("{}"),  # JSON string
):
    iface = get_interface(interface_id)
    if not iface:
        raise HTTPException(404, "Interface not found")

    # Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create job
    try:
        mapping = json.loads(output_field_mapping)
    except json.JSONDecodeError:
        mapping = {}

    job = create_job(
        job_type="dataprep",
        interface_id=interface_id,
        input_file=str(upload_path),
        config={"output_field_mapping": mapping},
    )

    background_tasks.add_task(run_dataprep, job["id"], iface, str(upload_path), mapping)
    return job


def run_dataprep(job_id: str, iface: dict, file_path: str, mapping: dict):
    """Background task: generate training pairs from uploaded file."""
    update_job(job_id, status="running")

    try:
        output_schema = json.loads(iface["output_schema"]) if isinstance(iface["output_schema"], str) else iface["output_schema"]

        if iface["file_type"] == "csv":
            pairs = generate_pairs_from_csv(
                file_path=file_path,
                input_template=iface["input_template"],
                output_schema=output_schema,
                instruction=iface["instruction"],
                output_field_mapping=mapping or None,
            )
        else:
            pairs = generate_pairs_from_text(
                file_path=file_path,
                instruction=iface["instruction"],
            )

        # Save both formats
        raw_path = save_pairs_jsonl(pairs, job_id)
        chatml_path = save_chatml_jsonl(pairs, job_id)

        update_job(
            job_id,
            status="completed",
            output_file=chatml_path,
            progress=1.0,
            config=json.dumps({
                "pairs_count": len(pairs),
                "raw_path": raw_path,
                "chatml_path": chatml_path,
            }),
        )
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


# ──────────────────────────────────────────────
#  Gemini Data Preparation
# ──────────────────────────────────────────────

@app.post("/api/dataprep/gemini")
async def api_dataprep_gemini(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pairs_per_batch: int = Form(10),
):
    if not GEMINI_API_KEY:
        raise HTTPException(400, "GEMINI_API_KEY not set")

    # Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Detect file type
    ext = Path(file.filename).suffix.lower()

    job = create_job(
        job_type="dataprep",
        input_file=str(upload_path),
        config={
            "mode": "gemini",
            "pairs_per_batch": pairs_per_batch,
            "file_type": ext,
        },
    )

    background_tasks.add_task(run_gemini_dataprep, job["id"], str(upload_path), ext, pairs_per_batch)
    return job


async def _run_gemini_dataprep_async(job_id: str, file_path: str, ext: str, pairs_per_batch: int):
    """Async Gemini data prep."""
    update_job(job_id, status="running")

    try:
        async def progress_cb(done, total, pair_count):
            update_job(job_id, progress=done / total if total else 0,
                       config=json.dumps({
                           "mode": "gemini",
                           "processed": done,
                           "total_batches": total,
                           "pairs_so_far": pair_count,
                       }))

        if ext == ".csv":
            pairs = await generate_pairs_gemini_csv(
                file_path=file_path,
                pairs_per_batch=pairs_per_batch,
                progress_callback=progress_cb,
            )
        else:
            pairs = await generate_pairs_gemini_text(
                file_path=file_path,
                pairs_per_chunk=pairs_per_batch,
                progress_callback=progress_cb,
            )

        raw_path = save_pairs_jsonl(pairs, job_id)
        chatml_path = save_chatml_jsonl(pairs, job_id)

        update_job(
            job_id,
            status="completed",
            output_file=chatml_path,
            progress=1.0,
            config=json.dumps({
                "mode": "gemini",
                "pairs_count": len(pairs),
                "raw_path": raw_path,
                "chatml_path": chatml_path,
            }),
        )
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


def run_gemini_dataprep(job_id: str, file_path: str, ext: str, pairs_per_batch: int):
    """Background task wrapper — runs async Gemini pipeline in its own event loop."""
    asyncio.run(_run_gemini_dataprep_async(job_id, file_path, ext, pairs_per_batch))


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

@app.post("/api/train")
def api_train(
    background_tasks: BackgroundTasks,
    adapter_id: str = Form(...),
    dataset_job_id: str = Form(...),
    epochs: int = Form(3),
    learning_rate: float = Form(2e-4),
    lora_rank: int = Form(16),
    batch_size: int = Form(4),
    continue_training: bool = Form(False),
):
    adapter = get_adapter(adapter_id)
    if not adapter:
        raise HTTPException(404, "Adapter not found")

    # If continuing, adapter must already be trained
    resume_from = None
    if continue_training:
        if adapter["status"] != "ready" or not adapter["lora_path"]:
            raise HTTPException(400, "Adapter must be trained (status=ready) to continue training")
        resume_from = adapter["lora_path"]

    dataset_job = get_job(dataset_job_id)
    if not dataset_job or dataset_job["status"] != "completed":
        raise HTTPException(400, "Dataset job not found or not completed")

    job_config = json.loads(dataset_job["config"]) if isinstance(dataset_job["config"], str) else dataset_job["config"]
    dataset_path = job_config.get("chatml_path")
    if not dataset_path:
        raise HTTPException(400, "No ChatML dataset found in job")

    prev_samples = adapter["train_samples"] or 0

    job = create_job(
        job_type="train",
        adapter_id=adapter_id,
        config={
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "dataset_path": dataset_path,
            "continue_training": continue_training,
            "resume_from": resume_from,
            "previous_samples": prev_samples,
        },
    )

    background_tasks.add_task(
        run_training, job["id"], adapter_id, dataset_path,
        epochs, learning_rate, lora_rank, batch_size, resume_from, prev_samples,
    )
    return job


def run_training(job_id: str, adapter_id: str, dataset_path: str,
                 epochs: int, lr: float, rank: int, batch_size: int,
                 resume_from: str = None, prev_samples: int = 0):
    """Background task: LoRA fine-tuning."""
    update_job(job_id, status="running")
    try:
        result = train_lora(
            adapter_id=adapter_id,
            dataset_path=dataset_path,
            resume_from=resume_from,
            epochs=epochs,
            learning_rate=lr,
            lora_rank=rank,
            batch_size=batch_size,
        )
        # Accumulate total samples across training rounds
        result["total_samples"] = prev_samples + result["samples"]
        update_job(job_id, status="completed", progress=1.0,
                   config=json.dumps(result))
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


# ──────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────

@app.post("/api/inference")
async def api_inference(
    adapter_id: str = Form(None),
    instruction: str = Form(...),
    input_text: str = Form(...),
    max_tokens: int = Form(1024),
    temperature: float = Form(0.3),
):
    adapter_path = None
    if adapter_id:
        adapter = get_adapter(adapter_id)
        if not adapter:
            raise HTTPException(404, "Adapter not found")
        if adapter["status"] != "ready":
            raise HTTPException(400, f"Adapter status is '{adapter['status']}', must be 'ready'")
        adapter_path = adapter["lora_path"]

    try:
        response = engine.generate(
            instruction=instruction,
            input_text=input_text,
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────
#  Jobs
# ──────────────────────────────────────────────

@app.get("/api/jobs")
def api_list_jobs(job_type: str = None):
    return list_jobs(job_type)


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/datasets/{job_id}/download")
def api_download_dataset(job_id: str, format: str = "chatml"):
    job = get_job(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "Job not found or not completed")

    config = json.loads(job["config"]) if isinstance(job["config"], str) else job["config"]
    path_key = "chatml_path" if format == "chatml" else "raw_path"
    file_path = config.get(path_key)

    if not file_path or not Path(file_path).exists():
        raise HTTPException(404, "Dataset file not found")

    return FileResponse(file_path, filename=Path(file_path).name)


# ──────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")
