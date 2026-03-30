"""Qwen Training Engine — FastAPI app with model registry, LoRA training, and inference."""

import asyncio
import json
import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
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
# Lazy imports — only used for local training/inference (requires GPU + torch)
# In server-only mode (GPU work on Colab), these are never called
train_lora = None
engine = None

def _load_gpu_modules():
    global train_lora, engine
    if train_lora is None:
        from server.trainer import train_lora as _tl
        from server.inference import engine as _eng
        train_lora = _tl
        engine = _eng

_DATA = Path(os.environ.get("DATA_DIR", "/app/data"))
SERVER_BASE_URL = os.environ.get("SERVER_BASE_URL", "http://localhost:8000")
UPLOAD_DIR = _DATA / "uploads"
DATASET_DIR = _DATA / "datasets"
ADAPTER_DIR = _DATA / "adapters"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

app = FastAPI(title="Qwen Training Engine", version="1.0.0")

# Live agent connections: agent_id -> {websocket, info}
live_agents: dict[str, dict] = {}


@app.on_event("startup")
def startup():
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)


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
    """Background task: LoRA fine-tuning (local GPU only)."""
    _load_gpu_modules()
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
    """Dispatch inference to Colab GPU agent and wait for result."""
    if not live_agents:
        raise HTTPException(400, "No GPU agent connected. Start the Colab notebook first.")

    adapter_url = None
    if adapter_id:
        adapter = get_adapter(adapter_id)
        if not adapter:
            raise HTTPException(404, "Adapter not found")
        if adapter["status"] != "ready":
            raise HTTPException(400, f"Adapter status is '{adapter['status']}', must be 'ready'")
        adapter_url = f"/api/adapters/{adapter_id}/download"

    # Create inference job
    job = create_job(
        job_type="inference",
        adapter_id=adapter_id,
        config={
            "instruction": instruction,
            "input_text": input_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "adapter_url": adapter_url,
        },
    )

    # Wait for agent to pick it up and complete (poll for up to 5 minutes)
    for _ in range(300):
        await asyncio.sleep(1)
        j = get_job(job["id"])
        if j["status"] == "completed":
            config = json.loads(j["config"]) if isinstance(j["config"], str) else j["config"]
            return {"response": config.get("response", "")}
        elif j["status"] == "failed":
            raise HTTPException(500, j.get("error", "Inference failed"))

    raise HTTPException(504, "Inference timed out")


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
#  GPU Agent — WebSocket + file transfer
# ──────────────────────────────────────────────

@app.get("/api/agents")
def api_list_agents():
    """List connected GPU agents."""
    return [
        {
            "agent_id": aid,
            "agent_name": info.get("agent_name", ""),
            "gpu_info": info.get("gpu_info"),
            "state": info.get("state", "UNKNOWN"),
            "connected": True,
        }
        for aid, info in live_agents.items()
    ]


@app.websocket("/agents/connect")
async def agent_connect(websocket: WebSocket):
    await websocket.accept()
    agent_id = None

    try:
        # First message = registration
        raw = await websocket.receive_text()
        data = json.loads(raw)

        if data.get("type") != "register":
            await websocket.send_text(json.dumps({"type": "error", "message": "Must register first"}))
            await websocket.close()
            return

        agent_id = data.get("agent_id")
        agent_name = data.get("agent_name", "unknown")

        live_agents[agent_id] = {
            "websocket": websocket,
            "agent_name": agent_name,
            "state": "AVAILABLE",
            "gpu_info": None,
        }

        print(f"[+] Agent connected: {agent_id} ({agent_name})")

        await websocket.send_text(json.dumps({
            "type": "registered",
            "message": f"Welcome {agent_name}",
        }))

        # Listen for messages
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data["type"] == "heartbeat":
                gpu_info = data.get("gpu_info")
                live_agents[agent_id]["gpu_info"] = gpu_info
                live_agents[agent_id]["state"] = data.get("state", "AVAILABLE")

                # Check if there's a pending job to assign
                pending_job = _find_pending_agent_job()
                if pending_job:
                    await websocket.send_text(json.dumps(pending_job))
                else:
                    await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

            elif data["type"] == "job_started":
                job_id = data["job_id"]
                update_job(job_id, status="running")
                live_agents[agent_id]["state"] = "WORKING"
                # Only update adapter status for training jobs, not inference
                job = get_job(job_id)
                if job and job.get("job_type") == "train" and job.get("adapter_id"):
                    update_adapter(job["adapter_id"], status="training")
                print(f"[*] Job {job_id[:8]} ({job.get('job_type','?')}) started on {agent_name}")

            elif data["type"] == "job_progress":
                job_id = data["job_id"]
                progress_str = data.get("progress", "")
                train_info = data.get("train_progress", {})
                progress_val = None
                if train_info.get("total_steps") and train_info.get("step"):
                    progress_val = train_info["step"] / train_info["total_steps"]
                update_job(job_id, status="running",
                           progress=progress_val,
                           config=json.dumps({"progress_detail": progress_str, **train_info}))

            elif data["type"] == "job_completed":
                job_id = data["job_id"]
                result = json.loads(data.get("result", "{}"))
                update_job(job_id, status="completed", progress=1.0,
                           config=json.dumps(result))
                live_agents[agent_id]["state"] = "AVAILABLE"

                # Update adapter for training jobs
                job = get_job(job_id)
                if job and job.get("job_type") == "train" and job.get("adapter_id"):
                    adapter_dir = ADAPTER_DIR / job["adapter_id"]
                    update_adapter(
                        job["adapter_id"],
                        status="ready",
                        lora_path=str(adapter_dir),
                        train_loss=result.get("final_loss"),
                        train_samples=result.get("samples", 0),
                    )
                print(f"[+] Job {job_id[:8]} completed on {agent_name}")

            elif data["type"] == "job_failed":
                job_id = data["job_id"]
                error = data.get("error", "")
                update_job(job_id, status="failed", error=error)
                live_agents[agent_id]["state"] = "AVAILABLE"

                job = get_job(job_id)
                if job and job.get("adapter_id"):
                    update_adapter(job["adapter_id"], status="failed")
                print(f"[-] Job {job_id[:8]} failed on {agent_name}: {error}")

    except WebSocketDisconnect:
        print(f"[-] Agent disconnected: {agent_id}")
    except Exception as e:
        print(f"[!] Agent error {agent_id}: {e}")
    finally:
        if agent_id:
            live_agents.pop(agent_id, None)


def _find_pending_agent_job() -> dict | None:
    """Find a pending training or inference job to assign to an agent."""
    # Check inference jobs first (higher priority — user is waiting)
    for job in list_jobs(job_type="inference", limit=5):
        if job["status"] == "pending":
            config = json.loads(job["config"]) if isinstance(job["config"], str) else (job["config"] or {})
            update_job(job["id"], status="assigned")
            return {
                "type": "job_assign",
                "job_id": job["id"],
                "job_type": "inference",
                "config": json.dumps({
                    "server_base_url": SERVER_BASE_URL,
                    "base_model": BASE_MODEL,
                    "instruction": config.get("instruction", ""),
                    "input_text": config.get("input_text", ""),
                    "max_tokens": config.get("max_tokens", 1024),
                    "temperature": config.get("temperature", 0.3),
                    "adapter_url": config.get("adapter_url"),
                }),
            }

    # Then training jobs
    jobs = list_jobs(job_type="train", limit=10)
    for job in jobs:
        if job["status"] == "pending":
            config = json.loads(job["config"]) if isinstance(job["config"], str) else (job["config"] or {})

            # Only dispatch remote jobs (ones without resume_from set locally)
            dataset_path = config.get("dataset_path", "")
            adapter_id = job.get("adapter_id", "")

            adapter = get_adapter(adapter_id) if adapter_id else None
            resume_from_url = None
            if config.get("continue_training") and adapter and adapter.get("lora_path"):
                resume_from_url = f"/api/adapters/{adapter_id}/download"

            update_job(job["id"], status="assigned")

            return {
                "type": "job_assign",
                "job_id": job["id"],
                "job_type": "train",
                "config": json.dumps({
                    "server_base_url": SERVER_BASE_URL,
                    "dataset_url": f"/api/datasets/{config.get('dataset_job_id', job['id'])}/download",
                    "adapter_upload_url": f"/api/adapters/{adapter_id}/upload" if adapter_id else "",
                    "base_model": BASE_MODEL,
                    "epochs": config.get("epochs", 3),
                    "batch_size": config.get("batch_size", 1),
                    "learning_rate": config.get("learning_rate", 2e-4),
                    "lora_rank": config.get("lora_rank", 8),
                    "lora_alpha": config.get("lora_alpha", 16),
                    "resume_from_url": resume_from_url,
                }),
            }
    return None


@app.post("/api/train/remote")
def api_train_remote(
    adapter_id: str = Form(...),
    dataset_job_id: str = Form(...),
    epochs: int = Form(3),
    learning_rate: float = Form(2e-4),
    lora_rank: int = Form(16),
    batch_size: int = Form(4),
    continue_training: bool = Form(False),
):
    """Queue a training job for a remote GPU agent."""
    adapter = get_adapter(adapter_id)
    if not adapter:
        raise HTTPException(404, "Adapter not found")

    if not live_agents:
        raise HTTPException(400, "No GPU agents connected")

    dataset_job = get_job(dataset_job_id)
    if not dataset_job or dataset_job["status"] != "completed":
        raise HTTPException(400, "Dataset job not found or not completed")

    job = create_job(
        job_type="train",
        adapter_id=adapter_id,
        config={
            "dataset_job_id": dataset_job_id,
            "dataset_path": "",  # remote — agent will download
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "continue_training": continue_training,
            "remote": True,
        },
    )
    return job


@app.post("/api/adapters/{adapter_id}/upload")
async def api_upload_adapter(adapter_id: str, file: UploadFile = File(...)):
    """Receive a trained adapter zip from the GPU agent."""
    adapter = get_adapter(adapter_id)
    if not adapter:
        raise HTTPException(404, "Adapter not found")

    adapter_dir = ADAPTER_DIR / adapter_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save and extract zip
    zip_path = ADAPTER_DIR / f"{adapter_id}.zip"
    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    import zipfile
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(adapter_dir))

    zip_path.unlink()  # cleanup zip

    print(f"[+] Adapter {adapter_id[:8]} uploaded ({len(content)} bytes)")
    return {"ok": True, "path": str(adapter_dir)}


@app.get("/api/adapters/{adapter_id}/download")
def api_download_adapter(adapter_id: str):
    """Download a trained adapter as zip (for continue-training on agent)."""
    adapter = get_adapter(adapter_id)
    if not adapter or not adapter.get("lora_path"):
        raise HTTPException(404, "Adapter not found or not trained")

    adapter_dir = Path(adapter["lora_path"])
    if not adapter_dir.exists():
        raise HTTPException(404, "Adapter files not found")

    import zipfile
    zip_path = ADAPTER_DIR / f"{adapter_id}_download.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for f in adapter_dir.rglob("*"):
            if f.is_file() and "checkpoint" not in str(f):
                zf.write(f, f.relative_to(adapter_dir))

    return FileResponse(str(zip_path), filename=f"adapter_{adapter_id}.zip")


# ──────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")
