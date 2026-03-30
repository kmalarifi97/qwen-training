import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/app/data/db/engine.db")


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS interfaces (
            id          TEXT PRIMARY KEY,
            name        TEXT UNIQUE NOT NULL,
            description TEXT,
            file_type   TEXT NOT NULL,  -- csv, txt, pdf
            input_template  TEXT NOT NULL,  -- jinja2 template for building the prompt input
            output_schema   TEXT NOT NULL,  -- JSON schema the model should produce
            instruction     TEXT NOT NULL,  -- system instruction for this interface
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adapters (
            id          TEXT PRIMARY KEY,
            name        TEXT UNIQUE NOT NULL,
            interface_id TEXT NOT NULL,
            base_model  TEXT NOT NULL,
            lora_path   TEXT,           -- path to adapter weights
            status      TEXT NOT NULL DEFAULT 'pending',  -- pending, training, ready, failed
            train_samples INTEGER DEFAULT 0,
            train_epochs  INTEGER DEFAULT 0,
            train_loss    REAL,
            metadata    TEXT,           -- JSON blob for extra info
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (interface_id) REFERENCES interfaces(id)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            job_type    TEXT NOT NULL,   -- dataprep, train, inference
            adapter_id  TEXT,
            interface_id TEXT,
            status      TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
            input_file  TEXT,
            output_file TEXT,
            config      TEXT,           -- JSON blob for job-specific config
            progress    REAL DEFAULT 0, -- 0.0 to 1.0
            error       TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (adapter_id) REFERENCES adapters(id),
            FOREIGN KEY (interface_id) REFERENCES interfaces(id)
        );
    """)
    conn.commit()
    conn.close()


# --- Interface CRUD ---

def create_interface(name: str, description: str, file_type: str,
                     input_template: str, output_schema: dict, instruction: str) -> dict:
    conn = get_db()
    row_id = uuid.uuid4().hex[:12]
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO interfaces (id, name, description, file_type, input_template, output_schema, instruction, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (row_id, name, description, file_type, input_template, json.dumps(output_schema), instruction, now, now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM interfaces WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return dict(row)


def list_interfaces() -> list[dict]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM interfaces ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_interface(interface_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM interfaces WHERE id = ?", (interface_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_interface(interface_id: str):
    conn = get_db()
    conn.execute("DELETE FROM interfaces WHERE id = ?", (interface_id,))
    conn.commit()
    conn.close()


# --- Adapter CRUD ---

def create_adapter(name: str, interface_id: str, base_model: str) -> dict:
    conn = get_db()
    row_id = uuid.uuid4().hex[:12]
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO adapters (id, name, interface_id, base_model, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
        (row_id, name, interface_id, base_model, now, now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM adapters WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return dict(row)


def update_adapter(adapter_id: str, **kwargs):
    conn = get_db()
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [adapter_id]
    conn.execute(f"UPDATE adapters SET {sets} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def list_adapters() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT a.*, i.name as interface_name FROM adapters a "
        "LEFT JOIN interfaces i ON a.interface_id = i.id "
        "ORDER BY a.created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_adapter(adapter_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM adapters WHERE id = ?", (adapter_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_adapter(adapter_id: str):
    conn = get_db()
    conn.execute("DELETE FROM adapters WHERE id = ?", (adapter_id,))
    conn.commit()
    conn.close()


# --- Job CRUD ---

def create_job(job_type: str, interface_id: str = None, adapter_id: str = None,
               input_file: str = None, config: dict = None) -> dict:
    conn = get_db()
    row_id = uuid.uuid4().hex[:12]
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO jobs (id, job_type, adapter_id, interface_id, status, input_file, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
        (row_id, job_type, adapter_id, interface_id, input_file, json.dumps(config or {}), now, now)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return dict(row)


def update_job(job_id: str, **kwargs):
    conn = get_db()
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    if "config" in kwargs and isinstance(kwargs["config"], dict):
        kwargs["config"] = json.dumps(kwargs["config"])
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def list_jobs(job_type: str = None, limit: int = 50) -> list[dict]:
    conn = get_db()
    if job_type:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE job_type = ? ORDER BY created_at DESC LIMIT ?",
            (job_type, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_job(job_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
