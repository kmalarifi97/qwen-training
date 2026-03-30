"""Data prep module: convert uploaded files into training JSONL using interface definitions."""

import csv
import json
import io
from pathlib import Path

from jinja2 import Template


DATA_DIR = Path("/app/data")
UPLOADS_DIR = DATA_DIR / "uploads"
DATASETS_DIR = DATA_DIR / "datasets"


def parse_csv(file_path: str) -> tuple[list[str], list[dict]]:
    """Read CSV, return (columns, rows)."""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    return columns, rows


def parse_txt(file_path: str, chunk_size: int = 1500) -> list[str]:
    """Read text file and split into chunks by character count, respecting line breaks."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > chunk_size and current:
            chunks.append(current.strip())
            current = line
        else:
            current += "\n" + line if current else line
    if current.strip():
        chunks.append(current.strip())
    return chunks


def render_input(template_str: str, row: dict) -> str:
    """Render a Jinja2 input template with row data."""
    tmpl = Template(template_str)
    return tmpl.render(**row)


def generate_pairs_from_csv(
    file_path: str,
    input_template: str,
    output_schema: dict,
    instruction: str,
    output_field_mapping: dict = None,
) -> list[dict]:
    """Generate instruction/input/output triples from CSV rows.

    output_field_mapping maps output schema keys to CSV column names.
    Example: {"fraud_type": "Fraud Type", "reason": None}
    If a mapping value is None, that field is left empty for the model to learn to generate.
    """
    columns, rows = parse_csv(file_path)
    pairs = []

    for row in rows:
        input_text = render_input(input_template, row)

        # Build output from mapping
        output = {}
        if output_field_mapping:
            for out_key, csv_col in output_field_mapping.items():
                if csv_col and csv_col in row:
                    output[out_key] = row[csv_col]
                else:
                    output[out_key] = ""
        else:
            # If no mapping, use all schema keys and try to match column names
            for key in output_schema.get("properties", {}).keys():
                # Try exact match or case-insensitive match
                matched = None
                for col in columns:
                    if col.lower().replace(" ", "_") == key.lower():
                        matched = col
                        break
                output[key] = row.get(matched, "") if matched else ""

        pairs.append({
            "instruction": instruction,
            "input": input_text,
            "output": json.dumps(output, ensure_ascii=False),
        })

    return pairs


def generate_pairs_from_text(
    file_path: str,
    instruction: str,
    chunk_size: int = 1500,
) -> list[dict]:
    """Generate pairs from text file — each chunk becomes one input.
    Output is left empty (to be filled by teacher model or manually).
    """
    chunks = parse_txt(file_path, chunk_size)
    pairs = []

    for i, chunk in enumerate(chunks):
        pairs.append({
            "instruction": instruction,
            "input": chunk,
            "output": "",
        })

    return pairs


def save_pairs_jsonl(pairs: list[dict], dataset_id: str) -> str:
    """Save pairs as JSONL file, return path."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_DIR / f"dataset_{dataset_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return str(out_path)


def format_as_chatml(pairs: list[dict]) -> list[dict]:
    """Convert instruction/input/output triples to ChatML format for Qwen training."""
    formatted = []
    for pair in pairs:
        messages = [
            {"role": "system", "content": pair["instruction"]},
            {"role": "user", "content": pair["input"]},
            {"role": "assistant", "content": pair["output"]},
        ]
        formatted.append({"messages": messages})
    return formatted


def save_chatml_jsonl(pairs: list[dict], dataset_id: str) -> str:
    """Save pairs in ChatML format for Qwen fine-tuning."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    chatml = format_as_chatml(pairs)
    out_path = DATASETS_DIR / f"dataset_{dataset_id}_chatml.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in chatml:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(out_path)
