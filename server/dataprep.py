"""Data prep module: convert uploaded files into training JSONL using interface definitions.

Supports two modes:
1. Template-based: deterministic mapping from CSV columns → training pairs
2. Gemini-based: send data to Gemini API to generate rich, diverse training pairs
"""

import asyncio
import csv
import json
import io
import os
import re
from pathlib import Path

import httpx
from jinja2 import Template


DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
DATASETS_DIR = DATA_DIR / "datasets"

# --- Gemini config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

CSV_ROWS_PER_BATCH = 20

CSV_GENERATION_PROMPT = """You are an expert at creating fine-tuning training data from structured/tabular data.

Below is a CSV dataset with these columns:
{columns}

Here are {row_count} sample rows:
{rows}

Generate exactly {n} high-quality instruction/response training pairs based on this data.

Each pair should be diverse. Include these types:
- Record analysis: "Analyze this record for anomalies/patterns"
- Classification: "Classify this record" or "What category does this belong to?"
- Comparison: "Compare these two records"
- Summarization: "Summarize the key patterns in these records"
- Reasoning: "Why might this record be flagged as X?"
- Data interpretation: "What does this data tell us about Y?"

Rules:
- Use actual values from the rows in your instructions and outputs
- Outputs must be detailed and analytical, not just restating the data
- Match the language of the data (if data is in Arabic, pairs should be in Arabic)
- Output ONLY valid JSONL (one JSON object per line)
- No markdown, no code fences, no explanation before or after
- CRITICAL: All values must be plain strings, NOT objects or arrays

Each line must be exactly: {{"instruction": "a string", "input": "a string", "output": "a string"}}"""

TEXT_GENERATION_PROMPT = """أنت خبير في إنشاء بيانات تدريب للنماذج اللغوية. من النص التالي، قم بإنشاء {n} أزواج تدريبية بصيغة JSONL.

كل زوج يجب أن يكون بنفس لغة النص ويتضمن:
- instruction: سؤال أو طلب واضح ومتنوع
- input: سياق من النص إذا لزم (أو نص فارغ "")
- output: إجابة دقيقة ومفصلة مبنية على النص

نوّع في أنواع الأسئلة: فهم، تلخيص، شرح مفاهيم، مقارنة، تحليل، استنتاج، تعريف مصطلحات.

النص:
{text}

أعد النتيجة كـ JSONL فقط (سطر JSON واحد لكل زوج). بدون أي شرح إضافي أو markdown أو code fences.
مهم: جميع القيم يجب أن تكون نصوص (strings) وليس كائنات أو مصفوفات.
كل سطر يجب أن يكون بالضبط: {{"instruction": "نص", "input": "نص", "output": "نص"}}"""


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


# ──────────────────────────────────────────────
#  Gemini-based pair generation
# ──────────────────────────────────────────────

def _normalize_pair(obj: dict) -> dict | None:
    """Normalize a single training pair, converting non-string values."""
    if not isinstance(obj, dict):
        return None
    if "instruction" not in obj or "output" not in obj:
        return None

    def to_str(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        if not isinstance(val, str):
            return str(val)
        return val

    return {
        "instruction": to_str(obj["instruction"]),
        "input": to_str(obj.get("input", "")),
        "output": to_str(obj["output"]),
    }


def _parse_jsonl(raw: str) -> list[dict]:
    """Parse Gemini output into training pairs.
    Handles: markdown fences, thinking tags, JSON arrays, etc."""
    text = re.sub(r'```(?:json|jsonl)?\s*\n?', '', raw)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    pairs = []

    # Strategy 1: line-by-line JSONL
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            pair = _normalize_pair(obj)
            if pair:
                pairs.append(pair)
        except json.JSONDecodeError:
            continue

    # Strategy 2: regex for JSON objects
    if not pairs:
        for match in re.finditer(r'\{[^{}]*"instruction"[^{}]*\}', text, re.DOTALL):
            try:
                obj = json.loads(match.group())
                pair = _normalize_pair(obj)
                if pair:
                    pairs.append(pair)
            except json.JSONDecodeError:
                continue

    # Strategy 3: JSON array
    if not pairs:
        try:
            arr_match = re.search(r'\[.*\]', text, re.DOTALL)
            if arr_match:
                arr = json.loads(arr_match.group())
                for obj in arr:
                    pair = _normalize_pair(obj)
                    if pair:
                        pairs.append(pair)
        except (json.JSONDecodeError, TypeError):
            pass

    return pairs


def batch_csv_rows(columns: list[str], rows: list[dict],
                   batch_size: int = CSV_ROWS_PER_BATCH) -> list[str]:
    """Convert CSV rows into text batches for Gemini."""
    batches = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        lines = []
        for row in batch_rows:
            line = " | ".join(f"{col}: {row.get(col, '')}" for col in columns)
            lines.append(line)
        batches.append("\n".join(lines))
    return batches


async def _call_gemini(prompt: str) -> str | None:
    """Call Gemini API, return raw text response."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    async with httpx.AsyncClient(timeout=120) as client:
        url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            print(f"[gemini] API error: {data['error'].get('message', '?')}")
            return None

        return data["candidates"][0]["content"]["parts"][0]["text"]


async def generate_pairs_gemini_csv(
    file_path: str,
    pairs_per_batch: int = 10,
    progress_callback=None,
) -> list[dict]:
    """Send CSV rows to Gemini in batches, collect training pairs."""
    columns, rows = parse_csv(file_path)
    batches = batch_csv_rows(columns, rows)
    all_pairs = []
    columns_str = ", ".join(columns)

    for i, batch in enumerate(batches):
        row_count = batch.count("\n") + 1
        prompt = CSV_GENERATION_PROMPT.format(
            columns=columns_str,
            row_count=row_count,
            rows=batch,
            n=pairs_per_batch,
        )

        try:
            text = await _call_gemini(prompt)
            if text:
                pairs = _parse_jsonl(text)
                all_pairs.extend(pairs)
                print(f"[gemini-csv] Batch {i+1}/{len(batches)}: {len(pairs)} pairs")
            else:
                print(f"[gemini-csv] Batch {i+1}/{len(batches)}: no response")
        except Exception as e:
            print(f"[gemini-csv] Batch {i+1}/{len(batches)} ERROR: {e}")

        if progress_callback:
            await progress_callback(i + 1, len(batches), len(all_pairs))

        if i < len(batches) - 1:
            await asyncio.sleep(0.5)

    return all_pairs


async def generate_pairs_gemini_text(
    file_path: str,
    pairs_per_chunk: int = 10,
    chunk_size: int = 2500,
    progress_callback=None,
) -> list[dict]:
    """Send text chunks to Gemini, collect training pairs."""
    chunks = parse_txt(file_path, chunk_size)
    all_pairs = []

    for i, chunk in enumerate(chunks):
        prompt = TEXT_GENERATION_PROMPT.format(text=chunk, n=pairs_per_chunk)

        try:
            text = await _call_gemini(prompt)
            if text:
                pairs = _parse_jsonl(text)
                all_pairs.extend(pairs)
                print(f"[gemini-text] Chunk {i+1}/{len(chunks)}: {len(pairs)} pairs")
            else:
                print(f"[gemini-text] Chunk {i+1}/{len(chunks)}: no response")
        except Exception as e:
            print(f"[gemini-text] Chunk {i+1}/{len(chunks)} ERROR: {e}")

        if progress_callback:
            await progress_callback(i + 1, len(chunks), len(all_pairs))

        if i < len(chunks) - 1:
            await asyncio.sleep(0.5)

    return all_pairs
