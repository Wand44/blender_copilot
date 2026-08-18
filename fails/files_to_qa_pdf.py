#!/usr/bin/env python3
"""
Blender PDF reference -> Q&A dataset by chunking extracted text and
sending each chunk to Ollama for targeted QA generation.

This version reads a local PDF file instead of scraping the Blender docs site.
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import List

from pypdf import PdfReader

DEFAULT_PDF = "blender_python_reference_2_61_0.pdf"
DEFAULT_OUTPUT = "training_data/blender_pdf_qa_pairs.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:1.5b"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LIMIT = None
REQUEST_TIMEOUT = 600
MAX_RETRIES = 4
RETRY_BACKOFF = 2
MAX_PROMPT_CHARS = 8000

SYSTEM_PROMPT = """You are building a fine-tuning dataset for a Blender Python API coding assistant.

You will be given a chunk of text from the Blender Python Reference PDF. Focus on concrete API usage, practical examples, and code-oriented documentation. Ignore generic welcome text, contents pages, tutorial prose, and boilerplate that does not describe a specific API call.

Hard requirements:
1. Each question must be specific and developer-oriented.
2. Each answer must include a short explanation plus a fenced Python code block showing a realistic usage example.
3. Prefer practical API usage over generic descriptions.
4. Do not invent undocumented parameters or behavior. If the chunk does not support a strong pair, skip it.
5. The answer should be structured and code-first; if there is a relevant code example in the chunk, use it directly or adapt it.
6. Return only JSONL, one object per line, with this shape:
{"question": "...", "answer": "..."}
"""


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_boilerplate_paragraph(text: str) -> bool:
    if not text:
        return True

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return True

    lowered = cleaned.lower()
    skipped_markers = [
        "contents",
        "welcome, this document",
        "this document isn't intended",
        "this api is generally stable",
        "the blender/python api can do the following",
        "quickstart introduction",
        "before starting",
        "key concepts",
        "see also:",
        "for more examples",
        "example of a data path",
        "custom properties",
        "this document is an api reference",
    ]
    if any(marker in lowered for marker in skipped_markers):
        return True

    if len(re.sub(r"[^a-z0-9]", "", lowered)) < 20:
        return True

    code_signal = any(token in cleaned for token in ["import bpy", "bpy.", ">>>", "bpy.ops", "bpy.data", "bpy.context"])
    if code_signal:
        return False

    structural_signal = any(token in lowered for token in ["operator", "property", "method", "class", "function", "attribute"])
    if structural_signal and len(cleaned.split()) > 6:
        return False

    return False


def prepare_text_for_qa(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept = []
    for para in paragraphs:
        if is_boilerplate_paragraph(para):
            continue
        kept.append(para)
    return "\n\n".join(kept).strip()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    raw_text = normalize_text("\n\n".join(pages))
    return prepare_text_for_qa(raw_text)


def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraphs = [p for p in paragraphs if not is_boilerplate_paragraph(p)]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:]
            else:
                current = ""

        if len(paragraph) > max_chars:
            # Fall back to sentence-based slicing for very large paragraphs.
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            temp = ""
            for sentence in sentences:
                if len(temp) + len(sentence) + 1 <= max_chars:
                    temp = f"{temp} {sentence}".strip()
                else:
                    if temp:
                        chunks.append(temp)
                    temp = sentence
            if temp:
                current = temp
        else:
            current = paragraph

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def _valid_pair(obj):
    if not isinstance(obj, dict):
        return False
    q, a = obj.get("question"), obj.get("answer")
    if not isinstance(q, str) or not isinstance(a, str):
        return False
    q, a = q.strip(), a.strip()
    if len(q) < 8 or len(a) < 10:
        return False
    if "```" not in a:
        return False
    prose = a.split("```")[0].strip()
    if len(prose) < 10:
        return False
    return True


def parse_jsonl(text: str):
    text = text.strip()
    if text.startswith("["):
        try:
            arr = json.loads(text)
            return [{"question": o["question"].strip(), "answer": o["answer"].strip()} for o in arr if _valid_pair(o)]
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    pairs = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line or line.lower() == "json":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _valid_pair(obj):
            pairs.append({"question": obj["question"].strip(), "answer": obj["answer"].strip()})
    return pairs


def call_ollama(doc_text: str, timeout: int = REQUEST_TIMEOUT, retries: int = MAX_RETRIES) -> str:
    prompt_text = doc_text[:MAX_PROMPT_CHARS].strip()
    if not prompt_text:
        return ""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                wait_time = RETRY_BACKOFF * attempt
                print(f"  retry {attempt}/{retries} in {wait_time}s: {exc}")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"failed after {retries} attempts: {exc}") from exc

    if last_error is not None:
        raise RuntimeError(f"failed to reach Ollama: {last_error}")
    return ""


def write_pairs(output_path: Path, pairs: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out:
        for pair in pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a local Blender PDF reference into chunked QA pairs")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the local Blender PDF file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to the output JSONL file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Maximum characters in each text chunk")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Optional limit on how many chunks to process")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="Number of retries for each chunk")
    parser.add_argument("--dry-run", action="store_true", help="Extract text and show chunking without calling Ollama")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_path = Path(args.output)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = extract_pdf_text(pdf_path)


    chunks = chunk_text(text, max_chars=args.chunk_size)

    if args.dry_run:
        for idx, chunk in enumerate(chunks[:5], 1):
            print(f"\n--- Chunk {idx} ---")
            print(chunk[:800])
        return

    if output_path.exists():
        output_path.unlink()

    total_pairs = 0
    for idx, chunk in enumerate(chunks[: args.limit] if args.limit else chunks, 1):
        if len(chunk) < 200:
            continue
        print(f"Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
        try:
            raw = call_ollama(chunk, timeout=args.timeout, retries=args.retries)
        except Exception as exc:
            print(f"  error: {exc}")
            continue

        pairs = parse_jsonl(raw)
        if pairs:
            write_pairs(output_path, pairs)
            total_pairs += len(pairs)
            print(f"  wrote {len(pairs)} pairs")
        else:
            print("  no valid pairs returned")

    print(f"Done. Wrote {total_pairs} Q&A pairs to {output_path}")


if __name__ == "__main__":
    main()
