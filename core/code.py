"""Extraction, hashing, and text helpers shared by every pipeline stage."""
import hashlib
import re
import warnings

FENCE_RE = re.compile(r"```python(.*?)```", re.DOTALL)

# Old generator template; stripped when merging old rows in combine_data.
LEGACY_HEADING_RE = re.compile(
    r"^[12]\) (?:Architecture Overview|The Code)[ \t]*\n(?:[ \t]*\n)?", re.MULTILINE)

FIM_RE = re.compile(
    r"<\|fim_prefix\|>(.*?)<\|fim_suffix\|>(.*?)<\|fim_middle\|>", re.DOTALL)


def extract_fences(text: str):
    """Return the contents of every ```python block, in order."""
    return list(FENCE_RE.findall(text or ""))


def first_fence(text: str) -> str:
    blocks = extract_fences(text)
    return blocks[0] if blocks else ""


def parse_ok(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            __import__("ast").parse(code)
        return True
    except SyntaxError:
        return False


def normalize_answer(answer: str) -> str:
    return LEGACY_HEADING_RE.sub("", answer or "").lstrip("\n")


def is_raw_code(text: str) -> bool:
    """True when a block of text is pure code (used when auto-fencing raw sources)."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    codeish = sum(1 for line in lines if re.match(
        r"^\s*(class |def |import |from |@|[A-Za-z_]\w*\s*=|bpy\.|#)", line))
    return codeish / len(lines) >= 0.6


def code_hash(answer: str) -> str:
    blocks = extract_fences(answer)
    if not blocks:
        blocks = [answer or ""]
    body = "\n---\n".join(block.strip() for block in blocks)
    return hashlib.md5(re.sub(r"\s+", " ", body).encode()).hexdigest()


def content_hash(question: str, answer: str) -> str:
    text = re.sub(r"\s+", " ", f"{question or ''}\n{answer or ''}").strip()
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fim_full_code(question: str, answer: str) -> "str | None":
    """Rebuild the full snippet from a FIM question and its middle-span answer."""
    match = FIM_RE.search(question or "")
    if not match:
        return None
    middle = first_fence(answer)
    return "\n".join(part.strip("\n")
                     for part in (match.group(1), middle, match.group(2)) if part)


def norm_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (dedupe/similarity)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()
