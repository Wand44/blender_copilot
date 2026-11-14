"""Training-row schema, JSONL I/O, API-area extraction, and length stats."""
import ast
import json
import math
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from core import code as code_mod

SOURCE_NAMES = {"docs", "synthetic", "correction", "addon", "general", "replay"}

_SHADER_RE = re.compile(r"^ShaderNode[A-Za-z0-9_]+$")
_FALLBACK_AREA_RE = re.compile(
    r"\bbpy\.(?:ops\.[a-z_]+|types\.[A-Za-z_]\w*|data\.[A-Za-z_]\w*"
    r"|context(?:\.[A-Za-z_]\w*){0,3})"
    r"|\bbmesh\b|\bmathutils\.\w+|ShaderNode\w+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_mtime_iso(path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def estimate_tokens(text: str) -> int:
    """Approximate token count using words + punctuation (no tokenizer dependency)."""
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _area(path: str) -> str:
    parts = path.split(".")
    if parts[0] == "bpy":
        rest = parts[1:]
        if not rest:
            return "bpy"
        if rest[0] == "ops":
            return "bpy.ops." + rest[1] if len(rest) > 1 else "bpy.ops"
        if rest[0] == "types":
            return "bpy.types." + rest[1] if len(rest) > 1 else "bpy.types"
        if rest[0] == "data":
            return "bpy.data." + rest[1] if len(rest) > 1 else "bpy.data"
        if rest[0] == "context":
            inner = rest[1:]
            if not inner:
                return "context"
            if inner[0] == "scene":
                return "scene." + inner[1] if len(inner) > 1 else "scene"
            if inner[0] in ("object", "active_object") and len(inner) > 1 \
                    and inner[1] in ("modifiers", "animation_data", "data", "constraints"):
                return "context.object." + inner[1]
            return "context." + inner[0]
        if rest[0] in ("utils", "props", "app", "path"):
            return "bpy." + rest[0]
        return "bpy." + rest[0]
    if parts[0] in ("bmesh", "mathutils", "bpy_extras", "gpu", "aud", "idprop", "imbuf"):
        return parts[0]
    return ".".join(parts[:2])


def _code_for_analysis(question: str, answer: str) -> str:
    full = code_mod.fim_full_code(question, answer)
    if full is not None:
        return full
    blocks = code_mod.extract_fences(answer)
    return "\n\n".join(blocks) if blocks else (answer or "")


def api_areas(question: str, answer: str) -> list:
    """Collect API areas from code with ast; regex fallback for non-Python listings."""
    source = _code_for_analysis(question, answer)
    areas = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = _chain(node)
                if chain and chain[0] in ("bpy", "bmesh", "mathutils", "bpy_extras",
                                          "gpu", "aud", "idprop", "imbuf"):
                    areas.add(_area(".".join(chain)))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _SHADER_RE.match(node.value):
                    areas.add(node.value)
    except SyntaxError:
        for match in _FALLBACK_AREA_RE.finditer(source):
            areas.add(_area(match.group(0)))
    # Attribute walking also yields parent prefixes; keep only useful areas.
    areas -= {"bpy", "bpy.ops", "bpy.types", "bpy.data", "context", "scene"}
    return sorted(areas)


API_AREA_GROUPS = {
    "mesh": re.compile(r"bpy\.ops\.mesh|bpy\.types\.Mesh|primitive_|from_pydata|vertices|polygons"),
    "bmesh": re.compile(r"\bbmesh\b"),
    "materials/nodes": re.compile(r"material|shadernode|node_tree|principled|bsdf|nodes\.new"),
    "render/cycles": re.compile(r"\brender\b|\bcycles\b|eevee|samples|denois|image_settings"),
    "animation": re.compile(r"keyframe|animation_data|fcurve|frame_set|\baction\b"),
    "modifiers": re.compile(r"modifier|subsurf|bevel|boolean|\barray\b"),
    "lights/camera": re.compile(r"bpy\.types\.(?:Light|Camera)|bpy\.data\.(?:lights|cameras)|light_add|camera_add"),
    "UI/addons": re.compile(r"bpy\.types\.(?:Operator|Panel|Menu)|register_class|bpy\.props|bl_info|bl_idname"),
}


def api_categories(areas, question: str = "", answer: str = "") -> list:
    text = (question or "") + "\n" + (answer or "") + "\n" + "\n".join(areas or [])
    return sorted(name for name, pattern in API_AREA_GROUPS.items()
                  if pattern.search(text))


def training_row(question: str, answer: str, *, id, source: str,
                 producing_model=None, verified: bool = False,
                 attempts: int = 0, timestamp: str = None,
                 areas=None, content: str = None) -> dict:
    if source not in SOURCE_NAMES:
        raise ValueError(f"unknown source: {source}")
    areas = list(areas) if areas is not None else api_areas(question, answer)
    return {
        "id": id,
        "question": question,
        "answer": answer,
        "source": source,
        "producing_model": producing_model or "unknown",
        "verified": bool(verified),
        "attempts": int(attempts or 0),
        "api_areas": areas,
        "content_hash": content or code_mod.content_hash(question, answer),
        "token_length": estimate_tokens((question or "") + " " + (answer or "")),
        "timestamp": timestamp or utc_now(),
    }


LENGTH_BINS = [
    (0, 128), (128, 256), (256, 512), (512, 1024),
    (1024, 2048), (2048, 4096), (4096, 8192), (8192, None),
]


def length_stats(values) -> dict:
    values = sorted(int(value or 0) for value in values)
    if not values:
        return {"n": 0, "min": 0, "mean": 0, "median": 0, "p90": 0, "max": 0,
                "histogram": {"labels": [], "counts": []}}
    def percentile(p):
        index = min(len(values) - 1, max(0, int(math.ceil(p * len(values))) - 1))
        return values[index]
    hist = Counter()
    for value in values:
        for low, high in LENGTH_BINS:
            if high is None or value < high:
                hist[f"{low}-{high}" if high else f"{low}+"] += 1
                break
    labels = [f"{low}-{high}" if high else f"{low}+" for low, high in LENGTH_BINS]
    return {
        "n": len(values),
        "min": values[0],
        "mean": round(sum(values) / len(values), 1),
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "max": values[-1],
        "histogram": {"labels": labels, "counts": [hist[label] for label in labels]},
    }
