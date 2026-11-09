"""Static validation for synthetic Blender Q&A samples.

Ground truth is data/bpy_api_cache.json built by build_api_cache.py.
"""
import ast
import hashlib
import json
import os
import re
import subprocess
import warnings
from collections import Counter
from typing import Optional

import blender_config

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "bpy_api_cache.json")
DOC_FILE = os.path.join(DATA_DIR, "doc_data.jsonl")
BLENDER_BIN = blender_config.BLENDER_BIN
TARGET_BLENDER = blender_config.TARGET_BLENDER

FENCE_RE = re.compile(r"```python(.*?)```", re.DOTALL)
BPY_RE = re.compile(r"\bbpy\.([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)")
BL_RE = re.compile(r"^\s*(bl_\w+)\s*[:=]", re.MULTILINE)
FAKE_NAMES = {"add_per_event", "bl_wide", "bl_height"}
VALID_BL_EXTRA = {"bl_info"}
VALID_STATUS = {"FINISHED", "CANCELLED", "PASS_THROUGH", "RUNNING_MODAL", "INTERFACE"}

_cache = None


def load_cache():
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(CACHE_PATH):
        subprocess.run(
            [BLENDER_BIN, "-b", "--python",
             os.path.join(HERE, "build_api_cache.py"), "--", CACHE_PATH],
            check=True, timeout=300)
    with open(CACHE_PATH) as f:
        _cache = json.load(f)
    # valid bl_* attrs
    _cache["_valid_bl"] = {a for attrs in _cache["types"].values() for a in attrs
                           if a.startswith("bl_")} | VALID_BL_EXTRA
    # augment from docs
    doc_types, doc_ops = set(), set()
    if os.path.exists(DOC_FILE):
        text = open(DOC_FILE, encoding="utf-8").read()
        for m in re.finditer(r"bpy\.types\.([A-Za-z_]\w*)", text):
            doc_types.add(m.group(1))
        for m in re.finditer(r"bpy\.ops\.([a-z_]+)\.([a-z_]+)", text):
            doc_ops.add(f"{m.group(1)}.{m.group(2)}")
    _cache["_doc_types"] = doc_types
    _cache["_doc_ops"] = doc_ops
    return _cache


def code_hash(answer: str) -> str:
    blocks = FENCE_RE.findall(answer)
    if not blocks:
        blocks = [answer]
    body = "\n---\n".join(b.strip() for b in blocks)
    return hashlib.md5(re.sub(r"\s+", " ", body).encode()).hexdigest()


def extract_fences(text: str):
    return [b for b in FENCE_RE.findall(text)]


def fim_full_code(question: str, answer: str) -> "str | None":
    """Rebuild the full snippet from a FIM question and its middle-span answer."""
    m = re.search(r"<\|fim_prefix\|>(.*?)<\|fim_suffix\|>(.*?)<\|fim_middle\|>",
                  question, re.DOTALL)
    if not m:
        return None
    blocks = extract_fences(answer)
    middle = blocks[0] if blocks else ""
    return "\n".join(part.strip("\n")
                     for part in (m.group(1), middle, m.group(2)) if part)


SIG_LINE_RE = re.compile(
    r"^\s*(?:"
    r"class\s+[\w.]+\s*\(.*\)"
    r"|(?:classmethod|staticmethod)\s+[\w.]+\s*\(.*\)"
    r"|def\s+[\w.]+\s*\(.*\)"
    r"|bpy\.ops\.[\w.]+\s*\(.*\)"
    r"|bpy\.types\.[\w.]+"
    r"|\*?[\w.]+\s*\(.*\)"
    r")\s*[,;:]?\s*$")


def is_reference_listing(block: str) -> bool:
    lines = [l for l in block.splitlines() if l.strip()]
    if not lines:
        return False
    sig = sum(1 for l in lines if SIG_LINE_RE.match(l))
    return sig / len(lines) > 0.5


def validate_reference(block: str, classes: set) -> Optional[str]:
    cache = load_cache()
    for m in re.finditer(r"bpy\.types\.([A-Za-z_]\w*)", block):
        n = m.group(1)
        if n not in cache["types"] and n not in cache["_doc_types"] and n not in classes:
            return f"invented_api_reference:bpy.types.{n}"
    for m in re.finditer(r"bpy\.ops\.([a-z_0-9]+)\.([a-z_0-9]+)", block):
        p = f"{m.group(1)}.{m.group(2)}"
        if p not in set(cache["ops"]) and p not in cache["_doc_ops"]:
            return f"invented_api_reference:bpy.ops.{p}"
    return None


def is_raw_code(text: str) -> bool:
    """True when a block of text is pure code (used when auto-fencing raw sources)."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    codeish = sum(1 for l in lines if re.match(
        r"^\s*(class |def |import |from |@|[A-Za-z_]\w*\s*=|bpy\.|#)", l))
    return codeish / len(lines) >= 0.6


def parse_ok(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        return False


def _sample_defined_names(code: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    classes, props = set(), set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.add(node.name)
            for item in node.body:
                targets = []
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    targets = [item.target.id]
                elif isinstance(item, ast.Assign):
                    targets = [t.id for t in item.targets if isinstance(t, ast.Name)]
                for t in targets:
                    props.add(t)
    return classes, props


def _bpy_paths(code: str):
    for m in BPY_RE.finditer(code):
        p = "bpy." + m.group(1)
        yield re.sub(r"\.\d+$", "", p)


def _other_paths(code: str):
    """bmesh, mathutils, bpy_extras etc. references."""
    patterns = [
        r"\b(bmesh)\b",
        r"\b(mathutils)\.(\w+)",
        r"\b(bpy_extras)\.(\w+)",
        r"\b(gpu)\.(\w+)",
        r"\b(aud)\.(\w+)",
        r"\b(idprop)\.(\w+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, code):
            yield m.group(0)


def path_valid(p: str, classes: set, props: set, deep: bool = True) -> bool:
    cache = load_cache()
    parts = p.split(".")[1:]
    if not parts:
        return True
    root, rest = parts[0], parts[1:]
    if any(seg in FAKE_NAMES for seg in parts):
        return False
    if root == "types":
        if not rest:
            return True
        name = rest[0]
        if name in classes:
            return True
        if name in cache["types"]:
            if not deep or len(rest) == 1:
                return True
            attr_set = set(cache["types"][name])
            return all(seg in attr_set or seg in props for seg in rest[1:])
        if name in cache["_doc_types"]:
            return True
        return False
    if root == "ops":
        if not rest:
            return True
        if len(rest) == 1:
            return any(o.startswith(rest[0] + ".") for o in cache["ops"]) or \
                   any(o.startswith(rest[0] + ".") for o in cache["_doc_ops"])
        return ".".join(rest) in set(cache["ops"]) or ".".join(rest) in cache["_doc_ops"]
    if root == "context":
        return not rest or rest[0] in set(cache["roots"]["context"])
    if root == "data":
        return not rest or rest[0] in set(cache["roots"]["data"])
    if root in ("props", "utils", "app"):
        return not rest or rest[0] in set(cache["roots"][root])
    if root in cache.get("modules", {}):
        if not deep:
            return True
        mods = cache["modules"][root]
        return all(seg in mods for seg in rest)
    return False


def _check_bl_attrs(code: str):
    cache = load_cache()
    valid_bl = cache["_valid_bl"]
    valid_regions = set(cache.get("bl_region_type", []))
    valid_spaces = set(cache.get("bl_space_type", []))
    reasons = []
    for m in BL_RE.finditer(code):
        attr = m.group(1)
        if attr not in valid_bl:
            reasons.append(f"invented_bl_attr:{attr}")
    for m in re.finditer(r"bl_region_type\s*=\s*['\"](\w+)['\"]", code):
        if m.group(1) not in valid_regions:
            reasons.append(f"invalid_bl_region_type:{m.group(1)}")
    for m in re.finditer(r"bl_space_type\s*=\s*['\"](\w+)['\"]", code):
        if m.group(1) not in valid_spaces:
            reasons.append(f"invalid_bl_space_type:{m.group(1)}")
    return reasons


def _ops_path(func) -> "str | None":
    parts = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "bpy":
        parts.reverse()
        if len(parts) == 3 and parts[0] == "ops":
            return f"{parts[1]}.{parts[2]}"
    return None


def _usage_checks(code: str):
    reasons = []
    cache = load_cache()
    ops_props = cache.get("ops_props", {})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return reasons

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _ops_path(node.func)
        if not path or path not in ops_props:
            continue
        valid = ops_props[path]
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if kw.arg not in valid:
                reasons.append(f"invalid_op_kwarg:{path}.{kw.arg}")
            elif isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str) \
                    and valid[kw.arg] and kw.value.value not in valid[kw.arg]:
                reasons.append(f"invalid_enum_value:{path}.{kw.arg}={kw.value.value}")

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    for cls in classes:
        base_names = {ast.unparse(b) for b in cls.bases}
        is_operator = "bpy.types.Operator" in base_names
        is_panel = "bpy.types.Panel" in base_names
        if is_operator:
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name in ("execute", "modal"):
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Set):
                            for el in sub.value.elts:
                                if isinstance(el, ast.Constant) and isinstance(el.value, str) \
                                        and el.value not in VALID_STATUS:
                                    reasons.append(f"invalid_return_status:{el.value}")
        if is_panel:
            draws = [i for i in cls.body if isinstance(i, ast.FunctionDef) and i.name == "draw"]
            if not draws or len(draws[0].args.args) != 2:
                reasons.append("panel_missing_draw")
            else:
                src = ast.get_source_segment(code, cls) or ""
                if "self.layout" not in src:
                    reasons.append("panel_draw_without_layout")

    reg = next((n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "register"), None)
    if reg:
        order = []
        for stmt in reg.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                    and ast.unparse(stmt.value.func) == "bpy.utils.register_class" \
                    and stmt.value.args and isinstance(stmt.value.args[0], ast.Name):
                order.append(stmt.value.args[0].id)
        for cls in classes:
            for node in ast.walk(cls):
                if isinstance(node, ast.Call) and \
                        ast.unparse(node.func) in ("bpy.props.PointerProperty",
                                                   "bpy.props.CollectionProperty"):
                    for kw in node.keywords:
                        if kw.arg == "type" and isinstance(kw.value, ast.Name) \
                                and cls.name in order and kw.value.id in order \
                                and order.index(kw.value.id) > order.index(cls.name):
                            reasons.append(f"register_order:{kw.value.id}_after_{cls.name}")
    return reasons


def _check_deprecated(code: str):
    cache = load_cache()
    deprecated = cache.get("deprecated", {})
    deprecated_attrs = {"use_auto_smooth", "calc_normals", "edges_sharp_to_custom_normals"}
    reasons = []
    for path in _bpy_paths(code):
        parts = path.split(".")
        for i in range(2, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub in deprecated:
                reasons.append(f"deprecated:{sub}")
        # attribute-level heuristics for common 4.x removals
        if parts[-1] in deprecated_attrs:
            reasons.append(f"deprecated_attr:{parts[-1]}")
    return reasons


COT_MARKERS = ("wait,", "let's look closer", "hmm", "step 1")


def validate_format(answer: str, task_type: str) -> "str | None":
    """Enforce the output schema: exactly one ```python block.

    Explanation answers may carry short prose before the code. Coder answers
    (bugfix/refactor) must be the code block alone, matching the generator
    prompt. Any unclosed/extra fence is a truncation signal.
    """
    if answer.count("```") % 2:
        return "unbalanced_fence"
    blocks = extract_fences(answer)
    if not blocks:
        if task_type == "fim":
            return "fim_no_code"
        return "no_code_block"
    if len(blocks) > 1:
        return "multiple_code_blocks"
    if not blocks[0].strip():
        return "empty_code"
    lower = answer.lower()
    if any(marker in lower for marker in COT_MARKERS):
        return "chain_of_thought"
    if not answer.rstrip().endswith("```"):
        return "trailing_text"
    if task_type == "fim":
        return None
    if task_type in ("bugfix", "refactor"):
        if not answer.lstrip().startswith("```python"):
            return "unexpected_format"
        return None
    if task_type in ("explanation", "addon", "bugfix_explained"):
        return None
    return "unexpected_format"


def validate_static(question: str, answer: str, task_type: str,
                    refs: list = None, strict_format: bool = True) -> "tuple[bool, str]":
    if not isinstance(question, str) or not isinstance(answer, str) \
            or not question.strip() or not answer.strip():
        return False, "empty_or_non_string"

    # pipeline instruction leakage
    leak_markers = ("provided reference", "===question===", "===answer===",
                    "the reference snippet", "the only allowed API source")
    if any(m in question.lower() for m in leak_markers):
        return False, "question_leak"

    if strict_format:
        fmt = validate_format(answer, task_type)
        if fmt:
            return False, fmt

    blocks = extract_fences(answer)
    code = blocks[0] if blocks else ""
    if task_type == "fim":
        # The answer is only the middle span; rebuild the full snippet before
        # parsing or API checking it.
        code = fim_full_code(question, answer)
        if code is None:
            return False, "fim_markers_missing"
    if not code.strip():
        # prose-only answer (allowed for non-synthetic sources when not strict)
        return True, "ok"
    if task_type in ("bugfix", "refactor", "bugfix_explained"):
        q_blocks = extract_fences(question)
        if q_blocks and re.sub(r"\s+", "", q_blocks[0]) == re.sub(r"\s+", "", code):
            return False, "unchanged_code"
    if not parse_ok(code):
        if is_reference_listing(code):
            err = validate_reference(code, set())
            if err:
                return False, err
            return True, "ok_reference"
        return False, "ast_parse_error"

    classes, props = _sample_defined_names(code)
    for p in _bpy_paths(code):
        if not path_valid(p, classes, props, deep=True):
            return False, f"invented_api:{p}"

    for r in _check_bl_attrs(code):
        return False, r

    for r in _check_deprecated(code):
        return False, r

    for r in _usage_checks(code):
        return False, r

    # addon structure checks (not applicable to FIM middle-span answers)
    code_text = code
    if task_type != "fim":
        has_reg = re.search(r"^def register\s*\(", code_text, re.MULTILINE)
        has_unreg = re.search(r"^def unregister\s*\(", code_text, re.MULTILINE)
        if has_reg:
            if not re.search(r"bpy\.utils\.register_class\s*\(", code_text):
                return False, "register_without_register_class"
            if has_unreg and not re.search(r"bpy\.utils\.unregister_class\s*\(", code_text):
                return False, "unregister_without_unregister_class"
            looks_addon = (
                not re.search(r'__name__\s*==\s*[\'"]__main__[\'"]', code_text)
                and re.search(r"class\s+\w+\s*\(\s*bpy\.types\.(?:Operator|Panel)", code_text))
            if looks_addon and "bl_info" not in code_text:
                return False, "addon_missing_bl_info"

    for m in re.finditer(r"class\s+\w+\s*\(\s*(bpy\.types\.\w+)\s*\)", code_text):
        base = m.group(1).split(".")[-1]
        if base not in load_cache()["types"]:
            return False, f"invented_base_class:{m.group(1)}"

    return True, "ok"


# Backward-compatible alias for older callers (keeps old combine_data working)
def validate_sample(question: str, answer: str, strategy, seen_hashes: set = None,
                    strict_format: bool = False):
    if isinstance(strategy, str) and strategy in (
            "explanation", "bugfix", "bugfix_explained", "refactor", "addon", "fim"):
        task_type = strategy
    else:
        try:
            strategy = int(strategy)
        except (TypeError, ValueError):
            pass
        task_type = {3: "explanation", 4: "explanation", 5: "bugfix_explained",
                     6: "fim", 7: "bugfix", 8: "refactor"}.get(strategy, "explanation")
    ok, reason = validate_static(question, answer, task_type, strict_format=strict_format)
    if ok and seen_hashes is not None:
        h = code_hash(answer)
        if h in seen_hashes:
            return False, "duplicate_code"
        seen_hashes.add(h)
    return ok, reason
