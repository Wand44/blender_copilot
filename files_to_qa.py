#!/usr/bin/env python
"""
Blender API Docs -> Q&A Dataset, using a local Ollama model to read each
doc page and pull out real, useful Q&A pairs. No bs4, no multi-agent
framework - just a plain loop: strip tags -> ask the model -> save.
"""
import html
import json
import os
import re
import urllib.request
from html.parser import HTMLParser

DOC_DIR = "/home/mintlover/Documents/blender_fine-tune/blender_python_reference_5_2"
OUTPUT_FILE = "/home/mintlover/Documents/blender_fine-tune/training_data/blender_QA_pairs.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mrthp/omnicoder2:latest"
LIMIT = None  # set to e.g. 20 to sanity-check quality before a full run

SYSTEM_PROMPT = """You are building a fine-tuning dataset for a Blender Python API coding assistant. Quality bar is high - this is training data for a coding model, not a trivia set.

You will be given the plain-text content of one Blender API documentation page. It has this shape:
- A title (what module/class/area this page covers)
- One or more definitions (a name, e.g. bpy.data.data or Object.select_get, followed by its
  description and sometimes a Type/Parameters/Returns field list)
- Sometimes a code block near the end (```python ... ```) - a real usage example for
  this page, not necessarily tied to just one definition above it

Skip these entirely, on every page, no matter what class they belong to:
- bl_rna_get_subclass, bl_rna_get_subclass_py, input_template, output_template,
  is_registered_node_type, driver_add, driver_remove, keyframe_insert, keyframe_delete,
  path_resolve, path_from_id, type_recast, as_pointer, id_properties_clear
- anything else that is clearly generic RNA/introspection plumbing inherited from a base class
  (bpy_struct, Struct, Node, ID) rather than something specific to this page's own subject
These are auto-documented on nearly every single class in the API (over a thousand pages) purely
because of inheritance. Writing a pair for them here just produces a near-duplicate of the same
pair from every other class's page, and they are rarely called directly by addon developers
anyway. Only cover one of these if the page's own prose gives it real, specific meaning beyond
the boilerplate.

Hard requirements for every pair you output:

1. QUESTION: must be a clear, specific, complete developer question - the kind someone would
   actually type while writing a Blender script ("How do I remove a mesh from bpy.data.meshes?",
   "What's the difference between select_get() and select_set()?"). Never a fragment, never
   generic filler repeated across entries ("What does X do?" for every single item is lazy -
   only use it when nothing more specific fits).

2. ANSWER: must contain a fenced python code block (```python ... ```) showing exactly how to
   call or use the thing, using its real name/signature as documented on the page. This is
   non-negotiable - a prose-only answer is not acceptable output, skip the pair instead.
   - If the page's own example covers this definition, use or adapt it.
   - Otherwise construct a minimal, correct usage snippet yourself, but stay strictly within
     what the page actually documents: real parameter names, real types, real return
     behavior. Never invent a parameter, method, socket name, or behavior that isn't stated
     on the page - if you're not sure of an exact detail, leave it out rather than guess.
   - The code must be complete enough to actually run, not a fragment that assumes an object
     already exists out of nowhere. If the thing being documented is an instance member (like a
     collection's .new()/.remove()/.find()), briefly show a realistic way to obtain that
     instance too (e.g. via bpy.context or bpy.data), don't just write "assuming X is an
     instance of Y" and start from there.
   - Write it the way an experienced developer actually would: direct and idiomatic, not
     padded with throwaway intermediate variables (prefer `obj.volume_display.density = 0.5`
     over `x = obj.volume_display` / `v = 0.5` / `x.density = v` for something this simple).
   - The code must be preceded by a sentence or two of real explanation - code with no
     explanation, or explanation with no code, both fail the requirement.

Be picky beyond that too:
- Skip anything where the page gives no real explanation - a property that's just a type and a
  default value ("(default \"\", never None)") is NOT useful on its own. Skip it unless the page
  gives you enough surrounding context to explain what it's actually for.
- When a page documents several small, closely related properties (e.g. a handful of simple
  settings on the same class), don't write one shallow pair per property. Combine the related
  ones into fewer, richer pairs that configure several of them together in one realistic
  example - that's both less redundant and closer to real code.

Never output null, missing, or empty question/answer fields. If you can't meet every requirement
above for a given definition, leave it out entirely rather than including a partial pair.

If the page has nothing worth extracting, output nothing at all.

Output ONLY plain JSONL, one object per line, nothing else - no markdown fences, no commentary, no numbering:
{"question": "...", "answer": "..."}
"""


# Tags that should force a line break so prose/definitions/headings don't
# all get mashed into one paragraph.
_BLOCK_TAGS = {"p", "dt", "dd", "dl", "div", "li", "tr", "br",
               "h1", "h2", "h3", "h4", "h5", "h6", "section"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0
        self._in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag == "pre":
            self._in_pre = True
            self.parts.append("\n```python\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "pre":
            self._in_pre = False
            self.parts.append("\n```\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        # Inside <pre> keep the text exactly as-is (real newlines/indentation
        # are what make the example code readable); elsewhere it gets
        # whitespace-collapsed below.
        self.parts.append(data)


_ARTICLE_RE = re.compile(r"<article\b[^>]*>(.*?)</article>", re.IGNORECASE | re.DOTALL)


def extract_main_content(raw_html):
    """Furo (Sphinx theme docs.blender.org uses) puts the real page content
    in <article role="main">...</article>. Everything outside it is the
    site sidebar (every class in the whole API, thousands of entries),
    header, and footer - pure noise for our purposes."""
    match = _ARTICLE_RE.search(raw_html)
    return match.group(1) if match else raw_html


def html_to_text(raw_html):
    parser = _TextExtractor()
    parser.feed(raw_html)
    text = html.unescape("".join(parser.parts)).replace("\u00b6", "")
    # Collapse runs of horizontal whitespace but keep the newlines we
    # inserted at block/pre boundaries, then trim excess blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


MAX_DOC_CHARS = 12000


def call_ollama(doc_text):
    if len(doc_text) > MAX_DOC_CHARS:
        print(f"    (truncating page from {len(doc_text)} to {MAX_DOC_CHARS} chars)")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": doc_text[:MAX_DOC_CHARS]},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content", "")


_BOILERPLATE_METHODS = (
    "bl_rna_get_subclass_py", "bl_rna_get_subclass", "input_template", "output_template",
    "is_registered_node_type", "driver_add", "driver_remove", "keyframe_insert",
    "keyframe_delete", "path_resolve", "path_from_id", "type_recast", "as_pointer",
    "id_properties_clear",
)


def _valid_pair(obj):
    if not isinstance(obj, dict):
        return False
    q, a = obj.get("question"), obj.get("answer")
    # Reject missing keys, None, empty strings, whitespace-only strings, and
    # non-string types - all of which showed up as garbage in the output.
    if not isinstance(q, str) or not isinstance(a, str):
        return False
    q, a = q.strip(), a.strip()
    if len(q) < 8 or len(a) < 10:
        return False
    # Hard requirement: this is a coding fine-tune, so a code-free answer
    # (or an answer that's only a code block with no explanation) doesn't
    # meet the bar - reject rather than trust the model's compliance.
    if "```" not in a:
        return False
    prose = a.split("```")[0].strip()
    if len(prose) < 10:
        return False
    # Backstop for the prompt instruction to skip inherited RNA plumbing -
    # these repeat near-identically across ~1500 unrelated class pages, so
    # even a few slipping past the prompt adds up fast.
    combined = (q + " " + a).lower()
    if any(name in combined for name in _BOILERPLATE_METHODS):
        return False
    return True


def parse_jsonl(text):
    text = text.strip()
    if text.startswith("["):
        try:
            arr = json.loads(text)
            return [{"question": o["question"].strip(), "answer": o["answer"].strip()}
                     for o in arr if _valid_pair(o)]
        except (json.JSONDecodeError, TypeError):
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


_DT_ID_RE = re.compile(r"<dt\b[^>]*\bid=")


def find_files():
    html_files = []
    for root, _dirs, files in os.walk(DOC_DIR):
        for name in files:
            if name.endswith(".html"):
                html_files.append(os.path.join(root, name))
    print(f"Found {len(html_files)} HTML files")

    skip_keywords = ["overview", "index", "nav", "search", "copyright"]
    after_keywords = 0
    kept = []
    for fpath in html_files:
        fname = os.path.basename(fpath).lower()
        if any(kw in fname for kw in skip_keywords):
            continue
        after_keywords += 1
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except OSError:
            continue
        # Isolate the real content before filtering/storing - the sidebar
        # nav (every class in the whole API) lives outside <article> and
        # would otherwise dominate everything downstream.
        article = extract_main_content(raw)
        if not _DT_ID_RE.search(article):  # skip hub/nav pages with no real entries
            continue
        kept.append((fpath, article))
    print(f"  {after_keywords} survived filename filter, {len(kept)} have real API entries")
    return kept


def main():
    kept = find_files()
    print(f"Kept {len(kept)} files with real API content")
    if LIMIT:
        kept = kept[:LIMIT]
        print(f"LIMIT set - only processing first {len(kept)}")

    total_pairs = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, (fpath, content) in enumerate(kept, 1):
            text = html_to_text(content)
            try:
                raw = call_ollama(text)
            except Exception as e:
                print(f"  [{i}/{len(kept)}] {os.path.basename(fpath)}: error - {e}")
                continue
            pairs = parse_jsonl(raw)
            for p in pairs:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            out.flush()
            total_pairs += len(pairs)
            print(f"  [{i}/{len(kept)}] {os.path.basename(fpath)}: {len(pairs)} pairs (total {total_pairs})")

    print(f"\nDone. {total_pairs} Q&A pairs written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()