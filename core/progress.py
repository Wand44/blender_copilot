"""One-line machine-readable progress events for pipeline scripts."""
import json
import sys

PREFIX = "##PROGRESS##"


def emit(done, total, message="", **extra) -> None:
    payload = {"done": int(done), "total": int(total), "message": message}
    payload.update(extra)
    print(PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def parse(line: str):
    if not line.startswith(PREFIX):
        return None
    try:
        return json.loads(line[len(PREFIX):])
    except json.JSONDecodeError:
        return None
