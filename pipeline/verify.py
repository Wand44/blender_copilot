#!/usr/bin/env python3
"""Re-run verification over generated rows using the shared core verifier."""
import argparse

from core import config, execution_validator, validator
from core.gpu import warn_if_busy
from core.progress import emit
from core.rows import read_jsonl, utc_now, write_jsonl
from pipeline import model_correction


def revalidate_synthetic() -> None:
    validator.load_cache()
    rows = read_jsonl(config.SYNTHETIC_FILE)
    passed = 0
    for index, row in enumerate(rows, 1):
        code = validator.extract_fences(row.get("answer") or "")
        code = code[0] if code else ""
        ok, reason = validator.validate_sample(row.get("question") or "", row.get("answer") or "",
                                               row.get("type") or "explanation",
                                               strict_format=True)
        if ok and code:
            ok, reason = execution_validator.validate(code)
        row["validated"] = bool(ok)
        row["verified"] = bool(ok)
        row["status"] = "passed" if ok else "failed"
        row["verification_error"] = None if ok else reason
        row["timestamp"] = row.get("timestamp") or utc_now()
        passed += int(bool(ok))
        emit(index, len(rows), f"synthetic row {index}: {'pass' if ok else 'fail'}")
    write_jsonl(config.SYNTHETIC_FILE, rows)
    print(f"Revalidated {len(rows)} synthetic rows: {passed} pass, {len(rows) - passed} fail")


def main():
    parser = argparse.ArgumentParser(description="Re-run verification on generated data")
    parser.add_argument("--target", choices=("corrections", "synthetic", "all"),
                        default="corrections")
    args = parser.parse_args()
    warn_if_busy()
    if args.target in ("corrections", "all"):
        model_correction.revalidate()
    if args.target in ("synthetic", "all"):
        revalidate_synthetic()


if __name__ == "__main__":
    main()
