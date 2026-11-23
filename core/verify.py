"""Single answer-verification path used by correction, verify, and tests."""
import os
import sys

# Allow `python3 core/verify.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import execution_validator, validator


def _empty_info() -> dict:
    return {"coverage": 0.0, "exec_nonfatal": False, "check_ok": None,
            "code_ok": False, "unverified": False}


def evaluate(question: str, answer: str, check: str = "",
             require_check: bool = False):
    """Static + coverage + hidden-check evaluation of one candidate answer."""
    info = _empty_info()
    ok, reason = validator.validate_sample(question, answer, "explanation",
                                           strict_format=True)
    if not ok:
        return False, reason, info
    code = validator.extract_fences(answer)[0]
    result = execution_validator.run_with_check(code, check)
    info.update({
        "coverage": float(result.get("coverage") or 0.0),
        "exec_nonfatal": bool(result.get("nonfatal")),
        "check_ok": result.get("check_ok"),
        "code_ok": bool(result.get("code_ok")),
    })
    class_reg = execution_validator.is_class_register(code)
    if not result.get("code_ok"):
        if result.get("nonfatal"):
            return False, f"nonfatal:{result.get('error_type', 'RuntimeError')}", info
        reason = result.get("error") or "execution failed"
        return False, f"exec_{str(result.get('error_type', 'ExecutionError')).lower()}:{reason[:60]}", info
    # The check comes before coverage: a broken check is the actionable reason.
    if require_check:
        if not check:
            info["unverified"] = True
            return False, "missing_check", info
        if result.get("check_ok") is not True:
            detail = (result.get("check_error") or result.get("error") or "")[:120]
            return False, f"check_failed:{detail}", info
    if not class_reg and info["coverage"] < execution_validator.COVERAGE_FLOOR:
        return False, f"low_coverage:{int(round(info['coverage'] * 100))}", info
    # Vacuous check: does the check also pass when no answer code ran at all?
    if check and result.get("check_ok") is True and execution_validator.is_vacuous_check(check):
        return False, "vacuous_check", info
    return True, "ok", info


def revalidate_row(row: dict) -> dict:
    answer = row.get("answer") or ""
    check = row.get("check") or ""
    ok, reason, info = evaluate(row.get("question") or "", answer, check,
                                require_check=bool(check.strip()))
    row["coverage"] = round(float(info.get("coverage", 0.0)), 3)
    row["exec_nonfatal"] = bool(info.get("exec_nonfatal"))
    row["small_ok"] = bool(row.get("small_ok")) if "small_ok" in row else None
    if ok:
        row["validated"] = True
        row["verified"] = True
        row["status"] = "passed"
    elif info.get("unverified"):
        row["validated"] = "unverified"
        row["verified"] = False
        row["status"] = "unverified"
    else:
        row["validated"] = False
        row["verified"] = False
        row["status"] = "failed"
        row["correction_error"] = reason
    return row
