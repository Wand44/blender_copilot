"""Duplicate detection shared by combine_data and tests."""
from core import code as code_mod


def norm_q(question: str) -> str:
    return code_mod.norm_text(question)


def dedupe(rows):
    """Drop duplicate code, duplicate questions, then near-duplicate questions."""
    seen_hash, seen_q, seen_words = set(), set(), set()
    kept = []
    stats = {"duplicate_code": 0, "duplicate_question": 0, "near_duplicate": 0}
    for row in rows:
        question = row.get("question", "")
        answer = row.get("answer", "")
        # Task types are different training signals even when they share code.
        key = (row.get("_type"), code_mod.code_hash(answer))
        normalized = norm_q(question)
        words = set(normalized.split())
        sorted_words = " ".join(sorted(words)) if len(words) >= 4 else None
        if key in seen_hash:
            stats["duplicate_code"] += 1
            continue
        if normalized in seen_q:
            stats["duplicate_question"] += 1
            continue
        if sorted_words and sorted_words in seen_words:
            stats["near_duplicate"] += 1
            continue
        seen_hash.add(key)
        seen_q.add(normalized)
        if sorted_words:
            seen_words.add(sorted_words)
        kept.append(row)
    return kept, stats
