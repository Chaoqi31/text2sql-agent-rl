# src/sqlrl/dataset.py
import json
from pathlib import Path

from sqlrl.schema import Question


def _load_rows(split_json: str) -> list[dict]:
    """Accept either a JSON array (BIRD raw train.json/dev.json) or JSONL
    (bird23-train-filtered ships line-delimited)."""
    text = Path(split_json).read_text()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_bird(split_json: str, db_root: str, *, difficulties: set[str] | None = None) -> list[Question]:
    raw = _load_rows(split_json)
    out: list[Question] = []
    for r in raw:
        diff = r.get("difficulty", "") or ""
        if difficulties is not None and diff not in difficulties:
            continue
        db_id = r["db_id"]
        out.append(Question(
            db_id=db_id,
            question=r["question"],
            gold_sql=(r.get("SQL") or r.get("query") or ""),
            db_path=str(Path(db_root) / db_id / f"{db_id}.sqlite"),
            evidence=(r.get("evidence") or ""),
            difficulty=diff,
        ))
    return out
