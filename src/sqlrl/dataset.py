# src/sqlrl/dataset.py
import json
from pathlib import Path

from sqlrl.schema import Question


def load_bird(split_json: str, db_root: str, *, difficulties: set[str] | None = None) -> list[Question]:
    raw = json.loads(Path(split_json).read_text())
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
