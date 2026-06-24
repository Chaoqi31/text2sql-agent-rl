#!/usr/bin/env python3
"""Build SFT cold-start data (plan WS2, D1=SynSQL+BIRD gold, D2=single-turn schema-in-prompt).

Each example: {"messages": [system, user(schema + question), assistant(reasoning + 'FINAL SQL: …')]}
- BIRD-train gold: schema serialized from the real sqlite; assistant = bare 'FINAL SQL: <gold>'
  (BIRD ships no CoT).
- SynSQL-2.5M: schema + CoT ship in the record; assistant = '<cot>\\nFINAL SQL: <sql>'.

    python scripts/prepare_sft.py --bird-json <train.json> --db-root <train_databases> \
        --synsql-jsonl <synsql_subset.jsonl> --out <sft.jsonl>
Either source is optional; pass BIRD, SynSQL, or both.

NOTE: SynSQL field names are validated against one real record on the box (see _synsql_fields).
"""
import argparse
import json
import sqlite3
from pathlib import Path

from sqlrl.dataset import load_bird
from sqlrl.prompts import SYSTEM_PROMPT


def db_schema_text(db_path: str) -> str:
    """Serialize a sqlite db's CREATE TABLE DDL (the schema the model must read)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(r[0] for r in rows)


def _messages(schema: str, question: str, evidence: str, assistant: str) -> dict:
    user = f"Schema:\n{schema}\n\nQuestion: {question}"
    if evidence:
        user += f"\nEvidence: {evidence}"
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def bird_to_sft(q) -> dict:
    return _messages(db_schema_text(q.db_path), q.question, q.evidence,
                     f"FINAL SQL: {q.gold_sql}")


def load_tables(tables_json: str) -> dict[str, str]:
    """tables.json -> {db_id: schema DDL text}. Each entry: {db_id, ddls:[CREATE TABLE...]}"""
    rows = json.loads(Path(tables_json).read_text())
    return {r["db_id"]: "\n".join(r.get("ddls", [])) for r in rows}


def synsql_to_sft(rec: dict, tables: dict[str, str]) -> dict:
    # SynSQL fields: db_id, question, external_knowledge, cot, sql; schema via tables.json join
    schema = tables.get(rec.get("db_id", ""), "")
    assistant = f"{rec.get('cot', '')}\nFINAL SQL: {rec.get('sql', '')}".strip()
    return _messages(schema, rec.get("question", ""), rec.get("external_knowledge", ""), assistant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-json", default=None, help="BIRD train json/jsonl (gold)")
    ap.add_argument("--db-root", default=None, help="BIRD train_databases root")
    ap.add_argument("--synsql-jsonl", default=None, help="SynSQL subset jsonl (already stratified)")
    ap.add_argument("--synsql-tables", default=None, help="SynSQL tables.json (schema, joined by db_id)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_bird = n_syn = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        if args.bird_json and args.db_root:
            for q in load_bird(args.bird_json, args.db_root):
                if not q.gold_sql or not Path(q.db_path).exists():
                    continue
                f.write(json.dumps(bird_to_sft(q)) + "\n")
                n_bird += 1
        if args.synsql_jsonl:
            if not args.synsql_tables:
                raise SystemExit("--synsql-jsonl requires --synsql-tables (schema source)")
            tables = load_tables(args.synsql_tables)
            for line in Path(args.synsql_jsonl).read_text().splitlines():
                if line.strip():
                    f.write(json.dumps(synsql_to_sft(json.loads(line), tables)) + "\n")
                    n_syn += 1
    print(f"wrote {n_bird + n_syn} SFT examples ({n_bird} BIRD + {n_syn} SynSQL) -> {out}")


if __name__ == "__main__":
    main()
