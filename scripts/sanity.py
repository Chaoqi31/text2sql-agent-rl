# scripts/sanity.py
import argparse
import json
from pathlib import Path

from sqlrl.config import SmokeConfig, load_config
from sqlrl.ex import execution_match
from sqlrl.schema import Question
from sqlrl.tools import SqlToolset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, SmokeConfig)

    rows = [json.loads(l) for l in Path(cfg.questions_jsonl).read_text().splitlines()][: cfg.n_questions]
    ok = 0
    for d in rows:
        q = Question(**d)
        if not Path(q.db_path).exists():
            print(f"MISSING db: {q.db_path}")
            continue
        ts = SqlToolset(q.db_path)
        try:
            ex = execution_match(q.gold_sql, q.gold_sql, q.db_path)
            tables = ts.list_tables()
        finally:
            ts.close()
        ok += ex
        print(f"[{q.db_id}] gold_runs={bool(ex)} | {tables[:80]}")
    print(f"\n{ok}/{len(rows)} gold queries execute & self-match (offline wiring OK)")


if __name__ == "__main__":
    main()
