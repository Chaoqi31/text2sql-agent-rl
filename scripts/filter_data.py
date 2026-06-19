#!/usr/bin/env python3
"""Arctic-style data filtering: keep only questions whose GOLD SQL executes cleanly and
returns a non-empty result (drops dirty/erroring/empty-result/timeout gold). Runs locally
on the BIRD databases — no GPU. Cuts reward-signal noise (a broken gold = unwinnable, EX
always 0, which poisons GRPO groups).

    python scripts/filter_data.py \
      --in-json  data/bird/bird23-train-filtered/data/train-00000-of-00001.jsonl \
      --db-root  data/bird/train/train_databases \
      --out-jsonl data/bird/train_exec_filtered.jsonl --workers 8
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlrl.db import connect_ro
from sqlrl.dataset import load_bird
from sqlrl.ex import _fetch_all


def gold_ok(q, timeout_s: float = 30.0):
    if not q.gold_sql or not q.gold_sql.strip():
        return False, "empty_sql"
    if not Path(q.db_path).exists():
        return False, "no_db"
    conn = connect_ro(q.db_path)
    try:
        rows, err = _fetch_all(conn, q.gold_sql, timeout_s)
    except Exception:
        return False, "exec_error"
    finally:
        conn.close()
    if err is not None:
        return False, "exec_error"
    if not rows:
        return False, "empty_result"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    qs = load_bird(args.in_json, args.db_root)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(lambda q: gold_ok(q), qs))

    reasons = Counter(why for _, why in verdicts)
    kept = [q for q, (ok, _) in zip(qs, verdicts) if ok]

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q in kept:
            f.write(json.dumps(q.__dict__) + "\n")

    print(f"kept {len(kept)}/{len(qs)} ({len(kept)/max(1,len(qs)):.1%}) -> {out}")
    print("drop reasons:", dict(reasons))


if __name__ == "__main__":
    main()
