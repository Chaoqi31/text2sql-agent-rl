#!/usr/bin/env python3
"""Stratified subset of a LOCAL SynSQL-2.5M data.json (run locally; AutoDL HF is flaky —
download the raw json here, subset, upload the small result). ijson streams the 9.36GB
array so it never loads into RAM.

    python scripts/extract_synsql_subset.py --data /path/SynSQL/data.json \
        --out synsql_subset.jsonl --total 50000

Record fields (validated against the real repo): db_id, sql_complexity (Simple / Moderate /
Complex / Highly Complex), question, external_knowledge, cot, sql. Schema lives separately
in tables.json (joined later by scripts/prepare_sft.py).
"""
import argparse
import json
from collections import Counter

import ijson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="local SynSQL data.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=50000)
    ap.add_argument("--key", default="sql_complexity", help="field to stratify on")
    args = ap.parse_args()

    per = max(1, args.total // 4)                       # 4 complexity tiers, balanced
    written: Counter = Counter()
    n = 0
    with open(args.data, "rb") as fin, open(args.out, "w") as fout:
        for rec in ijson.items(fin, "item"):
            bucket = rec.get(args.key, "_")
            if written[bucket] >= per:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written[bucket] += 1
            n += 1
            if n >= args.total:
                break

    print(f"wrote {n} records -> {args.out}")
    print("per-bucket:", dict(written))


if __name__ == "__main__":
    main()
