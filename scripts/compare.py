#!/usr/bin/env python3
"""Compare base vs tuned eval results — the Phase 1 minimum-shippable check.

    python scripts/compare.py --baseline runs/eval_baseline.jsonl --tuned runs/eval_tuned.jsonl

Success = ex_delta > 0 (tuned EX beats base on the same agent loop, weights-only change).
Multi-seed CIs are plan-3; this is the single-run signal.
"""
import argparse
import json
from pathlib import Path


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _summary(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "ex_rate": sum(r["ex"] for r in rows) / n,
        "ex_rate_fallback": sum(r.get("breakdown", {}).get("ex_fallback", r["ex"]) for r in rows) / n,
        "mean_reward": sum(r["reward"] for r in rows) / n,
        "valid_sql_rate": sum(1 for r in rows if r.get("final_sql")) / n,
        "finished_rate": sum(1 for r in rows if r.get("finished")) / n,
        "avg_turns": sum(r["turns"] for r in rows) / n,
    }


def compare(baseline_rows: list[dict], tuned_rows: list[dict]) -> dict:
    b, t = _summary(baseline_rows), _summary(tuned_rows)
    return {"baseline": b, "tuned": t, "ex_delta": t["ex_rate"] - b["ex_rate"],
            "ex_delta_fallback": t["ex_rate_fallback"] - b["ex_rate_fallback"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="runs/eval_baseline.jsonl")
    ap.add_argument("--tuned", default="runs/eval_tuned.jsonl")
    args = ap.parse_args()
    out = compare(_load(args.baseline), _load(args.tuned))
    print(json.dumps(out, indent=2))
    d = out["ex_delta"]
    print(f"\nEX delta (tuned - base): {d:+.3f}  -> {'PASS (RL improved EX)' if d > 0 else 'NOT YET'}")


if __name__ == "__main__":
    main()
