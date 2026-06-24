#!/usr/bin/env python3
"""Build AGENTIC SFT data via gold-constructed tool-use trajectories (STaR-style cold start;
paper ref: Reasoning-SQL STaR-SFT). The first SFT was single-turn (schema-in-prompt) and HURT
agentic EX (0.422->0.326) because the eval/RL loop is multi-turn agentic. This builds SFT
examples in the EXACT agentic format: each trajectory inspects the schema with tools, then
commits the gold SQL — teaching the tool-use pattern with correct answers for every train Q
(no weak-base ceiling: the answer is gold).

    python scripts/prepare_sft_agentic.py --bird-json <train.jsonl> --db-root <train_databases> --out agentic_sft.jsonl

Trajectory: [system, user(question), assistant(list_tables), tool, assistant(describe gold
tables), tool..., assistant("FINAL SQL: <gold>")] — same message shape as src/sqlrl/agent.run_agent.
"""
import argparse
import json
from pathlib import Path

import sqlglot
from sqlglot import exp

from sqlrl.agent import _build_messages
from sqlrl.dataset import load_bird
from sqlrl.tools import SqlToolset


def _gold_tables(sql: str, available: list[str]) -> list[str]:
    """Tables referenced by the gold SQL, restricted to ones that exist in the db."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    low = {t.lower(): t for t in available}
    out, seen = [], set()
    for t in tree.find_all(exp.Table):
        n = (t.name or "").lower()
        if n in low and n not in seen:
            seen.add(n)
            out.append(low[n])
    return out


def build_trajectory(q) -> dict:
    ts = SqlToolset(q.db_path)
    try:
        all_tables = ts._table_names()
        tables = _gold_tables(q.gold_sql, all_tables) or all_tables[:3]
        msgs = list(_build_messages(q))                       # [system, user]
        cid = 0

        def call(name, args):
            nonlocal cid
            cid += 1
            # arguments as a dict — the HF chat template iterates it (.items()); a JSON
            # string trips "Can only get item pairs from a mapping" at render time.
            return {"id": f"call_{cid}", "type": "function",
                    "function": {"name": name, "arguments": args}}

        # turn 1: list the tables
        c1 = call("list_tables", {})
        msgs.append({"role": "assistant", "content": "Let me see the available tables.",
                     "tool_calls": [c1]})
        msgs.append({"role": "tool", "tool_call_id": c1["id"], "name": "list_tables",
                     "content": ts.list_tables()})
        # turn 2: describe the tables the gold query uses
        dcalls = [call("describe_table", {"table": t}) for t in tables]
        msgs.append({"role": "assistant", "content": "Let me inspect the relevant tables.",
                     "tool_calls": dcalls})
        for dc, t in zip(dcalls, tables):
            msgs.append({"role": "tool", "tool_call_id": dc["id"], "name": "describe_table",
                         "content": ts.describe_table(t)})
        # turn 3: commit the answer
        msgs.append({"role": "assistant", "content": f"FINAL SQL: {q.gold_sql}"})
        return {"messages": msgs}
    finally:
        ts.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-json", required=True)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = load_bird(args.bird_json, args.db_root)
    if args.limit:
        qs = qs[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = skip = 0
    with out.open("w") as f:
        for q in qs:
            if not q.gold_sql or not Path(q.db_path).exists():
                skip += 1
                continue
            try:
                f.write(json.dumps(build_trajectory(q)) + "\n")
                n += 1
            except Exception:
                skip += 1
    print(f"wrote {n} agentic SFT trajectories -> {out}  (skipped {skip})")


if __name__ == "__main__":
    main()
