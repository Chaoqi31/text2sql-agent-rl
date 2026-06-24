#!/usr/bin/env python3
"""Single-turn eval (DIAGNOSTIC): schema in the prompt, NO tool loop — matches how the SFT
data was built (scripts/prepare_sft._messages). Tells us whether the SFT model is actually
good at SQL in its native format, vs the agentic-eval mismatch seen in scripts/evaluate.py.

    python scripts/evaluate_singleturn.py --config configs/eval_mini.yaml --tag sft-st --model-name sqlrl-lora

Reuses execution_match + extract_final_sql + the SFT prompt format. GPU-box script.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from sqlrl.agent import extract_final_sql
from sqlrl.config import EvalSettings, load_config
from sqlrl.dataset import load_bird
from sqlrl.ex import execution_match
from sqlrl.prompts import SYSTEM_PROMPT
from scripts.prepare_sft import db_schema_text


def _messages(q):
    user = f"Schema:\n{db_schema_text(q.db_path)}\n\nQuestion: {q.question}"
    if q.evidence:
        user += f"\nEvidence: {q.evidence}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _run_one(client, model, q, max_tokens):
    try:
        resp = client.chat.completions.create(
            model=model, messages=_messages(q), temperature=0.0, max_tokens=max_tokens)
        final_sql = extract_final_sql(resp.choices[0].message.content or "")
        ex = execution_match(final_sql, q.gold_sql, q.db_path) if final_sql else 0
        return {"db_id": q.db_id, "final_sql": final_sql, "ex": ex, "finished": final_sql is not None}
    except Exception as e:
        return {"db_id": q.db_id, "final_sql": None, "ex": 0, "finished": False, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_mini.yaml")
    ap.add_argument("--tag", default="sft-st")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, EvalSettings)
    model = args.model_name or cfg.model_name

    questions = load_bird(cfg.eval_json, cfg.db_root)
    if cfg.n_questions:
        questions = questions[: cfg.n_questions]
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        results = list(pool.map(lambda q: _run_one(client, model, q, cfg.max_tokens), questions))

    n = len(results) or 1
    ex = sum(r["ex"] for r in results)
    fin = sum(1 for r in results if r["finished"])
    print(f"[{args.tag}] n={n} EX={ex/n:.4f} ({ex}/{n}) finished={fin/n:.4f}")

    out = Path("runs") / f"eval_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {n} -> {out}")


if __name__ == "__main__":
    main()
