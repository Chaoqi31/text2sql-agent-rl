#!/usr/bin/env python3
"""Eval: run the agent loop against a vllm-served model, score by BIRD EX.

Serve first (scripts/serve_vllm.sh base | lora <adapter>), then:
    python scripts/evaluate.py --config configs/eval.yaml --tag baseline
    python scripts/evaluate.py --config configs/eval.yaml --tag tuned --model-name sqlrl-lora

Reuses the SAME run_agent loop + reward_r1 as offline tests — only the served weights
differ between baseline and tuned (honest comparison, spec §11). GPU-box script.
"""
import argparse
import json
from pathlib import Path

from openai import OpenAI

from sqlrl.config import EvalSettings, load_config
from sqlrl.dataset import load_bird
from sqlrl.reward import reward_r1
from sqlrl.runner import run_questions, summarize
from sqlrl.schema import AgentConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, EvalSettings)
    if args.model_name:
        cfg.model_name = args.model_name

    questions = load_bird(cfg.eval_json, cfg.db_root)
    if cfg.n_questions:
        questions = questions[: cfg.n_questions]
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    agent_cfg = AgentConfig(max_turns=cfg.max_turns, temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens)

    results = run_questions(client, cfg.model_name, questions,
                            agent_cfg=agent_cfg, reward_fn=reward_r1,
                            concurrency=cfg.concurrency)
    s = summarize(results)
    print(f"[{args.tag}] {json.dumps(s, indent=2)}")

    out = Path("runs") / f"eval_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")
    print(f"wrote {len(results)} rollouts -> {out}")


if __name__ == "__main__":
    main()
