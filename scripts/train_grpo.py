#!/usr/bin/env python3
"""R1 agentic GRPO training with TRL. GPU only (Phase 1, plan-2).

TRL's GRPOTrainer does native multi-turn tool calling: we hand it environment_factory
(each public method of the env becomes a tool) + reward_funcs (our R1 = BIRD EX), and it
drives generate -> call tool -> feed result -> repeat, masks tool-result tokens from the
loss, runs GRPO, and keeps the colocated vLLM weights in sync.

    python scripts/train_grpo.py --config configs/smoke_train.yaml   # 2-step smoke
    python scripts/train_grpo.py --config configs/train.yaml         # mini-scale R1 run

LoRA adapter -> runs/<run_name>/final. Serve with scripts/serve_vllm.sh lora <path>,
evaluate with scripts/evaluate.py --tag tuned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from sqlrl.agent import _build_messages  # noqa: E402  (same prompt construction as eval)
from sqlrl.config import TrainSettings, load_config  # noqa: E402
from sqlrl.dataset import load_bird  # noqa: E402
from sqlrl.train_env import make_sql_env_factory  # noqa: E402
from sqlrl.train_reward import make_r1_reward  # noqa: E402

_REWARDS = {"r1": make_r1_reward}   # r2/r3/s0 added in plan-3


def build_dataset(questions) -> Dataset:
    # prompt built by the SAME helper run_agent uses -> identical prompting train vs eval.
    rows = [{"prompt": _build_messages(q), "gold_sql": q.gold_sql, "db_path": q.db_path}
            for q in questions]
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume GRPO from")
    args = ap.parse_args()

    cfg = load_config(args.config, TrainSettings)
    if args.base_model:
        cfg.base_model = args.base_model
    if args.steps:
        cfg.steps = args.steps
    if cfg.reward_arm not in _REWARDS:
        raise SystemExit(f"reward_arm {cfg.reward_arm!r} not in plan-2 (r1 only); {sorted(_REWARDS)}")

    questions = load_bird(cfg.train_jsonl, cfg.db_root)
    dataset = build_dataset(questions)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    output_dir = str(REPO / "runs" / cfg.run_name)
    save_strategy, save_steps = ("no", 0) if cfg.steps <= 3 else ("steps", max(5, cfg.steps // 6))
    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        num_generations=cfg.group_size,
        per_device_train_batch_size=cfg.group_size,
        gradient_accumulation_steps=cfg.scenarios_per_step,
        max_steps=cfg.steps,
        learning_rate=cfg.learning_rate,
        beta=cfg.kl_coef,
        temperature=cfg.temperature,
        scale_rewards=True,
        max_tool_calling_iterations=cfg.max_turns,
        max_completion_length=cfg.max_tokens,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=save_steps,
        report_to=[],
        log_completions=True,
        num_completions_to_print=2,
    )
    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM",
    )

    print(f"[train] TRL GRPO | {cfg.base_model} | arm={cfg.reward_arm} | "
          f"{len(questions)} prompts | steps {cfg.steps} | group {cfg.group_size}")

    trainer = GRPOTrainer(
        model=cfg.base_model,
        reward_funcs=[_REWARDS[cfg.reward_arm]()],
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=make_sql_env_factory(),
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume) if args.resume else trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    print(f"[train] done. LoRA adapter -> {final_dir}")


if __name__ == "__main__":
    main()
