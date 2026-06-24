#!/usr/bin/env python3
"""SFT cold-start (plan WS2). GPU only. Single-turn: schema-in-prompt -> reasoning -> FINAL SQL.

    python scripts/train_sft.py --config configs/sft.yaml

LoRA adapter -> runs/<run_name>/final; merged model -> runs/<run_name>/merged (if merge_after).
Then point configs/train.yaml base_model at the merged dir and run R2 GRPO unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from sqlrl.config import SftSettings, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config, SftSettings)
    if args.base_model:
        cfg.base_model = args.base_model

    ds = load_dataset("json", data_files=cfg.sft_jsonl, split="train")  # conversational {messages}
    # load tokenizer explicitly — SFTTrainer's auto-loaded processor dropped the chat_template
    # on this base (transformers 5.x); the explicit one carries it and renders {messages}.
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    output_dir = str(REPO / "runs" / cfg.run_name)
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        max_length=cfg.max_seq_len,            # TRL>=0.12 (older: max_seq_length); not pinned
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        assistant_only_loss=cfg.assistant_only_loss,
        logging_steps=10,
        save_strategy="steps",          # periodic checkpoints — long run, survive disconnect
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                      target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM")

    print(f"[sft] {cfg.base_model} | {len(ds)} examples | {cfg.epochs} epochs")
    trainer = SFTTrainer(model=cfg.base_model, args=sft_cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=args.resume) if args.resume else trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    print(f"[sft] adapter -> {final_dir}")

    if cfg.merge_after:
        merged = trainer.model.merge_and_unload()
        merged_dir = Path(output_dir) / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[sft] merged -> {merged_dir}  (set configs/train.yaml base_model to this)")


if __name__ == "__main__":
    main()
