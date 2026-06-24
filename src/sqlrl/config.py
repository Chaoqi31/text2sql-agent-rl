# src/sqlrl/config.py
from pathlib import Path
import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    bird_json: str
    db_root: str
    out_jsonl: str
    difficulties: list[str] | None = None      # None = keep all


class SmokeConfig(BaseModel):
    questions_jsonl: str
    n_questions: int = 2
    max_turns: int = 8


class TrainSettings(BaseModel):
    # model / run
    base_model: str                                     # path on the box, e.g. .../models/Qwen3.5-9B
    run_name: str = "sqlrl-r1-v1"
    train_jsonl: str                                    # prepared/filtered question jsonl
    db_root: str                                        # train_databases root (for db_path resolve)
    reward_arm: str = "r1"                              # plan-2 = r1 only; r2/r3/s0 -> plan-3
    # GRPO loop
    steps: int = 60
    scenarios_per_step: int = 8                         # prompts per optimizer step (grad accum)
    group_size: int = 8                                 # rollouts per prompt (group-relative baseline)
    learning_rate: float = 1.0e-5
    kl_coef: float = 0.0
    # rollout
    max_turns: int = 8
    temperature: float = 0.7                            # exploration during training
    max_tokens: int = 1024
    # LoRA (target_modules fixed in Phase -1 by dumping the module tree)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = ["gate_proj", "up_proj", "down_proj",
                                      "q_proj", "k_proj", "v_proj", "o_proj"]
    # rollout engine
    vllm_gpu_memory_utilization: float = 0.40


class SftSettings(BaseModel):
    # SFT cold-start before R2 GRPO (plan WS2). Single-turn: schema-in-prompt -> reasoning -> FINAL SQL.
    base_model: str                                     # box path, e.g. .../models/Qwen3.5-9B
    run_name: str = "sqlrl-sft-v1"
    sft_jsonl: str                                      # output of scripts/prepare_sft.py
    epochs: float = 2.0
    learning_rate: float = 1.0e-5
    max_seq_len: int = 4096
    per_device_batch_size: int = 4
    grad_accum: int = 8
    assistant_only_loss: bool = False                  # True for agentic data: train only on
                                                       # assistant turns (tool calls + FINAL SQL)
    # LoRA — same targets as GRPO (config consistency)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = ["gate_proj", "up_proj", "down_proj",
                                      "q_proj", "k_proj", "v_proj", "o_proj"]
    merge_after: bool = True                            # merge adapter into base -> runs/<run_name>/merged


class EvalSettings(BaseModel):
    base_url: str = "http://localhost:8000/v1"          # vllm serve OpenAI-compat endpoint
    api_key: str = "EMPTY"                              # vLLM ignores it; SDK requires a value
    model_name: str                                     # served model id (base path or lora module name)
    eval_json: str                                      # bird_mini_dev sqlite json
    db_root: str                                        # dev_databases root
    n_questions: int = 0                                # 0 = all
    max_turns: int = 8
    temperature: float = 0.0                            # greedy for reproducible eval
    max_tokens: int = 1024
    concurrency: int = 8                                # parallel rollouts (vLLM batches them)


def load_config(path: str, model_cls):
    return model_cls(**yaml.safe_load(Path(path).read_text()))
