# text2sql-agent-rl

Agentic **Text-to-SQL** trained with reinforcement learning (GRPO). A general-purpose
**Qwen3.5-9B** base — *not* a SQL-specialized model — is taught to explore a database with
tools and write correct SQL, lifting execution accuracy **+7.2 points** on the full BIRD dev set.

## Result (full BIRD-dev, n=1534)

| Model | Execution Accuracy | Δ vs base | Finished |
|---|---|---|---|
| Qwen3.5-9B base | 0.4498 | — | 82% |
| + RL, R1 reward (execution-only) | 0.4967 | +4.7 pt (p≈0.009) | 89% |
| **+ RL, R2 reward (partial credit)** | **0.5215** | **+7.2 pt (p<0.0001)** | **90%** |

EX = BIRD official execution-match (result set-equality), greedy decoding. RL also makes the
agent more decisive: the *finished*-rate (emits a final answer) rises 82% → 90%. Single training
seed; deltas are statistically significant at n=1534 (binomial z-test), a multi-seed CI is future work.

## What this is

Most Text-to-SQL RL work starts from a SQL-specialized base (e.g. OmniSQL). This project starts
from a **general** instruction model and shows that RL alone closes a meaningful chunk of the gap.
The model runs as an **agent**: it inspects the schema, runs trial queries against a read-only
SQLite copy, observes the results, and self-corrects before committing a final query.

### Agent loop

Read-only tools:
- `list_tables` — list tables in the database
- `describe_table(table)` — schema for one table
- `run_sql(query)` — execute a **SELECT** (read-only, timeout + row cap) and observe the result

The agent reasons over tool outputs across up to 8 turns, then emits `FINAL SQL: <query>`.

### Training

- **Algorithm:** GRPO (group-relative policy optimization) via TRL, LoRA adapters, colocated
  vLLM for rollout generation.
- **LoRA:** r16 / α32 on MLP `{gate,up,down}_proj` + full-attention `{q,k,v,o}_proj`
  (Qwen3.5's DeltaNet linear-attention layers are left un-adapted).
- **Scale:** 150 steps, group size 4, lr 1e-5, KL 0 (fits a single 96 GB GPU).
- **Data:** BIRD train, gold-execution-filtered (drop gold SQL that errors or returns empty, so
  GRPO groups are not poisoned) — 6583 / 6601 kept.

### Reward

The recommended reward is **R2** — faithful partial credit, fully deterministic, no LLM judge:

```
reward = 3·exec + 1·syntax + 1·schema + 1·ngram + 1·format
```

- `exec` — execution match vs gold (0/1)
- `syntax` — query runs read-only without error
- `schema` — Jaccard of {tables ∪ columns} vs gold (sqlglot-parsed)
- `ngram` — token-bigram Jaccard vs gold SQL
- `format` — a valid SELECT was emitted

Partial terms score 0 unless a valid SELECT is produced, and `w_exec=3` keeps a correct query
strictly above any incorrect one. The partial credit gives a denser gradient than execution-only
(R1) and trains faster. Other arms in the repo: `r1` (execution-only 0/1), `s0` (single-shot, no
tool loop — ablation), `r3` (an intentionally gameable process-reward foil).

## Layout

```
src/sqlrl/          core (offline, CPU-testable)
  agent.py            agent loop (run_agent)
  tools.py            SqlToolset: list_tables / describe_table / run_sql
  db.py               read-only sqlite engine (write-guard, timeout, truncation)
  ex.py               BIRD execution-match
  reward.py           R1 / R2 / R3 reward functions
  prompts.py          single shared system prompt (eval + train use the same one)
  dataset.py          BIRD loader
  runner.py           eval runner + summarize
  train_env.py        TRL environment factory (tools as rollout actions)
  train_reward.py     TRL reward adapters
scripts/            entry points (data prep, train, serve, eval, compare)
configs/            yaml configs (train / eval / data)
tests/              51 offline tests
```

## Setup

Core + tests (offline, CPU only):

```bash
pip install -e ".[dev]"
```

Evaluation additionally needs `openai` (talks to the vLLM OpenAI-compatible server). Training
additionally needs `torch`, `trl`, `peft`, and `vllm` on a CUDA box — these are intentionally not
pinned here; install versions matching your GPU/CUDA. See `scripts/setup_gpu.sh` for the exact
setup used (AutoDL, Blackwell sm_120).

## Usage

```bash
# 1. prepare + execution-filter the BIRD training data
python scripts/prepare_data.py --config configs/data.yaml
python scripts/filter_data.py --in-json <prepared.jsonl> --db-root <train_databases> \
    --out-jsonl train_exec_filtered.jsonl

# 2. train (R2 reward)
python scripts/train_grpo.py --config configs/train.yaml --reward-arm r2 --run-name sqlrl-r2

# 3. serve a checkpoint with vLLM (base or +LoRA adapter)
bash scripts/serve_vllm.sh base
bash scripts/serve_vllm.sh lora runs/sqlrl-r2/final

# 4. evaluate on full BIRD-dev and compare
python scripts/evaluate.py --config configs/eval.yaml --tag base
python scripts/evaluate.py --config configs/eval.yaml --tag r2 --model-name sqlrl-lora
python scripts/compare.py --baseline runs/eval_base.jsonl --tuned runs/eval_r2.jsonl
```

Config paths point at the GPU-box layout used during development — edit them for your environment.

## Tests

```bash
pytest -q          # 51 offline tests, no GPU required
```

## Limitations

- Single training seed (n=1534 makes the deltas significant, but a multi-seed CI would fully
  nail the headline number).
- Config files are pinned to a specific GPU-box path layout; edit them before running.
- Remaining eval losses are mostly context-overflow on long conversations — raising max_tokens /
  max_turns / context length is the obvious next lever.

## References

- BIRD benchmark — Li et al., *Can LLM Already Serve as a Database Interface?* (arXiv:2305.03111)
- R2 partial-reward shaping adapted from *Reasoning-SQL* (arXiv:2503.23157), with the AI-feedback term omitted.
- Base model: Qwen3.5-9B.
