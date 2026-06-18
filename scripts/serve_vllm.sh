#!/usr/bin/env bash
# Serve Qwen3.5-9B (base or +LoRA) with vLLM's OpenAI server for eval. Phase 0/1.
#
#   bash scripts/serve_vllm.sh base
#   bash scripts/serve_vllm.sh lora runs/sqlrl-r1-v1/final
# Then: python scripts/evaluate.py --tag baseline      (or --tag tuned --model-name sqlrl-lora)
set -euo pipefail

MODE="${1:-base}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/text2sql/models/Qwen3.5-9B}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-8192}"
# Qwen2.5 used hermes; CONFIRM Qwen3.5's parser in Phase -1.3 and override if needed.
PARSER="${TOOL_PARSER:-hermes}"

# Blackwell sm_120 workarounds (prior-project-confirmed): flashinfer JIT sampler mis-detects the
# arch and aborts; enforce-eager + sampler off make `vllm serve` start.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

COMMON=(--port "$PORT" --max-model-len "$MAXLEN" --enforce-eager
        --enable-auto-tool-choice --tool-call-parser "$PARSER")

if [[ "$MODE" == "lora" ]]; then
  LORA_PATH="${2:?usage: serve_vllm.sh lora <lora_path>}"
  echo "serving $BASE_MODEL + LoRA($LORA_PATH) as 'sqlrl-lora' on :$PORT"
  exec vllm serve "$BASE_MODEL" "${COMMON[@]}" --enable-lora --lora-modules "sqlrl-lora=$LORA_PATH"
else
  echo "serving base $BASE_MODEL on :$PORT"
  exec vllm serve "$BASE_MODEL" "${COMMON[@]}"
fi
