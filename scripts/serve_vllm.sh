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
# Qwen3.5 tool-call parser is qwen3_coder (per official model card, Phase -1.2).
PARSER="${TOOL_PARSER:-qwen3_coder}"

# Blackwell sm_120 workarounds (empirically confirmed): flashinfer JIT sampler mis-detects the
# arch and aborts; enforce-eager + sampler off make `vllm serve` start.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# --language-model-only: Qwen3.5 is multimodal; skip the vision encoder for text-only SQL
# (frees VRAM for KV cache). Per the official model card.
COMMON=(--port "$PORT" --max-model-len "$MAXLEN" --enforce-eager --language-model-only
        --enable-auto-tool-choice --tool-call-parser "$PARSER")

if [[ "$MODE" == "lora" ]]; then
  LORA_PATH="${2:?usage: serve_vllm.sh lora <lora_path>}"
  echo "serving $BASE_MODEL + LoRA($LORA_PATH) as 'sqlrl-lora' on :$PORT"
  exec vllm serve "$BASE_MODEL" "${COMMON[@]}" --enable-lora --lora-modules "sqlrl-lora=$LORA_PATH"
else
  echo "serving base $BASE_MODEL on :$PORT"
  exec vllm serve "$BASE_MODEL" "${COMMON[@]}"
fi
