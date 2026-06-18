#!/usr/bin/env bash
# One-time GPU env setup on the AutoDL box (sm_120 Blackwell, RTX PRO 6000). Phase -1.1.
# AutoDL notes: disable the academic proxy (it breaks domestic mirrors), install via a
# domestic pip mirror, sm_120 needs CUDA 12.9+/driver >= 575 for vLLM.
set -euo pipefail
cd "$(dirname "$0")/.."

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP="pip install -i $MIRROR"

echo "== python ==" && python --version
python -m pip install -U -i "$MIRROR" pip

echo "== sqlrl (offline core, editable) ==" && $PIP -e .
echo "== TRL training stack ==" && $PIP transformers trl peft accelerate datasets openai
echo "== vLLM (pulls a matching torch; sm_120 needs a recent build) ==" && $PIP vllm

echo "== versions + cuda =="
python - <<'PY'
for m in ["torch","transformers","trl","peft","accelerate","datasets","vllm","openai"]:
    try:
        mod = __import__(m); print(f"{m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:14s} MISSING: {e}")
import torch
print("cuda:", torch.cuda.is_available(),
      "| cap:", torch.cuda.get_device_capability() if torch.cuda.is_available() else "-",
      "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "setup done. Next: python scripts/dump_modules.py   (Phase -1.2: confirm LoRA targets)"
echo "then:        python scripts/train_grpo.py --config configs/smoke_train.yaml   (Phase -1.4 green gate)"
