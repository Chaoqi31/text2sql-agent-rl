#!/usr/bin/env python3
"""Phase -1.2: dump the Qwen3.5-9B module tree to choose LoRA target_modules.

Qwen3.5 is a hybrid: 3x linear_attention (DeltaNet) + 1x full_attention per block.
DeltaNet layers lack the usual q/k/v/o_proj, so this prints the Linear-leaf names that
actually exist (the LoRA-targetable set) plus one block's full tree for layer-type context.

    python scripts/dump_modules.py --model /root/autodl-tmp/text2sql/models/Qwen3.5-9B
"""
import argparse
import collections

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/text2sql/models/Qwen3.5-9B")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    suffixes = collections.Counter()
    for name, mod in model.named_modules():
        if "Linear" in mod.__class__.__name__:
            suffixes[name.split(".")[-1]] += 1
    print("Linear-leaf suffixes (candidate LoRA target_modules):")
    for s, n in suffixes.most_common():
        print(f"  {s:28s} x{n}")

    print("\nfirst 80 module names (layer-type context):")
    for i, (name, _) in enumerate(model.named_modules()):
        if i >= 80:
            break
        print("  ", name)


if __name__ == "__main__":
    main()
