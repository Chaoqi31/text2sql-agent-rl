# scripts/prepare_data.py
import argparse
import json
from collections import Counter
from pathlib import Path

from sqlrl.config import DataConfig, load_config
from sqlrl.dataset import load_bird


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config, DataConfig)

    diffs = set(cfg.difficulties) if cfg.difficulties else None
    questions = load_bird(cfg.bird_json, cfg.db_root, difficulties=diffs)

    missing = sorted({q.db_id for q in questions if not Path(q.db_path).exists()})
    out = Path(cfg.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q in questions:
            f.write(json.dumps(q.__dict__) + "\n")

    print(f"wrote {len(questions)} questions -> {out}")
    print("difficulty histogram:", dict(Counter(q.difficulty for q in questions)))
    if missing:
        print(f"WARNING: {len(missing)} db_ids have no sqlite file, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
