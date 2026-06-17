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


def load_config(path: str, model_cls):
    return model_cls(**yaml.safe_load(Path(path).read_text()))
