# tests/test_dataset.py
import json
from sqlrl.dataset import load_bird


def test_load_bird_filters_and_builds_path(tmp_path):
    j = tmp_path / "train.json"
    j.write_text(json.dumps([
        {"db_id": "shop", "question": "q1", "evidence": "e", "SQL": "SELECT 1", "difficulty": "simple"},
        {"db_id": "shop", "question": "q2", "evidence": "", "SQL": "SELECT 2", "difficulty": "challenging"},
    ]))
    qs = load_bird(str(j), str(tmp_path / "dbs"), difficulties={"challenging"})
    assert len(qs) == 1
    assert qs[0].question == "q2" and qs[0].gold_sql == "SELECT 2"
    assert qs[0].db_path.replace("\\", "/").endswith("dbs/shop/shop.sqlite")


def test_load_bird_keeps_all_when_no_filter(tmp_path):
    j = tmp_path / "t.json"
    j.write_text(json.dumps([{"db_id": "a", "question": "x", "SQL": "SELECT 1", "difficulty": "simple"}]))
    assert len(load_bird(str(j), str(tmp_path), difficulties=None)) == 1


def test_load_bird_reads_jsonl(tmp_path):
    # bird23-train-filtered (the clean train source) ships as line-delimited JSONL, not a JSON array
    j = tmp_path / "filtered.jsonl"
    j.write_text(
        '{"db_id": "a", "question": "q1", "SQL": "SELECT 1", "evidence": ""}\n'
        '{"db_id": "b", "question": "q2", "SQL": "SELECT 2", "evidence": "e"}\n'
    )
    qs = load_bird(str(j), str(tmp_path), difficulties=None)
    assert len(qs) == 2
    assert qs[1].db_id == "b" and qs[1].gold_sql == "SELECT 2" and qs[1].evidence == "e"
