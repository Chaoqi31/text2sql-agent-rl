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
