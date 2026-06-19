# tests/test_runner.py
from sqlrl.runner import run_questions, summarize
from sqlrl.schema import Question, AgentConfig
from sqlrl.reward import reward_r1
from tests.mock_client import MockClient, _Choice, _Msg, _Resp

GOLD = "SELECT name FROM customers WHERE city='NYC'"


class _StatelessClient:
    """Thread-safe stand-in: every create() returns the same FINAL SQL (no script index)."""
    def __init__(self, sql):
        self._content = f"FINAL SQL: {sql}"
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return _Resp([_Choice(_Msg(content=self._content, tool_calls=None))])


def test_end_to_end_offline_smoke(tiny_db):
    q = Question(db_id="shop", question="NYC names?", gold_sql=GOLD, db_path=tiny_db)
    client = MockClient([
        ("tool", "list_tables", {}),
        ("final", f"FINAL SQL: {GOLD}"),
    ])
    res = run_questions(client, "mock", [q], agent_cfg=AgentConfig(max_turns=4), reward_fn=reward_r1)
    assert len(res) == 1
    assert res[0].ex == 1 and res[0].reward == 1.0 and res[0].finished
    s = summarize(res)
    assert s["ex_rate"] == 1.0 and s["finished_rate"] == 1.0 and s["valid_sql_rate"] == 1.0


def test_run_questions_concurrent_matches_sequential_and_preserves_order(tiny_db):
    qs = [Question(db_id=f"q{i}", question="x", gold_sql=GOLD, db_path=tiny_db) for i in range(6)]
    client = _StatelessClient(GOLD)
    cfg = AgentConfig(max_turns=2)
    seq = run_questions(client, "m", qs, agent_cfg=cfg, reward_fn=reward_r1, concurrency=1)
    par = run_questions(client, "m", qs, agent_cfg=cfg, reward_fn=reward_r1, concurrency=4)
    assert [r.ex for r in seq] == [1] * 6 == [r.ex for r in par]
    assert [r.db_id for r in par] == [f"q{i}" for i in range(6)]   # order preserved


class _RaisingClient:
    chat = None
    def __init__(self):
        self.chat = self
        self.completions = self
    def create(self, **kwargs):
        raise RuntimeError("simulated transient API error")


def test_run_questions_survives_a_rollout_error(tiny_db):
    qs = [Question(db_id="ok", question="x", gold_sql=GOLD, db_path=tiny_db),
          Question(db_id="bad", question="x", gold_sql=GOLD, db_path=tiny_db)]
    # one good (stateless), one that raises — eval must not crash; the bad one scores 0
    good = _StatelessClient(GOLD)
    res_bad = run_questions(_RaisingClient(), "m", [qs[1]], agent_cfg=AgentConfig(max_turns=2),
                            reward_fn=reward_r1, concurrency=2)
    assert len(res_bad) == 1 and res_bad[0].ex == 0 and "error" in res_bad[0].breakdown
    res_good = run_questions(good, "m", [qs[0]], agent_cfg=AgentConfig(max_turns=2),
                             reward_fn=reward_r1, concurrency=2)
    assert res_good[0].ex == 1
