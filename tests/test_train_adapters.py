# tests/test_train_adapters.py
from sqlrl.prompts import SYSTEM_PROMPT
from sqlrl.tools import SqlToolset
from sqlrl.train_env import make_sql_env_factory
from sqlrl.train_reward import make_r1_reward

GOLD = "SELECT name FROM customers WHERE city='NYC'"


def test_env_factory_binds_db_per_rollout_and_exposes_tools(tiny_db):
    Env = make_sql_env_factory()
    env = Env()
    env.reset(db_path=tiny_db)                    # per-rollout DB binding
    assert "customers" in env.list_tables() and "orders" in env.list_tables()
    assert "city" in env.describe_table("customers")
    assert "Ann" in env.run_sql("SELECT name FROM customers WHERE city='NYC'")
    env.reset()                                   # rebinding closes prior + clears
    assert env.calls == []


def test_env_tool_names_match_eval_toolset(tiny_db):
    # fairness invariant: training env exposes the same tool names as the eval toolset
    Env = make_sql_env_factory()
    env_tools = {n for n in dir(Env) if n in {"list_tables", "describe_table", "run_sql"}}
    eval_tools = {t["function"]["name"] for t in SqlToolset(tiny_db).tool_specs}
    assert env_tools == eval_tools


def test_r1_reward_adapter_scores_completions(tiny_db):
    reward = make_r1_reward()
    comps = [
        [{"role": "assistant", "content": f"let me think\nFINAL SQL: {GOLD}"}],  # correct -> 1
        [{"role": "assistant", "content": "FINAL SQL: SELECT name FROM customers"}],  # wrong -> 0
        [{"role": "assistant", "content": "no sql here"}],                       # no SQL -> 0
        "FINAL SQL: " + GOLD,                                                     # plain-str completion -> 1
    ]
    out = reward(prompts=None, completions=comps,
                 gold_sql=[GOLD, GOLD, GOLD, GOLD], db_path=[tiny_db] * 4)
    assert out == [1.0, 0.0, 0.0, 1.0]


def test_system_prompt_is_shared():
    from sqlrl import agent
    assert agent.SYSTEM_PROMPT is SYSTEM_PROMPT     # agent imports the single source


def test_train_and_eval_configs_parse():
    # committed configs must validate against the schema (catches yaml/schema drift)
    from sqlrl.config import EvalSettings, TrainSettings, load_config

    t = load_config("configs/train.yaml", TrainSettings)
    assert t.reward_arm == "r1" and t.group_size >= 4 and t.lora_target_modules
    s = load_config("configs/smoke_train.yaml", TrainSettings)
    assert s.steps <= 3                              # smoke must stay tiny
    e = load_config("configs/eval.yaml", EvalSettings)
    assert e.base_url.endswith("/v1") and e.eval_json and e.temperature == 0.0
