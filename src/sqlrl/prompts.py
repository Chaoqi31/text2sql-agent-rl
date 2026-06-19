# src/sqlrl/prompts.py
# Single source of truth for the agent's system prompt — imported by BOTH the eval
# loop (agent.run_agent) and the training dataset builder (scripts/train_grpo.py),
# so base and tuned see identical prompting (fairness invariant, spec §11).
# NOTE: a more verbose "inspect->test->verify" prompt was tried (v2) and HURT — it pushed
# rollouts to clip at max_tokens without emitting FINAL SQL (reward collapsed to 0). Reverted
# to this terse known-good prompt. Prompt optimization is a separate careful task: A/B base EX
# on a held-out slice BEFORE training, don't change it blind.
SYSTEM_PROMPT = (
    "You are a Text-to-SQL agent. Inspect the database with the tools, then answer.\n"
    "Tools: list_tables; describe_table(table); run_sql(query) — read-only SELECT only.\n"
    "When confident, output the final query on its own line exactly as:\n"
    "FINAL SQL: <single SELECT query>"
)
