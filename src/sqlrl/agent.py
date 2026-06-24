# src/sqlrl/agent.py
import json
import re

from sqlrl.prompts import SYSTEM_PROMPT
from sqlrl.schema import AgentRollout

_FINAL_RE = re.compile(r"FINAL\s+SQL:\s*(.*)", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.+?)```", re.IGNORECASE | re.DOTALL)


def extract_final_sql(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_RE.search(text)
    if not m:
        return None
    rest = m.group(1).lstrip()
    fence = _FENCE_RE.match(rest)
    if fence:
        sql = fence.group(1).strip()          # fenced ```sql ...``` keeps multi-line SQL
    else:
        # unfenced: first line only (drop next-line prose), then cut same-line trailing
        # prose after the statement terminator
        sql = rest.split("\n", 1)[0].split(";", 1)[0].strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql or None


def _build_messages(question):
    user = f"Question: {question.question}\nDatabase: {question.db_id}"
    if question.evidence:
        user += f"\nEvidence: {question.evidence}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def run_agent(client, model_name: str, question, toolset, cfg) -> AgentRollout:
    toolset.reset()
    messages = _build_messages(question)
    tool_calls: list[tuple[str, dict]] = []
    final_sql, finished, turns = None, False, 0

    for turns in range(1, cfg.max_turns + 1):
        resp = client.chat.completions.create(
            model=model_name, messages=messages, tools=toolset.tool_specs,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        assistant_msg = {"role": "assistant", "content": msg.content}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ]
        messages.append(assistant_msg)

        fs = extract_final_sql(msg.content or "")
        if fs is not None:
            final_sql, finished = fs, True
            break

        if calls:
            for c in calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append((name, args))
                if cfg.single_shot and name == "run_sql":
                    final_sql, finished = args.get("query"), True
                    break
                result = toolset.dispatch(name, args)
                messages.append({"role": "tool", "tool_call_id": c.id, "name": name, "content": result})
            if finished:
                break
            continue

        messages.append({"role": "user",
                         "content": "Continue. Use a tool or output 'FINAL SQL: <query>'."})

    # salvage: if the agent never committed a FINAL SQL, fall back to its last run_sql query
    # (a query it actually executed) so a correct-but-uncommitted answer isn't auto-zeroed.
    fallback_sql = next((args.get("query") for name, args in reversed(tool_calls)
                         if name == "run_sql" and args.get("query")), None)
    return AgentRollout(messages=messages, final_sql=final_sql, turns=turns,
                        finished=finished, tool_calls=tool_calls, fallback_sql=fallback_sql)
