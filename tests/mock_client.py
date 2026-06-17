# tests/mock_client.py
import json
from dataclasses import dataclass


@dataclass
class _Func:
    name: str
    arguments: str          # JSON string, like the OpenAI SDK


@dataclass
class _ToolCall:
    id: str
    function: _Func


@dataclass
class _Msg:
    content: str | None
    tool_calls: list | None


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list


class _Completions:
    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def create(self, **kwargs):
        step = self._script[self._i]
        self._i += 1
        if step[0] == "tool":
            _, name, args = step
            tc = _ToolCall(id=f"call_{self._i}",
                           function=_Func(name=name, arguments=json.dumps(args)))
            return _Resp([_Choice(_Msg(content=None, tool_calls=[tc]))])
        # ("text", str) or ("final", str): plain assistant content
        return _Resp([_Choice(_Msg(content=step[1], tool_calls=None))])


class _Chat:
    def __init__(self, script):
        self.completions = _Completions(script)


class MockClient:
    """Script entries: ("tool", name, args_dict) | ("text", str) | ("final", str)."""
    def __init__(self, script):
        self.chat = _Chat(script)
