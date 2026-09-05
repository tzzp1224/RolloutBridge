"""A plain OpenAI SDK tool loop. It intentionally does not import RolloutBridge."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
from pathlib import Path
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

MAX_MODEL_CALLS = 4
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> float:
    """Evaluate bounded arithmetic without names, calls, attributes, or subscripts."""
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST, depth: int = 0) -> float:
        if depth > 20:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(cast("int | float", node.value))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            value = _BINARY[type(node.op)](
                evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
            )
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = _UNARY[type(node.op)](evaluate(node.operand, depth + 1))
        else:
            raise ValueError("only basic numeric arithmetic is allowed")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("calculation is outside the allowed range")
        return value

    return evaluate(tree)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable: {name}")
    return value


def _answer_number(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(matches[-1]) if matches else None


def main() -> None:
    task = json.loads(Path(_required_env("RB_TASK_FILE")).read_text(encoding="utf-8"))
    question = str(task["question"])
    expected = float(task["answer"])
    client = OpenAI(
        base_url=_required_env("RB_GATEWAY_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "rolloutbridge-local"),
    )
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "Solve the arithmetic problem. You must use the calculator tool, then return "
                "only the final numeric value."
            ),
        },
        {"role": "user", "content": question},
    ]
    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    answer = ""
    calls = 0
    tool_calls_executed = 0
    while calls < MAX_MODEL_CALLS:
        response = client.chat.completions.create(
            model="gateway-owned-model",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1.2,
        )
        calls += 1
        if len(response.choices) != 1:
            raise RuntimeError("Gateway returned an invalid choice count")
        message = response.choices[0].message
        messages.append(cast("ChatCompletionMessageParam", message.model_dump(exclude_none=True)))
        if not message.tool_calls:
            answer = message.content or ""
            break
        for tool_call in message.tool_calls:
            if tool_call.type != "function" or tool_call.function.name != "calculator":
                raise RuntimeError("model requested an unsupported tool")
            arguments = json.loads(tool_call.function.arguments)
            if set(arguments) != {"expression"} or not isinstance(arguments["expression"], str):
                raise RuntimeError("model supplied invalid calculator arguments")
            observation = calculate(arguments["expression"])
            tool_calls_executed += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                }
            )
    else:
        raise RuntimeError("model call limit exceeded")

    predicted = _answer_number(answer)
    reward = float(predicted is not None and math.isclose(predicted, expected, abs_tol=1e-8))
    result = {
        "reward": reward,
        "metrics": {
            "model_calls": float(calls),
            "tool_calls": float(tool_calls_executed),
        },
    }
    Path(_required_env("RB_RESULT_FILE")).write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
