"""A LangChain create_agent harness. It intentionally owns its own loop."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
from pathlib import Path
from typing import cast

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

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


def _safe_calculate(expression: str) -> float:
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


@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression."""
    return _safe_calculate(expression)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable: {name}")
    return value


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def main() -> None:
    task = json.loads(Path(_required_env("RB_TASK_FILE")).read_text(encoding="utf-8"))
    model = ChatOpenAI(
        model="gateway-owned-model",
        base_url=_required_env("RB_GATEWAY_BASE_URL"),
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "rolloutbridge-local")),
        temperature=0.8,
    )
    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt=(
            "Solve the arithmetic problem. You must use the calculator tool, then return only "
            "the final numeric value."
        ),
    )
    state = agent.invoke(
        {"messages": [{"role": "user", "content": str(task["question"])}]},
        config={"recursion_limit": MAX_MODEL_CALLS * 2 + 1},
    )
    ai_messages = [message for message in state["messages"] if message.type == "ai"]
    if not ai_messages or len(ai_messages) > MAX_MODEL_CALLS:
        raise RuntimeError("model call limit exceeded")
    answer = _message_text(ai_messages[-1])
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", answer)
    predicted = float(matches[-1]) if matches else None
    expected = float(task["answer"])
    reward = float(predicted is not None and math.isclose(predicted, expected, abs_tol=1e-8))
    tool_messages = sum(message.type == "tool" for message in state["messages"])
    result = {
        "reward": reward,
        "metrics": {
            "model_calls": float(len(ai_messages)),
            "tool_calls": float(tool_messages),
        },
    }
    Path(_required_env("RB_RESULT_FILE")).write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
