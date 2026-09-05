from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException

from rolloutbridge.compiler import compile_trajectory
from rolloutbridge.controller import LocalController
from rolloutbridge.gateway import create_app
from rolloutbridge.types import RawCapture, RolloutSpec


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(app: FastAPI, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2).is_success:
                return server, thread
        except httpx.HTTPError:
            time.sleep(0.02)
    raise RuntimeError(f"server on port {port} did not start")


def fake_vllm() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completion(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model") != "fake-model" or payload.get("return_token_ids") is not True:
            raise HTTPException(400, "Gateway did not enforce exact-token request fields")
        messages = payload.get("messages", [])
        has_observation = any(message.get("role") == "tool" for message in messages)
        if not has_observation:
            return {
                "id": "first",
                "object": "chat.completion",
                "created": 0,
                "model": "fake-model",
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "token_ids": [10, 11],
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "calculation",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": json.dumps({"expression": "6 * 7"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        return {
            "id": "second",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "prompt_token_ids": [1, 2, 10, 11, 20, 21],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": [30],
                    "message": {"role": "assistant", "content": "42"},
                }
            ],
        }

    return app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_both_real_harnesses_cross_http_and_compile_exact_tokens(tmp_path: Path) -> None:
    upstream_port = free_port()
    gateway_port = free_port()
    upstream_server, upstream_thread = start_server(fake_vllm(), upstream_port)
    gateway_server, gateway_thread = start_server(
        create_app(f"http://127.0.0.1:{upstream_port}", "fake-model"), gateway_port
    )
    root = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    try:
        for harness_name, script in (
            ("openai", root / "examples/openai_tool_agent/run.py"),
            ("langgraph", root / "examples/langgraph_agent/run.py"),
        ):
            rollout_id = f"integration-{harness_name}"
            spec = RolloutSpec(
                rollout_id=rollout_id,
                task_id="arithmetic",
                task={"question": "What is 6 * 7?", "answer": 42},
            )
            controller = LocalController(
                f"http://127.0.0.1:{gateway_port}", artifacts, "fake-model", timeout_seconds=30
            )
            result = await controller.run(spec, [sys.executable, str(script)], cwd=root)
            assert result.status == "succeeded"
            assert result.reward == 1
            assert result.metrics == {"model_calls": 2, "tool_calls": 1}

            capture = RawCapture.model_validate_json(
                (artifacts / f"rollout_{rollout_id}.json").read_text(encoding="utf-8")
            )
            assert len(capture.events) == 2
            row = compile_trajectory(capture.events, capture.result, capture.spec.task_id)[0]
            assert row.continuation_ids == [10, 11, 20, 21, 30]
            assert row.loss_mask == [1, 1, 0, 0, 1]
            assert [segment.kind for segment in row.segments] == [
                "initial_context",
                "model_response",
                "context_delta",
                "model_response",
            ]
    finally:
        gateway_server.should_exit = True
        upstream_server.should_exit = True
        gateway_thread.join(timeout=5)
        upstream_thread.join(timeout=5)
