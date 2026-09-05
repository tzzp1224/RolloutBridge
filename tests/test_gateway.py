from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import httpx
import pytest

from rolloutbridge.gateway import create_app


def upstream_response(
    *,
    status: int = 200,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    logprobs: object = None,
) -> httpx.Response:
    body: dict[str, Any] = {
        "id": "completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    }
    if prompt_ids is not None:
        body["prompt_token_ids"] = prompt_ids
    if response_ids is not None:
        body["choices"][0]["token_ids"] = response_ids
    if logprobs is not None:
        body["choices"][0]["logprobs"] = logprobs
    return httpx.Response(
        status,
        json=body if status < 400 else {"error": "upstream"},
        request=httpx.Request("POST", "http://upstream/v1/chat/completions"),
    )


class StubUpstream:
    def __init__(self, responses: list[httpx.Response] | None = None, delay: float = 0) -> None:
        self.responses = responses or []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.delay = delay
        self.active: defaultdict[str, int] = defaultdict(int)
        self.max_active: defaultdict[str, int] = defaultdict(int)
        self.total_active = 0
        self.max_total_active = 0

    async def post(self, url: str, *, json: dict[str, Any]) -> httpx.Response:
        marker = str(json.get("marker", "default"))
        self.active[marker] += 1
        self.max_active[marker] = max(self.max_active[marker], self.active[marker])
        self.total_active += 1
        self.max_total_active = max(self.max_total_active, self.total_active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.calls.append((url, json))
            if self.responses:
                return self.responses.pop(0)
            return upstream_response(prompt_ids=[1], response_ids=[2])
        finally:
            self.active[marker] -= 1
            self.total_active -= 1

    async def aclose(self) -> None:
        return None


async def app_client(app: Any, stub: StubUpstream):
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    await app.state.http_client.aclose()
    app.state.http_client = stub
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return lifespan, client


@pytest.mark.asyncio
async def test_rewrites_request_captures_exact_tokens_and_logprobs() -> None:
    app = create_app("http://upstream", "fixed-model")
    stub = StubUpstream(
        [
            upstream_response(
                prompt_ids=[1, 2],
                response_ids=[3, 4],
                logprobs={"token_logprobs": [-0.1, -0.2]},
            )
        ]
    )
    lifespan, client = await app_client(app, stub)
    try:
        response = await client.post(
            "/rollouts/r1/v1/chat/completions",
            json={"model": "wrong", "messages": [], "return_token_ids": False},
        )
        assert response.status_code == 200
        assert stub.calls[0][0] == "http://upstream/v1/chat/completions"
        assert stub.calls[0][1]["model"] == "fixed-model"
        assert stub.calls[0][1]["return_token_ids"] is True
        captured = (await client.get("/internal/rollouts/r1/events")).json()
        assert captured[0]["sequence_id"] == 0
        assert captured[0]["prompt_token_ids"] == [1, 2]
        assert captured[0]["response_token_ids"] == [3, 4]
        assert captured[0]["response_logprobs"] == [-0.1, -0.2]
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_upstream_error_is_preserved_without_event() -> None:
    app = create_app("http://upstream", "model")
    lifespan, client = await app_client(app, StubUpstream([upstream_response(status=429)]))
    try:
        response = await client.post("/rollouts/r/v1/chat/completions", json={"messages": []})
        assert response.status_code == 429
        assert response.json() == {"error": "upstream"}
        assert (await client.get("/internal/rollouts/r/events")).json() == []
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_missing_tokens_fails_fast_and_bad_optional_logprobs_are_dropped() -> None:
    app = create_app("http://upstream", "model")
    stub = StubUpstream(
        [
            upstream_response(prompt_ids=[1]),
            upstream_response(
                prompt_ids=[1],
                response_ids=[2, 3],
                logprobs={"token_logprobs": [-0.1]},
            ),
        ]
    )
    lifespan, client = await app_client(app, stub)
    try:
        missing = await client.post("/rollouts/r/v1/chat/completions", json={"messages": []})
        assert missing.status_code == 502
        good = await client.post("/rollouts/r/v1/chat/completions", json={"messages": []})
        assert good.status_code == 200
        captured = (await client.get("/internal/rollouts/r/events")).json()
        assert len(captured) == 1
        assert captured[0]["response_logprobs"] is None
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"stream": True}, {"n": 2}, {"n": True}])
async def test_rejects_streaming_and_multiple_choices(payload: dict[str, Any]) -> None:
    app = create_app("http://upstream", "model")
    stub = StubUpstream()
    lifespan, client = await app_client(app, stub)
    try:
        response = await client.post("/rollouts/r/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert stub.calls == []
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_per_rollout_serialization_and_cross_rollout_concurrency() -> None:
    app = create_app("http://upstream", "model")
    stub = StubUpstream(delay=0.02)
    lifespan, client = await app_client(app, stub)
    try:
        await asyncio.gather(
            *(
                client.post("/rollouts/a/v1/chat/completions", json={"marker": "same"})
                for _ in range(3)
            )
        )
        assert stub.max_active["same"] == 1
        assert [
            item["sequence_id"] for item in (await client.get("/internal/rollouts/a/events")).json()
        ] == [0, 1, 2]

        await asyncio.gather(
            client.post("/rollouts/b/v1/chat/completions", json={"marker": "b"}),
            client.post("/rollouts/c/v1/chat/completions", json={"marker": "c"}),
        )
        assert stub.max_total_active == 2
        assert len((await client.get("/internal/rollouts/b/events")).json()) == 1
        assert len((await client.get("/internal/rollouts/c/events")).json()) == 1
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_get_delete_are_isolated_and_delete_is_idempotent() -> None:
    app = create_app("http://upstream", "model")
    lifespan, client = await app_client(app, StubUpstream())
    try:
        await client.post("/rollouts/a/v1/chat/completions", json={})
        await client.post("/rollouts/b/v1/chat/completions", json={})
        assert (await client.delete("/internal/rollouts/a")).json() == {"deleted": 1}
        assert (await client.delete("/internal/rollouts/a")).json() == {"deleted": 0}
        assert (await client.get("/internal/rollouts/a/events")).json() == []
        assert len((await client.get("/internal/rollouts/b/events")).json()) == 1
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)
