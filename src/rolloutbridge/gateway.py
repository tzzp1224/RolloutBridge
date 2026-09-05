"""Exact-token Chat Completions proxy with rollout-scoped capture."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Response

from .types import ModelCallEvent


def _token_list(value: object) -> list[int] | None:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        return None
    return value


def _response_logprobs(choice: dict[str, Any]) -> Any:
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        if "token_logprobs" in logprobs:
            return logprobs["token_logprobs"]
        content = logprobs.get("content")
        if isinstance(content, list):
            return [item.get("logprob") if isinstance(item, dict) else None for item in content]
    return None


def create_app(upstream_base_url: str, model_id: str) -> FastAPI:
    """Create a single-process Gateway app."""
    upstream_url = f"{upstream_base_url.rstrip('/')}/v1/chat/completions"
    events: dict[str, list[ModelCallEvent]] = {}
    locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.http_client = httpx.AsyncClient(timeout=300.0)
        try:
            yield
        finally:
            await app.state.http_client.aclose()

    app = FastAPI(title="RolloutBridge Gateway", version="0.1.0", lifespan=lifespan)
    app.state.events = events
    app.state.locks = locks

    def rollout_lock(rollout_id: str) -> asyncio.Lock:
        return locks.setdefault(rollout_id, asyncio.Lock())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/rollouts/{rollout_id}/v1/chat/completions")
    async def chat_completions(rollout_id: str, payload: dict[str, Any]) -> Response:
        if payload.get("stream", False) is not False:
            raise HTTPException(status_code=400, detail="streaming is not supported")
        n = payload.get("n", 1)
        if type(n) is not int or n != 1:
            raise HTTPException(status_code=400, detail="only n=1 is supported")

        forwarded = dict(payload)
        forwarded["model"] = model_id
        forwarded["return_token_ids"] = True

        async with rollout_lock(rollout_id):
            try:
                upstream = await app.state.http_client.post(upstream_url, json=forwarded)
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502, detail=f"upstream request failed: {exc}"
                ) from exc

            content_type = upstream.headers.get("content-type", "application/json")
            if not 200 <= upstream.status_code < 300:
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=content_type.split(";", 1)[0],
                )

            try:
                body = upstream.json()
                choices = body["choices"]
                choice = choices[0]
                prompt_ids = _token_list(body.get("prompt_token_ids"))
                response_ids = _token_list(choice.get("token_ids"))
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                prompt_ids = response_ids = None
                choice = {}
            if prompt_ids is None or response_ids is None:
                raise HTTPException(
                    status_code=502,
                    detail="upstream response omitted serving-time token IDs",
                )

            rollout_events = events.setdefault(rollout_id, [])
            rollout_events.append(
                ModelCallEvent(
                    event_id=uuid.uuid4().hex,
                    rollout_id=rollout_id,
                    sequence_id=len(rollout_events),
                    prompt_token_ids=prompt_ids,
                    response_token_ids=response_ids,
                    response_logprobs=_response_logprobs(choice),
                )
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=content_type.split(";", 1)[0],
            )

    @app.get("/internal/rollouts/{rollout_id}/events")
    async def get_events(rollout_id: str) -> list[dict[str, Any]]:
        async with rollout_lock(rollout_id):
            return [event.model_dump(mode="json") for event in events.get(rollout_id, [])]

    @app.delete("/internal/rollouts/{rollout_id}")
    async def delete_events(rollout_id: str) -> dict[str, int]:
        async with rollout_lock(rollout_id):
            deleted = len(events.pop(rollout_id, []))
        return {"deleted": deleted}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RolloutBridge exact-token Gateway")
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(create_app(args.upstream_base_url, args.model_id), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
