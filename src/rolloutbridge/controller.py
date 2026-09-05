"""Local subprocess ownership and RawCapture persistence."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import ModelCallEvent, ModelIdentity, RawCapture, RolloutResult, RolloutSpec


class _HarnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    reward: float | None
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("reward", mode="before")
    @classmethod
    def reject_boolean_reward(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("reward must be a number or null")
        return value


class LocalController:
    def __init__(
        self,
        gateway_url: str,
        artifacts_dir: str | Path,
        model_id: str,
        model_revision: str | None = None,
        timeout_seconds: float = 300,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.gateway_url = gateway_url.rstrip("/")
        self.artifacts_dir = Path(artifacts_dir)
        self.model_id = model_id
        self.model_revision = model_revision
        self.timeout_seconds = timeout_seconds

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def run(
        self,
        spec: RolloutSpec,
        command: Sequence[str],
        cwd: str | Path | None = None,
    ) -> RolloutResult:
        argv = self._validate_command(command)
        with tempfile.TemporaryDirectory(prefix="rolloutbridge-") as temp_name:
            temp_dir = Path(temp_name)
            task_file = temp_dir / "task.json"
            result_file = temp_dir / "result.json"
            self._write_json(task_file, spec.task)

            env = os.environ.copy()
            env.update(
                {
                    "RB_ROLLOUT_ID": spec.rollout_id,
                    "RB_GATEWAY_BASE_URL": (f"{self.gateway_url}/rollouts/{spec.rollout_id}/v1"),
                    "RB_TASK_FILE": str(task_file),
                    "RB_RESULT_FILE": str(result_file),
                    "RB_MODE": spec.mode,
                }
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                start_new_session=True,
            )
            timed_out = False
            try:
                exit_code = await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
            except TimeoutError:
                timed_out = True
                await self._terminate_process_group(process)
                exit_code = 124

            result = self._derive_result(spec.rollout_id, exit_code, timed_out, result_file)

        await self._capture_and_cleanup(spec, result)
        return result

    @staticmethod
    def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
        if isinstance(command, (str, bytes)) or not command:
            raise TypeError("command must be a non-empty argv sequence")
        if any(not isinstance(arg, str) or not arg for arg in command):
            raise TypeError("every command argument must be a non-empty string")
        return tuple(command)

    @staticmethod
    async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    @staticmethod
    def _derive_result(
        rollout_id: str,
        exit_code: int,
        timed_out: bool,
        result_file: Path,
    ) -> RolloutResult:
        if timed_out:
            return RolloutResult(rollout_id=rollout_id, status="failed", reward=None, exit_code=124)
        if exit_code != 0:
            return RolloutResult(
                rollout_id=rollout_id, status="failed", reward=None, exit_code=exit_code
            )
        try:
            payload: Any = json.loads(result_file.read_text(encoding="utf-8"))
            harness_result = _HarnessResult.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError):
            return RolloutResult(rollout_id=rollout_id, status="failed", reward=None, exit_code=1)
        return RolloutResult(
            rollout_id=rollout_id,
            status="succeeded",
            reward=harness_result.reward,
            exit_code=0,
            metrics=harness_result.metrics,
        )

    async def _capture_and_cleanup(self, spec: RolloutSpec, result: RolloutResult) -> None:
        event_url = f"{self.gateway_url}/internal/rollouts/{spec.rollout_id}/events"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(event_url)
            response.raise_for_status()
            events = [ModelCallEvent.model_validate(item) for item in response.json()]
            capture = RawCapture(
                model=ModelIdentity(id=self.model_id, revision=self.model_revision),
                spec=spec,
                result=result,
                events=events,
            )
            self._persist_capture(capture)
            cleanup = await client.delete(f"{self.gateway_url}/internal/rollouts/{spec.rollout_id}")
            cleanup.raise_for_status()

    def _persist_capture(self, capture: RawCapture) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        target = self.artifacts_dir / f"rollout_{capture.spec.rollout_id}.json"
        if target.resolve().parent != self.artifacts_dir.resolve():
            raise ValueError("rollout_id is not safe for an artifact filename")
        temporary = target.with_suffix(".json.tmp")
        self._write_json(temporary, capture.model_dump(mode="json"))
        temporary.replace(target)

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
