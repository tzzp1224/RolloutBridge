from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

import rolloutbridge.controller as controller_module
from rolloutbridge.controller import LocalController
from rolloutbridge.types import RolloutSpec


class FakeClient:
    events: ClassVar[list[dict[str, Any]]] = []
    get_status: ClassVar[int] = 200
    deleted: ClassVar[list[str]] = []

    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get(self, url: str) -> httpx.Response:
        return httpx.Response(
            self.get_status,
            json=self.events if self.get_status == 200 else {"error": "down"},
            request=httpx.Request("GET", url),
        )

    async def delete(self, url: str) -> httpx.Response:
        self.deleted.append(url)
        return httpx.Response(
            200, json={"deleted": len(self.events)}, request=httpx.Request("DELETE", url)
        )


@pytest.fixture(autouse=True)
def fake_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.events = []
    FakeClient.get_status = 200
    FakeClient.deleted = []
    monkeypatch.setattr(controller_module.httpx, "AsyncClient", FakeClient)


def spec(rollout_id: str = "r1") -> RolloutSpec:
    return RolloutSpec(
        rollout_id=rollout_id,
        task_id="task",
        task={"question": "1+1", "answer": 2},
        mode="train",
    )


def writer_program(payload: str) -> str:
    return (
        "import json,os,pathlib;"
        "task=json.loads(pathlib.Path(os.environ['RB_TASK_FILE']).read_text());"
        "assert task['answer']==2;"
        "assert os.environ['RB_ROLLOUT_ID']=='r1';"
        "assert os.environ['RB_GATEWAY_BASE_URL'].endswith('/rollouts/r1/v1');"
        "assert os.environ['RB_MODE']=='train';"
        f"pathlib.Path(os.environ['RB_RESULT_FILE']).write_text({payload!r})"
    )


@pytest.mark.asyncio
async def test_injects_contract_derives_success_persists_and_cleans(tmp_path: Path) -> None:
    FakeClient.events = [
        {
            "event_id": "e0",
            "rollout_id": "r1",
            "sequence_id": 0,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
        }
    ]
    controller = LocalController("http://gateway", tmp_path, "model", "rev")
    command = [sys.executable, "-c", writer_program('{"reward":1,"metrics":{"x":2}}')]
    result = await controller.run(spec(), command)
    assert result.status == "succeeded"
    assert result.reward == 1
    assert result.metrics == {"x": 2}
    artifact = tmp_path / "rollout_r1.json"
    first = artifact.read_bytes()
    payload = json.loads(first)
    assert payload["schema_version"] == "0.1"
    assert payload["model"] == {"id": "model", "revision": "rev"}
    assert payload["events"][0]["response_token_ids"] == [2]
    assert FakeClient.deleted == ["http://gateway/internal/rollouts/r1"]

    await controller.run(spec(), command)
    assert artifact.read_bytes() == first


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("program", "expected_code"),
    [
        ("import sys;sys.exit(7)", 7),
        (writer_program("not-json"), 1),
        ("pass", 1),
        (writer_program('{"reward":1,"metrics":{},"status":"fake"}'), 1),
    ],
)
async def test_failure_modes_still_capture(
    tmp_path: Path, program: str, expected_code: int
) -> None:
    controller = LocalController("http://gateway", tmp_path, "model")
    result = await controller.run(spec(), [sys.executable, "-c", program])
    assert result.status == "failed"
    assert result.reward is None
    assert result.exit_code == expected_code
    assert json.loads((tmp_path / "rollout_r1.json").read_text())["result"]["status"] == "failed"


@pytest.mark.asyncio
async def test_timeout_kills_the_process_group_and_uses_124(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    program = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    controller = LocalController(
        "http://gateway", tmp_path / "artifacts", "model", timeout_seconds=0.2
    )
    result = await controller.run(spec(), [sys.executable, "-c", program, str(pid_file)])
    assert result.exit_code == 124
    assert result.status == "failed"
    child_pid = int(pid_file.read_text())
    await asyncio.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.asyncio
async def test_gateway_failure_is_infrastructure_error_not_empty_capture(tmp_path: Path) -> None:
    FakeClient.get_status = 503
    controller = LocalController("http://gateway", tmp_path, "model")
    with pytest.raises(httpx.HTTPStatusError):
        await controller.run(
            spec(),
            [sys.executable, "-c", writer_program('{"reward":0,"metrics":{}}')],
        )
    assert not (tmp_path / "rollout_r1.json").exists()


@pytest.mark.asyncio
async def test_command_must_be_argv(tmp_path: Path) -> None:
    controller = LocalController("http://gateway", tmp_path, "model")
    with pytest.raises(TypeError):
        await controller.run(spec(), "echo unsafe")  # type: ignore[arg-type]
