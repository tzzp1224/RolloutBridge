from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from rolloutbridge.compiler import compile_trajectory
from rolloutbridge.types import (
    ModelCallEvent,
    RolloutResult,
    TrainingRow,
    TrajectorySegment,
)


def event(
    sequence: int,
    prompt: list[int],
    response: list[int],
    *,
    rollout_id: str = "r1",
    event_id: str | None = None,
) -> ModelCallEvent:
    return ModelCallEvent(
        event_id=event_id or f"e{sequence}",
        rollout_id=rollout_id,
        sequence_id=sequence,
        prompt_token_ids=prompt,
        response_token_ids=response,
    )


def result(rollout_id: str = "r1", status: str = "succeeded") -> RolloutResult:
    return RolloutResult(rollout_id=rollout_id, status=status, reward=1, exit_code=0)


def test_single_call_and_failed_rollout_compile_identically() -> None:
    events = [event(0, [1, 2], [3, 4])]
    row = compile_trajectory(events, result(status="failed"), "task")[0]
    assert row.prompt_ids == [1, 2]
    assert row.continuation_ids == [3, 4]
    assert row.loss_mask == [1, 1]
    assert [(segment.kind, segment.event_id) for segment in row.segments] == [
        ("initial_context", "e0"),
        ("model_response", "e0"),
    ]


def test_three_calls_merge_context_deltas_and_empty_delta() -> None:
    events = [
        event(0, [1, 2], [3]),
        event(1, [1, 2, 3], [4]),
        event(2, [1, 2, 3, 4, 5, 6], [7, 8]),
    ]
    row = compile_trajectory(events, result(), "task")[0]
    assert row.continuation_ids == [3, 4, 5, 6, 7, 8]
    assert row.loss_mask == [1, 1, 0, 0, 1, 1]
    assert [segment.kind for segment in row.segments] == [
        "initial_context",
        "model_response",
        "model_response",
        "context_delta",
        "model_response",
    ]


@pytest.mark.parametrize(
    "new_prompt",
    [
        [1, 9],  # broken exact prefix
        [3],  # summarized context
        [2, 3],  # left-truncated context
    ],
)
def test_discontinuity_splits_rows(new_prompt: list[int]) -> None:
    rows = compile_trajectory([event(0, [1, 2], [3]), event(1, new_prompt, [4])], result(), "task")
    assert [row.row_index for row in rows] == [0, 1]
    assert rows[1].prompt_ids == new_prompt
    assert rows[1].segments[0].event_id == "e1"


def test_input_is_sorted_by_sequence_id() -> None:
    rows = compile_trajectory(
        [event(2, [1, 2, 3, 4], [5]), event(0, [1], [2]), event(1, [1, 2, 3], [4])],
        result(),
        "task",
    )
    assert rows[0].continuation_ids == [2, 3, 4, 5]
    assert [segment.event_id for segment in rows[0].segments] == ["e0", "e0", "e1", "e1", "e2"]


def test_empty_events_return_empty() -> None:
    assert compile_trajectory([], result("unrelated"), "task") == []


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([event(0, [1], [2]), event(0, [1, 2], [3], event_id="other")], "sequence"),
        ([event(0, [1], [2]), event(1, [1, 2], [3], event_id="e0")], "event_id"),
        ([event(0, [1], [2]), event(1, [1, 2], [3], rollout_id="r2")], "mixed"),
    ],
)
def test_rejects_ambiguous_event_identity(events: list[ModelCallEvent], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compile_trajectory(events, result(), "task")


def test_rejects_result_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="result rollout_id"):
        compile_trajectory([event(0, [1], [2])], result("r2"), "task")


def test_optional_logprobs_are_discarded_when_unusable() -> None:
    malformed = ModelCallEvent(
        event_id="e",
        rollout_id="r",
        sequence_id=0,
        prompt_token_ids=[1],
        response_token_ids=[2, 3],
        response_logprobs=[-0.1],
    )
    nonfinite = ModelCallEvent(
        event_id="e",
        rollout_id="r",
        sequence_id=0,
        prompt_token_ids=[1],
        response_token_ids=[2],
        response_logprobs=[float("nan")],
    )
    assert malformed.response_logprobs is None
    assert nonfinite.response_logprobs is None


def test_training_row_rejects_mask_and_segment_corruption() -> None:
    base = {
        "rollout_id": "r",
        "task_id": "t",
        "row_index": 0,
        "prompt_ids": [1],
        "continuation_ids": [2],
        "loss_mask": [1],
        "segments": [
            {
                "stream": "prompt",
                "start": 0,
                "end": 1,
                "event_id": "e",
                "kind": "initial_context",
                "trainable": False,
            },
            {
                "stream": "continuation",
                "start": 0,
                "end": 1,
                "event_id": "e",
                "kind": "model_response",
                "trainable": True,
            },
        ],
    }
    TrainingRow.model_validate(base)
    for patch in (
        {"loss_mask": [1, 0]},
        {"loss_mask": [2]},
        {"continuation_ids": [2, 3]},
        {
            "segments": [
                base["segments"][0],
                {**base["segments"][1], "start": 1, "end": 2},
            ]
        },
    ):
        with pytest.raises(ValidationError):
            TrainingRow.model_validate({**base, **patch})


def test_segment_ownership_is_validated() -> None:
    with pytest.raises(ValidationError):
        TrajectorySegment(
            stream="continuation",
            start=0,
            end=1,
            event_id="e",
            kind="context_delta",
            trainable=True,
        )


def test_compilation_is_canonical_and_repeatable() -> None:
    events = [event(0, [1], [2]), event(1, [1, 2, 3], [4])]
    encodings = []
    for _ in range(2):
        rows = compile_trajectory(events, result(), "task")
        value = [row.model_dump(mode="json") for row in rows]
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        encodings.append((encoded, hashlib.sha256(encoded).hexdigest()))
    assert encodings[0] == encodings[1]


def test_mutable_defaults_are_not_shared() -> None:
    first = RolloutResult(rollout_id="a", status="failed", exit_code=1)
    second = RolloutResult(rollout_id="b", status="failed", exit_code=1)
    first.metrics["x"] = 1
    assert second.metrics == {}
