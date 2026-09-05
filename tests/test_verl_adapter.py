from __future__ import annotations

import pytest

from rolloutbridge.types import RolloutResult, RolloutSpec, TrainingRow, TrajectorySegment
from rolloutbridge.verl_adapter import _join_rollouts, build_verl_batch


def spec(rollout_id: str, task_id: str = "task") -> RolloutSpec:
    return RolloutSpec(rollout_id=rollout_id, task_id=task_id, task={})


def result(rollout_id: str, reward: float | None = 1) -> RolloutResult:
    return RolloutResult(
        rollout_id=rollout_id,
        status="succeeded" if reward is not None else "failed",
        reward=reward,
        exit_code=0 if reward is not None else 1,
    )


def row(rollout_id: str, task_id: str = "task", row_index: int = 0) -> TrainingRow:
    return TrainingRow(
        rollout_id=rollout_id,
        task_id=task_id,
        row_index=row_index,
        prompt_ids=[1],
        continuation_ids=[2, 3],
        loss_mask=[0, 1],
        segments=[
            TrajectorySegment(
                stream="prompt",
                start=0,
                end=1,
                event_id="e",
                kind="initial_context",
                trainable=False,
            ),
            TrajectorySegment(
                stream="continuation",
                start=0,
                end=1,
                event_id="e",
                kind="context_delta",
                trainable=False,
            ),
            TrajectorySegment(
                stream="continuation",
                start=1,
                end=2,
                event_id="e",
                kind="model_response",
                trainable=True,
            ),
        ],
    )


def test_same_task_can_have_multiple_independent_rollouts() -> None:
    joined = _join_rollouts(
        [spec("r1"), spec("r2")],
        [result("r1", 0), result("r2", 1)],
        [row("r1"), row("r2")],
    )
    assert [item.spec.task_id for item in joined] == ["task", "task"]
    assert [item.spec.rollout_id for item in joined] == ["r1", "r2"]


def test_different_tasks_do_not_change_rollout_join_key() -> None:
    joined = _join_rollouts(
        [spec("r1", "a"), spec("r2", "b")],
        [result("r2"), result("r1")],
        [row("r2", "b"), row("r1", "a")],
    )
    assert [(item.spec.rollout_id, item.row.task_id) for item in joined] == [
        ("r1", "a"),
        ("r2", "b"),
    ]


@pytest.mark.parametrize(
    ("specs", "results", "rows"),
    [
        ([spec("r")], [], [row("r")]),
        ([spec("r"), spec("r")], [result("r")], [row("r")]),
        ([spec("r")], [result("r"), result("r")], [row("r")]),
        ([spec("r")], [result("r")], [row("r"), row("r")]),
    ],
)
def test_rejects_missing_and_duplicate_entities(specs, results, rows) -> None:
    with pytest.raises(ValueError):
        _join_rollouts(specs, results, rows)


def test_rejects_failed_identity_conflict_and_nonzero_row_index() -> None:
    with pytest.raises(ValueError, match="successful reward"):
        _join_rollouts([spec("r")], [result("r", None)], [row("r")])
    with pytest.raises(ValueError, match="conflicting identity"):
        _join_rollouts([spec("r", "a")], [result("r")], [row("r", "b")])
    with pytest.raises(ValueError, match="single-row"):
        _join_rollouts([spec("r")], [result("r")], [row("r", row_index=1)])


def test_build_requires_lazy_train_dependencies_or_returns_dataproto() -> None:
    try:
        batch = build_verl_batch([spec("r")], [result("r")], [row("r")], pad_token_id=0)
    except RuntimeError as exc:
        assert "train" in str(exc)
    else:
        assert batch.batch["responses"].shape == (1, 2)
        assert batch.batch["response_mask"].tolist() == [[0, 1]]
