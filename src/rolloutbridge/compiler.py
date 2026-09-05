"""Pure exact-token trajectory compilation."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ModelCallEvent, RolloutResult, TrainingRow, TrajectorySegment


def compile_trajectory(
    events: Sequence[ModelCallEvent],
    result: RolloutResult,
    task_id: str,
) -> list[TrainingRow]:
    """Compile ordered model calls into maximal exact-prefix training rows.

    A response is always trainable. Tokens newly appearing between one complete
    interaction and the next prompt are environment context and are masked out.
    Any broken prefix starts a new row rather than guessing token alignment.
    """
    if not events:
        return []

    ordered = sorted(events, key=lambda event: event.sequence_id)
    if len({event.sequence_id for event in ordered}) != len(ordered):
        raise ValueError("duplicate sequence_id")
    if len({event.event_id for event in ordered}) != len(ordered):
        raise ValueError("duplicate event_id")

    rollout_ids = {event.rollout_id for event in ordered}
    if len(rollout_ids) != 1:
        raise ValueError("events contain mixed rollout_id values")
    rollout_id = next(iter(rollout_ids))
    if result.rollout_id != rollout_id:
        raise ValueError("result rollout_id does not match events")

    rows: list[TrainingRow] = []
    prompt_ids: list[int] = []
    continuation_ids: list[int] = []
    loss_mask: list[int] = []
    segments: list[TrajectorySegment] = []

    def start_row(event: ModelCallEvent) -> None:
        nonlocal prompt_ids, continuation_ids, loss_mask, segments
        prompt_ids = list(event.prompt_token_ids)
        continuation_ids = list(event.response_token_ids)
        loss_mask = [1] * len(event.response_token_ids)
        segments = [
            TrajectorySegment(
                stream="prompt",
                start=0,
                end=len(prompt_ids),
                event_id=event.event_id,
                kind="initial_context",
                trainable=False,
            ),
            TrajectorySegment(
                stream="continuation",
                start=0,
                end=len(continuation_ids),
                event_id=event.event_id,
                kind="model_response",
                trainable=True,
            ),
        ]

    def finish_row() -> None:
        rows.append(
            TrainingRow(
                rollout_id=rollout_id,
                task_id=task_id,
                row_index=len(rows),
                prompt_ids=prompt_ids,
                continuation_ids=continuation_ids,
                loss_mask=loss_mask,
                segments=segments,
            )
        )

    start_row(ordered[0])
    for event in ordered[1:]:
        complete_prefix = prompt_ids + continuation_ids
        event_prompt = event.prompt_token_ids
        if event_prompt[: len(complete_prefix)] != complete_prefix:
            finish_row()
            start_row(event)
            continue

        context_delta = event_prompt[len(complete_prefix) :]
        if context_delta:
            start = len(continuation_ids)
            continuation_ids.extend(context_delta)
            loss_mask.extend([0] * len(context_delta))
            segments.append(
                TrajectorySegment(
                    stream="continuation",
                    start=start,
                    end=len(continuation_ids),
                    event_id=event.event_id,
                    kind="context_delta",
                    trainable=False,
                )
            )

        start = len(continuation_ids)
        continuation_ids.extend(event.response_token_ids)
        loss_mask.extend([1] * len(event.response_token_ids))
        segments.append(
            TrajectorySegment(
                stream="continuation",
                start=start,
                end=len(continuation_ids),
                event_id=event.event_id,
                kind="model_response",
                trainable=True,
            )
        )

    finish_row()
    return rows
