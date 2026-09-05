"""Narrow, lazily imported VERL DataProto adapter."""

# pyright: reportMissingImports=false

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import RolloutResult, RolloutSpec, TrainingRow


@dataclass(frozen=True)
class _JoinedRollout:
    spec: RolloutSpec
    result: RolloutResult
    row: TrainingRow


def _unique_by_rollout_id(items: Sequence[Any], label: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        rollout_id = item.rollout_id
        if rollout_id in indexed:
            raise ValueError(f"duplicate {label} for rollout {rollout_id!r}")
        indexed[rollout_id] = item
    return indexed


def _join_rollouts(
    specs: Sequence[RolloutSpec],
    results: Sequence[RolloutResult],
    rows: Sequence[TrainingRow],
) -> list[_JoinedRollout]:
    """Join entities by rollout identity without importing numerical libraries."""
    spec_by_id = _unique_by_rollout_id(specs, "spec")
    result_by_id = _unique_by_rollout_id(results, "result")
    row_by_id = _unique_by_rollout_id(rows, "training row")
    identities = set(spec_by_id)
    if set(result_by_id) != identities or set(row_by_id) != identities:
        raise ValueError("specs, results, and rows must contain the same rollout IDs")
    if not identities:
        raise ValueError("cannot build an empty VERL batch")

    joined: list[_JoinedRollout] = []
    for spec in specs:
        result = result_by_id[spec.rollout_id]
        row = row_by_id[spec.rollout_id]
        reward = result.reward
        if result.status != "succeeded" or reward is None:
            raise ValueError(f"rollout {spec.rollout_id!r} has no successful reward")
        if not math.isfinite(reward):
            raise ValueError(f"rollout {spec.rollout_id!r} reward is not finite")
        if row.task_id != spec.task_id or row.rollout_id != spec.rollout_id:
            raise ValueError(f"rollout {spec.rollout_id!r} has conflicting identity")
        if row.row_index != 0:
            raise ValueError(f"rollout {spec.rollout_id!r} is not a single-row trajectory")
        if not any(row.loss_mask):
            raise ValueError(f"rollout {spec.rollout_id!r} has no trainable token")
        joined.append(_JoinedRollout(spec=spec, result=result, row=row))
    return joined


def build_verl_batch(
    specs: Sequence[RolloutSpec],
    results: Sequence[RolloutResult],
    rows: Sequence[TrainingRow],
    *,
    pad_token_id: int,
    max_prompt_length: int | None = None,
    max_response_length: int | None = None,
) -> Any:
    """Build a VERL DataProto, rejecting every lossy or ambiguous conversion."""
    joined = _join_rollouts(specs, results, rows)
    prompt_width = max(len(item.row.prompt_ids) for item in joined)
    response_width = max(len(item.row.continuation_ids) for item in joined)
    if max_prompt_length is not None and prompt_width > max_prompt_length:
        raise ValueError("prompt exceeds max_prompt_length; truncation is forbidden")
    if max_response_length is not None and response_width > max_response_length:
        raise ValueError("response exceeds max_response_length; truncation is forbidden")

    try:
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto
    except ImportError as exc:
        raise RuntimeError("build_verl_batch requires the 'train' dependency group") from exc

    prompts: list[list[int]] = []
    responses: list[list[int]] = []
    attention_masks: list[list[int]] = []
    response_masks: list[list[int]] = []
    scores: list[list[float]] = []
    for item in joined:
        prompt_padding = prompt_width - len(item.row.prompt_ids)
        response_padding = response_width - len(item.row.continuation_ids)
        prompts.append([pad_token_id] * prompt_padding + item.row.prompt_ids)
        responses.append(item.row.continuation_ids + [pad_token_id] * response_padding)
        attention_masks.append(
            [0] * prompt_padding
            + [1] * len(item.row.prompt_ids)
            + [1] * len(item.row.continuation_ids)
            + [0] * response_padding
        )
        response_masks.append(item.row.loss_mask + [0] * response_padding)
        token_scores = [0.0] * response_width
        last_trainable = max(index for index, value in enumerate(item.row.loss_mask) if value)
        reward = item.result.reward
        if reward is None:
            raise AssertionError("joined rollout unexpectedly lost its reward")
        token_scores[last_trainable] = float(reward)
        scores.append(token_scores)

    prompt_tensor = torch.tensor(prompts, dtype=torch.long)
    response_tensor = torch.tensor(responses, dtype=torch.long)
    attention_tensor = torch.tensor(attention_masks, dtype=torch.long)
    position_ids = (attention_tensor.cumsum(dim=-1) - 1).clamp_min(0)
    tensors = TensorDict(
        {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": torch.cat((prompt_tensor, response_tensor), dim=-1),
            "attention_mask": attention_tensor,
            "position_ids": position_ids,
            "response_mask": torch.tensor(response_masks, dtype=torch.long),
            "token_level_scores": torch.tensor(scores, dtype=torch.float32),
        },
        batch_size=[len(joined)],
    )
    return DataProto(
        batch=tensors,
        non_tensor_batch={
            "uid": np.asarray([item.spec.task_id for item in joined], dtype=object),
            "rollout_id_list": np.asarray([item.spec.rollout_id for item in joined], dtype=object),
        },
    )
