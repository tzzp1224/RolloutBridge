"""Validated wire and training contracts for RolloutBridge."""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RolloutSpec(_Contract):
    rollout_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["train", "eval"] = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCallEvent(_Contract):
    event_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    sequence_id: int = Field(ge=0)
    prompt_token_ids: list[int] = Field(min_length=1)
    response_token_ids: list[int] = Field(min_length=1)
    response_logprobs: list[float] | None = None

    @field_validator("response_logprobs", mode="before")
    @classmethod
    def discard_unusable_logprobs(cls, value: object) -> list[float] | None:
        if value is None or not isinstance(value, (list, tuple)):
            return None
        try:
            converted = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return converted if all(math.isfinite(item) for item in converted) else None

    @model_validator(mode="after")
    def discard_misaligned_logprobs(self) -> Self:
        if self.response_logprobs is not None and len(self.response_logprobs) != len(
            self.response_token_ids
        ):
            self.response_logprobs = None
        return self


class RolloutResult(_Contract):
    rollout_id: str = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    reward: float | None = None
    exit_code: int
    metrics: dict[str, float] = Field(default_factory=dict)


class TrajectorySegment(_Contract):
    stream: Literal["prompt", "continuation"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    event_id: str = Field(min_length=1)
    kind: Literal["initial_context", "context_delta", "model_response"]
    trainable: bool

    @model_validator(mode="after")
    def validate_interval_and_ownership(self) -> Self:
        if self.end <= self.start:
            raise ValueError("segment must be a non-empty half-open interval")
        expected_trainable = self.kind == "model_response"
        if self.trainable != expected_trainable:
            raise ValueError(f"{self.kind} trainable must be {expected_trainable}")
        if self.kind == "initial_context" and self.stream != "prompt":
            raise ValueError("initial_context belongs to the prompt stream")
        if self.kind != "initial_context" and self.stream != "continuation":
            raise ValueError(f"{self.kind} belongs to the continuation stream")
        return self


class TrainingRow(_Contract):
    rollout_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    prompt_ids: list[int] = Field(min_length=1)
    continuation_ids: list[int] = Field(min_length=1)
    loss_mask: list[int] = Field(min_length=1)
    segments: list[TrajectorySegment] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_alignment_and_segments(self) -> Self:
        if len(self.continuation_ids) != len(self.loss_mask):
            raise ValueError("continuation_ids and loss_mask must have equal length")
        if any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask values must be 0 or 1")

        continuation_started = False
        for segment in self.segments:
            continuation_started = continuation_started or segment.stream == "continuation"
            if continuation_started and segment.stream == "prompt":
                raise ValueError("prompt segments must precede continuation segments")

        limits = {"prompt": len(self.prompt_ids), "continuation": len(self.continuation_ids)}
        for stream, limit in limits.items():
            stream_segments = [segment for segment in self.segments if segment.stream == stream]
            if not stream_segments:
                raise ValueError(f"segments must cover the {stream} stream")
            cursor = 0
            for segment in stream_segments:
                if segment.start != cursor:
                    raise ValueError(f"{stream} segments must be ordered with no gaps or overlap")
                cursor = segment.end
            if cursor != limit:
                raise ValueError(f"segments must completely cover the {stream} stream")

        mask_from_segments = [0] * len(self.continuation_ids)
        for segment in self.segments:
            if segment.stream == "continuation" and segment.trainable:
                mask_from_segments[segment.start : segment.end] = [1] * (
                    segment.end - segment.start
                )
        if mask_from_segments != self.loss_mask:
            raise ValueError("loss_mask must exactly match trainable continuation segments")
        return self


class ModelIdentity(_Contract):
    id: str = Field(min_length=1)
    revision: str | None = None


class RawCapture(_Contract):
    schema_version: Literal["0.1"] = "0.1"
    model: ModelIdentity
    spec: RolloutSpec
    result: RolloutResult
    events: list[ModelCallEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        rollout_id = self.spec.rollout_id
        if self.result.rollout_id != rollout_id:
            raise ValueError("result rollout_id does not match spec")
        if any(event.rollout_id != rollout_id for event in self.events):
            raise ValueError("event rollout_id does not match spec")
        return self
