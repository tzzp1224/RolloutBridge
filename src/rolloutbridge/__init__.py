"""RolloutBridge public API."""

from .compiler import compile_trajectory
from .controller import LocalController
from .types import (
    ModelCallEvent,
    ModelIdentity,
    RawCapture,
    RolloutResult,
    RolloutSpec,
    TrainingRow,
    TrajectorySegment,
)
from .verl_adapter import build_verl_batch

__all__ = [
    "LocalController",
    "ModelCallEvent",
    "ModelIdentity",
    "RawCapture",
    "RolloutResult",
    "RolloutSpec",
    "TrainingRow",
    "TrajectorySegment",
    "build_verl_batch",
    "compile_trajectory",
]
