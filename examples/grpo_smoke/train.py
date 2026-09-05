"""One-GPU rollout-to-GRPO optimizer smoke test for the pinned VERL stack."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rolloutbridge import LocalController, RawCapture, RolloutSpec, TrainingRow, build_verl_batch
from rolloutbridge.compiler import compile_trajectory

_TASK_CANDIDATES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "arithmetic-grpo-0",
        {
            "question": "Use the calculator to evaluate (37 * 19) - (144 / 12).",
            "answer": 691,
        },
    ),
    (
        "arithmetic-grpo-1",
        {
            "question": "Use the calculator to evaluate (98765 % 97) * 13 - (444 // 7).",
            "answer": 184,
        },
    ),
    (
        "arithmetic-grpo-2",
        {
            "question": "Use the calculator to evaluate ((83 * 17) - (1440 / 12)) + 29.",
            "answer": 1320,
        },
    ),
    (
        "arithmetic-grpo-3",
        {
            "question": "Use the calculator to evaluate (19 * 23 + 17) / 6.",
            "answer": 75.66666666666667,
        },
    ),
)
_MAX_SAMPLES_PER_TASK = 16


def _wait_for(
    url: str,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and (exit_code := process.poll()) is not None:
            raise RuntimeError(f"service process exited with code {exit_code}: {url}")
        try:
            if httpx.get(url, timeout=2).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"service did not become ready: {url}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


async def _collect(
    gateway_url: str,
    artifacts: Path,
    model_id: str,
    model_revision: str | None,
    harness: Path,
) -> tuple[list[RolloutSpec], list[RawCapture]]:
    controller = LocalController(
        gateway_url,
        artifacts,
        model_id,
        model_revision=model_revision,
        timeout_seconds=180,
    )
    for task_index, (task_id, task) in enumerate(_TASK_CANDIDATES):
        specs: list[RolloutSpec] = []
        captures: list[RawCapture] = []
        for sample_index in range(_MAX_SAMPLES_PER_TASK):
            spec = RolloutSpec(
                rollout_id=f"gpu-smoke-{task_index}-{sample_index}",
                task_id=task_id,
                task=task,
            )
            await controller.run(spec, [sys.executable, str(harness)])
            capture_path = artifacts / f"rollout_{spec.rollout_id}.json"
            capture = RawCapture.model_validate_json(capture_path.read_text(encoding="utf-8"))
            if capture.result.status != "succeeded" or capture.result.reward is None:
                continue

            compiled = compile_trajectory(capture.events, capture.result, spec.task_id)
            if len(compiled) != 1:
                print(
                    f"skipping rollout {spec.rollout_id}: exact-prefix compiler produced "
                    f"{len(compiled)} rows",
                    file=sys.stderr,
                )
                continue
            row = compiled[0]
            has_tool_turn = (
                len(capture.events) >= 2
                and capture.result.metrics.get("model_calls", 0) >= 2
                and capture.result.metrics.get("tool_calls", 0) >= 1
                and any(segment.kind == "context_delta" for segment in row.segments)
                and 0 in row.loss_mask
            )
            if not has_tool_turn:
                continue

            specs.append(spec)
            captures.append(capture)
            rewards = {item.result.reward for item in captures}
            if 0.0 in rewards and 1.0 in rewards:
                return specs, captures

    raise RuntimeError(
        "workload calibration inconclusive: no candidate task produced both reward 0 and 1 "
        f"within {_MAX_SAMPLES_PER_TASK} rollouts"
    )


def _optimizer_step(
    model_id: str,
    model_revision: str | None,
    batch: Any,
) -> dict[str, float | str]:
    try:
        from verl.trainer.ppo.core_algos import (
            compute_grpo_outcome_advantage,
            compute_policy_loss,
        )
    except ImportError as exc:
        raise RuntimeError("this smoke test requires VERL 0.8.0") from exc

    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_scores"],
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
    )
    cpu_response_mask = batch.batch["response_mask"]
    if not torch.isfinite(advantages).all():
        raise RuntimeError("GRPO advantage is not finite")
    if torch.count_nonzero(advantages[cpu_response_mask == 0]).item() != 0:
        raise RuntimeError("GRPO advantage is non-zero on a masked response token")
    if torch.count_nonzero(advantages[cpu_response_mask == 1]).item() == 0:
        raise RuntimeError("GRPO advantage is zero on every trainable token")
    trainable_advantages = advantages[cpu_response_mask == 1]

    actor = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).cuda()
    actor.train()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)
    probe_name, probe = min(
        (
            (name, parameter)
            for name, parameter in actor.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: (item[1].numel(), item[0]),
    )
    input_ids = batch.batch["input_ids"].cuda()
    attention_mask = batch.batch["attention_mask"].cuda()
    responses = batch.batch["responses"].cuda()
    response_mask = batch.batch["response_mask"].cuda()
    prompt_width = batch.batch["prompts"].shape[1]
    logits = actor(input_ids=input_ids, attention_mask=attention_mask).logits
    response_logits = logits[:, prompt_width - 1 : -1].float()
    log_prob = (
        torch.log_softmax(response_logits, dim=-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    )
    loss, *_ = compute_policy_loss(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=advantages.cuda(),
        response_mask=response_mask,
        cliprange=0.2,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("policy loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradient_norms = [
        parameter.grad.detach().norm(2)
        for parameter in actor.parameters()
        if parameter.grad is not None
    ]
    if not gradient_norms:
        raise RuntimeError("backward produced no actor gradients")
    gradient_norm = torch.linalg.vector_norm(torch.stack(gradient_norms)).item()
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("actor gradient norm is not finite and positive")

    if probe.grad is None or torch.count_nonzero(probe.grad).item() == 0:
        raise RuntimeError(f"fixed parameter probe {probe_name!r} has no non-zero gradient")
    before = probe.detach().clone()
    optimizer.step()
    parameter_max_delta = (probe.detach() - before).abs().max().item()
    if not math.isfinite(parameter_max_delta) or parameter_max_delta <= 0:
        raise RuntimeError("optimizer step did not change the parameter probe")
    return {
        "loss": float(loss.detach()),
        "advantage_min": float(trainable_advantages.min()),
        "advantage_max": float(trainable_advantages.max()),
        "advantage_nonzero_trainable_tokens": int(torch.count_nonzero(trainable_advantages).item()),
        "advantage_nonzero_masked_tokens": 0,
        "gradient_norm": gradient_norm,
        "parameter_probe": probe_name,
        "parameter_max_delta": parameter_max_delta,
    }


def _validate_batch(batch: Any, rows: list[TrainingRow]) -> dict[str, Any]:
    group_ids = batch.non_tensor_batch["uid"].tolist()
    expected_group = rows[0].task_id
    if not group_ids or any(group_id != expected_group for group_id in group_ids):
        raise RuntimeError("VERL uid values do not form one task_id group")

    response_mask = batch.batch["response_mask"]
    response_width = response_mask.shape[1]
    for index, row in enumerate(rows):
        expected_mask = row.loss_mask + [0] * (response_width - len(row.loss_mask))
        if response_mask[index].tolist() != expected_mask:
            raise RuntimeError(f"VERL response_mask disagrees with row {row.rollout_id!r}")

    keys = ("prompts", "responses", "input_ids", "attention_mask", "response_mask")
    return {
        "task_id": expected_group,
        "batch_shapes": {key: list(batch.batch[key].shape) for key in keys},
        "trainable_tokens": int(response_mask.sum().item()),
        "masked_response_tokens": int((response_mask == 0).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/gpu-smoke"))
    parser.add_argument("--vllm-port", type=int, default=8100)
    parser.add_argument("--gateway-port", type=int, default=8101)
    args = parser.parse_args()
    if args.revision == "":
        raise RuntimeError("--revision received an empty value; resolve the model commit again")
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires Linux with an NVIDIA CUDA GPU")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    harness = root / "examples/openai_tool_agent/run.py"
    vllm_command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(args.vllm_port),
        "--gpu-memory-utilization",
        "0.45",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    if args.revision is not None:
        vllm_command.extend(("--revision", args.revision))
    vllm = subprocess.Popen(vllm_command, start_new_session=True)
    gateway: subprocess.Popen[bytes] | None = None
    try:
        _wait_for(f"http://127.0.0.1:{args.vllm_port}/health", 600, vllm)
        gateway = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rolloutbridge.gateway",
                "--upstream-base-url",
                f"http://127.0.0.1:{args.vllm_port}",
                "--model-id",
                args.model,
                "--port",
                str(args.gateway_port),
            ],
            start_new_session=True,
        )
        gateway_url = f"http://127.0.0.1:{args.gateway_port}"
        _wait_for(f"{gateway_url}/health", 60, gateway)
        specs, captures = asyncio.run(
            _collect(gateway_url, args.artifacts, args.model, args.revision, harness)
        )
    finally:
        if gateway is not None:
            _stop(gateway)
        _stop(vllm)

    rows = [
        compile_trajectory(capture.events, capture.result, capture.spec.task_id)[0]
        for capture in captures
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if not isinstance(pad_token_id, int):
        raise RuntimeError("tokenizer does not define a numeric pad token ID")
    batch = build_verl_batch(
        specs,
        [capture.result for capture in captures],
        rows,
        pad_token_id=pad_token_id,
        max_prompt_length=2048,
        max_response_length=2048,
    )
    batch_summary = _validate_batch(batch, rows)
    optimization_summary = _optimizer_step(args.model, args.revision, batch)
    summary = {
        "model": args.model,
        "model_revision": args.revision,
        "batch_size": len(captures),
        "rewards": [capture.result.reward for capture in captures],
        "artifacts": str(args.artifacts.resolve()),
        **batch_summary,
        **optimization_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
