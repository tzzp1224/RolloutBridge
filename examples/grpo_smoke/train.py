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

from rolloutbridge import LocalController, RawCapture, RolloutSpec, build_verl_batch
from rolloutbridge.compiler import compile_trajectory


def _wait_for(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
    harness: Path,
) -> tuple[list[RolloutSpec], list[RawCapture]]:
    specs: list[RolloutSpec] = []
    captures: list[RawCapture] = []
    controller = LocalController(gateway_url, artifacts, model_id, timeout_seconds=180)
    task = {
        "question": "Use the calculator to evaluate (37 * 19) - (144 / 12).",
        "answer": 691,
    }
    for index in range(8):
        spec = RolloutSpec(rollout_id=f"gpu-smoke-{index}", task_id="arithmetic-grpo", task=task)
        await controller.run(spec, [sys.executable, str(harness)])
        capture_path = artifacts / f"rollout_{spec.rollout_id}.json"
        capture = RawCapture.model_validate_json(capture_path.read_text(encoding="utf-8"))
        compiled = compile_trajectory(capture.events, capture.result, spec.task_id)
        if capture.result.status != "succeeded" or capture.result.reward is None:
            continue
        if len(compiled) != 1:
            raise RuntimeError(f"rollout {spec.rollout_id} did not compile to exactly one row")
        specs.append(spec)
        captures.append(capture)
        rewards = [item.result.reward for item in captures]
        if len(rewards) >= 2 and len(set(rewards)) > 1:
            return specs, captures
    raise RuntimeError("eight samples produced no group with non-zero reward variance")


def _optimizer_step(model_id: str, batch: Any) -> float:
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
    actor = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    actor.train()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)
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
    probe = next(parameter for parameter in actor.parameters() if parameter.requires_grad)
    before = probe.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    changed = (probe.detach() - before).abs().max().item()
    if not math.isfinite(changed) or changed == 0:
        raise RuntimeError("optimizer step did not change the actor parameter checksum")
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/gpu-smoke"))
    parser.add_argument("--vllm-port", type=int, default=8100)
    parser.add_argument("--gateway-port", type=int, default=8101)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires Linux with an NVIDIA CUDA GPU")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    harness = root / "examples/openai_tool_agent/run.py"
    vllm = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--port",
            str(args.vllm_port),
            "--gpu-memory-utilization",
            "0.45",
        ],
        start_new_session=True,
    )
    gateway: subprocess.Popen[bytes] | None = None
    try:
        _wait_for(f"http://127.0.0.1:{args.vllm_port}/health", 300)
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
        _wait_for(f"{gateway_url}/health", 60)
        specs, captures = asyncio.run(_collect(gateway_url, args.artifacts, args.model, harness))
    finally:
        if gateway is not None:
            _stop(gateway)
        _stop(vllm)

    rows = [
        compile_trajectory(capture.events, capture.result, capture.spec.task_id)[0]
        for capture in captures
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    loss = _optimizer_step(args.model, batch)
    summary = {
        "batch_size": len(captures),
        "loss": loss,
        "rewards": [capture.result.reward for capture in captures],
        "artifacts": str(args.artifacts.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
