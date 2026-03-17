#!/usr/bin/env python3
"""Standalone benchmark runner for cost_aware vs cache_aware vs round_robin.

Launches 2 SGLang workers + gateway, runs the "Prefix-Pinned Output Skew"
workload, and compares policies.

Usage:
    python3 run_cost_aware_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefixes — ~9000 chars each, diverge at char 10 ("You are a[n]...")
# ---------------------------------------------------------------------------

_PREFIX_A = (
    "You are an expert code reviewer. Your task is to carefully review "
    "the following code snippet for correctness, performance, security, "
    "and code style. Provide a detailed, thorough review with specific "
    "line-by-line comments. Be comprehensive and cover every possible "
    "issue you can find in the code.\n\n"
    + "REVIEW GUIDELINES:\n"
    + "\n".join(
        f"- Guideline {i}: Check for common pattern #{i} in the code "
        f"and verify it follows best practice recommendation {i}."
        for i in range(1, 81)
    )
    + "\n\nNow review the following code:\n"
)

_PREFIX_B = (
    "You are a text classifier. Classify the following text into exactly "
    "one category. Respond with ONLY the category name, nothing else. "
    "Do not explain your reasoning. Just output the single category.\n\n"
    + "CATEGORIES:\n"
    + "\n".join(
        f"- Category_{i}: Description of category {i} which covers texts "
        f"about subject area {i} and related domain topics."
        for i in range(1, 81)
    )
    + "\n\nClassify this text:\n"
)

TASKS = [
    (_PREFIX_A, 2000, "code_review"),   # LONG output
    (_PREFIX_B, 10, "classification"),   # SHORT output
]

USER_MESSAGES = [
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    "The quarterly earnings report showed a 15% increase in revenue driven by strong consumer demand in the electronics sector.",
]

NUM_REQUESTS_PER_TASK = 40
CONCURRENCY = 16


# ---------------------------------------------------------------------------
# Result collection (same as test file)
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    task_name: str
    max_tokens: int
    ttft: float
    e2e_latency: float
    output_tokens: int
    error: str | None = None


@dataclass
class ScenarioResult:
    policy: str
    results: list[RequestResult] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.error is None]

    @property
    def long_tasks(self) -> list[RequestResult]:
        return [r for r in self.successful if r.max_tokens >= 2000]

    @property
    def short_tasks(self) -> list[RequestResult]:
        return [r for r in self.successful if r.max_tokens < 100]

    def percentile(self, values: list[float], p: int) -> float:
        if not values:
            return float("inf")
        sorted_v = sorted(values)
        k = (len(sorted_v) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_v) else f
        return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])

    def summary(self) -> dict:
        all_e2e = [r.e2e_latency for r in self.successful]
        all_ttft = [r.ttft for r in self.successful]
        long_e2e = [r.e2e_latency for r in self.long_tasks]
        short_e2e = [r.e2e_latency for r in self.short_tasks]
        return {
            "policy": self.policy,
            "total": len(self.results),
            "ok": len(self.successful),
            "errors": len(self.results) - len(self.successful),
            "wall_time": self.wall_time,
            "all_e2e_p50": self.percentile(all_e2e, 50),
            "all_e2e_p95": self.percentile(all_e2e, 95),
            "all_e2e_p99": self.percentile(all_e2e, 99),
            "all_e2e_mean": statistics.mean(all_e2e) if all_e2e else 0,
            "all_ttft_p50": self.percentile(all_ttft, 50),
            "all_ttft_p95": self.percentile(all_ttft, 95),
            "all_ttft_p99": self.percentile(all_ttft, 99),
            "long_e2e_p50": self.percentile(long_e2e, 50),
            "long_e2e_p95": self.percentile(long_e2e, 95),
            "long_e2e_p99": self.percentile(long_e2e, 99),
            "long_e2e_mean": statistics.mean(long_e2e) if long_e2e else 0,
            "short_e2e_p50": self.percentile(short_e2e, 50),
            "short_e2e_p95": self.percentile(short_e2e, 95),
            "short_e2e_p99": self.percentile(short_e2e, 99),
            "short_e2e_mean": statistics.mean(short_e2e) if short_e2e else 0,
            "long_output_tokens_mean": statistics.mean(
                [r.output_tokens for r in self.long_tasks]
            ) if self.long_tasks else 0,
            "short_output_tokens_mean": statistics.mean(
                [r.output_tokens for r in self.short_tasks]
            ) if self.short_tasks else 0,
        }


def print_summary(s: dict):
    logger.info("=" * 72)
    logger.info("Policy: %s", s["policy"])
    logger.info("Requests: %d total, %d ok, %d errors | wall=%.1fs", s["total"], s["ok"], s["errors"], s["wall_time"])
    logger.info("ALL   e2e: p50=%.2fs  p95=%.2fs  p99=%.2fs  mean=%.2fs", s["all_e2e_p50"], s["all_e2e_p95"], s["all_e2e_p99"], s["all_e2e_mean"])
    logger.info("ALL  ttft: p50=%.2fs  p95=%.2fs  p99=%.2fs", s["all_ttft_p50"], s["all_ttft_p95"], s["all_ttft_p99"])
    logger.info("LONG  e2e: p50=%.2fs  p95=%.2fs  p99=%.2fs  mean=%.2fs  (avg out_tok=%.0f)", s["long_e2e_p50"], s["long_e2e_p95"], s["long_e2e_p99"], s["long_e2e_mean"], s["long_output_tokens_mean"])
    logger.info("SHORT e2e: p50=%.2fs  p95=%.2fs  p99=%.2fs  mean=%.2fs  (avg out_tok=%.0f)", s["short_e2e_p50"], s["short_e2e_p95"], s["short_e2e_p99"], s["short_e2e_mean"], s["short_output_tokens_mean"])
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Traffic generator
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    task_idx: int,
    use_estimated_output: bool,
) -> RequestResult:
    prefix, max_tokens, task_name = TASKS[task_idx]
    user_msg = USER_MESSAGES[task_idx]

    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    if use_estimated_output:
        body["estimated_output_tokens"] = max_tokens

    t_start = time.monotonic()
    ttft = 0.0
    output_tokens = 0
    first_token_seen = False

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer not-used"},
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                if not first_token_seen:
                    ttft = time.monotonic() - t_start
                    first_token_seen = True
                output_tokens += 1

        e2e = time.monotonic() - t_start
        return RequestResult(task_name=task_name, max_tokens=max_tokens, ttft=ttft, e2e_latency=e2e, output_tokens=output_tokens)
    except Exception as e:
        e2e = time.monotonic() - t_start
        return RequestResult(task_name=task_name, max_tokens=max_tokens, ttft=ttft, e2e_latency=e2e, output_tokens=output_tokens, error=str(e))


async def run_workload(
    base_url: str,
    model: str,
    policy: str,
    use_estimated_output: bool = True,
    num_requests_per_task: int = NUM_REQUESTS_PER_TASK,
    concurrency: int = CONCURRENCY,
) -> ScenarioResult:
    result = ScenarioResult(policy=policy)
    sem = asyncio.Semaphore(concurrency)

    requests = []
    for req_id in range(num_requests_per_task):
        for task_idx in range(len(TASKS)):
            requests.append(task_idx)

    async def _throttled(client, task_idx):
        async with sem:
            return await send_request(client, base_url, model, task_idx, use_estimated_output)

    logger.info("Starting workload: policy=%s, %d requests, concurrency=%d", policy, len(requests), CONCURRENCY)
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [_throttled(client, ti) for ti in requests]
        completed = await asyncio.gather(*tasks)
        result.results = list(completed)
    result.wall_time = time.monotonic() - t0

    logger.info("Workload complete in %.1fs (%d requests)", result.wall_time, len(requests))
    print_summary(result.summary())
    return result


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: int = 300):
    """Wait for an HTTP server to respond to /health."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Server ready at %s", url)
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s")


def launch_sglang_worker(model: str, port: int, gpu_id: int, log_dir: str = "/tmp") -> subprocess.Popen:
    """Launch an SGLang worker on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Use cached model only — avoid network calls during benchmark
    env["HF_HUB_OFFLINE"] = "1"
    # Propagate proxy if set in env.sh but not in current environment
    for key in ["http_proxy", "https_proxy", "no_proxy", "socks_proxy"]:
        if key not in env and key.upper() not in env:
            pass  # Offline mode handles this
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--mem-fraction-static", "0.85",
    ]
    logger.info("Launching worker on GPU %d, port %d: %s", gpu_id, port, model)
    stdout_f = open(f"{log_dir}/worker_gpu{gpu_id}.stdout.log", "w")
    stderr_f = open(f"{log_dir}/worker_gpu{gpu_id}.stderr.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=stderr_f, start_new_session=True)
    return proc


def launch_router(worker_urls: list[str], model: str, port: int, policy: str, extra_args: list[str] | None = None, log_dir: str = "/tmp") -> subprocess.Popen:
    """Launch the sglang router."""
    cmd = [
        sys.executable, "-m", "sglang_router.launch_router",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--model-path", model,
        "--policy", policy,
        "--worker-urls", *worker_urls,
    ]
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Launching router: policy=%s, port=%d, workers=%s", policy, port, worker_urls)
    stdout_f = open(f"{log_dir}/router_{policy}.stdout.log", "w")
    stderr_f = open(f"{log_dir}/router_{policy}.stderr.log", "w")
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, start_new_session=True)
    return proc


def kill_proc_tree(proc: subprocess.Popen):
    """Kill a process and all its children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cost-aware LB benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gpu0", type=int, default=1, help="GPU id for worker 0")
    parser.add_argument("--gpu1", type=int, default=2, help="GPU id for worker 1")
    parser.add_argument("--policies", nargs="+", default=["cache_aware", "cost_aware", "round_robin"])
    parser.add_argument("--requests-per-task", type=int, default=NUM_REQUESTS_PER_TASK)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--skip-workers", action="store_true", help="Workers already running (provide --worker-urls)")
    parser.add_argument("--worker-urls", nargs="+", default=None)
    args = parser.parse_args()

    num_rpt = args.requests_per_task
    concurrency = args.concurrency

    worker_procs = []
    router_proc = None
    all_results: dict[str, ScenarioResult] = {}
    log_dir = "/tmp/sglang_benchmark"
    os.makedirs(log_dir, exist_ok=True)

    try:
        # --- Launch workers (or reuse existing) ---
        if args.skip_workers and args.worker_urls:
            worker_urls = args.worker_urls
            logger.info("Reusing existing workers: %s", worker_urls)
        else:
            w0_port = get_open_port()
            w1_port = get_open_port()
            worker_urls = [f"http://127.0.0.1:{w0_port}", f"http://127.0.0.1:{w1_port}"]

            w0 = launch_sglang_worker(args.model, w0_port, args.gpu0, log_dir=log_dir)
            w1 = launch_sglang_worker(args.model, w1_port, args.gpu1, log_dir=log_dir)
            worker_procs = [w0, w1]

            logger.info("Waiting for workers to start...")
            wait_for_server(worker_urls[0], timeout=300)
            wait_for_server(worker_urls[1], timeout=300)
            logger.info("Both workers ready!")

        # --- Run each policy ---
        for policy in args.policies:
            router_port = get_open_port()
            extra_args = []
            if policy == "cost_aware":
                extra_args = [
                    "--w-prefill", "6",
                    "--w-cache-read", "1",
                    "--w-decode", "30",
                    "--default-estimated-output-tokens", "256",
                ]

            router_proc = launch_router(worker_urls, args.model, router_port, policy, extra_args, log_dir=log_dir)
            router_url = f"http://127.0.0.1:{router_port}"

            try:
                wait_for_server(router_url, timeout=60)
            except TimeoutError:
                if router_proc.poll() is not None:
                    log_file = f"{log_dir}/router_{policy}.stderr.log"
                    logger.error("Router died. See %s", log_file)
                raise

            # Short warmup — 2 requests per task
            logger.info("Warming up %s...", policy)
            asyncio.run(run_workload(router_url, args.model, f"{policy}_warmup",
                                     use_estimated_output=(policy == "cost_aware"),
                                     num_requests_per_task=2, concurrency=2))

            # Main run
            use_est = policy == "cost_aware"
            result = asyncio.run(run_workload(router_url, args.model, policy,
                                              use_estimated_output=use_est,
                                              num_requests_per_task=num_rpt,
                                              concurrency=concurrency))
            all_results[policy] = result

            # Shutdown router between policies to avoid tree contamination
            kill_proc_tree(router_proc)
            router_proc = None
            time.sleep(2)

        # --- Comparison ---
        logger.info("\n\n" + "=" * 72)
        logger.info("FINAL COMPARISON")
        logger.info("=" * 72)
        for policy, result in all_results.items():
            print_summary(result.summary())

        if "cache_aware" in all_results and "cost_aware" in all_results:
            ca = all_results["cache_aware"].summary()
            co = all_results["cost_aware"].summary()

            def ratio(a, b):
                return a / b if b > 0 else float("inf")

            logger.info("\n" + "-" * 72)
            logger.info("IMPROVEMENT (cache_aware / cost_aware — >1.0 means cost_aware wins)")
            logger.info("-" * 72)
            logger.info("  long_e2e_p99:  %.2f×", ratio(ca["long_e2e_p99"], co["long_e2e_p99"]))
            logger.info("  long_e2e_p95:  %.2f×", ratio(ca["long_e2e_p95"], co["long_e2e_p95"]))
            logger.info("  long_e2e_mean: %.2f×", ratio(ca["long_e2e_mean"], co["long_e2e_mean"]))
            logger.info("  all_e2e_p99:   %.2f×", ratio(ca["all_e2e_p99"], co["all_e2e_p99"]))
            logger.info("  all_ttft_p99:  %.2f×", ratio(ca["all_ttft_p99"], co["all_ttft_p99"]))
            logger.info("  wall_time:     %.2f×", ratio(ca["wall_time"], co["wall_time"]))
            logger.info("-" * 72)

        # Save results to JSON
        output_path = Path("benchmark_results.json")
        with open(output_path, "w") as f:
            json.dump({p: r.summary() for p, r in all_results.items()}, f, indent=2)
        logger.info("Results saved to %s", output_path)

    finally:
        # Cleanup
        if router_proc:
            kill_proc_tree(router_proc)
        for w in worker_procs:
            kill_proc_tree(w)
        logger.info("All processes cleaned up")


if __name__ == "__main__":
    main()
