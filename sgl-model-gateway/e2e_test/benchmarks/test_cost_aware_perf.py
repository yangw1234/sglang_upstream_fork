"""Cost-aware vs cache_aware extreme scenario benchmark.

Scenario: "Prefix-Pinned Output Skew" (2-worker variant)
  - 2 task types with distinct system prompts (prefixes)
  - Task A produces LONG output (max_tokens=2000)
  - Task B produces SHORT output (max_tokens=10)
  - cache_aware pins each prefix to one worker by cache affinity,
    leaving one worker saturated with long-decode and one idle
  - cost_aware redistributes long-output tasks by weighted load,
    balancing decode work across both workers

Run: pytest e2e_test/benchmarks/test_cost_aware_perf.py -v -s
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field

import httpx
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefixes — each ~3000 chars to ensure >50% cache match ratio on 4000-char
# inputs.  Must be distinct so the radix tree maps each to a different worker.
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

# Task definitions: (prefix, max_tokens, task_name)
# With 2 workers, cache_aware pins A→Worker0, B→Worker1.
# Worker0 gets all long-decode work, Worker1 sits idle.
TASKS = [
    (_PREFIX_A, 2000, "code_review"),  # LONG output
    (_PREFIX_B, 10, "classification"),  # SHORT output
]

# User messages — short, so the prefix dominates the input
USER_MESSAGES = [
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    "The quarterly earnings report showed a 15% increase in revenue driven by strong consumer demand in the electronics sector.",
]


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Timing result for a single request."""

    task_name: str
    max_tokens: int
    ttft: float  # seconds
    e2e_latency: float  # seconds
    output_tokens: int
    worker_url: str = ""  # from response headers if available
    error: str | None = None


@dataclass
class ScenarioResult:
    """Aggregated results for one policy run."""

    policy: str
    results: list[RequestResult] = field(default_factory=list)

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
        """Compute summary statistics."""
        all_e2e = [r.e2e_latency for r in self.successful]
        all_ttft = [r.ttft for r in self.successful]
        long_e2e = [r.e2e_latency for r in self.long_tasks]
        short_e2e = [r.e2e_latency for r in self.short_tasks]

        return {
            "policy": self.policy,
            "total_requests": len(self.results),
            "successful": len(self.successful),
            "errors": len(self.results) - len(self.successful),
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
        }

    def log_summary(self):
        s = self.summary()
        logger.info("=" * 72)
        logger.info("Policy: %s", s["policy"])
        logger.info(
            "Requests: %d total, %d ok, %d errors",
            s["total_requests"],
            s["successful"],
            s["errors"],
        )
        logger.info(
            "ALL   e2e: p50=%.2fs  p95=%.2fs  p99=%.2fs  mean=%.2fs",
            s["all_e2e_p50"],
            s["all_e2e_p95"],
            s["all_e2e_p99"],
            s["all_e2e_mean"],
        )
        logger.info(
            "ALL  ttft: p50=%.2fs  p95=%.2fs  p99=%.2fs",
            s["all_ttft_p50"],
            s["all_ttft_p95"],
            s["all_ttft_p99"],
        )
        logger.info(
            "LONG  e2e: p50=%.2fs  p95=%.2fs  p99=%.2fs  mean=%.2fs",
            s["long_e2e_p50"],
            s["long_e2e_p95"],
            s["long_e2e_p99"],
            s["long_e2e_mean"],
        )
        logger.info(
            "SHORT e2e: p50=%.2fs  p95=%.2fs  p99=%.2fs  mean=%.2fs",
            s["short_e2e_p50"],
            s["short_e2e_p95"],
            s["short_e2e_p99"],
            s["short_e2e_mean"],
        )
        logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Traffic generator
# ---------------------------------------------------------------------------

NUM_REQUESTS_PER_TASK = 40  # 40 × 2 tasks = 80 total
CONCURRENCY = 16


async def _send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    task_idx: int,
    request_id: int,
    use_estimated_output: bool = True,
) -> RequestResult:
    """Send a single chat completion request with streaming and measure timing."""
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
        return RequestResult(
            task_name=task_name,
            max_tokens=max_tokens,
            ttft=ttft,
            e2e_latency=e2e,
            output_tokens=output_tokens,
        )
    except Exception as e:
        e2e = time.monotonic() - t_start
        return RequestResult(
            task_name=task_name,
            max_tokens=max_tokens,
            ttft=ttft,
            e2e_latency=e2e,
            output_tokens=output_tokens,
            error=str(e),
        )


async def run_workload(
    base_url: str,
    model: str,
    policy: str,
    use_estimated_output: bool = True,
) -> ScenarioResult:
    """Run the full mixed workload and collect results."""
    result = ScenarioResult(policy=policy)
    sem = asyncio.Semaphore(CONCURRENCY)

    # Build request list: interleaved task types for realistic arrival pattern
    # A, B, C, D, A, B, C, D, ... (not AAAA...BBBB... which would be too easy)
    requests: list[tuple[int, int]] = []  # (task_idx, request_id)
    for req_id in range(NUM_REQUESTS_PER_TASK):
        for task_idx in range(len(TASKS)):
            requests.append((task_idx, req_id))

    async def _throttled(client, task_idx, req_id):
        async with sem:
            return await _send_request(
                client, base_url, model, task_idx, req_id, use_estimated_output
            )

    logger.info(
        "Starting workload: policy=%s, %d requests, concurrency=%d",
        policy,
        len(requests),
        CONCURRENCY,
    )
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [_throttled(client, ti, ri) for ti, ri in requests]
        completed = await asyncio.gather(*tasks)
        result.results = list(completed)

    elapsed = time.monotonic() - t0
    logger.info("Workload complete in %.1fs (%d requests)", elapsed, len(requests))
    result.log_summary()
    return result


def run_workload_sync(
    base_url: str,
    model: str,
    policy: str,
    use_estimated_output: bool = True,
) -> ScenarioResult:
    """Synchronous wrapper for run_workload."""
    return asyncio.run(run_workload(base_url, model, policy, use_estimated_output))


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_results(cache_aware: ScenarioResult, cost_aware: ScenarioResult) -> dict:
    """Compare two scenario results and return improvement ratios."""
    ca = cache_aware.summary()
    co = cost_aware.summary()

    def ratio(ca_val, co_val):
        if co_val == 0:
            return float("inf")
        return ca_val / co_val

    comparison = {
        "long_e2e_p99_improvement": ratio(ca["long_e2e_p99"], co["long_e2e_p99"]),
        "long_e2e_p95_improvement": ratio(ca["long_e2e_p95"], co["long_e2e_p95"]),
        "long_e2e_mean_improvement": ratio(ca["long_e2e_mean"], co["long_e2e_mean"]),
        "all_e2e_p99_improvement": ratio(ca["all_e2e_p99"], co["all_e2e_p99"]),
        "all_e2e_p95_improvement": ratio(ca["all_e2e_p95"], co["all_e2e_p95"]),
        "all_ttft_p99_improvement": ratio(ca["all_ttft_p99"], co["all_ttft_p99"]),
    }

    logger.info("=" * 72)
    logger.info("COMPARISON: cache_aware vs cost_aware")
    logger.info("-" * 72)
    for key, val in comparison.items():
        logger.info("  %s: %.2f×", key, val)
    logger.info("=" * 72)

    return comparison


# ---------------------------------------------------------------------------
# E2E benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.workers(count=2)
@pytest.mark.gateway(policy="cache_aware")
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestCacheAwareBaseline:
    """Baseline: run the extreme scenario under cache_aware (2 workers)."""

    def test_prefix_pinned_output_skew(self, setup_backend):
        _, model_path, client, gateway = setup_backend
        result = run_workload_sync(gateway.base_url, model_path, "cache_aware")
        s = result.summary()

        # Sanity: most requests should succeed
        assert (
            s["errors"] <= s["total_requests"] * 0.05
        ), f"Too many errors: {s['errors']}/{s['total_requests']}"


@pytest.mark.e2e
@pytest.mark.workers(count=2)
@pytest.mark.gateway(
    policy="cost_aware",
    extra_args=[
        "--w-prefill",
        "6",
        "--w-cache-read",
        "1",
        "--w-decode",
        "30",
        "--default-estimated-output-tokens",
        "256",
    ],
)
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestCostAwareExtreme:
    """Run the extreme scenario under cost_aware — should dominate cache_aware."""

    def test_prefix_pinned_output_skew(self, setup_backend):
        _, model_path, client, gateway = setup_backend
        result = run_workload_sync(gateway.base_url, model_path, "cost_aware")
        s = result.summary()

        assert (
            s["errors"] <= s["total_requests"] * 0.05
        ), f"Too many errors: {s['errors']}/{s['total_requests']}"


@pytest.mark.e2e
@pytest.mark.workers(count=2)
@pytest.mark.gateway(policy="round_robin")
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestRoundRobinBaseline:
    """Baseline: round_robin for comparison (no cache, no cost awareness, 2 workers)."""

    def test_prefix_pinned_output_skew(self, setup_backend):
        _, model_path, client, gateway = setup_backend
        result = run_workload_sync(gateway.base_url, model_path, "round_robin")
        s = result.summary()

        assert (
            s["errors"] <= s["total_requests"] * 0.05
        ), f"Too many errors: {s['errors']}/{s['total_requests']}"
