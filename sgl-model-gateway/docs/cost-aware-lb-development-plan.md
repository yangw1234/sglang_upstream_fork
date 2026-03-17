# Development Plan: Cost-Aware Load Balancing Policy

**Status**: Phase 2 complete — ready for GPU-backed performance benchmark
**Target module**: `sgl-model-gateway/src/policies/cost_aware.rs`
**Scope**: Regular (non-disaggregated) HTTP routing only

---

## Design

### Score Function

```
effective_score(worker_i) = existing_load_i - matched_chars_i × S
```

Lowest score wins. `S = W_prefill - W_cache_read` = savings per cached token.

- `existing_load_i` = sum of cost-units across all in-flight requests on worker_i
- `matched_chars_i` = prefix match length from radix tree (converted via `chars / 4`)
- Request cost = `uncached_tokens × W_prefill + cached_tokens × W_cache_read + est_output × W_decode`

### Default Weights (Kimi K2.5 pricing × 10)

| Weight | Default | Meaning |
|--------|---------|---------|
| `W_prefill` | 6 | Cost per uncached prefill token |
| `W_cache_read` | 1 | Cost per cached prefill token |
| `W_decode` | 30 | Cost per output/decode token |
| `S` (savings) | 5 | W_prefill − W_cache_read |

### Output Estimate Priority

```
1. estimated_output_tokens  — explicit client field (best signal)
2. max_completion_tokens / max_tokens / max_output_tokens — upper bound
3. None → config default_estimated_output_tokens (256)
```

### Key Differences from Cache-Aware

| Aspect | cache_aware | cost_aware |
|--------|------------|------------|
| Load metric | `worker.load()` (request count) | `worker.weighted_load()` (cost-units) |
| Cache vs load | Binary switch (threshold) | Continuous score (always both) |
| Output awareness | None | Uses `estimated_output_tokens` |

---

## What's Implemented

### Rust Core (Phase 1)

| Area | Key Files |
|------|-----------|
| Policy | `src/policies/cost_aware.rs` — `CostAwarePolicy`, `score_worker()`, radix tree management |
| Worker | `src/core/worker.rs` — `weighted_load_counter`, `WorkerLoadGuard::with_cost()` + RAII drop |
| Config | `src/config/types.rs` — `PolicyConfig::CostAware` with 6 fields; `src/config/validation.rs` |
| Factory | `src/policies/factory.rs` — `create_from_config()` + `create_by_name()` |
| Router | `src/routers/http/router.rs` — `extract_estimated_output()`, cost-aware guard wiring |

### Python CLI Bindings (Phase 2)

Enables `--policy cost_aware --w-prefill 6 --w-cache-read 1 --w-decode 30 --default-estimated-output-tokens 256`:

| File | Change |
|------|--------|
| `bindings/python/src/lib.rs` | `CostAware` in `PolicyType` enum + `convert_policy` + 4 config fields on `Router` |
| `bindings/python/src/sglang_router/router.py` | `"cost_aware"` in `policy_from_str()` |
| `bindings/python/src/sglang_router/router_args.py` | `cost_aware` in all `--policy` choices; 4 new CLI args |

### Integration Tests (Phase 2 — 7 passing)

File: `tests/routing/cost_aware_test.rs` + helper `TestRouterConfig::cost_aware()` in `test_config.rs`

| Test | What it validates |
|------|-------------------|
| `test_cost_aware_basic_routing` | 10 requests through 2 workers succeed |
| `test_cost_aware_single_worker` | Single-worker degeneracy case |
| `test_cost_aware_chat_completions_with_estimate` | `/v1/chat/completions` + `estimated_output_tokens` field |
| `test_cost_aware_chat_completions_no_estimate` | Fallback when no estimate provided |
| `test_cost_aware_distributes_long_requests` | Concurrent mixed short+long workload vs round_robin |
| `test_weighted_load_increases_and_decreases` | RAII: weighted_load > 0 in-flight, returns to 0 |
| `test_cost_aware_streaming` | SSE streaming works through cost_aware routing |

### Unit Tests (Phase 1 — 8 in `cost_aware.rs`, 3 in factory/worker)

Score math, cache preference, load balance, edge cases (single worker, all unhealthy,
no estimate), RAII guard drop, factory creation.

---

## Phase 3: GPU-Backed Performance Benchmark (In Progress)

Goal: prove cost_aware reduces tail latency vs cache_aware/round_robin with real GPU workers.

### Extreme Scenario: "Prefix-Pinned Output Skew"

Full design: `docs/cost-aware-extreme-scenario.md`

**Why cache_aware fails**: When 4 task types have distinct prefixes but 2 produce
long output (2000 tokens) and 2 produce short output (10 tokens), cache_aware pins
each prefix to one worker. Request counts balance (equal across workers), so cache_aware
never triggers imbalance detection and keeps routing by prefix. Workers handling long
tasks are saturated while short-task workers idle. cost_aware sees the 200× output cost
difference via `weighted_load` and redistributes long tasks to idle workers.

### Implementation

File: `e2e_test/benchmarks/test_cost_aware_perf.py` — custom traffic generator
(genai-bench only supports uniform `D(input,output)`, can't model bimodal output).

| Test class | Policy | What it proves |
|------------|--------|----------------|
| `TestCacheAwareBaseline` | cache_aware | Baseline — shows high P99 from output skew |
| `TestCostAwareExtreme` | cost_aware | Lower tail latency via weighted load balancing |
| `TestRoundRobinBaseline` | round_robin | Comparison — no cache, no cost awareness |

Traffic: 160 requests (40 per task type), 32 concurrency, streaming, interleaved arrival.
Measures per-request TTFT + e2e latency, reports P50/P95/P99 for long/short/all tasks.

### Metrics to Compare

- P50 / P95 / P99 TTFT and e2e latency (overall + per output-length class)
- Long-task P99 improvement ratio (cache_aware / cost_aware)
- Error rate

### Run

```bash
pytest e2e_test/benchmarks/test_cost_aware_perf.py -v -s
```

---

## Open Questions

1. **Weight calibration**: Defaults from Kimi pricing. Auto-calibration from observed
   latency is a future enhancement.

2. **`max_tokens` fallback is lossy**: `max_tokens=4096` for a 10-token classification
   overweights by 400×. The `estimated_output_tokens` field is strongly preferred.

3. **Mid-flight progress**: `weighted_load` overestimates for requests near completion.
   Still more accurate than request-count. Acceptable for V1.

4. **Prefill cost lingers in decode phase**: Guard only drops when entire request completes.
   Conservative but safe. Future: split prefill_guard + decode_guard.

5. **PD disaggregation**: Out of scope. Would require a different cost model split across
   prefill/decode worker pools.
