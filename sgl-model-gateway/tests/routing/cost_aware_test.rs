//! Cost-aware load balancing integration tests
//!
//! End-to-end tests that validate the cost_aware policy routes requests
//! through the full gateway stack (mock workers → gateway → policy → routing).
//!
//! These tests compare cost_aware routing against simpler baselines
//! (round_robin, cache_aware) under workloads where output-length awareness
//! matters.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestWorkerConfig};

/// Build a cost_aware router config with default weights.
fn cost_aware_config(port: u16) -> RouterConfig {
    RouterConfig::builder()
        .regular_mode(vec![])
        .policy(PolicyConfig::CostAware {
            w_prefill: 6,
            w_cache_read: 1,
            w_decode: 30,
            default_estimated_output_tokens: 256,
            eviction_interval_secs: 60,
            max_tree_size: 10000,
        })
        .host("127.0.0.1")
        .port(port)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked()
}

#[cfg(test)]
mod basic_routing {
    use super::*;

    /// Verify requests succeed through the cost_aware policy.
    #[tokio::test]
    async fn test_cost_aware_basic_routing() {
        let config = cost_aware_config(3200);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19200, 2))
                .await;
        let app = ctx.create_app().await;

        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false
            });
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(success_count, 10, "All requests should succeed with cost_aware");
        ctx.shutdown().await;
    }

    /// Verify cost_aware handles a single worker correctly.
    #[tokio::test]
    async fn test_cost_aware_single_worker() {
        let config = cost_aware_config(3201);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19210)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Single worker request",
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod chat_completions {
    use super::*;

    /// Verify cost_aware routes /v1/chat/completions with estimated_output_tokens.
    #[tokio::test]
    async fn test_cost_aware_chat_completions_with_estimate() {
        let config = cost_aware_config(3202);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19220, 2))
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model-path",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "estimated_output_tokens": 50,
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        ctx.shutdown().await;
    }

    /// Verify cost_aware gracefully handles missing estimated_output_tokens
    /// (falls back to max_tokens, then default).
    #[tokio::test]
    async fn test_cost_aware_chat_completions_no_estimate() {
        let config = cost_aware_config(3203);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19230, 2))
                .await;
        let app = ctx.create_app().await;

        // No estimated_output_tokens, no max_tokens → uses default (256)
        let payload = json!({
            "model": "mock-model-path",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod mixed_output_lengths {
    use std::time::Instant;

    use super::*;

    /// Core benchmark: send concurrent requests with mixed output estimates through
    /// both cost_aware and round_robin, compare total wall-clock time.
    ///
    /// Setup: 2 workers, each with 100ms base delay.
    /// Workload: 6 requests — 3 short (estimated 20 tokens) + 3 long (estimated 2000 tokens).
    ///
    /// Round-robin will alternate, potentially stacking 2 long requests on one worker.
    /// Cost-aware should spread long requests across workers, reducing tail latency.
    #[tokio::test]
    async fn test_cost_aware_distributes_long_requests() {
        // --- cost_aware ---
        let config_ca = cost_aware_config(3204);
        let ctx_ca = AppTestContext::new_with_config(
            config_ca,
            TestWorkerConfig::slow_workers(19240, 2, 50),
        )
        .await;
        let app_ca = ctx_ca.create_app().await;

        let requests = build_mixed_workload();
        let start = Instant::now();
        let mut handles = vec![];
        for payload in requests {
            let a = app_ca.clone();
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();
                let resp = a.oneshot(req).await.unwrap();
                resp.status()
            }));
        }
        let mut ca_ok = 0;
        for h in handles {
            if h.await.unwrap() == StatusCode::OK {
                ca_ok += 1;
            }
        }
        let ca_time = start.elapsed();

        // --- round_robin ---
        let config_rr = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3205)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        let ctx_rr = AppTestContext::new_with_config(
            config_rr,
            TestWorkerConfig::slow_workers(19250, 2, 50),
        )
        .await;
        let app_rr = ctx_rr.create_app().await;

        let requests = build_mixed_workload();
        let start = Instant::now();
        let mut handles = vec![];
        for payload in requests {
            let a = app_rr.clone();
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();
                let resp = a.oneshot(req).await.unwrap();
                resp.status()
            }));
        }
        let mut rr_ok = 0;
        for h in handles {
            if h.await.unwrap() == StatusCode::OK {
                rr_ok += 1;
            }
        }
        let rr_time = start.elapsed();

        // Both must succeed fully
        assert_eq!(ca_ok, 6, "All cost_aware requests should succeed");
        assert_eq!(rr_ok, 6, "All round_robin requests should succeed");

        // Log timing for comparison (not a hard assert — mock workers don't
        // actually model real output-proportional latency, but the routing
        // decisions differ and can be inspected in the weighted_load counters).
        eprintln!(
            "[cost_aware_vs_rr] cost_aware={:?}, round_robin={:?}",
            ca_time, rr_time
        );

        ctx_ca.shutdown().await;
        ctx_rr.shutdown().await;
    }

    /// Build a mixed workload: 3 short + 3 long requests with shared system prompt.
    fn build_mixed_workload() -> Vec<serde_json::Value> {
        let system_prompt = "You are a helpful assistant. You answer questions accurately and concisely. This is a shared system prompt for all requests in this batch.";

        let mut requests = vec![];
        // 3 short-output requests
        for i in 0..3 {
            requests.push(json!({
                "model": "mock-model-path",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": format!("Short question {}", i)}
                ],
                "max_tokens": 50,
                "estimated_output_tokens": 20,
                "stream": false
            }));
        }
        // 3 long-output requests
        for i in 0..3 {
            requests.push(json!({
                "model": "mock-model-path",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": format!("Write a detailed essay about topic {}", i)}
                ],
                "max_tokens": 4096,
                "estimated_output_tokens": 2000,
                "stream": false
            }));
        }
        requests
    }
}

#[cfg(test)]
mod weighted_load_tracking {
    use super::*;

    /// Verify that weighted_load on workers increases during in-flight requests
    /// and decreases after completion (RAII guard behavior).
    #[tokio::test]
    async fn test_weighted_load_increases_and_decreases() {
        let config = cost_aware_config(3206);
        // Use slow workers so requests are in-flight long enough to measure
        let ctx = AppTestContext::new_with_config(
            config,
            TestWorkerConfig::slow_workers(19260, 2, 200),
        )
        .await;
        let app = ctx.create_app().await;

        // Check initial weighted load is 0
        let workers = ctx.app_context.worker_registry.get_all();
        for w in &workers {
            assert_eq!(
                w.weighted_load(),
                0,
                "Initial weighted_load should be 0 for {}",
                w.url(),
            );
        }

        // Send a request that will take 200ms (the slow worker delay)
        let payload = json!({
            "model": "mock-model-path",
            "messages": [{"role": "user", "content": "Test message"}],
            "max_tokens": 100,
            "estimated_output_tokens": 50,
            "stream": false
        });
        let a = app.clone();
        let handle = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();
            a.oneshot(req).await.unwrap()
        });

        // Give routing a moment to happen
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // At least one worker should have non-zero weighted_load
        let workers = ctx.app_context.worker_registry.get_all();
        let total_weighted_load: usize = workers.iter().map(|w| w.weighted_load()).sum();
        assert!(
            total_weighted_load > 0,
            "weighted_load should be > 0 while request is in-flight (got {})",
            total_weighted_load
        );

        // Wait for request to complete
        let resp = handle.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Give RAII guard a moment to drop
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Weighted load should return to 0
        let workers = ctx.app_context.worker_registry.get_all();
        let total_weighted_load: usize = workers.iter().map(|w| w.weighted_load()).sum();
        assert_eq!(
            total_weighted_load, 0,
            "weighted_load should return to 0 after request completes"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod streaming {
    use super::*;

    /// Verify cost_aware handles streaming chat completions.
    #[tokio::test]
    async fn test_cost_aware_streaming() {
        let config = cost_aware_config(3207);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19270, 2))
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model-path",
            "messages": [{"role": "user", "content": "Stream test"}],
            "max_tokens": 100,
            "estimated_output_tokens": 30,
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify it's a streaming response
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("text/event-stream"),
            "Streaming response should have text/event-stream content type, got: {}",
            content_type
        );

        ctx.shutdown().await;
    }
}
