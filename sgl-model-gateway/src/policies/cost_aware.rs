//! Cost-Aware Load Balancing Policy
//!
//! Routes requests by minimizing cost-weighted queue depth across workers.
//! Combines cache affinity (via radix tree prefix matching) with load awareness
//! using a continuous score rather than binary mode switching.
//!
//! Score: effective_score(worker_i) = existing_load_i - matched_tokens_i × S
//! where S = W_prefill - W_cache_read (the savings per cached token).

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use smg_mesh::OptionalMeshSyncManager;
use tracing::debug;

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;

/// Configuration for cost-aware policy
#[derive(Debug, Clone)]
pub struct CostAwareConfig {
    /// Cost weight per uncached prefill token (default: 6, from $0.60/M × 10).
    pub w_prefill: usize,
    /// Cost weight per cached prefill token (default: 1, from $0.10/M × 10).
    pub w_cache_read: usize,
    /// Cost weight per decode/output token (default: 30, from $3.00/M × 10).
    pub w_decode: usize,
    /// Default estimated output tokens when not provided in request.
    pub default_estimated_output_tokens: usize,
    /// Eviction interval for background tree cleanup (seconds).
    pub eviction_interval_secs: u64,
    /// Max tree nodes per tenant before LRU eviction.
    pub max_tree_size: usize,
}

impl Default for CostAwareConfig {
    fn default() -> Self {
        Self {
            w_prefill: 6,
            w_cache_read: 1,
            w_decode: 30,
            default_estimated_output_tokens: 256,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
        }
    }
}

/// Cost-aware load balancing policy.
///
/// Selects the worker with the lowest effective score:
///   effective_score = existing_weighted_load - matched_chars × S
/// where S = w_prefill - w_cache_read.
#[derive(Debug)]
pub struct CostAwarePolicy {
    config: CostAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    mesh_sync: OptionalMeshSyncManager,
    _eviction_task: Option<PeriodicTask>,
}

impl CostAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CostAwareConfig::default())
    }

    pub fn with_config(config: CostAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "CostAwareEviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);
                        debug!(
                            "Cost-aware eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            trees,
            mesh_sync: None,
            _eviction_task: eviction_task,
        }
    }

    /// Initialize the tree with worker URLs (used during initial setup).
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        let mut model_workers: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }
        for (tree_key, model_workers) in model_workers {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }

    /// Add a single worker to the tree.
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Remove a worker from the tree.
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL from all model trees.
    pub fn remove_worker_by_url(&self, url: &str) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// Approximate token count from char count (~4 chars per token for English).
    fn chars_to_tokens(char_count: usize) -> usize {
        char_count / 4
    }

    /// Compute effective score for a worker.
    /// Lower score = better candidate.
    fn score_worker(
        &self,
        worker: &dyn Worker,
        text: &str,
        tree: &Option<Arc<Tree>>,
    ) -> f64 {
        let existing_load = worker.weighted_load() as f64;

        let matched_tokens = if let Some(tree) = tree {
            let matched_text = tree.prefix_match_tenant(text, worker.url());
            Self::chars_to_tokens(matched_text.chars().count())
        } else {
            0
        };

        let s = (self.config.w_prefill as f64) - (self.config.w_cache_read as f64);
        existing_load - matched_tokens as f64 * s
    }
}

#[async_trait]
impl LoadBalancingPolicy for CostAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        let model_id = normalize_model_key(workers[healthy[0]].model_id());
        let text = info.request_text.unwrap_or("");
        let tree = self.trees.get(model_id).map(|e| e.value().clone());

        // Score each healthy worker: lowest effective_score wins
        let best_idx = healthy.iter().copied().min_by(|&a, &b| {
            let score_a = self.score_worker(workers[a].as_ref(), text, &tree);
            let score_b = self.score_worker(workers[b].as_ref(), text, &tree);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })?;

        // Update tree with selected worker
        if let Some(tree) = &tree {
            tree.insert(text, workers[best_idx].url());
        } else if !text.is_empty() {
            // Create tree on first request with text
            let new_tree = Arc::new(Tree::new());
            new_tree.insert(text, workers[best_idx].url());
            self.trees
                .entry(model_id.to_string())
                .or_insert(new_tree);
        }

        Some(best_idx)
    }

    fn name(&self) -> &'static str {
        "cost_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        self.mesh_sync = mesh_sync;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compute_request_cost(
        &self,
        workers: &[Arc<dyn Worker>],
        worker_idx: usize,
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let text = info.request_text.unwrap_or("");
        let est_output = info
            .estimated_output_tokens
            .unwrap_or(self.config.default_estimated_output_tokens);
        let input_tokens = Self::chars_to_tokens(text.chars().count());

        let model_id = normalize_model_key(workers[worker_idx].model_id());
        let matched_tokens = self
            .trees
            .get(model_id)
            .map(|tree| {
                Self::chars_to_tokens(
                    tree.prefix_match_tenant(text, workers[worker_idx].url())
                        .chars()
                        .count(),
                )
            })
            .unwrap_or(0);

        let uncached = input_tokens.saturating_sub(matched_tokens);
        Some(
            uncached * self.config.w_prefill
                + matched_tokens * self.config.w_cache_read
                + est_output * self.config.w_decode,
        )
    }
}

impl Default for CostAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .api_key("test")
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn make_policy() -> CostAwarePolicy {
        CostAwarePolicy::with_config(CostAwareConfig {
            w_prefill: 6,
            w_cache_read: 1,
            w_decode: 30,
            default_estimated_output_tokens: 256,
            eviction_interval_secs: 0, // disable background task in tests
            max_tree_size: 10000,
        })
    }

    #[test]
    fn test_score_function_math() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        // Set up tree with known prefix for w1
        let tree = Arc::new(Tree::new());
        tree.insert("hello world this is a test", workers[0].url());
        let tree_opt = Some(tree);

        // w1: weighted_load=0, matched="hello world this is a test" (26 chars → 6 tokens), S=5
        // score = 0 - 6 * 5 = -30
        let score1 = policy.score_worker(workers[0].as_ref(), "hello world this is a test", &tree_opt);
        assert!((score1 - (-30.0)).abs() < 0.01, "Expected -30, got {}", score1);

        // w2: weighted_load=0, matched="" (0 tokens), S=5
        // score = 0 - 0 * 5 = 0
        let score2 = policy.score_worker(workers[1].as_ref(), "hello world this is a test", &tree_opt);
        assert!((score2 - 0.0).abs() < 0.01, "Expected 0, got {}", score2);
    }

    #[tokio::test]
    async fn test_equal_load_prefers_cache() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        // Insert text into w1's tree
        let model_id = normalize_model_key(workers[0].model_id());
        if let Some(tree) = policy.trees.get(model_id) {
            tree.insert("hello world this is a long prefix", workers[0].url());
        }

        let info = SelectWorkerInfo {
            request_text: Some("hello world this is a long prefix with more text"),
            ..Default::default()
        };

        // Both have 0 load, but w1 has cache match → should prefer w1
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 0, "Should prefer worker with cache match when loads are equal");
    }

    #[tokio::test]
    async fn test_short_output_prefers_cache() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        // Insert long prefix into w1's tree
        let model_id = normalize_model_key(workers[0].model_id());
        if let Some(tree) = policy.trees.get(model_id) {
            tree.insert("a]b]c]d]e]f]g]h]i]j]k]l]m]n]o]p]q]r]s]t]u]v]w]x]y]z]", workers[0].url());
        }

        // Give w1 slightly more weighted load
        workers[0].increment_weighted_load(50);

        let info = SelectWorkerInfo {
            request_text: Some("a]b]c]d]e]f]g]h]i]j]k]l]m]n]o]p]q]r]s]t]u]v]w]x]y]z]"),
            estimated_output_tokens: Some(10), // short output
            ..Default::default()
        };

        // w1 has more load but big cache match → savings should dominate for short output
        // 56 chars → 14 tokens, S=5: savings = 14 * 5 = 70
        // w1: score = 50 - 70 = -20
        // w2: score = 0 - 0 = 0
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 0, "Short output should prefer cache-hit worker even with more load");
    }

    #[tokio::test]
    async fn test_long_output_prefers_load_balance() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        // Insert some prefix into w1's tree (20 chars)
        let model_id = normalize_model_key(workers[0].model_id());
        if let Some(tree) = policy.trees.get(model_id) {
            tree.insert("12345678901234567890", workers[0].url());
        }

        // Give w1 significantly more weighted load (simulating heavy decode work)
        // 20 chars → 5 tokens, savings = 5 * 5 = 25
        // Make w1 load > 25 so load overwhelms cache savings
        workers[0].increment_weighted_load(200);

        let info = SelectWorkerInfo {
            request_text: Some("12345678901234567890extra"),
            estimated_output_tokens: Some(4000), // long output
            ..Default::default()
        };

        // w1: score = 200 - 5 * 5 = 200 - 25 = 175
        // w2: score = 0 - 0 * 5 = 0
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 1, "Long output with heavy load should prefer least-loaded worker");
    }

    #[tokio::test]
    async fn test_zero_cache_match() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        // Give w2 more load, no cache anywhere
        workers[1].increment_weighted_load(500);

        let info = SelectWorkerInfo {
            request_text: Some("brand new text never seen before"),
            ..Default::default()
        };

        // No prefix match → purely load-based: w1 (load=0) should win
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 0, "Zero cache match should route to least-loaded worker");
    }

    #[tokio::test]
    async fn test_single_worker() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000"]);

        let info = SelectWorkerInfo {
            request_text: Some("anything"),
            ..Default::default()
        };

        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 0, "Single worker should always be selected");
    }

    #[tokio::test]
    async fn test_all_unhealthy() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].set_healthy(false);
        workers[1].set_healthy(false);

        let info = SelectWorkerInfo {
            request_text: Some("anything"),
            ..Default::default()
        };

        let result = policy.select_worker(&workers, &info).await;
        assert!(result.is_none(), "All unhealthy should return None");
    }

    #[tokio::test]
    async fn test_no_estimate_uses_default() {
        let policy = make_policy();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        // Insert prefix for w1
        let model_id = normalize_model_key(workers[0].model_id());
        if let Some(tree) = policy.trees.get(model_id) {
            tree.insert("test prefix text", workers[0].url());
        }

        let info = SelectWorkerInfo {
            request_text: Some("test prefix text"),
            estimated_output_tokens: None, // no estimate → uses default (256)
            ..Default::default()
        };

        // compute_request_cost should use default_estimated_output_tokens (256)
        let cost = policy.compute_request_cost(&workers, 0, &info).unwrap();
        // "test prefix text" = 16 chars → 4 tokens, all matched on w1
        // cost = 0 * 6 + 4 * 1 + 256 * 30 = 0 + 4 + 7680 = 7684
        assert_eq!(cost, 7684, "Should use default estimated output tokens when None");

        let cost_w2 = policy.compute_request_cost(&workers, 1, &info).unwrap();
        // w2: no match, so all uncached (4 tokens)
        // cost = 4 * 6 + 0 * 1 + 256 * 30 = 24 + 0 + 7680 = 7704
        assert_eq!(cost_w2, 7704, "Worker without cache should have higher cost");
    }
}
