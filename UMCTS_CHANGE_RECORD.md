# UMCTS-GRPO Change Record

## 2026-06-12 - Active search result alignment

Purpose:
- Fix a rollout bookkeeping bug where inactive branches could trigger retriever calls and leave unused search results.

Files changed:
- `search_r1/llm_agent/generation_ts.py`

Compared with the previous implementation:
- Search queries are now collected only when `active_mask` is true and the parsed action is `search`.
- The consistency assertion now checks the number of active search actions.
- The `do_search=False` path now creates placeholder search results only for active search actions.

Reason:
- The environment loop only consumes `search_results` for `active=True and action == "search"`.
- UMCTS tree expansion should execute environment actions only for selected active representative branches.
- Using the same condition for search collection and search consumption prevents inactive branches from corrupting rollout state.

## 2026-06-12 - Leaf sampling fallback for sparse UMCTS trees

Purpose:
- Prevent UMCTS rollout from crashing when a tree produces fewer valid leaf trajectories than the requested `ts_k` budget.

Files changed:
- `search_r1/llm_agent/tree_node.py`

Compared with the previous implementation:
- `sample_leaf()` no longer asserts that the number of leaf candidates is at least `n`.
- Candidate leaf uids are still used to prune the tree, preserving the original pruning flow.
- After pruning, if the collected result leaves are fewer than `n`, leaves are sampled with replacement to preserve the requested per-tree output count.

Reason:
- `ts_k` is a rollout budget, but small models or clustered UMCTS expansions can yield fewer valid terminal branches.
- UMCTS should consume the available candidate trajectories instead of aborting the entire training job.
- Keeping the per-tree sample count stable preserves downstream GRPO/UMCTS grouping assumptions.
