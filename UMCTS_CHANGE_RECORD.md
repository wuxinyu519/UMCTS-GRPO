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
