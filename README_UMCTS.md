# UMCTS-GRPO Notes

This directory is a working copy of `Tree-GRPO` for UMCTS-GRPO experiments.
The original `Tree-GRPO` directory is left unchanged.

The implementation target is `UMCTS/umcts_grpo_neurips2026.pdf`. Details from
`UMCTS/3381_Uncertainty_Aware_Test_Ti.pdf` are used only where the NeurIPS draft
states the high-level method but does not specify engineering details.

## Current UMCTS changes

- Adds `expand_mode=umcts` as a rollout option.
- During expansion, samples `umcts_candidate_k` candidate next agent steps per selected prefix.
- Clusters candidate actions before environment execution. Supported modes:
  - `exact`: normalized action type/content match.
  - `lexical`: greedy Jaccard similarity clustering.
  - `embedding`: HuggingFace encoder cosine-similarity clustering.
- Executes one representative action per cluster and assigns `policy_prior = cluster_size / candidate_count`.
- Tracks per-node visit count, value sum, policy prior, interaction cost, and uncertainty in `search_r1/llm_agent/tree_node.py`.
- Estimates node uncertainty from rollout token log probability:
  `uncertainty = 1 - exp(mean_token_logprob)`.
- Selects expansion nodes with a PUCT-style score:
  `value + ucb + uncertainty_bonus + prior_bonus - cost_penalty`.
- Backpropagates final leaf reward through ancestor nodes after reward evaluation, tracking mean and variance.
- Adds `adv_estimator=umcts`, which combines inter-tree GRPO advantage with confidence-calibrated local process advantages.
- Adds UMCTS launch scripts:
  - `train_multihopqa_umcts.sh`
  - `train_singlehopqa_umcts.sh`
  - `train_webagent_umcts.sh`

## Important implementation note

For strict semantic-equivalence clustering that can merge paraphrased search
queries, run with:

```bash
UMCTS_CLUSTER_MODE=embedding UMCTS_EMBEDDING_MODEL=/path/to/local/encoder bash train_multihopqa_umcts.sh
```

Without an embedding model, `exact` and `lexical` are runnable approximations,
not a full semantic paraphrase detector.
