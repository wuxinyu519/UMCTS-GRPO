# UMCTS Semantic Clustering Environment

## Current login-node check

The current shell is on `dt-login02.delta.ncsa.illinois.edu`. The system Python
visible from this shell does not have `torch`, `transformers`, `ray`, or `vllm`.
`nvidia-smi` also cannot see a GPU driver on the login node. This is expected
for login-node work; actual rollout/training should run through the GPU job.

## Environment YAML files

`UMCTS/environment_treegrpo.yml` is the training environment. It contains the
core packages needed by Tree-GRPO and UMCTS-GRPO:

- `torch==2.6.0`
- `transformers==4.51.3`
- `ray==2.55.1`
- `vllm==0.8.5.post1`
- `flash-attn==2.7.4.post1`
- `swanlab`, `datasets`, `hydra-core`, `tensordict`, `requests`

This is sufficient for the current semantic clustering implementation because
it uses HuggingFace `AutoTokenizer` and `AutoModel`; it does not require
`sentence-transformers`.

`UMCTS/environment_retriever_faiss124.yml` is the retriever/server environment.
It contains `faiss-gpu`, `pyserini`, `fastapi`, `uvicorn`, `torch`, and
`transformers`, so it fits the local retrieval service side.

## Required model path

Semantic clustering requires a local embedding model path. The UMCTS scripts now
default to `UMCTS_CLUSTER_MODE=embedding` and intentionally fail if
`UMCTS_EMBEDDING_MODEL` is not set. Example:

```bash
UMCTS_CLUSTER_MODE=embedding \
UMCTS_EMBEDDING_MODEL=/path/to/local/e5-or-bge-model \
bash train_multihopqa_umcts.sh
```

Recommended encoder families are E5, BGE, or GTE-style text embedding models.
The path must be visible on the compute node running the Ray job.

## Why no exact fallback

UMCTS-GRPO specifies semantic action clustering. Exact string matching cannot
identify paraphrased search queries, so the UMCTS scripts do not silently fall
back to exact clustering when `embedding` is selected.
