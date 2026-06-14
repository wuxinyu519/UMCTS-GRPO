# UMCTS-GRPO

This repository contains the UMCTS-GRPO code forked from Tree-GRPO. Large runtime files are not tracked by git. Put datasets, indexes, models, logs, and training outputs under local directories such as `data/`, `logs/`, and `results/`.

## 1. Create Environments

The current working environments are exported under `envs/`. To reproduce the
same package set, create both environments from these YAML files:

```bash
conda env create -f envs/treegrpo-full.yml
conda env create -f envs/retriever-full.yml
```

The manual installation commands below are kept as a fallback.

Training environment:

```bash
conda create -n treegrpo python=3.12.9
conda activate treegrpo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install vllm==0.8.5.post1
pip install -e .
pip install flash-attn --no-build-isolation
pip install swanlab
```

Retriever environment:

```bash
conda create -n retriever python=3.10.13
conda activate retriever
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install transformers datasets pyserini uvicorn fastapi
```

Install FAISS GPU using the version available on your platform. On some clusters, `faiss-gpu==1.7.3` is not available via pip; use conda or the cluster-supported FAISS GPU package.

## 2. Prepare Directories

Run from the repository root:

```bash
mkdir -p data/wiki18 data/models logs results run_info
```

## 3. Download Retrieval Data

Download the Wiki18 index and corpus:

```bash
python scripts/download.py --save_path data/wiki18
cat data/wiki18/part_* > data/wiki18/e5_Flat.index
gzip -d data/wiki18/wiki-18.jsonl.gz
```

Expected files:

```text
data/wiki18/e5_Flat.index
data/wiki18/wiki-18.jsonl
```

## 4. Prepare Single-Hop QA Dataset

Run in the `treegrpo` environment:

```bash
conda activate treegrpo
bash scripts/data_process/data_process_singlehop.sh
```

Expected files:

```text
data/singlehopqa/train.parquet
data/singlehopqa/test.parquet
```

## 5. Prepare Models

Download the policy model and embedding model under `data/models/`.

If the cluster can access Hugging Face directly:

```bash
conda activate treegrpo
pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir data/models/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False

huggingface-cli download intfloat/e5-base-v2 \
  --local-dir data/models/e5-base-v2 \
  --local-dir-use-symlinks False
```

If Hugging Face requires authentication on your cluster, run this first:

```bash
huggingface-cli login
```

Expected model directories:

```text
data/models/Qwen2.5-1.5B-Instruct
data/models/e5-base-v2
```

`Qwen2.5-1.5B-Instruct` is used as the policy model. `e5-base-v2` is used by the retriever and by UMCTS semantic clustering when `UMCTS_CLUSTER_MODE=embedding`.

## 6. Run Experiments

The repository exposes three top-level bash entry points. By default they run on
one local 8-GPU node, start a local retriever, start Ray with the dashboard
enabled, and submit training through Ray Jobs at `http://127.0.0.1:8265`.

Run from the repository root:

```bash
cd UMCTS-GRPO
conda activate treegrpo
```

Single-hop, all configured models, default UMCTS parameters:

```bash
bash submit_sweep_singlehop_all_models.sh
```

Multi-hop, all configured models, default UMCTS parameters:

```bash
bash submit_sweep_multihop_all_models.sh
```

Single-hop and multi-hop, all configured models, all six parameter settings:

```bash
bash submit_sweep_all_models_all_params.sh
```

The all-parameter sweep submits/runs:

```text
5 models x 2 datasets x 6 parameter settings = 60 runs
```

The scripts expect these model directories:

```text
data/models/Qwen2.5-1.5B-Instruct
data/models/Qwen2.5-3B
data/models/Llama-3.2-3B
data/models/Qwen2.5-7B
data/models/Qwen2.5-14B
data/models/e5-base-v2
```

Missing model directories are skipped by default. To fail immediately when a
model or dataset is missing:

```bash
STRICT_MODE=1 bash submit_sweep_singlehop_all_models.sh
```

## 7. Optional Slurm Backend

For Slurm clusters, use the same three bash scripts with `BACKEND=slurm`. Each
training run requests one `gpuA100x8` node by default. Inside that single node,
the launcher starts a local retriever on GPUs `0,1`, waits for it to answer
`/retrieve`, then starts training on GPUs `2,3,4,5,6,7`.

```bash
BACKEND=slurm bash submit_sweep_singlehop_all_models.sh
BACKEND=slurm bash submit_sweep_multihop_all_models.sh
BACKEND=slurm bash submit_sweep_all_models_all_params.sh
```

Internal Slurm helper scripts live under `scripts/slurm/`. They are not the
main user entry points.

The default Slurm training helper is:

```text
scripts/slurm/train_8gpu.sbatch
```

It uses `N_GPUS_PER_NODE=6` for training because two of the eight allocated GPUs
are reserved for the retriever. To use an external shared retriever instead,
set `RETRIEVER_PER_TRAIN_JOB=0` and provide or start a retriever with
`scripts/slurm/retriever_2gpu.sbatch`.

## 8. Results

Training outputs are saved under:

```text
results/<dataset>/<model>/<run_name>/
```

Examples:

```text
results/singlehopqa/Qwen2.5-1.5B-Instruct/<run_name>/
results/multihopqa/Qwen2.5-7B/<run_name>/
```

Every training result directory contains:

```text
logs/verl.log
config/run.env
config/train_script.sh
config/git_commit.txt
config/git_diff.patch
checkpoints/
```

Use a new `RUN_ID` or a different preset script for each run. Existing result
directories are never overwritten.
