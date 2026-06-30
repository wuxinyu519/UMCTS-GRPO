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

huggingface-cli download Qwen/Qwen2.5-1.5B \
  --local-dir data/models/Qwen2.5-1.5B \
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
data/models/Qwen2.5-1.5B
data/models/e5-base-v2
```

`Qwen2.5-1.5B` is used as the policy model. `e5-base-v2` is used by the retriever and by UMCTS semantic clustering when `UMCTS_CLUSTER_MODE=embedding`.

## 6. Run Order On The Cluster

Run commands from the repository root:

```bash
cd /projects/bhnl/xwu20/UMCTS-GRPO
```

First start exactly one shared retriever:

```bash
sbatch submit_retriever_2gpu.sbatch
```

The retriever uses:

```text
data/wiki18/e5_Flat.index
data/wiki18/wiki-18.jsonl
data/models/e5-base-v2/
```

It writes its endpoint to:

```bash
cat run_info/retriever_url.txt
```

Then submit training jobs. The recommended training scripts do not start a new
retriever; they read `run_info/retriever_url.txt` automatically and wait for it
if the retriever is still starting.

## 7. Recommended Training Submits

Do not start Ray manually. Each training Slurm job starts a single-node Ray head
inside its allocated `gpuA100x8` node, then runs the selected training script.

Default single-hop Qwen2.5-1.5B run:

```bash
sbatch submit_train_8gpu.sbatch
```

Single-hop Qwen2.5-1.5B parameter presets:

```bash
sbatch submit_train_8gpu.sbatch    # default: l=1, c_u=1.0, lambda_l=1.0, tau=1.0
sbatch submit_train_param1.sbatch  # c_u=0.5
sbatch submit_train_param2.sbatch  # c_u=2.0
sbatch submit_train_param3.sbatch  # lambda_l=2.0
sbatch submit_train_param4.sbatch  # tau=5.0
sbatch submit_train_param5.sbatch  # tau=0.1
```

All-model default sweeps:

```bash
bash submit_sweep_singlehop_all_models.sh
bash submit_sweep_multihop_all_models.sh
```

All models, both datasets, all six parameter settings:

```bash
bash submit_sweep_all_models_all_params.sh
```

The all-parameter sweep submits 60 training jobs:

```text
5 models x 2 datasets x 6 parameter settings = 60 jobs
```

The model sweep scripts expect these local model directories:

```text
data/models/Qwen2.5-1.5B
data/models/Qwen2.5-3B
data/models/Llama-3.2-3B
data/models/Qwen2.5-7B
data/models/Qwen2.5-14B
```

Missing models are skipped by default. Set `STRICT_MODE=1` to fail instead:

```bash
STRICT_MODE=1 bash submit_sweep_singlehop_all_models.sh
```

## 8. Results And Logs

Training results are saved under:

```text
results/<dataset>/<model>/<run_name>/
```

Examples:

```text
results/singlehopqa/Qwen2.5-1.5B/<run_name>/
results/multihopqa/Qwen2.5-7B/<run_name>/
```

Each training result directory contains:

```text
logs/verl.log
config/run.env
config/train_script.sh
config/git_commit.txt
config/git_diff.patch
checkpoints/
```

Slurm stdout/stderr logs are saved separately:

```text
logs/umcts_retriever_<JOBID>.out
logs/umcts_retriever_<JOBID>.err
logs/umcts_train_<JOBID>.out
logs/umcts_train_<JOBID>.err
logs/umcts_train_p<NUM>_<JOBID>.out
logs/umcts_train_p<NUM>_<JOBID>.err
```

Existing result directories are never overwritten. If a result directory already
exists, the training script exits instead of replacing it. Use a new `RUN_ID` or
a different sweep tag for repeated runs.

## 9. Submit Script Summary

Recommended shared-retriever scripts:

| Script | Purpose | Default resources | Retriever behavior |
|---|---|---|---|
| `submit_retriever_2gpu.sbatch` | Starts the shared retrieval server | 1 `gpuA100x4` node, 2 GPUs | Writes `run_info/retriever_url.txt` |
| `submit_train_8gpu.sbatch` | Default single-hop Qwen2.5-1.5B UMCTS training | 1 `gpuA100x8` node, 8 GPUs | Reads shared retriever URL |
| `submit_train_param1.sbatch` | Single-hop preset: `c_u=0.5` | 1 `gpuA100x8` node, 8 GPUs | Reads shared retriever URL |
| `submit_train_param2.sbatch` | Single-hop preset: `c_u=2.0` | 1 `gpuA100x8` node, 8 GPUs | Reads shared retriever URL |
| `submit_train_param3.sbatch` | Single-hop preset: `lambda_l=2.0` | 1 `gpuA100x8` node, 8 GPUs | Reads shared retriever URL |
| `submit_train_param4.sbatch` | Single-hop preset: `tau=5.0` | 1 `gpuA100x8` node, 8 GPUs | Reads shared retriever URL |
| `submit_train_param5.sbatch` | Single-hop preset: `tau=0.1` | 1 `gpuA100x8` node, 8 GPUs | Reads shared retriever URL |
| `submit_sweep_singlehop_all_models.sh` | Submits default single-hop jobs for all configured models | Each job uses `submit_train_8gpu.sbatch` | Reads shared retriever URL |
| `submit_sweep_multihop_all_models.sh` | Submits default multi-hop jobs for all configured models | Each job uses `submit_train_8gpu.sbatch` | Reads shared retriever URL |
| `submit_sweep_all_models_all_params.sh` | Submits all configured model, dataset, and parameter combinations | Each job uses `submit_train_8gpu.sbatch` | Reads shared retriever URL |
