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

## 6. Start Retriever On The Cluster

Start the retriever as a Slurm job from the repository root:

```bash
sbatch submit_retriever_2gpu.sbatch
```

The retriever job uses the local files below by default:

```text
data/wiki18/e5_Flat.index
data/wiki18/wiki-18.jsonl
data/models/e5-base-v2/
```

It writes the retriever endpoint here:

```bash
cat run_info/retriever_url.txt
```

The training jobs read this file automatically and wait for it if the retriever
has not finished starting.

## 7. Run Single-Hop UMCTS Training On The Cluster

Do not start Ray manually. The Slurm training scripts start Ray inside the
allocated job and then run `train_singlehopqa_umcts.sh`.

Submit the default single-hop training job:

```bash
sbatch submit_train_8gpu.sbatch
```

For the current parameter sweep, submit any of these independent single-hop jobs:

```bash
sbatch submit_train_8gpu.sbatch    # default baseline: l=1, c_u=1.0, lambda_l=1.0
sbatch submit_train_param1.sbatch  # conservative uncertainty: l=1, c_u=0.5
sbatch submit_train_param2.sbatch  # aggressive uncertainty: l=1, c_u=2.0
sbatch submit_train_param3.sbatch  # stronger local advantage: l=1, c_u=1.0, lambda_l=2.0
sbatch submit_train_param4.sbatch  # strong confidence gate: l=1, c_u=1.0, tau=5.0
sbatch submit_train_param5.sbatch  # weak confidence gate: l=1, c_u=1.0, tau=0.1
```

Each preset requests one `gpuA100x8` node by default. Existing result
directories are not overwritten. If a result directory already exists, the run
exits instead of replacing it.

Training outputs are saved under:

```text
results/<dataset>/<model>/<run_name>/
```

For example, the preset scripts save to directories like:

```text
results/singlehopqa/Qwen2.5-1.5B-Instruct/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_1/
results/singlehopqa/Qwen2.5-1.5B-Instruct/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_2/
```

Each result directory contains:

```text
logs/verl.log
config/run.env
config/train_script.sh
config/git_commit.txt
config/git_diff.patch
checkpoints/
```

## 8. Script Summary

| Script | What it runs | Default resources | Output location |
|---|---|---|---|
| `submit_retriever_2gpu.sbatch` | Starts the retrieval server and writes `run_info/retriever_url.txt` | 1 `gpuA100x4` node, 2 GPUs | Slurm logs in `logs/umcts_retriever_<JOBID>.out/.err` |
| `submit_train_8gpu.sbatch` | Default single-hop baseline: `l=1`, `c_u=1.0`, `lambda_l=1.0` | 1 `gpuA100x8` node, 8 GPUs | Results under `results/singlehopqa/<model>/<run_name>/`; Slurm logs in `logs/umcts_train_<JOBID>.out/.err` |
| `submit_train_param1.sbatch` | Single-hop UMCTS with conservative uncertainty: `l=1`, `c_u=0.5`, `lambda_l=1.0`, `tau=1.0` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_1/` |
| `submit_train_param2.sbatch` | Single-hop UMCTS with aggressive uncertainty: `l=1`, `c_u=2.0`, `lambda_l=1.0`, `tau=1.0` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_2/` |
| `submit_train_param3.sbatch` | Single-hop UMCTS with stronger local advantage: `l=1`, `c_u=1.0`, `lambda_l=2.0`, `tau=1.0` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_3/` |
| `submit_train_param4.sbatch` | Single-hop UMCTS with stronger confidence gate: `l=1`, `c_u=1.0`, `lambda_l=1.0`, `tau=5.0` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_4/` |
| `submit_train_param5.sbatch` | Single-hop UMCTS with weaker confidence gate: `l=1`, `c_u=1.0`, `lambda_l=1.0`, `tau=0.1` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_5/` |
| `train_singlehopqa_umcts.sh` | The actual single-hop UMCTS training command and hyperparameter overrides | Uses the Ray cluster started by the submit script | Writes `logs/verl.log`, config snapshots, and checkpoints under the result directory |

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
