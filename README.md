# UMCTS-GRPO

This repository contains the UMCTS-GRPO code forked from Tree-GRPO. Large runtime files are not tracked by git. Put datasets, indexes, models, logs, and training outputs under local directories such as `data/`, `logs/`, and `results/`.

## 1. Create Environments

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

## 4. Prepare QA Datasets

Run in the `treegrpo` environment:

```bash
conda activate treegrpo
bash scripts/data_process/data_process_singlehop.sh
bash scripts/data_process/data_process_multihop.sh
```

Expected files:

```text
data/singlehopqa/train.parquet
data/singlehopqa/test.parquet
data/multihopqa/train.parquet
data/multihopqa/test.parquet
```

## 5. Prepare Models

Place local Hugging Face model directories under `data/models/`, for example:

```text
data/models/Qwen2.5-1.5B-Instruct
data/models/e5-base-v2
```

`Qwen2.5-1.5B-Instruct` is used as the policy model. `e5-base-v2` is used by the retriever and by UMCTS semantic clustering when `UMCTS_CLUSTER_MODE=embedding`.

## 6. Start Retriever

On Delta Slurm, start the retriever first:

```bash
sbatch submit_retriever_2gpu.sbatch
```

The retriever job writes its URL to:

```text
run_info/retriever_url.txt
```

## 7. Submit UMCTS Training

The training job uses 8 GPUs. The retriever uses separate GPUs and is not included in the training GPU count.

Example: 4 nodes x 2 A100 GPUs:

```bash
sbatch \
  --partition=gpuA100x4 \
  --nodes=4 \
  --gpus-per-node=2 \
  --cpus-per-task=16 \
  --job-name=umcts_train_4x2 \
  --export=ALL,N_NODES=4,N_GPUS_PER_NODE=2,RUN_ID=qwen25-15b_singlehop_4x2_run1 \
  submit_train_8gpu.sbatch
```

Example: 1 node x 8 A100 GPUs:

```bash
sbatch \
  --partition=gpuA100x8 \
  --nodes=1 \
  --gpus-per-node=8 \
  --cpus-per-task=64 \
  --job-name=umcts_train_8gpu \
  --export=ALL,N_NODES=1,N_GPUS_PER_NODE=8,RUN_ID=qwen25-15b_singlehop_8gpu_run1 \
  submit_train_8gpu.sbatch
```

Use a new `RUN_ID` for each run. Existing result directories are not overwritten.

Training outputs are saved under:

```text
results/<dataset>/<model>/<run_name>/
```

## 8. Useful Commands

Check jobs:

```bash
squeue -u $USER
```

Cancel a job:

```bash
scancel <JOBID>
```

Watch logs:

```bash
tail -f logs/umcts_train_<JOBID>.out
tail -f logs/umcts_train_<JOBID>.err
```

Check retriever URL:

```bash
cat run_info/retriever_url.txt
```
