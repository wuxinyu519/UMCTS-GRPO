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

Place local Hugging Face model directories under `data/models/`, for example:

```text
data/models/Qwen2.5-1.5B-Instruct
data/models/e5-base-v2
```

`Qwen2.5-1.5B-Instruct` is used as the policy model. `e5-base-v2` is used by the retriever and by UMCTS semantic clustering when `UMCTS_CLUSTER_MODE=embedding`.

## 6. Start Retriever

Edit `local_retrieval_launch.sh` so the paths point to your local index,
corpus, and E5 model:

```bash
index_file=data/wiki18/e5_Flat.index
corpus_file=data/wiki18/wiki-18.jsonl
retriever_path=data/models/e5-base-v2
```

Then start the retriever in the `retriever` environment:

```bash
conda activate retriever
bash local_retrieval_launch.sh
```

By default, the training scripts expect:

```text
http://127.0.0.1:8000/retrieve
```

On Delta, the Slurm retriever helper starts the server and writes the URL to
`run_info/retriever_url.txt`:

```bash
sbatch submit_retriever_2gpu.sbatch
```

## 7. Start Ray For Training

UMCTS training follows the Tree-GRPO default launch style: start a Ray head with dashboard/job server, then run the training script. The retriever uses separate GPUs and is not included in the training GPU count.

For an 8-GPU training node:

```bash
conda activate treegrpo
ray stop --force
ray start --head --node-ip-address=127.0.0.1 --dashboard-host=127.0.0.1 --dashboard-port=8265 --num-gpus=8
```

The UMCTS scripts submit work through Ray's default job server:

```bash
ray job submit --address=http://127.0.0.1:8265 -- python3 -m verl.trainer.main_ppo_format_ts ...
```

## 8. Run UMCTS Training

Single-hop QA:

```bash
conda activate treegrpo
RUN_ID=qwen25-15b_singlehop_run1 bash train_singlehopqa_umcts.sh
```

On Delta, `submit_train_8gpu.sbatch` starts Ray inside the Slurm allocation and
runs single-hop QA. By default it requests one full A100 8-GPU node:

```bash
sbatch submit_train_8gpu.sbatch
```

Preset submit scripts are provided for common parameter sweeps. Each preset is
an independent single-hop QA Slurm job:

```bash
sbatch submit_train_param1.sbatch  # baseline: threshold 0.85, tau 1.0
sbatch submit_train_param2.sbatch  # stricter clustering: threshold 0.90
sbatch submit_train_param3.sbatch  # looser clustering: threshold 0.80
sbatch submit_train_param4.sbatch  # stronger confidence gate: tau 2.0
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

## 9. Useful Commands

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

## 10. Script Summary

| Script | What it runs | Default resources | Output location |
|---|---|---|---|
| `submit_retriever_2gpu.sbatch` | Starts the retrieval server and writes `run_info/retriever_url.txt` | 1 `gpuA100x4` node, 2 GPUs | Slurm logs in `logs/umcts_retriever_<JOBID>.out/.err` |
| `submit_train_8gpu.sbatch` | Starts Ray in the Slurm allocation and runs `TRAIN_SCRIPT`; defaults to `train_singlehopqa_umcts.sh` | 1 `gpuA100x8` node, 8 GPUs | Results under `results/singlehopqa/<model>/<run_name>/`; Slurm logs in `logs/umcts_train_<JOBID>.out/.err` |
| `submit_train_param1.sbatch` | Single-hop UMCTS baseline, threshold `0.85`, tau `1.0` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_1/` |
| `submit_train_param2.sbatch` | Single-hop UMCTS with stricter clustering, threshold `0.90` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_2/` |
| `submit_train_param3.sbatch` | Single-hop UMCTS with looser clustering, threshold `0.80` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_3/` |
| `submit_train_param4.sbatch` | Single-hop UMCTS with stronger confidence gate, tau `2.0` | 1 `gpuA100x8` node, 8 GPUs | `results/singlehopqa/Qwen2.5-1.5B-Instruct/..._singlehop_4/` |
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
