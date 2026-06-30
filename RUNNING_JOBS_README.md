# Current UMCTS-GRPO Runs

This file records the training jobs currently launched from this local checkout.

## Current Queue Snapshot

Last updated from `squeue -u xwu20` on 2026-06-16.

| Job ID | Partition | Job name | Status | Task | Dataset | Model | Parameter setting | Submit script / repo | Expected output |
|---|---|---|---|---|---|---|---|---|---|
| `19225360` | `gpuA100x8` | `umcts_train` | `PD (Priority)` | UMCTS-GRPO training | `singlehopqa` | `Qwen2.5-1.5B` | default | `/projects/bhnl/xwu20/UMCTS-GRPO/submit_train_8gpu.sbatch` | `results/singlehopqa/Qwen2.5-1.5B/...` |
| `<PARAM4_JOBID>` | `gpuA100x8` | `umcts_p4_a100` | `PD (after submit)` | UMCTS-GRPO training | `singlehopqa` | `Qwen2.5-1.5B` | `param4`: `tau=5.0` | `/projects/bhnl/xwu20/UMCTS-GRPO/submit_train_param4.sbatch` | `results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_param4_tau5p0_a100_run1` |
| `19296536` | `gpuA100x8` | `umcts_p2_a10` | `PD (Priority)` | UMCTS-GRPO training | `singlehopqa` | `Qwen2.5-1.5B` | `param2`: `uncertainty_coef=2.0` | `/projects/bhnl/xwu20/UMCTS-GRPO/submit_train_param2.sbatch` | `results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_param2_cu2p0_a100_run1` |
| `19294991` | `gpuH200x8` | `umcts_defaul` | `PD (Priority)` | UMCTS-GRPO training | `singlehopqa` | `Qwen2.5-1.5B` | default | `/projects/bhnl/xwu20/UMCTS-GRPO/submit_train_8gpu.sbatch` | `results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_umcts_default_h200_run1` |
| `19295053` | `gpuH200x8` | `treegrpo_q15` | `PD (Priority)` | official Tree-GRPO baseline | `singlehopqa` | `Qwen2.5-1.5B` | Tree-GRPO default, `expand_mode=random` | `/projects/bhnl/xwu20/Tree-GRPO/submit_official_treegrpo_qwen15_singlehop.sbatch` | `/projects/bhnl/xwu20/Tree-GRPO/results/singlehopqa/Qwen2.5-1.5B/singlehopqa-treesearch-qwen25-15b-official-qwen25-15b_singlehop_treegrpo_h200_run1` |

All pending jobs currently show `Reason=Priority`, so they are waiting for scheduling priority/resources rather than failing due to script errors.

## Default SingleHop Run

- Submit script: `submit_train_8gpu.sbatch`
- Slurm job id: `19202253`
- Node/GPU: `gpue05`, 8 x H200
- Dataset: `singlehopqa`
- Data directory: `/projects/bhnl/xwu20/UMCTS-GRPO/data/singlehopqa`
- Model: `Qwen2.5-1.5B`
- Model path: `/projects/bhnl/xwu20/UMCTS-GRPO/data/models/Qwen2.5-1.5B`
- Retriever URL: `http://gpub008.delta.ncsa.illinois.edu:8000/retrieve`
- Parameter setting: default
- Total training steps: `180`
- Last observed progress: around `step 44 / 180`
- Slurm log:
  - `logs/umcts_train_19202253.out`
  - `logs/umcts_train_19202253.err`
- Result directory:

```text
/projects/bhnl/xwu20/UMCTS-GRPO/results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-20260614_051339-19202253
```

Key UMCTS parameters:

```text
tree_search_m=2
tree_search_n=2
tree_search_l=1
tree_search_k=3
umcts_cluster_mode=embedding
umcts_cluster_threshold=0.85
umcts_uncertainty_coef=1.0
umcts_confidence_tau=1.0
umcts_local_advantage_weight=1.0
```

## Param1 SingleHop Run

- Submit script: `submit_train_param1.sbatch`
- Slurm job id: `19204678`
- Node/GPU: `gpuc06`
- Dataset: `singlehopqa`
- Data directory: `/projects/bhnl/xwu20/UMCTS-GRPO/data/singlehopqa`
- Model: `Qwen2.5-1.5B`
- Model path: `/projects/bhnl/xwu20/UMCTS-GRPO/data/models/Qwen2.5-1.5B`
- Retriever URL: `http://gpub008.delta.ncsa.illinois.edu:8000/retrieve`
- Parameter setting: `param1`
- Total training steps: `180`
- Last observed progress: around `step 38 / 180`
- Slurm log:
  - `logs/umcts_train_p1_19204678.out`
  - `logs/umcts_train_p1_19204678.err`
- Result directory:

```text
/projects/bhnl/xwu20/UMCTS-GRPO/results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_1
```

Key UMCTS parameters:

```text
tree_search_m=2
tree_search_n=2
tree_search_l=1
tree_search_k=3
umcts_cluster_mode=embedding
umcts_cluster_threshold=0.85
umcts_uncertainty_coef=0.5
umcts_confidence_tau=1.0
umcts_local_advantage_weight=1.0
```

## Pending A100x8 Param2 And Param4 Submissions

These are the next normal UMCTS-GRPO SingleHop jobs to submit on Delta A100x8.
Both use the current one-node launch layout:

```text
partition=gpuA100x8
node count=1
allocated GPUs=8
retriever GPUs=0,1
training GPUs=2,3,4,5,6,7
trainer.n_gpus_per_node=6
dataset=singlehopqa
model=Qwen2.5-1.5B
```

Param2 increases uncertainty exploration:

```text
submit script=submit_train_param2.sbatch
umcts_uncertainty_coef=2.0
umcts_confidence_tau=1.0
umcts_local_advantage_weight=1.0
```

Submit command:

```bash
cd /projects/bhnl/xwu20/UMCTS-GRPO

sbatch \
  --partition=gpuA100x8 \
  --job-name=umcts_p2_a100 \
  --output=logs/umcts_train_p2_a100_%j.out \
  --error=logs/umcts_train_p2_a100_%j.err \
  --export=ALL,RUN_ID=qwen25-15b_singlehop_param2_cu2p0_a100_run1 \
  submit_train_param2.sbatch
```

After submission, logs will be:

```text
logs/umcts_train_p2_a100_<JOBID>.out
logs/umcts_train_p2_a100_<JOBID>.err
```

Expected result directory:

```text
/projects/bhnl/xwu20/UMCTS-GRPO/results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_param2_cu2p0_a100_run1
```

Param4 strengthens confidence gating:

```text
submit script=submit_train_param4.sbatch
umcts_uncertainty_coef=1.0
umcts_confidence_tau=5.0
umcts_local_advantage_weight=1.0
```

Submit command:

```bash
cd /projects/bhnl/xwu20/UMCTS-GRPO

sbatch \
  --partition=gpuA100x8 \
  --job-name=umcts_p4_a100 \
  --output=logs/umcts_train_p4_a100_%j.out \
  --error=logs/umcts_train_p4_a100_%j.err \
  --export=ALL,RUN_ID=qwen25-15b_singlehop_param4_tau5p0_a100_run1 \
  submit_train_param4.sbatch
```

After submission, logs will be:

```text
logs/umcts_train_p4_a100_<JOBID>.out
logs/umcts_train_p4_a100_<JOBID>.err
```

Expected result directory:

```text
/projects/bhnl/xwu20/UMCTS-GRPO/results/singlehopqa/Qwen2.5-1.5B/singlehopqa-umcts-qwen2.5-3b-qwen25-15b_singlehop_param4_tau5p0_a100_run1
```
