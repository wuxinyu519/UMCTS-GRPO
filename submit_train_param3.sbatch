#!/bin/bash
#SBATCH --job-name=umcts_train_p3
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --output=/projects/bhnl/xwu20/UMCTS-GRPO/logs/umcts_train_p3_%j.out
#SBATCH --error=/projects/bhnl/xwu20/UMCTS-GRPO/logs/umcts_train_p3_%j.err

set -euo pipefail

export TRAIN_SCRIPT=train_singlehopqa_umcts.sh
export RUN_ID=${RUN_ID:-qwen25-15b_singlehop_3}
export tree_search_m=${tree_search_m:-2}
export tree_search_n=${tree_search_n:-2}
export tree_search_l=${tree_search_l:-1}
export tree_search_k=${tree_search_k:-3}
export UMCTS_CANDIDATE_K=${UMCTS_CANDIDATE_K:-4}
export UMCTS_UNCERTAINTY_COEF=${UMCTS_UNCERTAINTY_COEF:-2.0}
export UMCTS_LOCAL_ADVANTAGE_WEIGHT=${UMCTS_LOCAL_ADVANTAGE_WEIGHT:-1.0}
export UMCTS_CLUSTER_THRESHOLD=${UMCTS_CLUSTER_THRESHOLD:-0.85}
export UMCTS_CONFIDENCE_TAU=${UMCTS_CONFIDENCE_TAU:-1.0}
export N_NODES=${N_NODES:-4}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-16}

cd /projects/bhnl/xwu20/UMCTS-GRPO
exec bash submit_train_8gpu.sbatch
