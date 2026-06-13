#!/bin/bash
#SBATCH --job-name=umcts_train_p5
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --output=logs/umcts_train_p5_%j.out
#SBATCH --error=logs/umcts_train_p5_%j.err

set -euo pipefail

export TRAIN_SCRIPT=train_singlehopqa_umcts.sh
export RUN_ID=${RUN_ID:-qwen25-15b_singlehop_5}
export tree_search_m=${tree_search_m:-2}
export tree_search_n=${tree_search_n:-2}
export tree_search_l=${tree_search_l:-1}
export tree_search_k=${tree_search_k:-3}
export UMCTS_CANDIDATE_K=${UMCTS_CANDIDATE_K:-4}
export UMCTS_UNCERTAINTY_COEF=${UMCTS_UNCERTAINTY_COEF:-1.0}
export UMCTS_LOCAL_ADVANTAGE_WEIGHT=${UMCTS_LOCAL_ADVANTAGE_WEIGHT:-0}
export UMCTS_CLUSTER_THRESHOLD=${UMCTS_CLUSTER_THRESHOLD:-0.85}
export UMCTS_CONFIDENCE_TAU=${UMCTS_CONFIDENCE_TAU:-1.0}
export N_NODES=${N_NODES:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-64}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$REPO_DIR"
exec bash submit_train_8gpu.sbatch
