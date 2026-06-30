#!/bin/bash
#SBATCH --job-name=umcts_train_p12
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --constraint="projects"
#SBATCH --output=logs/umcts_train_p12_%j.out
#SBATCH --error=logs/umcts_train_p12_%j.err

# Ablation: UMCTS_UCB_C=2.0 (baseline=1.0), all others default.

set -euo pipefail

export TRAIN_SCRIPT=train_singlehopqa_umcts.sh
export RUN_ID=${RUN_ID:-qwen25-15b_singlehop_p12_ucb2}
export tree_search_m=${tree_search_m:-2}
export tree_search_n=${tree_search_n:-2}
export tree_search_l=${tree_search_l:-1}
export tree_search_k=${tree_search_k:-3}
export UMCTS_CANDIDATE_K=${UMCTS_CANDIDATE_K:-4}
export UMCTS_UNCERTAINTY_COEF=${UMCTS_UNCERTAINTY_COEF:-1.0}
export UMCTS_UCB_C=${UMCTS_UCB_C:-2.0}
export UMCTS_LOCAL_ADVANTAGE_WEIGHT=${UMCTS_LOCAL_ADVANTAGE_WEIGHT:-1.0}
export UMCTS_CLUSTER_THRESHOLD=${UMCTS_CLUSTER_THRESHOLD:-0.85}
export UMCTS_CONFIDENCE_TAU=${UMCTS_CONFIDENCE_TAU:-1.0}
export N_NODES=${N_NODES:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-6}
export N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-64}

REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
cd "$REPO_DIR"
export RETRIEVER_CUDA_VISIBLE_DEVICES=0,1
export TRAIN_CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export N_GPUS_PER_NODE=6
exec bash submit_train_8gpu.sbatch
