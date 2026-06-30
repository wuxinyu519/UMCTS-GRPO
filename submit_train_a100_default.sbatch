#!/bin/bash
#SBATCH --job-name=umcts_train_a100
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --output=logs/umcts_train_a100_%j.out
#SBATCH --error=logs/umcts_train_a100_%j.err

set -euo pipefail

export TRAIN_SCRIPT=train_singlehopqa_umcts.sh
export N_NODES=${N_NODES:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-6}
export N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-64}
export RETRIEVER_CUDA_VISIBLE_DEVICES=0,1
export TRAIN_CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash submit_train_8gpu.sbatch
