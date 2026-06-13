#!/bin/bash
#SBATCH --job-name=umcts_retriever
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gpus-per-node=2
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --output=logs/umcts_retriever_%j.out
#SBATCH --error=logs/umcts_retriever_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
INDEX_FILE=${INDEX_FILE:-${REPO_DIR}/data/wiki18/e5_Flat.index}
CORPUS_FILE=${CORPUS_FILE:-${REPO_DIR}/data/wiki18/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-${REPO_DIR}/data/models/e5-base-v2}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
RETRIEVER_PORT=${RETRIEVER_PORT:-8000}
RETRIEVER_ENV=${RETRIEVER_ENV:-retriever}

module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$REPO_DIR"
mkdir -p logs run_info
rm -f run_info/retriever_url.txt
trap 'rm -f "$REPO_DIR/run_info/retriever_url.txt"' ERR

HOSTNAME_FQDN=$(hostname -f 2>/dev/null || hostname)
RETRIEVER_URL="http://${HOSTNAME_FQDN}:${RETRIEVER_PORT}/retrieve"

echo "=== Retriever Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "URL: ${RETRIEVER_URL}"
conda run --no-capture-output -n "$RETRIEVER_ENV" python -c "import sys; print('Python executable:', sys.executable); import faiss; print('faiss:', faiss.__version__)"
nvidia-smi

echo "$RETRIEVER_URL" > run_info/retriever_url.txt
conda run --no-capture-output -n "$RETRIEVER_ENV" python search_r1/search/retrieval_server.py \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --topk "$RETRIEVER_TOPK" \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu
