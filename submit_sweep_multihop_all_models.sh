#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/slurm/sweep_common.sh"
cd "$REPO_DIR"
ensure_retriever

models=(
  "qwen25-15b|Qwen2.5-1.5B-Instruct"
  "qwen25-3b|Qwen2.5-3B"
  "llama32-3b|Llama-3.2-3B"
  "qwen25-7b|Qwen2.5-7B"
  "qwen25-14b|Qwen2.5-14B"
)

for item in "${models[@]}"; do
    IFS='|' read -r model_slug model_dir <<< "$item"
    submit_training_job \
        "multihopqa" "multihop" "train_multihopqa_umcts.sh" "${REPO_DIR}/data/multihopqa" \
        "$model_slug" "$model_dir" "default" "1" "1.0" "1.0" "1.0"
done

echo "Multi-hop all-model sweep submitted. SWEEP_TAG=${SWEEP_TAG}"
