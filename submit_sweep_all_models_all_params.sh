#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/slurm/sweep_common.sh"
cd "$REPO_DIR"
ensure_retriever

models=(
  "qwen25-15b|Qwen2.5-1.5B"
  "qwen25-3b|Qwen2.5-3B"
  "llama32-3b|Llama-3.2-3B"
  "qwen25-7b|Qwen2.5-7B"
  "qwen25-14b|Qwen2.5-14B"
)

datasets=(
  "singlehopqa|singlehop|train_singlehopqa_umcts.sh|${REPO_DIR}/data/singlehopqa"
  "multihopqa|multihop|train_multihopqa_umcts.sh|${REPO_DIR}/data/multihopqa"
)

params=(
  "default|1|1.0|1.0|1.0|1.0"
  "cu0p5|1|0.5|1.0|1.0|1.0"
  "cu2p0|1|2.0|1.0|1.0|1.0"
  "local2p0|1|1.0|2.0|1.0|1.0"
  "tau5p0|1|1.0|1.0|5.0|1.0"
  "tau0p1|1|1.0|1.0|0.1|1.0"
  "param6_cu0p5_ucb0p5|1|0.5|1.0|1.0|0.5"
  "param7_cu0p1_ucb0p1_to60|1|0.1|1.0|1.0|0.1|1|-1|60"
)

for dataset_item in "${datasets[@]}"; do
    IFS='|' read -r dataset dataset_slug train_script data_dir <<< "$dataset_item"
    for model_item in "${models[@]}"; do
        IFS='|' read -r model_slug model_dir <<< "$model_item"
        for param_item in "${params[@]}"; do
            IFS='|' read -r param_slug tree_l cu local_w tau ucb_c save_freq test_freq total_steps <<< "$param_item"
            submit_training_job "$dataset" "$dataset_slug" "$train_script" "$data_dir" "$model_slug" "$model_dir" "$param_slug" "$tree_l" "$cu" "$local_w" "$tau" "${ucb_c:-1.0}" "${save_freq:-60}" "${test_freq:-60}" "${total_steps:-180}"
        done
    done
done

echo "All-model all-parameter sweep submitted. SWEEP_TAG=${SWEEP_TAG}"
