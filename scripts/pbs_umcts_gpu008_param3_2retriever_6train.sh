#!/bin/bash
#PBS -N umcts_g008_p3
#PBS -o umcts_g008_p3.out
#PBS -e umcts_g008_p3.err
#PBS -l walltime=168:00:00
#PBS -q poderoso
#PBS -l select=1:ncpus=40:ngpus=8:host=gpu008

set -euo pipefail

cd /data/wux/tree_search/UMCTS-GRPO

export EXPERIMENT_NAME=singlehopqa-umcts-qwen2.5-1.5b-gpu008-param3-local2.0-2retriever-6train
export tree_search_m=${tree_search_m:-2}
export tree_search_n=${tree_search_n:-2}
export tree_search_l=${tree_search_l:-1}
export tree_search_k=${tree_search_k:-3}
export UMCTS_CANDIDATE_K=${UMCTS_CANDIDATE_K:-4}
export UMCTS_UNCERTAINTY_COEF=${UMCTS_UNCERTAINTY_COEF:-1.0}
export UMCTS_LOCAL_ADVANTAGE_WEIGHT=${UMCTS_LOCAL_ADVANTAGE_WEIGHT:-2.0}
export UMCTS_CLUSTER_THRESHOLD=${UMCTS_CLUSTER_THRESHOLD:-0.85}
export UMCTS_CONFIDENCE_TAU=${UMCTS_CONFIDENCE_TAU:-1.0}
export ACTOR_PPO_MICRO_BATCH_SIZE=${ACTOR_PPO_MICRO_BATCH_SIZE:-12}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-12}
export TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-5}

exec bash scripts/pbs_umcts_gpu008_2retriever_6train.sh
