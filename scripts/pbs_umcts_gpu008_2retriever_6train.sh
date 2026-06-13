#!/bin/bash
#PBS -N umcts_g008_2r6t
#PBS -o umcts_g008_2r6t.out
#PBS -e umcts_g008_2r6t.err
#PBS -l walltime=168:00:00
#PBS -q poderoso
#PBS -l select=1:ncpus=40:ngpus=8:host=gpu008

set -euo pipefail

BASE_DIR=/data/wux/tree_search/UMCTS-GRPO
DATA_DIR=$BASE_DIR/data/singlehopqa
INDEX_FILE=$BASE_DIR/data/wiki-18/e5_Flat.index
CORPUS_FILE=$BASE_DIR/data/wiki-18/wiki-18.jsonl
BASE_MODEL=/data/wux/huggingface/transformers/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
UMCTS_EMBEDDING_MODEL=/data/wux/huggingface/transformers/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd

RETRIEVER_GPUS=0,1
TRAIN_GPUS=2,3,4,5,6,7
RETRIEVER_PORT=8000
RETRIEVER_URL=http://127.0.0.1:${RETRIEVER_PORT}/retrieve

RETRIEVER_ENV=retriever_faiss124
TRAIN_ENV=treegrpo
TRAIN_ENV_PREFIX=/data/wux/anaconda3/envs/${TRAIN_ENV}
PYTHON_BIN=${TRAIN_ENV_PREFIX}/bin/python
RAY_BIN=${TRAIN_ENV_PREFIX}/bin/ray

N_NODES=1
N_GPUS_PER_NODE=6
N_CPUS_PER_NODE=32
EXPERIMENT_NAME=${EXPERIMENT_NAME:-singlehopqa-umcts-qwen2.5-1.5b-gpu008-2retriever-6train}
RUN_ID=$(date +%Y%m%d_%H%M%S)-pbs${PBS_JOBID%%.*}

cd "$BASE_DIR"
mkdir -p logs run_info results verl_log
exec > >(tee -a "$BASE_DIR/logs/${EXPERIMENT_NAME}_${RUN_ID}.pbs.log") 2>&1

source /data/wux/anaconda3/etc/profile.d/conda.sh

export HF_HOME=/data/wux/huggingface
export TRANSFORMERS_CACHE=/data/wux/huggingface/transformers
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gcs_rpc_server_reconnect_timeout_s=100
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0
export RAY_USAGE_STATS_ENABLED=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN

echo "=== PBS / Node Info ==="
echo "PBS_JOBID=${PBS_JOBID:-}"
echo "HOST=$(hostname)"
echo "BASE_DIR=$BASE_DIR"
echo "RETRIEVER_GPUS=$RETRIEVER_GPUS"
echo "TRAIN_GPUS=$TRAIN_GPUS"
nvidia-smi

for required_path in "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" "$INDEX_FILE" "$CORPUS_FILE" "$BASE_MODEL" "$UMCTS_EMBEDDING_MODEL"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path"
    exit 1
  fi
done

cleanup() {
  set +e
  echo "Cleaning up background services..."
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  if [[ -n "${RETRIEVER_PID:-}" ]]; then
    kill "$RETRIEVER_PID" >/dev/null 2>&1 || true
    wait "$RETRIEVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "=== Starting retriever on GPUs ${RETRIEVER_GPUS} ==="
(
  source /data/wux/anaconda3/etc/profile.d/conda.sh
  conda activate "$RETRIEVER_ENV"
  export CUDA_VISIBLE_DEVICES=$RETRIEVER_GPUS
  export HF_HOME=/data/wux/huggingface
  export TRANSFORMERS_CACHE=/data/wux/huggingface/transformers
  python search_r1/search/retrieval_server.py \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "$UMCTS_EMBEDDING_MODEL" \
    --faiss_gpu
) > "$BASE_DIR/logs/${EXPERIMENT_NAME}_${RUN_ID}.retriever.log" 2>&1 &
RETRIEVER_PID=$!
echo "Retriever PID: $RETRIEVER_PID"

echo "=== Waiting for retriever at ${RETRIEVER_URL} ==="
conda activate "$TRAIN_ENV"
python - <<PY
import requests
import sys
import time

url = "${RETRIEVER_URL}"
payload = {
    "queries": ["who got the first nobel prize in physics"],
    "topk": 3,
    "return_scores": True,
}

last_error = None
for _ in range(240):
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("retriever reachable")
        print(str(data)[:500])
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        print("waiting for retriever:", repr(last_error), flush=True)
        time.sleep(30)

print("retriever not reachable:", repr(last_error))
sys.exit(1)
PY

echo "=== Starting Ray for training on GPUs ${TRAIN_GPUS} ==="
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
"$RAY_BIN" stop --force || true
"$RAY_BIN" start --head \
  --node-ip-address=127.0.0.1 \
  --port=6379 \
  --include-dashboard=false \
  --num-gpus=$N_GPUS_PER_NODE \
  --num-cpus=$N_CPUS_PER_NODE

export RAY_ADDRESS=127.0.0.1:6379
export PYTHON_BIN=$PYTHON_BIN
export DATA_DIR=$DATA_DIR
export BASE_MODEL=$BASE_MODEL
export UMCTS_EMBEDDING_MODEL=$UMCTS_EMBEDDING_MODEL
export RETRIEVER_URL=$RETRIEVER_URL
export N_NODES=$N_NODES
export N_GPUS_PER_NODE=$N_GPUS_PER_NODE
export N_CPUS_PER_NODE=$N_CPUS_PER_NODE
export EXPERIMENT_NAME=$EXPERIMENT_NAME
export RUN_ID=$RUN_ID
export RUN_NAME=${EXPERIMENT_NAME}-${RUN_ID}
export RESULT_ROOT=$BASE_DIR/results
export SWANLAB_MODE=local
export UMCTS_CLUSTER_MODE=embedding
export UMCTS_CLUSTER_THRESHOLD=0.85
export UMCTS_CANDIDATE_K=4
export ACTOR_PPO_MICRO_BATCH_SIZE=${ACTOR_PPO_MICRO_BATCH_SIZE:-12}
export LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-12}
export TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-5}

echo "=== Training config ==="
echo "RUN_NAME=$RUN_NAME"
echo "DATA_DIR=$DATA_DIR"
echo "BASE_MODEL=$BASE_MODEL"
echo "UMCTS_EMBEDDING_MODEL=$UMCTS_EMBEDDING_MODEL"
echo "RETRIEVER_URL=$RETRIEVER_URL"
echo "N_GPUS_PER_NODE=$N_GPUS_PER_NODE"
echo "ACTOR_PPO_MICRO_BATCH_SIZE=$ACTOR_PPO_MICRO_BATCH_SIZE"
echo "LOG_PROB_MICRO_BATCH_SIZE=$LOG_PROB_MICRO_BATCH_SIZE"
echo "TRAINER_SAVE_FREQ=$TRAINER_SAVE_FREQ"
nvidia-smi

bash train_singlehopqa_umcts.sh
