#!/bin/bash
#SBATCH --job-name=umcts_train
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --output=logs/umcts_train_%j.out
#SBATCH --error=logs/umcts_train_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/projects/bhnl/xwu20/UMCTS-GRPO}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/data/singlehopqa}
BASE_MODEL=${BASE_MODEL:-${REPO_DIR}/data/models/Qwen2.5-1.5B}
UMCTS_EMBEDDING_MODEL=${UMCTS_EMBEDDING_MODEL:-${REPO_DIR}/data/models/e5-base-v2}
TRAIN_ENV=${TRAIN_ENV:-treegrpo}
TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX:-/u/xwu20/.conda/envs/${TRAIN_ENV}}
PYTHON_BIN=${PYTHON_BIN:-${TRAIN_ENV_PREFIX}/bin/python}
RAY_BIN=${RAY_BIN:-${TRAIN_ENV_PREFIX}/bin/ray}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_singlehopqa_umcts.sh}
RETRIEVER_ENV=${RETRIEVER_ENV:-retriever}
RETRIEVER_ENV_PREFIX=${RETRIEVER_ENV_PREFIX:-/u/xwu20/.conda/envs/${RETRIEVER_ENV}}
RETRIEVER_PYTHON_BIN=${RETRIEVER_PYTHON_BIN:-${RETRIEVER_ENV_PREFIX}/bin/python}
RETRIEVER_INDEX_FILE=${RETRIEVER_INDEX_FILE:-${REPO_DIR}/data/wiki18/e5_Flat.index}
RETRIEVER_CORPUS_FILE=${RETRIEVER_CORPUS_FILE:-${REPO_DIR}/data/wiki18/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-${REPO_DIR}/data/models/e5-base-v2}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
RETRIEVER_PORT=${RETRIEVER_PORT:-8000}
RETRIEVER_CUDA_VISIBLE_DEVICES=${RETRIEVER_CUDA_VISIBLE_DEVICES:-0,1}
TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}
N_NODES=1
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-6}
N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-${SLURM_CPUS_PER_TASK:-8}}

unset ${!CONDA_@} _CE_CONDA _CE_M
export PATH="${TRAIN_ENV_PREFIX}/bin:${PATH}"
export RAY_enable_open_telemetry=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${REPO_DIR}/swanlab_log}
export UMCTS_DUMP_METRICS=${UMCTS_DUMP_METRICS:-${REPO_DIR}/tree_metrics/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.jsonl}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}

cd "$REPO_DIR"
mkdir -p logs run_info tree_metrics

cleanup() {
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    if [[ -n "${RETRIEVER_PID:-}" ]]; then
        kill "$RETRIEVER_PID" >/dev/null 2>&1 || true
        wait "$RETRIEVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ -z "${RETRIEVER_URL:-}" ]]; then
    RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
    echo "$RETRIEVER_URL" > run_info/retriever_url.txt

    echo "Starting retriever on GPUs ${RETRIEVER_CUDA_VISIBLE_DEVICES}"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$RETRIEVER_CUDA_VISIBLE_DEVICES" "$RETRIEVER_PYTHON_BIN" search_r1/search/retrieval_server.py \
        --index_path "$RETRIEVER_INDEX_FILE" \
        --corpus_path "$RETRIEVER_CORPUS_FILE" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name "$RETRIEVER_NAME" \
        --retriever_model "$RETRIEVER_MODEL" \
        --faiss_gpu \
        > "logs/umcts_retriever_${SLURM_JOB_ID}.log" 2>&1 &
    RETRIEVER_PID=$!

    echo "Waiting for retriever at ${RETRIEVER_URL} ..."
    retriever_ready=0
    for attempt in $(seq 1 ${RETRIEVER_WAIT_ATTEMPTS:-360}); do
        if ! kill -0 "$RETRIEVER_PID" >/dev/null 2>&1; then
            echo "ERROR: Retriever process exited before becoming ready."
            tail -100 "logs/umcts_retriever_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        if curl -fsS -X POST "$RETRIEVER_URL" \
            -H "Content-Type: application/json" \
            -d '{"queries":["test query"],"topk":1,"return_scores":true}' >/dev/null 2>&1; then
            retriever_ready=1
            echo "Retriever is ready."
            break
        fi
        sleep 5
    done
    if [[ "$retriever_ready" -ne 1 ]]; then
        echo "ERROR: Retriever did not become ready at ${RETRIEVER_URL}."
        tail -100 "logs/umcts_retriever_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
else
    echo "$RETRIEVER_URL" > run_info/retriever_url.txt
fi
export DATA_DIR BASE_MODEL UMCTS_EMBEDDING_MODEL RETRIEVER_URL N_NODES N_GPUS_PER_NODE N_CPUS_PER_NODE PYTHON_BIN
export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"

echo "=== Training Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Retriever URL: ${RETRIEVER_URL}"
echo "Retriever GPUs: ${RETRIEVER_CUDA_VISIBLE_DEVICES}"
echo "Training GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Nodes: ${N_NODES}"
echo "GPUs per node: ${N_GPUS_PER_NODE}"
echo "CPUs per node exposed to Ray: ${N_CPUS_PER_NODE}"
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Ray: $("$RAY_BIN" --version)"
nvidia-smi

"$RAY_BIN" stop --force || true

RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-true}
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
ray_start_args=(
    --head
    --node-ip-address=127.0.0.1
    --port="$RAY_PORT"
    --include-dashboard="$RAY_INCLUDE_DASHBOARD"
    --num-gpus="$N_GPUS_PER_NODE"
    --num-cpus="$N_CPUS_PER_NODE"
)
if [[ "$RAY_INCLUDE_DASHBOARD" == "true" ]]; then
    ray_start_args+=(--dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT")
fi
"$RAY_BIN" start "${ray_start_args[@]}"
echo "Ray dashboard enabled: ${RAY_INCLUDE_DASHBOARD}"
if [[ "$RAY_INCLUDE_DASHBOARD" == "true" ]]; then
    echo "Ray dashboard URL: http://${SLURMD_NODENAME:-127.0.0.1}:${RAY_DASHBOARD_PORT}"
fi

echo "Waiting for Ray core at ${RAY_ADDRESS} ..."
ray_ready=0
for attempt in $(seq 1 60); do
    if "$RAY_BIN" status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
        ray_ready=1
        echo "Ray core is ready."
        break
    fi
    sleep 5
done

if [[ "$ray_ready" -ne 1 ]]; then
    echo "ERROR: Ray core did not become ready at ${RAY_ADDRESS}."
    if [[ "${N_NODES}" -gt 1 ]]; then
        echo "=== Ray logs from head node ${HEAD_NODE} ==="
        srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "ls -l /tmp/ray/session_latest/logs 2>/dev/null || true"
        srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "for f in /tmp/ray/session_latest/logs/gcs_server.err /tmp/ray/session_latest/logs/gcs_server.out /tmp/ray/session_latest/logs/raylet.err /tmp/ray/session_latest/logs/monitor.err; do echo === \$f ===; tail -100 \$f 2>/dev/null || true; done"
    else
        ls -l /tmp/ray/session_latest/logs 2>/dev/null || true
        for f in /tmp/ray/session_latest/logs/gcs_server.err /tmp/ray/session_latest/logs/gcs_server.out /tmp/ray/session_latest/logs/raylet.err /tmp/ray/session_latest/logs/monitor.err; do
            echo "=== $f ==="
            tail -100 "$f" 2>/dev/null || true
        done
    fi
    exit 1
fi

bash "$TRAIN_SCRIPT"
