#!/bin/bash

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-${REPO_DIR}/data/models}
TRAIN_SUBMIT_SCRIPT=${TRAIN_SUBMIT_SCRIPT:-${REPO_DIR}/scripts/slurm/train_8gpu.sbatch}
RETRIEVER_SUBMIT_SCRIPT=${RETRIEVER_SUBMIT_SCRIPT:-${REPO_DIR}/scripts/slurm/retriever_2gpu.sbatch}
BACKEND=${BACKEND:-local}
SWEEP_TAG=${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}
STRICT_MODE=${STRICT_MODE:-0}
DRY_RUN=${DRY_RUN:-0}
SLEEP_BETWEEN_SUBMITS=${SLEEP_BETWEEN_SUBMITS:-1}
N_NODES=1
N_GPUS_PER_NODE=8
N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-64}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}
TRAIN_ENV=${TRAIN_ENV:-treegrpo}
RETRIEVER_ENV=${RETRIEVER_ENV:-retriever}
TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX:-${CONDA_PREFIX:-}}
PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_BIN=${RAY_BIN:-ray}
RETRIEVER_PORT=${RETRIEVER_PORT:-8000}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
INDEX_FILE=${INDEX_FILE:-${REPO_DIR}/data/wiki18/e5_Flat.index}
CORPUS_FILE=${CORPUS_FILE:-${REPO_DIR}/data/wiki18/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-${REPO_DIR}/data/models/e5-base-v2}
UMCTS_EMBEDDING_MODEL=${UMCTS_EMBEDDING_MODEL:-${REPO_DIR}/data/models/e5-base-v2}
RETRIEVER_PER_TRAIN_JOB=${RETRIEVER_PER_TRAIN_JOB:-1}

check_retriever_url() {
    local url="$1"
    curl -fsS --max-time 10 \
        -H "Content-Type: application/json" \
        -d '{"queries":["test"],"topk":1,"return_scores":true}' \
        "$url" >/dev/null 2>&1
}

wait_for_retriever() {
    local retriever_ready=0
    for attempt in $(seq 1 240); do
        if [[ -s run_info/retriever_url.txt ]]; then
            RETRIEVER_URL=$(cat run_info/retriever_url.txt)
            if check_retriever_url "$RETRIEVER_URL"; then
                retriever_ready=1
                export RETRIEVER_URL
                echo "Retriever ready at: ${RETRIEVER_URL}"
                break
            fi
        fi
        sleep 5
    done

    if [[ "$retriever_ready" -ne 1 ]]; then
        echo "ERROR: Retriever did not become ready within 20 minutes."
        return 1
    fi
}

ensure_retriever() {
    cd "$REPO_DIR"
    mkdir -p logs run_info

    if [[ "$BACKEND" == "slurm" && "$RETRIEVER_PER_TRAIN_JOB" == "1" ]]; then
        echo "Slurm backend: each training job will start its own local retriever on the same node."
        unset RETRIEVER_URL
        rm -f run_info/retriever_url.txt
        return
    fi

    if [[ -n "${RETRIEVER_URL:-}" ]]; then
        echo "Using RETRIEVER_URL from environment: ${RETRIEVER_URL}"
        return
    fi

    if [[ -s run_info/retriever_url.txt ]]; then
        RETRIEVER_URL=$(cat run_info/retriever_url.txt)
        if check_retriever_url "$RETRIEVER_URL"; then
            export RETRIEVER_URL
            echo "Using existing retriever: ${RETRIEVER_URL}"
            return
        fi
    fi

    rm -f run_info/retriever_url.txt
    if [[ "$BACKEND" == "slurm" ]]; then
        echo "Submitting shared retriever job..."
        local retriever_job
        retriever_job=$(sbatch --parsable --export="ALL,REPO_DIR=${REPO_DIR}" "$RETRIEVER_SUBMIT_SCRIPT")
        echo "Retriever job ID: ${retriever_job}"
        wait_for_retriever || {
            scancel "$retriever_job" 2>/dev/null || true
            exit 1
        }
        return
    fi

    echo "Starting local retriever on port ${RETRIEVER_PORT}..."
    RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}/retrieve"
    echo "$RETRIEVER_URL" > run_info/retriever_url.txt
    conda run --no-capture-output -n "$RETRIEVER_ENV" python search_r1/search/retrieval_server.py \
        --index_path "$INDEX_FILE" \
        --corpus_path "$CORPUS_FILE" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name "$RETRIEVER_NAME" \
        --retriever_model "$RETRIEVER_MODEL" \
        --faiss_gpu > "logs/retriever_local_${SWEEP_TAG}.log" 2>&1 &
    RETRIEVER_PID=$!
    export RETRIEVER_URL RETRIEVER_PID
    wait_for_retriever
}

ensure_ray_dashboard() {
    if [[ "$BACKEND" == "slurm" ]]; then
        return
    fi

    if curl -fsS "${RAY_DASHBOARD_ADDRESS}/api/version" >/dev/null 2>&1; then
        echo "Using existing Ray dashboard: ${RAY_DASHBOARD_ADDRESS}"
        return
    fi

    echo "Starting local Ray dashboard at ${RAY_DASHBOARD_ADDRESS}..."
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    "$RAY_BIN" start --head \
        --node-ip-address=127.0.0.1 \
        --port="$RAY_PORT" \
        --dashboard-host=127.0.0.1 \
        --dashboard-port="$RAY_DASHBOARD_PORT" \
        --num-gpus="$N_GPUS_PER_NODE" \
        --num-cpus="$N_CPUS_PER_NODE"

    local ray_ready=0
    for attempt in $(seq 1 60); do
        if curl -fsS "${RAY_DASHBOARD_ADDRESS}/api/version" >/dev/null 2>&1; then
            ray_ready=1
            echo "Ray dashboard is ready."
            break
        fi
        sleep 5
    done

    if [[ "$ray_ready" -ne 1 ]]; then
        echo "ERROR: Ray dashboard did not become ready at ${RAY_DASHBOARD_ADDRESS}."
        exit 1
    fi
}

submit_training_job() {
    local dataset="$1"
    local dataset_slug="$2"
    local train_script="$3"
    local data_dir="$4"
    local model_slug="$5"
    local model_dir="$6"
    local param_slug="$7"
    local tree_l="$8"
    local cu="$9"
    local local_w="${10}"
    local tau="${11}"
    local ucb_c="${12:-1.0}"

    local model_path="${MODEL_ROOT}/${model_dir}"
    if [[ ! -d "$model_path" ]]; then
        echo "Missing model directory: $model_path"
        if [[ "$STRICT_MODE" == "1" ]]; then
            exit 1
        fi
        echo "Skipping ${dataset_slug}/${model_slug}/${param_slug}."
        return
    fi

    if [[ ! -d "$data_dir" ]]; then
        echo "Missing data directory: $data_dir"
        if [[ "$STRICT_MODE" == "1" ]]; then
            exit 1
        fi
        echo "Skipping ${dataset_slug}/${model_slug}/${param_slug}."
        return
    fi

    local run_id="${model_slug}_${dataset_slug}_${param_slug}_${SWEEP_TAG}"
    local experiment_name="${dataset}-umcts-${model_slug}-${param_slug}"
    echo "Running ${dataset_slug}/${model_slug}/${param_slug}"

    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi

    if [[ "$BACKEND" == "slurm" ]]; then
        local export_vars="ALL,REPO_DIR=${REPO_DIR},RETRIEVER_URL=${RETRIEVER_URL:-},DATASET_NAME=${dataset},DATA_DIR=${data_dir},TRAIN_SCRIPT=${train_script},BASE_MODEL=${model_path},EXPERIMENT_NAME=${experiment_name},RUN_ID=${run_id},tree_search_l=${tree_l},UMCTS_UNCERTAINTY_COEF=${cu},UMCTS_LOCAL_ADVANTAGE_WEIGHT=${local_w},UMCTS_CONFIDENCE_TAU=${tau},UMCTS_UCB_C=${ucb_c}"
        sbatch --job-name="umcts_${dataset_slug}_${model_slug}_${param_slug}" --export="$export_vars" "$TRAIN_SUBMIT_SCRIPT"
        sleep "$SLEEP_BETWEEN_SUBMITS"
        return
    fi

    DATASET_NAME="$dataset" \
    DATA_DIR="$data_dir" \
    BASE_MODEL="$model_path" \
    EXPERIMENT_NAME="$experiment_name" \
    RUN_ID="$run_id" \
    RETRIEVER_URL="$RETRIEVER_URL" \
    UMCTS_EMBEDDING_MODEL="$UMCTS_EMBEDDING_MODEL" \
    RAY_DASHBOARD_ADDRESS="$RAY_DASHBOARD_ADDRESS" \
    N_NODES="$N_NODES" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
    PYTHON_BIN="$PYTHON_BIN" \
    tree_search_l="$tree_l" \
    UMCTS_UNCERTAINTY_COEF="$cu" \
    UMCTS_UCB_C="$ucb_c" \
    UMCTS_LOCAL_ADVANTAGE_WEIGHT="$local_w" \
    UMCTS_CONFIDENCE_TAU="$tau" \
    bash "$train_script"
}
