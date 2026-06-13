#!/bin/bash
#SBATCH --job-name=umcts_train
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --account=bhnl-delta-gpu
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --output=/projects/bhnl/xwu20/UMCTS-GRPO/logs/umcts_train_%j.out
#SBATCH --error=/projects/bhnl/xwu20/UMCTS-GRPO/logs/umcts_train_%j.err

set -euo pipefail

DATA_DIR=${DATA_DIR:-/projects/bhnl/xwu20/UMCTS-GRPO/data/singlehopqa}
BASE_MODEL=${BASE_MODEL:-/projects/bhnl/xwu20/UMCTS-GRPO/data/models/Qwen2.5-1.5B-Instruct}
UMCTS_EMBEDDING_MODEL=${UMCTS_EMBEDDING_MODEL:-/projects/bhnl/xwu20/UMCTS-GRPO/data/models/e5-base-v2}
TRAIN_ENV=${TRAIN_ENV:-treegrpo}
TRAIN_ENV_PREFIX=${TRAIN_ENV_PREFIX:-/u/xwu20/.conda/envs/${TRAIN_ENV}}
PYTHON_BIN=${PYTHON_BIN:-${TRAIN_ENV_PREFIX}/bin/python}
RAY_BIN=${RAY_BIN:-${TRAIN_ENV_PREFIX}/bin/ray}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_singlehopqa_umcts.sh}
RETRIEVER_WAIT_SECONDS=${RETRIEVER_WAIT_SECONDS:-7200}
N_NODES=${N_NODES:-${SLURM_JOB_NUM_NODES:-1}}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
N_CPUS_PER_NODE=${N_CPUS_PER_NODE:-${SLURM_CPUS_PER_TASK:-8}}

unset ${!CONDA_@} _CE_CONDA _CE_M
export PATH="${TRAIN_ENV_PREFIX}/bin:${PATH}"
export RAY_enable_open_telemetry=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/projects/bhnl/xwu20/UMCTS-GRPO/swanlab_log}

cd /projects/bhnl/xwu20/UMCTS-GRPO
mkdir -p logs

if [[ -z "${RETRIEVER_URL:-}" ]]; then
    echo "Waiting up to ${RETRIEVER_WAIT_SECONDS}s for run_info/retriever_url.txt ..."
    waited=0
    while [[ ! -s run_info/retriever_url.txt && "$waited" -lt "$RETRIEVER_WAIT_SECONDS" ]]; do
        sleep 10
        waited=$((waited + 10))
    done
    if [[ ! -s run_info/retriever_url.txt ]]; then
        echo "ERROR: Retriever URL was not created within ${RETRIEVER_WAIT_SECONDS}s."
        exit 1
    fi
    RETRIEVER_URL=$(cat run_info/retriever_url.txt)
fi
export DATA_DIR BASE_MODEL UMCTS_EMBEDDING_MODEL RETRIEVER_URL N_NODES N_GPUS_PER_NODE N_CPUS_PER_NODE PYTHON_BIN

echo "=== Training Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Retriever URL: ${RETRIEVER_URL}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Nodes: ${N_NODES}"
echo "GPUs per node: ${N_GPUS_PER_NODE}"
echo "CPUs per node exposed to Ray: ${N_CPUS_PER_NODE}"
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Ray: $("$RAY_BIN" --version)"
nvidia-smi

"$RAY_BIN" stop --force || true

if [[ "$N_NODES" -gt 1 ]]; then
    mapfile -t RAY_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    HEAD_NODE=${RAY_NODES[0]}
    HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
    RAY_PORT=${RAY_PORT:-6379}
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

    echo "Ray head node: ${HEAD_NODE}"
    echo "Ray head IP: ${HEAD_IP}"
    echo "Ray address: ${RAY_ADDRESS}"

    srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
        unset \${!CONDA_@} _CE_CONDA _CE_M
        export PATH=\"${TRAIN_ENV_PREFIX}/bin:\${PATH}\"
        export RAY_enable_open_telemetry=0
        export RAY_USAGE_STATS_ENABLED=0
        export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s}
        export SWANLAB_MODE=${SWANLAB_MODE}
        export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR}
        cd /projects/bhnl/xwu20/UMCTS-GRPO
        \"${RAY_BIN}\" stop --force || true
        \"${RAY_BIN}\" start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --include-dashboard=false --num-gpus=${N_GPUS_PER_NODE} --num-cpus=${N_CPUS_PER_NODE} --block
    " &
    RAY_HEAD_PID=$!
    sleep 20

    for WORKER_NODE in "${RAY_NODES[@]:1}"; do
        echo "Starting Ray worker on ${WORKER_NODE}"
        srun --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -lc "
            unset \${!CONDA_@} _CE_CONDA _CE_M
            export PATH=\"${TRAIN_ENV_PREFIX}/bin:\${PATH}\"
            export RAY_enable_open_telemetry=0
            export RAY_USAGE_STATS_ENABLED=0
            export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s}
            export SWANLAB_MODE=${SWANLAB_MODE}
            export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR}
            cd /projects/bhnl/xwu20/UMCTS-GRPO
            \"${RAY_BIN}\" stop --force || true
            \"${RAY_BIN}\" start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=${N_GPUS_PER_NODE} --num-cpus=${N_CPUS_PER_NODE} --block
        " &
    done
else
    RAY_PORT=${RAY_PORT:-6379}
    export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
    "$RAY_BIN" start --head --node-ip-address=127.0.0.1 --port="$RAY_PORT" --include-dashboard=false --num-gpus="$N_GPUS_PER_NODE" --num-cpus="$N_CPUS_PER_NODE"
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

"$RAY_BIN" stop --force || true
