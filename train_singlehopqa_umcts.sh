export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DATA_DIR=${DATA_DIR:-''}

export WG_BACKEND="ray"
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export BASE_DIR=$(pwd)
export RETRIEVER_URL=${RETRIEVER_URL:-"http://127.0.0.1:8000/retrieve"}

WAND_PROJECT="UMCTS-GRPO"
DATASET_NAME=${DATASET_NAME:-singlehopqa}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-"http://127.0.0.1:8265"} # your head node address
N_NODES=${N_NODES:-1}

n_gpus_per_node=${N_GPUS_PER_NODE:-8}
total_gpus=$((N_NODES * n_gpus_per_node))
train_batch_size=$((total_gpus * 64))
val_batch_size=$((total_gpus * 16))
actor_ppo_mini_batch_size=$((total_gpus * 8))
actor_ppo_micro_batch_size=${ACTOR_PPO_MICRO_BATCH_SIZE:-$((n_gpus_per_node * 4))}
log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE:-$((n_gpus_per_node * 4))}
trainer_save_freq=${TRAINER_SAVE_FREQ:-60}

tree_search_m=${tree_search_m:-2}
tree_search_n=${tree_search_n:-2}
tree_search_l=${tree_search_l:-1}
tree_search_k=${tree_search_k:-3}
lr=${lr:-1e-6}
UMCTS_CLUSTER_MODE=${UMCTS_CLUSTER_MODE:-embedding}
UMCTS_CLUSTER_THRESHOLD=${UMCTS_CLUSTER_THRESHOLD:-0.85}
UMCTS_UCB_C=${UMCTS_UCB_C:-1.0}
UMCTS_UNCERTAINTY_COEF=${UMCTS_UNCERTAINTY_COEF:-1.0}
UMCTS_VALUE_COEF=${UMCTS_VALUE_COEF:-1.0}
UMCTS_PRIOR_COEF=${UMCTS_PRIOR_COEF:-1.0}
UMCTS_COST_COEF=${UMCTS_COST_COEF:-0.0}
UMCTS_CANDIDATE_K=${UMCTS_CANDIDATE_K:-4}
UMCTS_CONFIDENCE_TAU=${UMCTS_CONFIDENCE_TAU:-1.0}
UMCTS_INTER_ADVANTAGE_WEIGHT=${UMCTS_INTER_ADVANTAGE_WEIGHT:-1.0}
UMCTS_LOCAL_ADVANTAGE_WEIGHT=${UMCTS_LOCAL_ADVANTAGE_WEIGHT:-1.0}
UMCTS_GAMMA=${UMCTS_GAMMA:-1.0}
UMCTS_EMBEDDING_MODEL=${UMCTS_EMBEDDING_MODEL:-}
if [[ "$UMCTS_CLUSTER_MODE" == "embedding" && -z "$UMCTS_EMBEDDING_MODEL" ]]; then
    echo "ERROR: UMCTS_CLUSTER_MODE=embedding requires UMCTS_EMBEDDING_MODEL=/path/to/local/embedding-model"
    exit 1
fi
export BASE_MODEL=${BASE_MODEL:-'./Qwen2.5-3B'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-singlehopqa-umcts-qwen2.5-3b}
MODEL_NAME=$(basename "$BASE_MODEL")
MODEL_NAME=${MODEL_NAME//[^A-Za-z0-9._-]/_}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)-${SLURM_JOB_ID:-local}}
RUN_NAME=${RUN_NAME:-${EXPERIMENT_NAME}-${RUN_ID}}
RESULT_ROOT=${RESULT_ROOT:-${BASE_DIR}/results}
RESULT_DIR=${RESULT_DIR:-${RESULT_ROOT}/${DATASET_NAME}/${MODEL_NAME}/${RUN_NAME}}

if [[ -e "$RESULT_DIR" ]]; then
    echo "ERROR: RESULT_DIR already exists: $RESULT_DIR"
    exit 1
fi
mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/config" "$RESULT_DIR/checkpoints" verl_log

{
    echo "run_name=$RUN_NAME"
    echo "run_id=$RUN_ID"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "slurm_node=${SLURMD_NODENAME:-}"
    echo "dataset_name=$DATASET_NAME"
    echo "data_dir=$DATA_DIR"
    echo "base_model=$BASE_MODEL"
    echo "retriever_url=$RETRIEVER_URL"
    echo "umcts_embedding_model=$UMCTS_EMBEDDING_MODEL"
    echo "lr=$lr"
    echo "tree_search_m=$tree_search_m"
    echo "tree_search_n=$tree_search_n"
    echo "tree_search_l=$tree_search_l"
    echo "tree_search_k=$tree_search_k"
    echo "umcts_cluster_mode=$UMCTS_CLUSTER_MODE"
    echo "umcts_cluster_threshold=$UMCTS_CLUSTER_THRESHOLD"
    echo "umcts_ucb_c=$UMCTS_UCB_C"
    echo "umcts_uncertainty_coef=$UMCTS_UNCERTAINTY_COEF"
    echo "umcts_value_coef=$UMCTS_VALUE_COEF"
    echo "umcts_prior_coef=$UMCTS_PRIOR_COEF"
    echo "umcts_cost_coef=$UMCTS_COST_COEF"
    echo "umcts_candidate_k=$UMCTS_CANDIDATE_K"
    echo "umcts_confidence_tau=$UMCTS_CONFIDENCE_TAU"
    echo "umcts_inter_advantage_weight=$UMCTS_INTER_ADVANTAGE_WEIGHT"
    echo "umcts_local_advantage_weight=$UMCTS_LOCAL_ADVANTAGE_WEIGHT"
    echo "umcts_gamma=$UMCTS_GAMMA"
    echo "result_dir=$RESULT_DIR"
} > "$RESULT_DIR/config/run.env"
cp "$0" "$RESULT_DIR/config/train_script.sh"
git rev-parse HEAD > "$RESULT_DIR/config/git_commit.txt" 2>/dev/null || true
git diff > "$RESULT_DIR/config/git_diff.patch" 2>/dev/null || true
echo "Run name: $RUN_NAME"
echo "Result dir: $RESULT_DIR"

ulimit -n 65535

if [[ -n "${RAY_ADDRESS:-}" ]]; then
    echo "Running training directly with RAY_ADDRESS=$RAY_ADDRESS"
    CMD_PREFIX=("${PYTHON_BIN:-python3}" -m verl.trainer.main_ppo_format_ts)
else
    CMD_PREFIX=(ray job submit --address="$RAY_DASHBOARD_ADDRESS" --runtime-env=verl/trainer/runtime_env.yaml -- python3 -m verl.trainer.main_ppo_format_ts)
fi

"${CMD_PREFIX[@]}" \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=umcts \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tree_search=true \
    actor_rollout_ref.rollout.ts_m=$tree_search_m \
    actor_rollout_ref.rollout.ts_n=$tree_search_n \
    actor_rollout_ref.rollout.ts_l=$tree_search_l \
    actor_rollout_ref.rollout.ts_k=$tree_search_k \
    actor_rollout_ref.rollout.reward_mode=base \
    actor_rollout_ref.rollout.expand_mode=umcts \
    actor_rollout_ref.rollout.umcts_ucb_c=$UMCTS_UCB_C \
    actor_rollout_ref.rollout.umcts_uncertainty_coef=$UMCTS_UNCERTAINTY_COEF \
    actor_rollout_ref.rollout.umcts_value_coef=$UMCTS_VALUE_COEF \
    actor_rollout_ref.rollout.umcts_prior_coef=$UMCTS_PRIOR_COEF \
    actor_rollout_ref.rollout.umcts_cost_coef=$UMCTS_COST_COEF \
    actor_rollout_ref.rollout.umcts_candidate_k=$UMCTS_CANDIDATE_K \
    actor_rollout_ref.rollout.umcts_cluster_mode=$UMCTS_CLUSTER_MODE \
    actor_rollout_ref.rollout.umcts_cluster_threshold=$UMCTS_CLUSTER_THRESHOLD \
    actor_rollout_ref.rollout.umcts_embedding_model_path=$UMCTS_EMBEDDING_MODEL \
    actor_rollout_ref.rollout.umcts_confidence_tau=$UMCTS_CONFIDENCE_TAU \
    actor_rollout_ref.rollout.umcts_inter_advantage_weight=$UMCTS_INTER_ADVANTAGE_WEIGHT \
    actor_rollout_ref.rollout.umcts_local_advantage_weight=$UMCTS_LOCAL_ADVANTAGE_WEIGHT \
    actor_rollout_ref.rollout.umcts_gamma=$UMCTS_GAMMA \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="['console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=$trainer_save_freq \
    trainer.test_freq=60 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=180 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$RESULT_DIR/checkpoints \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    2>&1 | tee "$RESULT_DIR/logs/verl.log"
