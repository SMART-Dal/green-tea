#!/bin/bash
#SBATCH --job-name=grpo_vllm
#SBATCH --account=rrg-mrdal22
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/grpo_vllm_%j.out
#SBATCH --error=logs/grpo_vllm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Stage 3: GRPO Online RL (Multi-Objective Rewards)
# Goal: Learn online through direct Sniper simulation feedback
# Features:
#   - Reward: R = R_correct + tanh(energy_gain), epyc_9554p config
#   - Parallel Sniper simulation (48 CPUs, semaphore=16)

# ==============================================================================
# CONFIGURATION - Must match Stage 2 SFT Optimization model
# ==============================================================================
# User Configuration
MODEL_FAMILY="qwen-coder-base"
MODEL_SIZE="14b"

# Auto Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="${MODEL_FAMILY}-${MODEL_SIZE}_grpo_${TIMESTAMP}"

# W&B Configuration
export WANDB_PROJECT="energy-code-generation"
export WANDB_TAGS="${MODEL_FAMILY},${MODEL_SIZE},grpo,h100"
# WANDB_NAME and WANDB_RUN_ID set later (after checkpoint detection)
# ==============================================================================

echo "========================================="
echo "GRPO ONLINE RL TRAINING (vLLM)"
echo "Model: $MODEL_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "========================================="

# Setup environment
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0

# Navigate to project root
cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

# Install requirements (force standard torch to avoid CC incompatibilities)
echo "Installing/Verifying dependencies..."

# 1. Uninstall incompatible attention kernels (System xformers is broken for H100, binaries are ABI incompatible)
# Unsloth will fallback to PyTorch SDPA (Flash Attn via Torch) which works on H100
pip uninstall -y xformers flash-attn

# 2. Install Unsloth (Core only, no binary dependencies)
pip install --upgrade pip
pip install --no-deps 'unsloth-zoo==2026.1.4'
pip install --no-deps 'unsloth==2026.1.4'

# 3. Install other dependencies
pip install -r requirements.txt

# Sniper environment
export SNIPER_ROOT=/project/6090549/srajput/green-code-gen/sniper/sniper
export PATH=$SNIPER_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SNIPER_ROOT/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1  # Prevent Sniper OpenMP conflicts

SFT_LATEST="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning/checkpoints/qwen-coder-base-14b_sft_latest/final"
GRPO_LATEST="./checkpoints/qwen-coder-base-14b_grpo_latest"
GRPO_WANDB_ID_FILE="./checkpoints/qwen-coder-base-14b_grpo_wandb_run_id"

# Always load base from SFT (tokenizer + base weights never change)
INITIAL_MODEL="$SFT_LATEST"

# Resume logic: grpo_latest points to the previous run dir (mirrors sft_latest pattern)
# Inside that run dir: find latest checkpoint-N, or fall back to final/
RESUME_FLAG=""
if [ -L "$GRPO_LATEST" ] && [ -e "$GRPO_LATEST" ]; then
    PREV_RUN_DIR=$(readlink -f "$GRPO_LATEST")
    RESUME_CKPT=$(find $PREV_RUN_DIR -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    [ -z "$RESUME_CKPT" ] && [ -d "$PREV_RUN_DIR/final" ] && RESUME_CKPT="$PREV_RUN_DIR/final"
    if [ -n "$RESUME_CKPT" ]; then
        echo "Resuming GRPO from: $RESUME_CKPT"
        RESUME_FLAG="--resume-from-checkpoint $RESUME_CKPT"
        # Resume same W&B run for continuous metrics across jobs
        if [ -f "$GRPO_WANDB_ID_FILE" ]; then
            export WANDB_RUN_ID=$(cat $GRPO_WANDB_ID_FILE)
            export WANDB_RESUME="allow"
            echo "Resuming W&B run: $WANDB_RUN_ID"
        fi
    else
        echo "grpo_latest exists but no checkpoint found inside, starting fresh"
    fi
else
    echo "No GRPO checkpoint found, starting fresh from SFT"
fi

# On fresh start: create a new persistent W&B run ID
if [ -z "$WANDB_RUN_ID" ]; then
    export WANDB_RUN_ID="grpo-${MODEL_FAMILY}-${MODEL_SIZE}-$(date +%Y%m%d%H%M%S)"
    export WANDB_RESUME="allow"
    echo $WANDB_RUN_ID > $GRPO_WANDB_ID_FILE
fi
export WANDB_NAME="${MODEL_FAMILY}-${MODEL_SIZE}-grpo"

# Generation log: one file per training run (keyed by W&B run ID)
# All checkpoint-resumed jobs within this training run append to the same file
GENERATION_LOG="data/grpo_generations/grpo_generations_${WANDB_RUN_ID}.jsonl"

OUTPUT_DIR="./checkpoints/${MODEL_NAME}"
TRAIN_DATA="data/grpo_train.jsonl"
VAL_DATA=""  # No eval during GRPO: reward IS the signal; 2559 val evals would take 18+ hrs per eval step

# GRPO hyperparameters
NUM_GENERATIONS=16  # K=16 group size
LEARNING_RATE=1e-6
BETA=0.04
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=16  # effective batch = 2*16=32 prompts * 16 generations; halves generation KV cache vs batch=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Timeout configuration (conservative based on evaluation data)
COMPILE_TIMEOUT=30
SIMULATION_TIMEOUT=60  # 60s covers >95% of programs; 300s was causing tail latency per step

mkdir -p $OUTPUT_DIR logs

echo ""
echo "Configuration:"
echo "  Base model: $INITIAL_MODEL"
echo "  Resume: ${RESUME_CKPT:-none}"
echo "  Output: $OUTPUT_DIR"
echo "  Group size (K): $NUM_GENERATIONS"
echo "  Timeouts: compile=${COMPILE_TIMEOUT}s, simulation=${SIMULATION_TIMEOUT}s"
echo ""

python grpo_trainer_vllm.py \
    --model $INITIAL_MODEL \
    $RESUME_FLAG \
    --train-data $TRAIN_DATA \
    ${VAL_DATA:+--val-data "$VAL_DATA"} \
    --sniper-root $SNIPER_ROOT \
    --sniper-config epyc_9554p \
    --num-generations $NUM_GENERATIONS \
    --learning-rate $LEARNING_RATE \
    --beta $BETA \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --max-seq-length 4096 \
    --compile-timeout $COMPILE_TIMEOUT \
    --simulation-timeout $SIMULATION_TIMEOUT \
    --output-dir $OUTPUT_DIR \
    --use-wandb \
    --model-name $MODEL_NAME \
    --eval-during-training \
    --pie-dataset ../PIE_Dataset \
    --generation-log $GENERATION_LOG \
    2>&1 | tee logs/${MODEL_NAME}_grpo_vllm_${SLURM_JOB_ID}.log
PYTHON_EXIT=${PIPESTATUS[0]}

# Always update grpo_latest to point to this run dir (mirrors how sft_latest works)
GRPO_LATEST_LINK="./checkpoints/qwen-coder-base-14b_grpo_latest"
HAS_CKPT=$(find $OUTPUT_DIR -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
HAS_FINAL=$( [ -d "$OUTPUT_DIR/final" ] && echo 1 || echo 0 )

if [ "$HAS_CKPT" -gt 0 ] || [ "$HAS_FINAL" -eq 1 ]; then
    ln -sfn $(readlink -f $OUTPUT_DIR) $GRPO_LATEST_LINK
    echo "Updated $GRPO_LATEST_LINK -> $OUTPUT_DIR"
    echo "To continue: sbatch slurm/grpo_vllm_train.sh"
else
    echo "No checkpoints or final model saved this job, grpo_latest not updated."
fi

[ $PYTHON_EXIT -ne 0 ] && exit 1
echo "GRPO training complete."
