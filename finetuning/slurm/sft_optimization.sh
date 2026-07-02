#!/bin/bash
#SBATCH --job-name=sft_opt_def-tusharma
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sft_opt_32hrs_%j.out
#SBATCH --error=logs/sft_opt_32hrs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca


# Stage 2: SFT Optimization Training (Energy Focus)
# Goal: Learn energy-efficient transformations via contrastive learning

# ==============================================================================
# CONFIGURATION - Must match Stage 1 model
# ==============================================================================
# User Configuration
MODEL_FAMILY="qwen-coder-base"
MODEL_SIZE="14b"

# Auto Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="${MODEL_FAMILY}-${MODEL_SIZE}_sft_${TIMESTAMP}"

# W&B Configuration
export WANDB_PROJECT="energy-code-generation"
export WANDB_TAGS="${MODEL_FAMILY},${MODEL_SIZE},sft,h100"
export WANDB_NAME="${MODEL_NAME}"
# ==============================================================================

echo "========================================="
echo "SFT STAGE 2: OPTIMIZATION"
echo "Model: $MODEL_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "========================================="

# Setup environment
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# Model paths
BASE_MODEL="Qwen/Qwen2.5-Coder-14B"
# BASE_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"    #replacing instruct with base model, to improve the generation performance of the model
OUTPUT_DIR="./checkpoints/${MODEL_NAME}"
DATA_PATH="data"

mkdir -p $OUTPUT_DIR logs

echo ""
echo "Launching Energy Optimization SFT Training..."
echo "  Base model: $BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo ""

python sft_train_trl.py \
    --model $BASE_MODEL \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 4096 \
    --lora-r 64 \
    --lora-alpha 128 \
    --use-wandb \
    --model-name $MODEL_NAME \
    --eval-steps 150 \
    --save-steps 150 \
    --resume-from-checkpoint "checkpoints/qwen-coder-base-14b_sft_20260207_224437/checkpoint-150" \
    # Uncomment below to enable energy evaluation during training:
    # --eval-during-training \
    # --sniper-root "$SNIPER_ROOT" \
    # --eval-every-n-steps 150 \
    # --max-eval-samples 50 \
    # --eval-output-dir "data/sft_generations_base" \
    2>&1 | tee logs/${MODEL_NAME}_sft_opt_${SLURM_JOB_ID}.log
PYTHON_EXIT=${PIPESTATUS[0]}

if [ $PYTHON_EXIT -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "SFT OPTIMIZATION COMPLETE"
    echo "========================================="
    
    # Create symlink to latest for GRPO
    ln -sfn $(readlink -f $OUTPUT_DIR) ./checkpoints/qwen-coder-base-14b_sft_latest
    echo "Updated ./checkpoints/qwen-coder-base-14b_sft_latest to point to this run."
    
    echo "Next: Run GRPO online RL training"
    echo "  sbatch slurm/grpo_vllm_train.sh"
else
    echo ""
    echo "========================================="
    echo "SFT OPTIMIZATION FAILED"
    echo "========================================="
    exit 1
fi
