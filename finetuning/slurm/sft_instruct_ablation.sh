#!/bin/bash
#SBATCH --job-name=sft_instruct_ablation
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sft_instruct_ablation_%j.out
#SBATCH --error=logs/sft_instruct_ablation_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# SFT Ablation: Train from Instruct model (not base)
# Purpose: Isolate energy-contrastive training signal from instruction-following
# Reviewer W3: "Confounded baseline -- SFT trains from base model, baselines use instruct"
# Same hyperparameters as sft_optimization.sh, only model changes.

MODEL_FAMILY="qwen-coder-instruct"
MODEL_SIZE="14b"
MODEL_NAME="${MODEL_FAMILY}-${MODEL_SIZE}_sft_20260228_125105"
RESUME_CKPT="checkpoint-750"

export WANDB_PROJECT="energy-code-generation"
export WANDB_TAGS="${MODEL_FAMILY},${MODEL_SIZE},sft,h100,instruct-ablation,resumed"
export WANDB_NAME="${MODEL_NAME}_resumed"

echo "========================================="
echo "SFT INSTRUCT ABLATION"
echo "Model: $MODEL_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "========================================="

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

echo "Installing/Verifying dependencies..."
pip uninstall -y xformers flash-attn
pip install --upgrade pip
pip install --no-deps 'unsloth-zoo==2026.1.4'
pip install --no-deps 'unsloth==2026.1.4'
pip install -r requirements.txt

# Key difference from sft_optimization.sh: Instruct model, no resume from base checkpoint
BASE_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
OUTPUT_DIR="./checkpoints/${MODEL_NAME}"
DATA_PATH="data"

mkdir -p logs

echo ""
echo "Launching SFT Instruct Ablation Training (RESUMED from ${RESUME_CKPT})..."
echo "  Base model: $BASE_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Resuming: ${OUTPUT_DIR}/${RESUME_CKPT}"
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
    --resume-from-checkpoint ${OUTPUT_DIR}/${RESUME_CKPT} \
    2>&1 | tee logs/${MODEL_NAME}_sft_instruct_${SLURM_JOB_ID}.log
PYTHON_EXIT=${PIPESTATUS[0]}

if [ $PYTHON_EXIT -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "SFT INSTRUCT ABLATION COMPLETE"
    echo "========================================="
    ln -sfn $(readlink -f $OUTPUT_DIR) ./checkpoints/qwen-coder-instruct-14b_sft_latest
    echo "Updated ./checkpoints/qwen-coder-instruct-14b_sft_latest"
    echo ""
    echo "Next: Evaluate instruct SFT on 143 test problems"
    echo "  Compare compile%, tests%, ERR, CAERR against:"
    echo "    - SFT base (CAERR 4.45%)"
    echo "    - Zero-shot instruct (CAERR -2.92%)"
else
    echo ""
    echo "========================================="
    echo "SFT INSTRUCT ABLATION FAILED"
    echo "========================================="
    exit 1
fi
