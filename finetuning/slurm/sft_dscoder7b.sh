#!/bin/bash
#SBATCH --job-name=sft_dscoder7b
#SBATCH --account=rrg-mrdal22
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/sft_dscoder7b_%j.out
#SBATCH --error=logs/sft_dscoder7b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W1 second model: DeepSeek-Coder-7B-v1.5 energy-SFT (non-Qwen, cross-architecture generalizability)
# Next: sbatch slurm/grpo_dscoder7b.sh (after this completes)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="deepseek-coder-7b_sft_${TIMESTAMP}"
LATEST_LINK="./checkpoints/deepseek-coder-7b_sft_latest"

echo "SFT DeepSeek-Coder-7B | Model: $MODEL_NAME | Job: $SLURM_JOB_ID | Start: $(date)"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source ../config.env
source $HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv/bin/activate

pip uninstall -y xformers flash-attn
pip install --upgrade pip
pip install --no-deps 'unsloth-zoo==2026.1.4' 'unsloth==2026.1.4'
pip install -r requirements.txt

BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
OUTPUT_DIR="./checkpoints/${MODEL_NAME}"
mkdir -p $OUTPUT_DIR logs

export WANDB_PROJECT="energy-code-generation"
export WANDB_NAME="$MODEL_NAME"
export WANDB_TAGS="dscoder7b,sft,w1-second-model"

python sft_train_trl.py \
    --model $BASE_MODEL \
    --data-path data \
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
    2>&1 | tee logs/${MODEL_NAME}_${SLURM_JOB_ID}.log
PYTHON_EXIT=${PIPESTATUS[0]}

if [ $PYTHON_EXIT -eq 0 ]; then
    ln -sfn $(readlink -f $OUTPUT_DIR) $LATEST_LINK
    echo "Updated $LATEST_LINK -> $OUTPUT_DIR"
    echo "Next: sbatch slurm/grpo_dscoder7b.sh"
else
    echo "FAILED (exit=$PYTHON_EXIT)"
    exit 1
fi
echo "End: $(date)"
