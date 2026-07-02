#!/bin/bash
#SBATCH --job-name=grpo_dscoder7b
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/grpo_dscoder7b_%j.out
#SBATCH --error=logs/grpo_dscoder7b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W1 second model: DeepSeek-Coder-7B GRPO (non-Qwen cross-architecture)
# Prereq: sft_dscoder7b.sh. Next: sbatch slurm/eval_dscoder7b_gen.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ABL_NAME="grpo_dscoder7b"
MODEL_NAME="deepseek-coder-7b_${ABL_NAME}_${TIMESTAMP}"
LATEST_LINK="./checkpoints/deepseek-coder-7b_grpo_latest"
WANDB_ID_FILE="./checkpoints/deepseek-coder-7b_grpo_wandb_run_id"

echo "GRPO DeepSeek-Coder-7B | Model: $MODEL_NAME | Job: $SLURM_JOB_ID | Start: $(date)"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv
export CUDA_VISIBLE_DEVICES=0
cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source ../config.env
source $HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv/bin/activate

pip uninstall -y xformers flash-attn
pip install --upgrade pip
pip install --no-deps 'unsloth-zoo==2026.1.4' 'unsloth==2026.1.4'
pip install -r requirements.txt

export SNIPER_ROOT=/project/6090549/srajput/green-code-gen/sniper/sniper
export PATH=$SNIPER_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SNIPER_ROOT/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1

INITIAL_MODEL="./checkpoints/deepseek-coder-7b_sft_latest/final"
[ ! -d "$INITIAL_MODEL" ] && INITIAL_MODEL=$(find ./checkpoints/deepseek-coder-7b_sft_latest -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
[ -z "$INITIAL_MODEL" ] && echo "ERROR: No DeepSeek-7B SFT checkpoint found" && exit 1
echo "Init from: $INITIAL_MODEL"

OUTPUT_DIR="./checkpoints/${MODEL_NAME}"
RESUME_FLAG=""
if [ -L "$LATEST_LINK" ] && [ -e "$LATEST_LINK" ]; then
    PREV=$(readlink -f "$LATEST_LINK")
    RESUME_CKPT=$(find $PREV -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    [ -z "$RESUME_CKPT" ] && [ -d "$PREV/final" ] && RESUME_CKPT="$PREV/final"
    if [ -n "$RESUME_CKPT" ]; then
        echo "Resuming from: $RESUME_CKPT"
        RESUME_FLAG="--resume-from-checkpoint $RESUME_CKPT"
        [ -f "$WANDB_ID_FILE" ] && export WANDB_RUN_ID=$(cat $WANDB_ID_FILE) && export WANDB_RESUME="allow"
    fi
fi
if [ -z "$WANDB_RUN_ID" ]; then
    export WANDB_RUN_ID="${ABL_NAME}-$(date +%Y%m%d%H%M%S)"
    export WANDB_RESUME="allow"
    echo $WANDB_RUN_ID > $WANDB_ID_FILE
fi
export WANDB_PROJECT="energy-code-generation"
export WANDB_NAME="$ABL_NAME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $OUTPUT_DIR logs data/new_models/dscoder7b/grpo_generations

python grpo_trainer_vllm.py \
    --model $INITIAL_MODEL \
    $RESUME_FLAG \
    --train-data data/grpo_train.jsonl \
    --sniper-root $SNIPER_ROOT \
    --sniper-config epyc_9554p \
    --reward-type edp \
    --num-generations 16 \
    --learning-rate 1e-6 \
    --beta 0.04 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-seq-length 4096 \
    --compile-timeout 30 \
    --simulation-timeout 60 \
    --output-dir $OUTPUT_DIR \
    --use-wandb \
    --model-name $MODEL_NAME \
    --eval-during-training \
    --pie-dataset ../PIE_Dataset \
    --generation-log data/new_models/dscoder7b/grpo_generations/grpo_generations_${WANDB_RUN_ID}.jsonl \
    2>&1 | tee logs/${MODEL_NAME}_${SLURM_JOB_ID}.log
PYTHON_EXIT=${PIPESTATUS[0]}

HAS_CKPT=$(find $OUTPUT_DIR -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
HAS_FINAL=$( [ -d "$OUTPUT_DIR/final" ] && echo 1 || echo 0 )
if [ "$HAS_CKPT" -gt 0 ] || [ "$HAS_FINAL" -eq 1 ]; then
    ln -sfn $(readlink -f $OUTPUT_DIR) $LATEST_LINK
    echo "Updated $LATEST_LINK"
fi
[ $PYTHON_EXIT -ne 0 ] && exit 1
echo "GRPO DeepSeek-7B complete. Next: sbatch slurm/eval_dscoder7b_gen.sh | End: $(date)"
