#!/bin/bash
#SBATCH --job-name=sft_runtime
#SBATCH --account=rrg-mrdal22
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sft_runtime_%j.out
#SBATCH --error=logs/sft_runtime_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W1 ablation: SFT on runtime-contrastive pairs (same architecture, same HPO, runtime prompt)
# Prereq: python3 finetuning/build_runtime_pairs.py  (creates data/runtime/)
# Next: sbatch slurm/eval_sft_runtime_gen.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME="qwen-coder-base-14b_sft_runtime_${TIMESTAMP}"

export WANDB_PROJECT="energy-code-generation"
export WANDB_TAGS="qwen-coder-base,14b,sft,runtime,ablation,w1"
export WANDB_NAME="${MODEL_NAME}"

echo "========================================="
echo "W1 ABLATION: Runtime-SFT Training"
echo "Model: $MODEL_NAME  Job: $SLURM_JOB_ID  Start: $(date)"
echo "========================================="

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

pip uninstall -y xformers flash-attn
pip install --no-deps 'unsloth-zoo==2026.1.4' 'unsloth==2026.1.4'
pip install -r requirements.txt

OUTPUT_DIR="./checkpoints/${MODEL_NAME}"
mkdir -p $OUTPUT_DIR logs

python sft_train_trl.py \
    --model "Qwen/Qwen2.5-Coder-14B" \
    --data-path "data/runtime" \
    --template runtime \
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
    ln -sfn $(readlink -f $OUTPUT_DIR) ./checkpoints/qwen-coder-base-14b_sft_runtime_latest
    echo "Symlink updated: qwen-coder-base-14b_sft_runtime_latest -> $OUTPUT_DIR"
    echo "Next: sbatch slurm/eval_sft_runtime_gen.sh"
else
    echo "FAILED (exit=$PYTHON_EXIT)"
    exit 1
fi
echo "End: $(date)"
