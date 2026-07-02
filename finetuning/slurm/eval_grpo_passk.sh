#!/bin/bash
#SBATCH --job-name=grpo_passk_gen
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grpo_passk_gen_%j.out
#SBATCH --error=logs/grpo_passk_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Pass@k Phase 1 for GRPO: GPU generation of n=10 samples per problem with T=0.8
# Prereq: GRPO training (job 9400877) must have completed
# Next: sbatch slurm/eval_grpo_passk_test.sh (reuse eval_pass_at_k_test.sh with GENERATIONS_FILE override)

PROJECT_ROOT="$HOME/projects/rrg-mrdal22/srajput/green-code-gen"
GRPO_MODEL="${PROJECT_ROOT}/finetuning/checkpoints/qwen-coder-base-14b_grpo_latest/checkpoint-7900"
NUM_SAMPLES=10
TEMPERATURE=0.8

echo "========================================="
echo "GRPO PASS@K GENERATION"
echo "Model: ${GRPO_MODEL}"
echo "n=${NUM_SAMPLES}, T=${TEMPERATURE}"
echo "Job: $SLURM_JOB_ID  Start: $(date)"
echo "========================================="

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
cd "${PROJECT_ROOT}/finetuning"
source ../config.env

VENV="${PROJECT_ROOT}/venv"
source "${VENV}/bin/activate"

pip uninstall -y xformers flash-attn
pip install --upgrade pip
pip install --no-deps 'unsloth-zoo==2026.1.4'
pip install --no-deps 'unsloth==2026.1.4'
pip install -r requirements.txt

mkdir -p data/grpo_pass_at_k logs

python3 -u sft_baseline.py \
    --model "${GRPO_MODEL}" \
    --input "data/sft_pairs_test.jsonl" \
    --output-dir "data/grpo_pass_at_k" \
    --variants "grpo" \
    --num-samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top-p 0.95

GEN_EXIT=$?
if [ $GEN_EXIT -eq 0 ]; then
    echo "GRPO pass@k generation complete."
    echo "Next: submit correctness testing with GENERATIONS_FILE=data/grpo_pass_at_k/grpo/generations.jsonl"
else
    echo "GRPO pass@k generation failed (exit=$GEN_EXIT)"
fi
echo "End: $(date)"
exit $GEN_EXIT
