#!/bin/bash
#SBATCH --job-name=pass_at_k_gen
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pass_at_k_gen_%j.out
#SBATCH --error=logs/pass_at_k_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Pass@k Phase 1: GPU generation of n=10 samples per problem with temperature=0.8
# Addresses reviewer W4: greedy decoding only
# Next: sbatch slurm/eval_pass_at_k_test.sh (job array for correctness testing)

PROJECT_ROOT="$HOME/projects/rrg-mrdal22/srajput/green-code-gen"
SFT_MODEL="${PROJECT_ROOT}/finetuning/checkpoints/qwen-coder-base-14b_sft_latest/final"
NUM_SAMPLES=10
TEMPERATURE=0.8

echo "========================================="
echo "PASS@K GENERATION (Phase 1)"
echo "Model: ${SFT_MODEL}"
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

mkdir -p data/sft_pass_at_k logs

python3 -u sft_baseline.py \
    --model "${SFT_MODEL}" \
    --input "data/sft_pairs_test.jsonl" \
    --output-dir "data/sft_pass_at_k" \
    --variants "sft" \
    --num-samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top-p 0.95

GEN_EXIT=$?
if [ $GEN_EXIT -eq 0 ]; then
    echo ""
    echo "Generation complete. Next: sbatch slurm/eval_pass_at_k_test.sh"
else
    echo "Generation failed (exit=$GEN_EXIT)"
fi
echo "End: $(date)"
exit $GEN_EXIT
