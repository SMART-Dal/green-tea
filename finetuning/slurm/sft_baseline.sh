#!/bin/bash
#SBATCH --job-name=baseline-gen
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=logs/baseline_gen_%j.out
#SBATCH --error=logs/baseline_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Generate baselines using the instruct model with ChatML prompts.
# VARIANT selects which baseline to generate (run two jobs in parallel):
#   VARIANT=zero_shot_instruct sbatch slurm/sft_baseline.sh
#   VARIANT=green_prompt_instruct sbatch slurm/sft_baseline.sh
# Output: data/sft_baseline_generations/{zero_shot_instruct,green_prompt_instruct}/generations.jsonl
# Next step: VARIANT=zero_shot_instruct sbatch slurm/baseline_sim.sh (etc.)

VARIANT="${VARIANT:-zero_shot_instruct}"
MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

pip install -r requirements.txt -q

echo "========================================="
echo "BASELINE GENERATION: ${VARIANT}"
echo "Model: ${MODEL}"
echo "Job: $SLURM_JOB_ID  Start: $(date)"
echo "========================================="

mkdir -p data/sft_baseline_generations logs

python3 -u sft_baseline.py \
    --model "${MODEL}" \
    --input "data/sft_pairs_test.jsonl" \
    --output-dir "data/sft_baseline_generations" \
    --variants "${VARIANT}"

exit_code=$?
echo ""
echo "========================================="
if [ $exit_code -eq 0 ]; then
    echo "GENERATION COMPLETE: ${VARIANT}"
    echo "outputs: $(wc -l < data/sft_baseline_generations/${VARIANT}/generations.jsonl 2>/dev/null || echo 0)"
    echo ""
    echo "Next -- run Sniper simulation:"
    echo "  VARIANT=${VARIANT} sbatch slurm/baseline_sim.sh"
else
    echo "GENERATION FAILED (exit=$exit_code)"
fi
echo "End: $(date)"
echo "========================================="
exit $exit_code
