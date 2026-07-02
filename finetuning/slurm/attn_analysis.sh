#!/bin/bash
#SBATCH --job-name=attn_analysis
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/attn_analysis_%j.out
#SBATCH --error=logs/attn_analysis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

GRPO_MODEL="checkpoints/qwen-coder-base-14b_grpo_latest/checkpoint-best"
ABL1_MODEL="checkpoints/qwen-coder-base-14b_grpo_abl1_rsft_edp_20260313_014811/checkpoint-7375"
BASE_MODEL="Qwen/Qwen2.5-Coder-14B"
SIM_DIR="data/grpo_sim_results"
ABL1_SIM_DIR="data/grpo_abl1_rsft_edp_sim_results"
OUT_BASE="../analysis/attention_plots_compare_paper"
N_SAMPLES=200

echo "attn_analysis job $SLURM_JOB_ID start $(date)"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv
export CUDA_VISIBLE_DEVICES=0
source ../config.env
source "$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv/bin/activate"
pip install psutil -q
mkdir -p "$OUT_BASE/grpo" "$OUT_BASE/base" "$OUT_BASE/abl1" logs

for triple in "grpo|$GRPO_MODEL|$SIM_DIR" "base|$BASE_MODEL|$SIM_DIR" "abl1|$ABL1_MODEL|$ABL1_SIM_DIR"; do
    IFS='|' read tag model sim <<< "$triple"
    if compgen -G "$OUT_BASE/$tag/ex*_*/metadata.json" > /dev/null; then
        echo "=== $tag: already populated, skipping ==="; continue
    fi
    echo "=== $tag: $model (sim=$sim) ==="
    python analyze_grpo_checkpoint.py --mode attention \
        --model "$model" --sim-dir "$sim" \
        --output-dir "$OUT_BASE/$tag" \
        --n-samples $N_SAMPLES --n-layers 8 --min-reduction 0
done

echo "Done: $(date)"
