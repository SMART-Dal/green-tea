#!/bin/bash
#SBATCH --job-name=frontier_qwen32b_gen
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=logs/frontier_qwen32b_gen_%j.out
#SBATCH --error=logs/frontier_qwen32b_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W2 frontier baseline: Qwen2.5-Coder-32B-Instruct zero-shot + green-prompt (inference only, no training)
# Next: sbatch slurm/eval_frontier_qwen32b_sim.sh

MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
OUTPUT_DIR="data/new_models/frontier_qwen32b/generations"

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv
export CUDA_VISIBLE_DEVICES=0
source ../config.env
source $HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv/bin/activate
pip install -r requirements.txt -q

echo "Frontier baseline: $MODEL | Job: $SLURM_JOB_ID | Start: $(date)"
mkdir -p $OUTPUT_DIR logs

for VARIANT in zero_shot_instruct green_prompt_instruct; do
    echo "Running variant: $VARIANT"
    python3 -u sft_baseline.py \
        --model "$MODEL" \
        --input "data/sft_pairs_test.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        --variants "$VARIANT" \
        --use-transformers
    echo "Done $VARIANT: $(wc -l < $OUTPUT_DIR/$VARIANT/generations.jsonl 2>/dev/null || echo 0) samples"
done

echo "Complete. Next: sbatch slurm/eval_frontier_qwen32b_sim.sh"
echo "End: $(date)"
