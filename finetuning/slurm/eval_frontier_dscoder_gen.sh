#!/bin/bash
#SBATCH --job-name=frontier_dscoder_gen
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/frontier_dscoder_gen_%j.out
#SBATCH --error=logs/frontier_dscoder_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W2 frontier baseline: DeepSeek-Coder-V2-Lite-Instruct zero-shot + green-prompt (16B MoE, ~3B active)
# Non-Qwen frontier baseline for diversity. Next: sbatch slurm/eval_frontier_dscoder_sim.sh

MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
OUTPUT_DIR="data/new_models/frontier_dscoder/generations"

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

echo "Complete. Next: sbatch slurm/eval_frontier_dscoder_sim.sh"
echo "End: $(date)"
