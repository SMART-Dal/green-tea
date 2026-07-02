#!/bin/bash
#SBATCH --job-name=grpo_debug
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/debug_grpo_%j.out
#SBATCH --error=logs/debug_grpo_%j.err

# DEBUG GRPO: Verify parallel Sniper (K=4), reward computation, checkpointing
# Small model: Qwen2.5-Coder-0.5B, 20 samples, K=4 candidates
# Tests: ThreadPoolExecutor parallelization, reward hierarchical correctness+energy,
#        vLLM generation, checkpoint saving, wandb logging
# Output: checkpoints/Qwen_test_grpo, data/grpo_generations_debug/

MODEL_NAME="Qwen_test_grpo"

echo "========================================="
echo "DEBUG GRPO RUN"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: 0.5B model, K=4, parallel Sniper"
echo "========================================="

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

pip uninstall -y xformers flash-attn
pip install --upgrade pip
pip install --no-deps 'unsloth-zoo==2026.1.4'
pip install --no-deps 'unsloth==2026.1.4'
pip install -r requirements.txt
pip install "vllm<=0.11.2"

export SNIPER_ROOT=$HOME/projects/rrg-mrdal22/srajput/green-code-gen/sniper/sniper
export PATH=$SNIPER_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$SNIPER_ROOT/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1

OUTPUT_DIR="./checkpoints/$MODEL_NAME"
mkdir -p $OUTPUT_DIR logs data/grpo_generations_debug

TRAIN_DATA="data/grpo_train_debug.jsonl"
VAL_DATA="data/grpo_val_debug.jsonl"

python grpo_trainer_vllm.py \
    --model "Qwen/Qwen2.5-Coder-0.5B" \
    --train-data $TRAIN_DATA \
    --val-data $VAL_DATA \
    --sniper-root $SNIPER_ROOT \
    --output-dir $OUTPUT_DIR \
    --num-epochs 1 \
    --num-generations 4 \
    --learning-rate 1e-6 \
    --batch-size 1 \
    --gradient-accumulation-steps 2 \
    --max-seq-length 2048 \
    --sniper-config epyc_9554p \
    --compile-timeout 30 \
    --simulation-timeout 180 \
    --model-name $MODEL_NAME \
    --use-wandb \
    --eval-during-training \
    --pie-dataset "../PIE_Dataset" \
    --gpu-memory-utilization 0.50

PYTHON_EXIT=$?

echo ""
echo "========================================="
echo "DEBUG VERIFICATION"
echo "========================================="

if [ $PYTHON_EXIT -ne 0 ]; then
    echo "FAIL: Training crashed (exit=$PYTHON_EXIT)"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "FAIL: Output directory not created"
    exit 1
fi

CHECKPOINTS=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | wc -l)
echo "Checkpoints saved: $CHECKPOINTS"
if [ $CHECKPOINTS -eq 0 ]; then
    echo "WARN: No checkpoints saved"
fi

GEN_LOG="data/grpo_generations/grpo_generations.jsonl"
if [ -f "$GEN_LOG" ]; then
    NUM_GENS=$(wc -l < $GEN_LOG)
    echo "Generations logged: $NUM_GENS"

    python3 << 'VERIFY'
import json
from pathlib import Path

gen_file = Path("data/grpo_generations/grpo_generations.jsonl")
if not gen_file.exists():
    print("FAIL: Generation log missing")
    exit(1)

records = [json.loads(l) for l in open(gen_file)]
print(f"\nAnalyzing {len(records)} generation records:")

statuses = {}
rewards = []
compile_ok = 0
correct_100 = 0
energy_improved = 0

for r in records:
    status = r.get('status', 'unknown')
    statuses[status] = statuses.get(status, 0) + 1

    reward = r.get('reward', 0)
    rewards.append(reward)

    if r.get('compiled', False):
        compile_ok += 1
    if r.get('tests_passed', 0) == r.get('num_test_inputs', 1):
        correct_100 += 1
    if r.get('energy_reduction', 0) > 0:
        energy_improved += 1

print(f"  Statuses: {statuses}")
print(f"  Compiled: {compile_ok}/{len(records)} ({compile_ok/len(records)*100:.1f}%)")
print(f"  100% correct: {correct_100}/{len(records)} ({correct_100/len(records)*100:.1f}%)")
print(f"  Energy improved: {energy_improved}/{len(records)}")
print(f"  Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
print(f"  Mean reward: {sum(rewards)/len(rewards):.2f}")

if compile_ok == 0:
    print("\nFAIL: Zero compilations - check Sniper setup")
    exit(1)

if all(r == rewards[0] for r in rewards):
    print("\nWARN: All rewards identical - check diversity")

print("\nVERIFICATION PASSED")
VERIFY
else
    echo "FAIL: No generation log found at $GEN_LOG"
    exit 1
fi

echo ""
echo "========================================="
echo "DEBUG GRPO COMPLETE"
echo "========================================="
exit $PYTHON_EXIT
