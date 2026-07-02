#!/bin/bash
#SBATCH --job-name=sft_debug
#SBATCH --account=rrg-mrdal22
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/debug_sft_%j.out
#SBATCH --error=logs/debug_sft_%j.err

# DEBUG RUN: verifies EnergyEvaluationCallback fixes end-to-end
# Tests: precomputed baseline, compile-once, correctness check, Sniper on correct only,
#        gt_energy > 1e-9 threshold, full gen_record schema (compiled, tests_passed, etc.)
# 0.5B model, 20-sample debug data, eval every 10 steps, 3 eval samples per trigger
# Output: checkpoints/Qwen_test_sft, data/sft_generations_debug/
# Verification block at end inspects gen_record fields

MODEL_NAME="Qwen_test_sft"

echo "========================================="
echo "DEBUG SFT RUN"
echo "Job ID: $SLURM_JOB_ID"
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

OUTPUT_DIR="./checkpoints/Qwen_test_sft"
mkdir -p $OUTPUT_DIR logs

# Debug data: sft_train_trl.py loads sft_pairs_train.jsonl / sft_pairs_val.jsonl by name
mkdir -p data_debug
cp data/sft_pairs_train_debug.jsonl data_debug/sft_pairs_train.jsonl
cp data/sft_pairs_val_debug.jsonl data_debug/sft_pairs_val.jsonl

# Clean previous debug generation output
rm -rf data/sft_generations_debug

python sft_train_trl.py \
    --model "Qwen/Qwen2.5-Coder-0.5B" \
    --data-path data_debug \
    --output-dir $OUTPUT_DIR \
    --num-epochs 2 \
    --learning-rate 1e-5 \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --max-seq-length 512 \
    --lora-r 8 \
    --lora-alpha 16 \
    --model-name $MODEL_NAME \
    --eval-during-training \
    --sniper-root "$SNIPER_ROOT" \
    --eval-output-dir data/sft_generations_debug \
    --eval-every-n-steps 10 \
    --max-eval-samples 3 \
    --eval-steps 10 \
    --save-steps 20 \
    --use-wandb

PYTHON_EXIT=$?
rm -rf data_debug

# ==========================================================================
# VERIFICATION: inspect gen_records written by EnergyEvaluationCallback
# ==========================================================================
echo ""
echo "========================================="
echo "DEBUG VERIFICATION"
echo "========================================="

if [ ! -d data/sft_generations_debug ]; then
    echo "FAIL: data/sft_generations_debug/ not created -- callback never fired or crashed"
    exit 1
fi

GEN_FILES=$(ls data/sft_generations_debug/eval_generations_step*.jsonl 2>/dev/null)
if [ -z "$GEN_FILES" ]; then
    echo "FAIL: No generation files written"
    exit 1
fi

echo "Generation files:"
ls -la data/sft_generations_debug/
echo ""

python3 -u << 'VERIFY'
import json, sys
from pathlib import Path

gen_dir = Path("data/sft_generations_debug")
files = sorted(gen_dir.glob("eval_generations_step*.jsonl"))
print(f"Found {len(files)} generation files")

REQUIRED_CORE = ['step', 'problem_id', 'baseline_code', 'generated_code',
                 'optimized_code', 'status', 'compiled', 'tests_passed',
                 'num_inputs', 'generated_success_count']

all_ok = True
for gf in files:
    recs = [json.loads(l) for l in open(gf)]
    print(f"\n--- {gf.name} ({len(recs)} records) ---")

    for i, r in enumerate(recs):
        pid = r.get('problem_id', '?')
        # Check required core fields exist
        missing = [k for k in REQUIRED_CORE if k not in r]
        if missing:
            print(f"  [{pid}] FAIL: missing fields: {missing}")
            all_ok = False
            continue

        # Status breakdown
        status = r['status']
        compiled = r['compiled']
        tp = r['tests_passed']
        ni = r['num_inputs']
        gs = r['generated_success_count']

        line = f"  [{pid}] status={status} compiled={compiled} tests_passed={tp}/{ni} sniper_ok={gs}"

        # If success, check metric fields
        if status == 'success' and gs > 0:
            metric_fields = ['baseline_energy', 'baseline_avg_cycles', 'baseline_avg_ipc',
                             'generated_energy', 'generated_avg_cycles', 'generated_avg_ipc',
                             'baseline_edp', 'generated_edp', 'energy_reduction',
                             'speedup', 'ipc_improvement_pct']
            missing_m = [k for k in metric_fields if k not in r]
            if missing_m:
                print(f"{line}")
                print(f"    FAIL: success but missing metric fields: {missing_m}")
                all_ok = False
            else:
                ge = r['generated_energy']
                be = r['baseline_energy']
                er = r.get('energy_reduction', 0)
                print(f"{line}")
                print(f"    baseline_E={be:.6f}J  generated_E={ge:.6f}J  ERR={er*100:.1f}%")

                # Verify gt fields if optimized_energy present
                if 'optimized_energy' in r and r['optimized_energy'] > 1e-9:
                    gt_fields = ['optimized_cycles', 'optimized_ipc', 'optimized_edp', 'vs_gt_reduction']
                    missing_gt = [k for k in gt_fields if k not in r]
                    if missing_gt:
                        print(f"    FAIL: optimized_energy present but missing: {missing_gt}")
                        all_ok = False
                    else:
                        print(f"    optimized_E={r['optimized_energy']:.6f}J  vs_GT={r['vs_gt_reduction']*100:.1f}%")
                else:
                    print(f"    (no optimized_energy or below 1e-9)")
        elif not compiled:
            ce = r.get('compile_error', '(no error msg)')
            print(f"{line}")
            print(f"    compile_error: {ce[:80]}")
        else:
            print(f"{line}")

        # Sanity: compiled=True but tests_passed=0 and sniper_ok=0 is valid (all wrong answers)
        # compiled=False but status != compile_error is a bug
        if not compiled and status != 'compile_error':
            print(f"    FAIL: compiled=False but status={status}")
            all_ok = False
        # compiled=True, tests_passed>0 but sniper_ok=0 means Sniper failed on all correct -- warn
        if compiled and tp > 0 and gs == 0:
            print(f"    WARN: {tp} tests passed but Sniper returned 0 valid results")

print(f"\n{'='*50}")
if all_ok:
    print("VERIFICATION PASSED -- all gen_record fields consistent")
else:
    print("VERIFICATION FAILED -- see FAIL lines above")
    sys.exit(1)
VERIFY

echo ""
echo "========================================="
if [ $PYTHON_EXIT -eq 0 ]; then
    echo "DEBUG SFT RUN COMPLETE"
else
    echo "DEBUG SFT RUN FAILED (exit=$PYTHON_EXIT)"
fi
echo "========================================="
exit $PYTHON_EXIT
