#!/bin/bash
#SBATCH --job-name=grpo_gen
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/grpo_gen_%j.out
#SBATCH --error=logs/grpo_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# GPU: generate optimized code from GRPO checkpoint on test set
# Prereq: job 9400877 (GRPO training) must have completed
# Next: sbatch slurm/eval_grpo_sim.sh

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

MODEL_PATH="checkpoints/qwen-coder-base-14b_grpo_latest/checkpoint-7900"
INPUT_DATA="data/sft_pairs_test.jsonl"
OUTPUT_FILE="data/grpo_generations_test/eval_generations_test.jsonl"

echo "========================================="
echo "GRPO TEST SET GENERATION"
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_DATA ($(wc -l < $INPUT_DATA 2>/dev/null || echo '?') samples)"
echo "Output: $OUTPUT_FILE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================="

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

pip install -r requirements.txt -q

mkdir -p data/grpo_generations_test logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    ls -la checkpoints/
    exit 1
fi

[ ! -f "$INPUT_DATA" ] && echo "ERROR: Test data not found: $INPUT_DATA" && exit 1

export MODEL_PATH INPUT_DATA OUTPUT_FILE

python3 -u << 'PYTHON_SCRIPT'
import json, sys, os
import torch
from pathlib import Path
from collections import defaultdict

MODEL_PATH = os.environ['MODEL_PATH']
INPUT_DATA = os.environ['INPUT_DATA']
OUTPUT_FILE = os.environ['OUTPUT_FILE']

print("Loading model...", flush=True)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=4096, dtype=torch.bfloat16,
)
model.eval()
print(f"Model loaded from {MODEL_PATH}", flush=True)

samples = [json.loads(l) for l in open(INPUT_DATA)]
baseline_to_samples = defaultdict(list)
for s in samples:
    bc = s.get('inefficient_code', s.get('baseline_code', ''))
    if bc: baseline_to_samples[bc].append(s)

unique_baselines = list(baseline_to_samples.keys())
print(f"Loaded {len(samples)} samples -> {len(unique_baselines)} unique baselines", flush=True)

fence_token_id = tokenizer.encode('```', add_special_tokens=False)[0]
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

PROMPT_FMT = ("This is an energy inefficient program we want to optimize to score 10/10.\n"
              "### Program:\n{code}\n\n"
              "### Energy Optimized Version with score 10/10:\n```cpp\n")

def extract_code(gen):
    if '```cpp' in gen: return gen.split('```cpp')[1].split('```')[0].strip()
    if '```' in gen: return gen.split('```')[0].strip()
    return gen.strip()

completed_pids = set()
if Path(OUTPUT_FILE).exists():
    with open(OUTPUT_FILE) as f:
        for line in f: completed_pids.add(json.loads(line)['problem_id'])
    print(f"Found {len(completed_pids)} already completed, resuming...", flush=True)

skipped_long = skipped_done = total_outputs = 0
mode = 'a' if completed_pids else 'w'
with open(OUTPUT_FILE, mode) as outf:
    for idx, baseline_code in enumerate(unique_baselines, 1):
        samps = baseline_to_samples[baseline_code]
        if all(s.get('problem_id', '') in completed_pids for s in samps):
            skipped_done += len(samps); continue

        prompt = PROMPT_FMT.format(code=baseline_code)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        input_len = inputs['input_ids'].shape[1]
        if input_len + 2048 > 32768:
            skipped_long += len(samps); continue

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.2,
                do_sample=True, eos_token_id=[tokenizer.eos_token_id, fence_token_id],
                pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2
            )
        code = extract_code(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))

        for s in samps:
            if s.get('problem_id', '') in completed_pids: continue
            outf.write(json.dumps({
                'problem_id': s.get('problem_id', ''),
                'baseline_code': baseline_code,
                'generated_code': code,
                'optimized_code': s.get('optimized_code', ''),
            }) + '\n')
            total_outputs += 1
        outf.flush()
        if idx % 50 == 0 or idx == len(unique_baselines):
            print(f"[{idx}/{len(unique_baselines)}] -> {total_outputs} outputs (skip: done={skipped_done}, long={skipped_long})", flush=True)

print(f"\nComplete: {total_outputs} outputs -> {OUTPUT_FILE}", flush=True)
PYTHON_SCRIPT

exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo "GRPO GENERATION COMPLETE: $(wc -l < $OUTPUT_FILE) samples"
    echo "Next: sbatch slurm/eval_grpo_sim.sh"
else
    echo "FAILED (exit=$exit_code)"
fi
echo "End: $(date)"
exit $exit_code
