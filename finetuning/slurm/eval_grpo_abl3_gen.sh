#!/bin/bash
#SBATCH --job-name=grpo_abl3_gen
#SBATCH --account=rrg-mrdal22
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/grpo_abl3_gen_%j.out
#SBATCH --error=logs/grpo_abl3_gen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Generate test-set outputs from ABL3 (runtime-SFT init + runtime reward)

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

ABL_NAME="grpo_abl3_rsft_rt"
LATEST_LINK="./checkpoints/qwen-coder-base-14b_${ABL_NAME}_latest"
INPUT_DATA="data/sft_pairs_test.jsonl"
OUTPUT_DIR="data/${ABL_NAME}_test_generations"
OUTPUT_FILE="${OUTPUT_DIR}/eval_generations_test.jsonl"

echo "========================================="
echo "GRPO ABL3 GEN | Job: $SLURM_JOB_ID | Start: $(date)"
echo "========================================="

if [ ! -L "$LATEST_LINK" ] || [ ! -e "$LATEST_LINK" ]; then
    echo "ERROR: $LATEST_LINK not found"; exit 1
fi
MODEL_PATH=$(readlink -f "$LATEST_LINK")
CKPT=$(find $MODEL_PATH -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
[ -z "$CKPT" ] && [ -d "$MODEL_PATH/final" ] && CKPT="$MODEL_PATH/final"
[ -z "$CKPT" ] && echo "ERROR: no checkpoint in $MODEL_PATH" && exit 1
MODEL_PATH="$CKPT"
echo "Model: $MODEL_PATH"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv
export CUDA_VISIBLE_DEVICES=0
source ../config.env
VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"
pip install -r requirements.txt -q
mkdir -p $OUTPUT_DIR logs

[ ! -f "$INPUT_DATA" ] && echo "ERROR: $INPUT_DATA not found" && exit 1
export MODEL_PATH INPUT_DATA OUTPUT_FILE

python3 -u << 'PYTHON_SCRIPT'
import json, sys, os, torch
from pathlib import Path
from collections import defaultdict

MODEL_PATH = os.environ['MODEL_PATH']
INPUT_DATA = os.environ['INPUT_DATA']
OUTPUT_FILE = os.environ['OUTPUT_FILE']

print("Loading model...", flush=True)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_PATH, max_seq_length=4096, dtype=torch.bfloat16)
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
        if inputs['input_ids'].shape[1] + 2048 > 32768:
            skipped_long += len(samps); continue
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, temperature=0.2,
                do_sample=True, eos_token_id=[tokenizer.eos_token_id, fence_token_id],
                pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2)
        code = extract_code(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
        for s in samps:
            if s.get('problem_id', '') in completed_pids: continue
            outf.write(json.dumps({'problem_id': s.get('problem_id', ''), 'baseline_code': baseline_code,
                'generated_code': code, 'optimized_code': s.get('optimized_code', '')}) + '\n')
            total_outputs += 1
        outf.flush()
        if idx % 50 == 0 or idx == len(unique_baselines):
            print(f"[{idx}/{len(unique_baselines)}] -> {total_outputs} outputs (skip: done={skipped_done}, long={skipped_long})", flush=True)
print(f"\nComplete: {total_outputs} outputs -> {OUTPUT_FILE}", flush=True)
PYTHON_SCRIPT

exit_code=$?
[ $exit_code -eq 0 ] && echo "ABL3 GEN COMPLETE: $(wc -l < $OUTPUT_FILE) samples" || echo "FAILED (exit=$exit_code)"
echo "End: $(date)"
exit $exit_code
