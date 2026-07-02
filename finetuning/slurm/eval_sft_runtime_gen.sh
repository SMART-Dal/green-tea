#!/bin/bash
#SBATCH --job-name=sft_runtime_gen
#SBATCH --account=rrg-mrdal22
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/sft_runtime_gen_%j.out
#SBATCH --error=logs/sft_runtime_gen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W1 ablation: generate from runtime-SFT on test set, using ENERGY prompt at inference
# (we evaluate energy CAERR of the runtime-trained model, using energy prompt to match
#  the energy-SFT eval setup -- tests whether runtime training learns energy-relevant patterns)
# Prereq: sft_runtime_ablation.sh completed
# Next: sbatch slurm/eval_sft_runtime_sim.sh

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

MODEL_PATH="checkpoints/qwen-coder-base-14b_sft_runtime_latest/final"
INPUT_DATA="data/sft_pairs_test.jsonl"
OUTPUT_FILE="data/sft_runtime_generations_test/eval_generations_test.jsonl"

echo "========================================="
echo "W1 ABLATION: Runtime-SFT Generation"
echo "Model: $MODEL_PATH  Job: $SLURM_JOB_ID  Start: $(date)"
echo "========================================="

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv

export CUDA_VISIBLE_DEVICES=0
source ../config.env

VENV="$HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv"
source "$VENV/bin/activate"

pip install -r requirements.txt -q

mkdir -p data/sft_runtime_generations_test logs

[ ! -d "$MODEL_PATH" ] && echo "ERROR: $MODEL_PATH not found" && ls checkpoints/ && exit 1
[ ! -f "$INPUT_DATA" ] && echo "ERROR: $INPUT_DATA not found" && exit 1

export MODEL_PATH INPUT_DATA OUTPUT_FILE

python3 -u << 'PYTHON_SCRIPT'
import json, os, torch
from pathlib import Path
from collections import defaultdict

MODEL_PATH  = os.environ['MODEL_PATH']
INPUT_DATA  = os.environ['INPUT_DATA']
OUTPUT_FILE = os.environ['OUTPUT_FILE']

print("Loading runtime-SFT model...", flush=True)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=4096, dtype=torch.bfloat16)
model.eval()

# Use ENERGY prompt at inference -- evaluates whether runtime training
# learned energy-relevant transformations (matches energy-SFT eval setup)
PROMPT = ("This is an energy inefficient program we want to optimize to score 10/10.\n"
          "### Program:\n{code}\n\n"
          "### Energy Optimized Version with score 10/10:\n```cpp\n")

samples = [json.loads(l) for l in open(INPUT_DATA)]
baseline_to_samples = defaultdict(list)
for s in samples:
    bc = s.get('inefficient_code', s.get('baseline_code', ''))
    if bc: baseline_to_samples[bc].append(s)
unique_baselines = list(baseline_to_samples.keys())
print(f"Loaded {len(samples)} samples -> {len(unique_baselines)} unique baselines", flush=True)

fence_id = tokenizer.encode('```', add_special_tokens=False)[0]
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

completed = set()
if Path(OUTPUT_FILE).exists():
    for line in open(OUTPUT_FILE):
        completed.add(json.loads(line)['problem_id'])
    print(f"Resuming: {len(completed)} done", flush=True)

total = skipped_done = skipped_long = 0
mode = 'a' if completed else 'w'
with open(OUTPUT_FILE, mode) as out:
    for idx, bc in enumerate(unique_baselines, 1):
        samps = baseline_to_samples[bc]
        if all(s.get('problem_id','') in completed for s in samps):
            skipped_done += len(samps); continue
        prompt = PROMPT.format(code=bc)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        if inputs['input_ids'].shape[1] + 2048 > 32768:
            skipped_long += len(samps); continue
        with torch.no_grad():
            outt = model.generate(**inputs, max_new_tokens=2048, temperature=0.2,
                do_sample=True, eos_token_id=[tokenizer.eos_token_id, fence_id],
                pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2)
        gen = tokenizer.decode(outt[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        code = gen.split('```cpp')[1].split('```')[0].strip() if '```cpp' in gen else \
               gen.split('```')[0].strip() if '```' in gen else gen.strip()
        for s in samps:
            if s.get('problem_id','') in completed: continue
            out.write(json.dumps({'problem_id': s.get('problem_id',''), 'baseline_code': bc,
                                  'generated_code': code, 'optimized_code': s.get('optimized_code','')}) + '\n')
            total += 1
        out.flush()
        if idx % 50 == 0 or idx == len(unique_baselines):
            print(f"[{idx}/{len(unique_baselines)}] {total} outputs (skip: done={skipped_done} long={skipped_long})", flush=True)

print(f"Complete: {total} outputs -> {OUTPUT_FILE}", flush=True)
PYTHON_SCRIPT

exit_code=$?
[ $exit_code -eq 0 ] && echo "GEN COMPLETE: $(wc -l < $OUTPUT_FILE) samples. Next: sbatch slurm/eval_sft_runtime_sim.sh" || echo "FAILED"
echo "End: $(date)"
exit $exit_code
