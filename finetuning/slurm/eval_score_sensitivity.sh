#!/bin/bash
#SBATCH --job-name=score_sensitivity
#SBATCH --account=def-tusharma_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/score_sensitivity_%j.out
#SBATCH --error=logs/score_sensitivity_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Tech Q2: score-conditioned prompting sensitivity (8/10, 9/10, 10/10)
# Uses existing GRPO checkpoint. Outputs go to data/new_models/score_sensitivity/

cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

MODEL_PATH="checkpoints/qwen-coder-base-14b_grpo_latest/checkpoint-7900"
INPUT_DATA="data/sft_pairs_test.jsonl"
OUTPUT_BASE="data/new_models/score_sensitivity"

echo "Score Sensitivity | Job: $SLURM_JOB_ID | Start: $(date)"
[ ! -d "$MODEL_PATH" ] && echo "ERROR: $MODEL_PATH not found" && exit 1

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 arrow scipy-stack mpi4py opencv
export CUDA_VISIBLE_DEVICES=0
source ../config.env
source $HOME/projects/rrg-mrdal22/srajput/green-code-gen/venv/bin/activate
pip install -r requirements.txt -q

mkdir -p $OUTPUT_BASE logs

export MODEL_PATH INPUT_DATA OUTPUT_BASE

python3 -u << 'PYTHON_SCRIPT'
import json, sys, os, torch
from pathlib import Path
from collections import defaultdict

MODEL_PATH = os.environ['MODEL_PATH']
INPUT_DATA = os.environ['INPUT_DATA']
OUTPUT_BASE = os.environ['OUTPUT_BASE']

print(f"Loading {MODEL_PATH}...", flush=True)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_PATH, max_seq_length=4096, dtype=torch.bfloat16)
model.eval()

samples = [json.loads(l) for l in open(INPUT_DATA)]
baseline_to_samples = defaultdict(list)
for s in samples:
    bc = s.get('inefficient_code', s.get('baseline_code', ''))
    if bc: baseline_to_samples[bc].append(s)
unique_baselines = list(baseline_to_samples.keys())
print(f"{len(samples)} samples -> {len(unique_baselines)} unique baselines", flush=True)

fence_token_id = tokenizer.encode('```', add_special_tokens=False)[0]

def extract_code(gen):
    if '```cpp' in gen: return gen.split('```cpp')[1].split('```')[0].strip()
    if '```' in gen: return gen.split('```')[0].strip()
    return gen.strip()

for target_score in [8, 9, 10]:
    output_file = Path(OUTPUT_BASE) / f'score_{target_score}' / 'eval_generations_test.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_FMT = (f"This is an energy inefficient program we want to optimize to score {target_score}/10.\n"
                  f"### Program:\n{{code}}\n\n### Energy Optimized Version with score {target_score}/10:\n```cpp\n")

    completed_pids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f: completed_pids.add(json.loads(line)['problem_id'])
        print(f"Score {target_score}: resuming, {len(completed_pids)} done", flush=True)

    total_outputs = 0
    mode = 'a' if completed_pids else 'w'
    with open(output_file, mode) as outf:
        for idx, baseline_code in enumerate(unique_baselines, 1):
            samps = baseline_to_samples[baseline_code]
            if all(s.get('problem_id', '') in completed_pids for s in samps): continue
            prompt = PROMPT_FMT.format(code=baseline_code)
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            if inputs['input_ids'].shape[1] + 2048 > 32768: continue
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=2048, temperature=0.2, do_sample=True,
                                     eos_token_id=[tokenizer.eos_token_id, fence_token_id],
                                     pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2)
            code = extract_code(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
            for s in samps:
                if s.get('problem_id', '') in completed_pids: continue
                outf.write(json.dumps({'problem_id': s.get('problem_id', ''), 'baseline_code': baseline_code,
                                       'generated_code': code, 'optimized_code': s.get('optimized_code', ''),
                                       'target_score': target_score}) + '\n')
                total_outputs += 1
            outf.flush()
            if idx % 50 == 0: print(f"Score {target_score}: [{idx}/{len(unique_baselines)}] -> {total_outputs}", flush=True)
    print(f"Score {target_score}/10: {total_outputs} outputs -> {output_file}", flush=True)

print("Score sensitivity complete.", flush=True)
PYTHON_SCRIPT

echo "Score sensitivity done. Next: run sim array on each score_N/ directory. End: $(date)"
