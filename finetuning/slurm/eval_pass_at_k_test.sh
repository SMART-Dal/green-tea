#!/bin/bash
#SBATCH --job-name=pass_at_k_test
#SBATCH --account=rrg-mrdal22
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-14
#SBATCH --output=logs/pass_at_k_test_%A_%a.out
#SBATCH --error=logs/pass_at_k_test_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Pass@k Phase 2: CPU correctness testing (job array)
# Prereq: eval_pass_at_k.sh must have completed (generation phase)
# Each task tests a chunk of problems; final aggregation runs in task 0 after all complete

PROJECT_ROOT="$HOME/projects/rrg-mrdal22/srajput/green-code-gen"
export PIE_DATASET="${PROJECT_ROOT}/PIE_Dataset"
export GENERATIONS_FILE="${PROJECT_ROOT}/finetuning/data/grpo_pass_at_k/grpo/generations.jsonl"
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/grpo_pass_at_k/test_results"
export TASK_ID=${SLURM_ARRAY_TASK_ID}
export NUM_TASKS=15

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10

cd "${PROJECT_ROOT}/finetuning"
source ../config.env
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/grpo_pass_at_k/test_results"
VENV="${PROJECT_ROOT}/venv"
source "${VENV}/bin/activate"

mkdir -p "${OUTPUT_DIR}" logs

echo "========================================="
echo "PASS@K CORRECTNESS TEST: Task ${TASK_ID}/${NUM_TASKS}"
echo "Generations: ${GENERATIONS_FILE}"
echo "Start: $(date)"
echo "========================================="

[ ! -f "$GENERATIONS_FILE" ] && echo "ERROR: generations not found" && exit 1

python3 -u << 'PYTHON_SCRIPT'
import json, os, sys, subprocess, tempfile, shutil, math
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(line_buffering=True)

PIE_ROOT = Path(os.environ['PIE_DATASET'])
GENS = os.environ['GENERATIONS_FILE']
OUTPUT_DIR = Path(os.environ['OUTPUT_DIR'])
TASK_ID = int(os.environ['TASK_ID'])
NUM_TASKS = int(os.environ['NUM_TASKS'])

def get_test_io(pid):
    d = PIE_ROOT / 'extracted_testcases' / 'merged_test_cases' / pid
    if not d.exists(): return []
    return [(f.read_text().strip(), (d / f"output.{f.name.replace('input.', '')}").read_text().strip())
            for f in sorted(d.glob('input.*.txt'))
            if (d / f"output.{f.name.replace('input.', '')}").exists()]

def check_correctness(code, test_io):
    with tempfile.TemporaryDirectory(prefix='pk_') as td:
        cpp = Path(td) / 's.cpp'
        cpp.write_text(code)
        bn = Path(td) / 's.bin'
        cr = subprocess.run(['g++', '-O3', '-std=c++17', '-static', str(cpp), '-o', str(bn)],
                           capture_output=True, timeout=10)
        if cr.returncode != 0:
            return False, False, 0, len(test_io)
        passed = 0
        for inp, exp in test_io:
            try:
                r = subprocess.run([str(bn)], input=inp, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip() == exp:
                    passed += 1
            except Exception:
                pass
        return True, passed == len(test_io), passed, len(test_io)

by_problem = defaultdict(list)
for line in open(GENS):
    rec = json.loads(line)
    by_problem[rec['problem_id']].append(rec)

problems = sorted(by_problem.keys())
chunk_size = (len(problems) + NUM_TASKS - 1) // NUM_TASKS
my_problems = problems[TASK_ID * chunk_size : (TASK_ID + 1) * chunk_size]
print(f"Task {TASK_ID}: {len(my_problems)} problems (of {len(problems)} total)", flush=True)

output_file = OUTPUT_DIR / f'correctness_chunk_{TASK_ID}.jsonl'
completed = set()
if output_file.exists():
    for l in open(output_file):
        completed.add(json.loads(l)['problem_id'])
    print(f"Resuming: {len(completed)} done", flush=True)

with open(output_file, 'a') as f:
    for idx, pid in enumerate(my_problems, 1):
        if pid in completed:
            continue
        test_io = get_test_io(pid)
        if not test_io:
            continue
        gens = by_problem[pid]
        n = len(gens)
        compiled_count = 0
        correct_count = 0
        for g in gens:
            try:
                comp, ok, passed, total = check_correctness(g['generated_code'], test_io)
                if comp: compiled_count += 1
                if ok: correct_count += 1
            except Exception:
                pass
        f.write(json.dumps({
            'problem_id': pid, 'n': n,
            'compiled': compiled_count, 'correct': correct_count,
        }) + '\n')
        f.flush()
        if idx % 10 == 0:
            print(f"  [{idx}/{len(my_problems)}] {pid}: {correct_count}/{n} correct", flush=True)

print(f"Task {TASK_ID} complete: {len(my_problems)} problems", flush=True)

# Aggregation: only task 0 runs this after all chunks exist
if TASK_ID == 0:
    import time, glob
    expected = NUM_TASKS
    for _ in range(60):
        found = len(list(OUTPUT_DIR.glob('correctness_chunk_*.jsonl')))
        if found >= expected: break
        time.sleep(30)

    all_results = {}
    for chunk_file in sorted(OUTPUT_DIR.glob('correctness_chunk_*.jsonl')):
        for line in open(chunk_file):
            rec = json.loads(line)
            pid = rec['problem_id']
            n, c = rec['n'], rec['correct']
            all_results[pid] = {
                'n': n, 'correct': c,
                'pass_at_1': 1.0 - math.comb(n - c, 1) / math.comb(n, 1) if n >= 1 and n - c >= 1 else (1.0 if c > 0 else 0.0),
                'pass_at_5': 1.0 - math.comb(n - c, min(5, n)) / math.comb(n, min(5, n)) if n >= 5 and n - c >= 5 else (1.0 if c > 0 else 0.0),
                'pass_at_10': 1.0 - math.comb(n - c, min(10, n)) / math.comb(n, min(10, n)) if n >= 10 and n - c >= 10 else (1.0 if c > 0 else 0.0),
            }

    agg = {
        'num_problems': len(all_results),
        'mean_pass_at_1': sum(r['pass_at_1'] for r in all_results.values()) / len(all_results) if all_results else 0,
        'mean_pass_at_5': sum(r['pass_at_5'] for r in all_results.values()) / len(all_results) if all_results else 0,
        'mean_pass_at_10': sum(r['pass_at_10'] for r in all_results.values()) / len(all_results) if all_results else 0,
        'per_problem': all_results,
    }
    agg_file = OUTPUT_DIR.parent / 'pass_at_k_results.json'
    with open(agg_file, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAGGREGATE pass@k ({len(all_results)} problems):", flush=True)
    print(f"  pass@1:  {agg['mean_pass_at_1']:.4f}", flush=True)
    print(f"  pass@5:  {agg['mean_pass_at_5']:.4f}", flush=True)
    print(f"  pass@10: {agg['mean_pass_at_10']:.4f}", flush=True)
PYTHON_SCRIPT

echo "Task ${TASK_ID} done: $(date)"
