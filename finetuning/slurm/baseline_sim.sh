#!/bin/bash
#SBATCH --job-name=baseline-sim
#SBATCH --account=rrg-mrdal22
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-38
#SBATCH --output=logs/baseline_sim_%A_%a.out
#SBATCH --error=logs/baseline_sim_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Baseline simulation: compile -> correctness -> Sniper/McPAT -> comparison metrics
# VARIANT selects which baseline to simulate:
#   base model:     zero_shot | green_prompt
#   instruct model: zero_shot_instruct | green_prompt_instruct
# Submit: VARIANT=zero_shot_instruct sbatch slurm/baseline_sim.sh
#         VARIANT=green_prompt_instruct sbatch slurm/baseline_sim.sh
# Output: finetuning/data/baseline_sim_results/{VARIANT}/test_comparison_chunk_{N}.jsonl
# Prereq: sft_baseline.sh must have completed for the given VARIANT

export PROJECT_ROOT="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen"
export SNIPER_ROOT="${PROJECT_ROOT}/sniper/sniper"
export PIE_DATASET="${PROJECT_ROOT}/PIE_Dataset"

VARIANT="${VARIANT:-zero_shot}"
export GENERATIONS_FILE="${PROJECT_ROOT}/finetuning/data/sft_baseline_generations/${VARIANT}/generations.jsonl"
export DATASET_FILE="${PROJECT_ROOT}/finetuning/data/sft_pairs_test.jsonl"
export TASK_ID=${SLURM_ARRAY_TASK_ID}
export NUM_TASKS=39
export SAMPLES_PER_TASK=32

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10

mkdir -p logs

cd "${PROJECT_ROOT}/finetuning"
source ../config.env
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/baseline_sim_results/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

VENV="${PROJECT_ROOT}/venv"
source "${VENV}/bin/activate"

echo "========================================="
echo "BASELINE SIMULATION: ${VARIANT}"
echo "Task: ${TASK_ID}/${NUM_TASKS} (samples $((TASK_ID * SAMPLES_PER_TASK))-$(( (TASK_ID+1) * SAMPLES_PER_TASK - 1 )))"
echo "Sniper: ${SNIPER_ROOT}"
echo "Generations: ${GENERATIONS_FILE}"
echo "Dataset: ${DATASET_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "========================================="

for p in "$SNIPER_ROOT" "$PIE_DATASET"; do
    [ ! -d "$p" ] && echo "ERROR: $p not found" && exit 1
done
for p in "$GENERATIONS_FILE" "$DATASET_FILE"; do
    [ ! -f "$p" ] && echo "ERROR: $p not found" && exit 1
done

export TASK_ID NUM_TASKS SAMPLES_PER_TASK SNIPER_ROOT PIE_DATASET GENERATIONS_FILE DATASET_FILE OUTPUT_DIR

python3 -u << 'PYTHON_SCRIPT'
import json, sys, os, time
import numpy as np
import subprocess as _sub
import tempfile as _tmp
import shutil as _shu
from pathlib import Path
from dataclasses import dataclass, asdict

sys.stdout.reconfigure(line_buffering=True)

TASK_ID = int(os.environ['TASK_ID'])
NUM_TASKS = int(os.environ['NUM_TASKS'])
SAMPLES_PER_TASK = int(os.environ['SAMPLES_PER_TASK'])
SNIPER_ROOT = Path(os.environ['SNIPER_ROOT'])
PIE_ROOT = Path(os.environ['PIE_DATASET'])
GENERATIONS_FILE = os.environ['GENERATIONS_FILE']
DATASET_FILE = os.environ['DATASET_FILE']
OUTPUT_DIR = Path(os.environ['OUTPUT_DIR'])

SNIPER_CONFIG = str(SNIPER_ROOT / 'config' / 'epyc_9554p.cfg')
SNIPER_TIMEOUT = 120

@dataclass
class ComparisonResult:
    step: int
    problem_id: str
    num_inputs: int
    status: str
    compiled: bool
    tests_passed: int
    baseline_energy: float
    baseline_avg_cycles: int
    baseline_avg_instructions: int
    baseline_avg_ipc: float
    baseline_edp: float
    generated_success_count: int
    generated_energy: float
    generated_avg_cycles: int
    generated_avg_instructions: int
    generated_avg_ipc: float
    generated_edp: float
    energy_reduction: float
    edp_reduction: float
    speedup: float
    ipc_improvement_pct: float
    optimized_energy: float
    optimized_cycles: int
    optimized_ipc: float
    optimized_edp: float
    vs_gt_reduction: float
    baseline_code: str = None
    generated_code: str = None
    optimized_code: str = None
    compile_error: str = None

def run_sniper(binary_path: str, test_input: str) -> dict:
    tdir = _tmp.mkdtemp(prefix='sniper_')
    try:
        out_dir = Path(tdir) / 'out'
        out_dir.mkdir()
        inp_file = Path(tdir) / 'inp.txt'
        inp_file.write_text(test_input)
        cmd = [str(SNIPER_ROOT / 'run-sniper'), '-c', SNIPER_CONFIG,
               '-d', str(out_dir), '--power', '--', binary_path]
        with open(inp_file) as inp:
            r = _sub.run(cmd, stdin=inp, capture_output=True, text=True, timeout=SNIPER_TIMEOUT)
        if r.returncode != 0:
            return {'status': 'runtime_error'}
        cycles = instructions = 0
        sim_out = out_dir / 'sim.out'
        if sim_out.exists():
            for line in open(sim_out):
                line = line.strip()
                if line.startswith("Instructions") and "|" in line:
                    try: instructions = int(line.split("|")[1].strip())
                    except (ValueError, IndexError): pass
                elif line.startswith("Cycles") and "|" in line:
                    try: cycles = int(line.split("|")[1].strip())
                    except (ValueError, IndexError): pass
        energy = 0.0
        parsing = False
        for line in r.stdout.split('\n'):
            if "Power" in line and "Energy" in line and "Energy %" in line:
                parsing = True
                continue
            if parsing and "total" in line.lower():
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        energy = float(parts[3])
                    break
                except (ValueError, IndexError):
                    pass
        return {'status': 'success', 'energy_joules': energy, 'cycles': cycles, 'instructions': instructions}
    except _sub.TimeoutExpired:
        return {'status': 'timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:200]}
    finally:
        _shu.rmtree(tdir, ignore_errors=True)

def get_test_io(pid: str):
    d = PIE_ROOT / 'extracted_testcases' / 'merged_test_cases' / pid
    if not d.exists(): return []
    return [(f.read_text().strip(), (d / f"output.{f.name.replace('input.', '')}").read_text().strip())
            for f in sorted(d.glob('input.*.txt'))
            if (d / f"output.{f.name.replace('input.', '')}").exists()]

generations = [json.loads(l) for l in open(GENERATIONS_FILE)]
test_lookup = {}
for rec in (json.loads(l) for l in open(DATASET_FILE)):
    test_lookup[rec['inefficient_code']] = rec

offset = TASK_ID * SAMPLES_PER_TASK
chunk = generations[offset:offset + SAMPLES_PER_TASK]
print(f"Task {TASK_ID}: samples [{offset}, {offset+len(chunk)}) of {len(generations)} total", flush=True)

if not chunk:
    print(f"Task {TASK_ID}: nothing to process", flush=True)
    sys.exit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / f'test_comparison_chunk_{TASK_ID}.jsonl'

completed_pids = set()
if output_file.exists():
    with open(output_file) as f:
        for line in f:
            rec = json.loads(line)
            completed_pids.add(rec['problem_id'])
    print(f"Task {TASK_ID}: {len(completed_pids)} already done, resuming...", flush=True)

import hashlib
from collections import defaultdict

code_hash_to_samples = defaultdict(list)
for gen in chunk:
    if gen['problem_id'] in completed_pids:
        continue
    code_hash = hashlib.md5((gen['baseline_code'] + '|||' + gen['generated_code']).encode()).hexdigest()
    code_hash_to_samples[code_hash].append(gen)

unique_hashes = list(code_hash_to_samples.keys())
remaining = sum(len(s) for s in code_hash_to_samples.values())
print(f"Chunk {TASK_ID}: {remaining} remaining -> {len(unique_hashes)} unique pairs (skipped {len(completed_pids)} done)", flush=True)

results = []
start_time = time.time()
processed = 0

for idx, code_hash in enumerate(unique_hashes, 1):
    samples_for_hash = code_hash_to_samples[code_hash]
    gen = samples_for_hash[0]
    pid = gen['problem_id']
    baseline_code = gen['baseline_code']
    generated_code = gen['generated_code']
    test_io = get_test_io(pid)

    if not test_io:
        print(f"  [{idx}/{len(unique_hashes)}] [{pid}] SKIP: no test IO", flush=True)
        continue

    sample = test_lookup.get(baseline_code, {})
    b_energy = sample.get('baseline_energy', 0)
    b_cycles = sample.get('baseline_cycles', 0)
    b_instructions = sample.get('baseline_instructions', 0)
    b_ipc = sample.get('baseline_ipc', 0) or (b_instructions / b_cycles if b_cycles else 0)
    b_runtime = b_cycles / 1.5e9 if b_cycles > 0 else 0
    b_edp = b_energy * b_runtime
    opt_energy = sample.get('optimized_energy', 0)
    opt_cycles = sample.get('optimized_cycles', 0)
    opt_ipc = sample.get('optimized_ipc', 0)
    opt_edp = sample.get('optimized_edp', 0)

    tdir = _tmp.mkdtemp(prefix='tc_')
    compiled = False
    compile_err = ''
    tests_passed = 0
    g_e, g_c, g_i = [], [], []
    try:
        cpp = Path(tdir) / 's.cpp'
        cpp.write_text(generated_code)
        bn = Path(tdir) / 's.bin'
        cr = _sub.run(['g++', '-O3', '-std=c++17', '-static', str(cpp), '-o', str(bn)],
                      capture_output=True, text=True, timeout=10)
        compiled = cr.returncode == 0
        if not compiled:
            compile_err = cr.stderr[:500] if cr.stderr else 'Unknown compilation error'
        else:
            for inp_txt, exp_out in test_io:
                try:
                    r = _sub.run([str(bn)], input=inp_txt, capture_output=True, text=True, timeout=5)
                    if r.returncode != 0 or r.stdout.strip() != exp_out:
                        continue
                except Exception:
                    continue
                tests_passed += 1
                sim = run_sniper(str(bn), inp_txt)
                if sim['status'] == 'success' and sim.get('energy_joules', 0) > 1e-9:
                    g_e.append(sim['energy_joules'])
                    g_c.append(sim['cycles'])
                    g_i.append(sim['instructions'])
    except Exception as ex:
        print(f"  [{pid}] Error: {ex}", flush=True)
    finally:
        _shu.rmtree(tdir, ignore_errors=True)

    ge = float(np.mean(g_e)) if g_e else 0
    gc = int(np.mean(g_c)) if g_c else 0
    g_instr = int(np.mean(g_i)) if g_i else 0
    gi_ipc = float(np.mean([i / c for i, c in zip(g_i, g_c) if c])) if g_i else 0
    g_runtime = gc / 1.5e9 if gc > 0 else 0
    g_edp = ge * g_runtime

    if not compiled:
        status = 'compile_error'
    elif tests_passed == 0:
        status = 'correctness_error'
    elif len(g_e) == 0:
        status = 'simulation_error'
    else:
        status = 'success'

    err = (b_energy - ge) / b_energy * 100 if (b_energy > 1e-9 and ge > 1e-9) else 0.0
    edp_red = (b_edp - g_edp) / b_edp * 100 if (b_edp > 1e-9 and g_edp > 1e-9) else 0.0
    speedup = b_cycles / gc if gc > 0 else 0
    ipc_imp = (gi_ipc - b_ipc) / b_ipc * 100 if b_ipc > 0 else 0
    vs_gt = ((opt_energy - ge) / opt_energy * 100) if (opt_energy > 1e-9 and ge > 1e-9) else 0.0

    for sample_gen in samples_for_hash:
        s_data = test_lookup.get(sample_gen['baseline_code'], {})
        s_b_energy = s_data.get('baseline_energy', 0)
        s_b_cycles = s_data.get('baseline_cycles', 0)
        s_b_instructions = s_data.get('baseline_instructions', 0)
        s_b_ipc = s_data.get('baseline_ipc', 0) or (s_b_instructions / s_b_cycles if s_b_cycles else 0)
        s_b_runtime = s_b_cycles / 1.5e9 if s_b_cycles > 0 else 0
        s_b_edp = s_b_energy * s_b_runtime
        s_opt_energy = s_data.get('optimized_energy', 0)
        s_opt_cycles = s_data.get('optimized_cycles', 0)
        s_opt_ipc = s_data.get('optimized_ipc', 0)
        s_opt_edp = s_data.get('optimized_edp', 0)

        s_err = (s_b_energy - ge) / s_b_energy * 100 if (s_b_energy > 1e-9 and ge > 1e-9) else 0.0
        s_edp_red = (s_b_edp - g_edp) / s_b_edp * 100 if (s_b_edp > 1e-9 and g_edp > 1e-9) else 0.0
        s_speedup = s_b_cycles / gc if gc > 0 else 0
        s_ipc_imp = (gi_ipc - s_b_ipc) / s_b_ipc * 100 if s_b_ipc > 0 else 0
        s_vs_gt = ((s_opt_energy - ge) / s_opt_energy * 100) if (s_opt_energy > 1e-9 and ge > 1e-9) else 0.0

        result = ComparisonResult(
            step=0, problem_id=sample_gen['problem_id'], num_inputs=len(test_io),
            status=status, compiled=compiled, tests_passed=tests_passed,
            baseline_energy=s_b_energy, baseline_avg_cycles=s_b_cycles,
            baseline_avg_instructions=s_b_instructions, baseline_avg_ipc=s_b_ipc,
            baseline_edp=s_b_edp, generated_success_count=len(g_e),
            generated_energy=ge, generated_avg_cycles=gc, generated_avg_instructions=g_instr,
            generated_avg_ipc=gi_ipc, generated_edp=g_edp,
            energy_reduction=s_err, edp_reduction=s_edp_red, speedup=s_speedup,
            ipc_improvement_pct=s_ipc_imp, optimized_energy=s_opt_energy,
            optimized_cycles=s_opt_cycles, optimized_ipc=s_opt_ipc, optimized_edp=s_opt_edp,
            vs_gt_reduction=s_vs_gt, baseline_code=sample_gen['baseline_code'],
            generated_code=generated_code, optimized_code=sample_gen.get('optimized_code', ''),
            compile_error=compile_err if not compiled else None,
        )
        results.append(result)
        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')
        processed += 1

    if idx % 5 == 0 or idx == len(unique_hashes):
        elapsed = time.time() - start_time
        err_display = f"{err:.1f}%" if len(g_e) > 0 else "N/A"
        print(f"  [{idx}/{len(unique_hashes)}] -> {processed} total | "
              f"[{gen['problem_id']}] compiled={compiled} passed={tests_passed}/{len(test_io)} "
              f"sniper={len(g_e)} ERR={err_display} ({elapsed:.0f}s)", flush=True)

paired = [r for r in results if r.baseline_energy > 1e-9 and r.generated_energy > 1e-9]
err_vals = [r.energy_reduction for r in paired]
edp_vals = [r.edp_reduction for r in paired]
vs_gt_vals = [r.vs_gt_reduction for r in results if r.optimized_energy > 1e-9 and r.generated_success_count > 0]
total_tests = sum(r.num_inputs for r in results)
total_passed = sum(r.tests_passed for r in results)

summary = {
    'task_id': TASK_ID,
    'samples_processed': len(results),
    'compile_rate': sum(1 for r in results if r.compiled) / len(results) if results else 0,
    'success_rate': sum(1 for r in results if r.generated_success_count > 0) / len(results) if results else 0,
    'correctness_rate': total_passed / total_tests if total_tests else 0,
    'total_tests_passed': total_passed,
    'total_tests': total_tests,
    'mean_energy_reduction_pct': float(np.mean(err_vals)) if err_vals else 0,
    'median_energy_reduction_pct': float(np.median(err_vals)) if err_vals else 0,
    'mean_edp_reduction_pct': float(np.mean(edp_vals)) if edp_vals else 0,
    'median_edp_reduction_pct': float(np.median(edp_vals)) if edp_vals else 0,
    'num_improvements': sum(1 for e in err_vals if e > 0),
    'num_regressions': sum(1 for e in err_vals if e < -5),
    'mean_vs_gt_reduction_pct': float(np.mean(vs_gt_vals)) if vs_gt_vals else 0,
    'num_beats_gt': sum(1 for v in vs_gt_vals if v > 0),
    'num_vs_gt_compared': len(vs_gt_vals),
}

summary_file = OUTPUT_DIR / f'test_summary_chunk_{TASK_ID}.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

elapsed = time.time() - start_time
print(f"\nTask {TASK_ID} complete: {len(results)} samples in {elapsed:.0f}s", flush=True)
print(f"  compile={summary['compile_rate']:.0%} correct={total_passed}/{total_tests} "
      f"ERR={summary['mean_energy_reduction_pct']:.2f}% beats_GT={summary['num_beats_gt']}/{summary['num_vs_gt_compared']}", flush=True)
print(f"  Results: {output_file}", flush=True)
PYTHON_SCRIPT

echo ""
echo "========================================="
echo "TASK ${TASK_ID} COMPLETE (${VARIANT})"
echo "End: $(date)"
echo "========================================="
