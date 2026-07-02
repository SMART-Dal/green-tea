#!/bin/bash
#SBATCH --job-name=sft-runtime-sim
#SBATCH --account=rrg-mrdal22
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-38
#SBATCH --output=logs/sft_runtime_sim_%A_%a.out
#SBATCH --error=logs/sft_runtime_sim_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# W1 ablation: Sniper/McPAT simulation for runtime-SFT generated outputs
# Prereq: eval_sft_runtime_gen.sh completed
# Next: python3 analysis/rq1/rq1.py (runtime-SFT results auto-loaded if RUNTIME_SFT_DIR set)

export PROJECT_ROOT="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen"
export SNIPER_ROOT="${PROJECT_ROOT}/sniper/sniper"
export PIE_DATASET="${PROJECT_ROOT}/PIE_Dataset"
export GENERATIONS_FILE="${PROJECT_ROOT}/finetuning/data/sft_runtime_generations_test/eval_generations_test.jsonl"
export DATASET_FILE="${PROJECT_ROOT}/finetuning/data/sft_pairs_test.jsonl"
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/sft_runtime_sim_results"
export TASK_ID=${SLURM_ARRAY_TASK_ID}
export NUM_TASKS=39
export SAMPLES_PER_TASK=32

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10

cd "${PROJECT_ROOT}/finetuning"
source ../config.env
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/sft_runtime_sim_results"
mkdir -p "${OUTPUT_DIR}" logs

VENV="${PROJECT_ROOT}/venv"
source "${VENV}/bin/activate"

echo "========================================="
echo "SFT RUNTIME SIM: Task ${TASK_ID}/${NUM_TASKS}"
echo "Generations: ${GENERATIONS_FILE}"
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
import json, sys, os, time, hashlib
import numpy as np
import subprocess as _sub
import tempfile as _tmp
import shutil as _shu
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
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
    step: int; problem_id: str; num_inputs: int; status: str; compiled: bool
    tests_passed: int; baseline_energy: float; baseline_avg_cycles: int
    baseline_avg_instructions: int; baseline_avg_ipc: float; baseline_edp: float
    generated_success_count: int; generated_energy: float; generated_avg_cycles: int
    generated_avg_instructions: int; generated_avg_ipc: float; generated_edp: float
    energy_reduction: float; edp_reduction: float; speedup: float
    ipc_improvement_pct: float; optimized_energy: float; optimized_cycles: int
    optimized_ipc: float; optimized_edp: float; vs_gt_reduction: float
    baseline_code: str = None; generated_code: str = None
    optimized_code: str = None; compile_error: str = None

def run_sniper(binary_path, test_input):
    tdir = _tmp.mkdtemp(prefix='sniper_')
    try:
        out_dir = Path(tdir) / 'out'; out_dir.mkdir()
        inp_file = Path(tdir) / 'inp.txt'; inp_file.write_text(test_input)
        cmd = [str(SNIPER_ROOT / 'run-sniper'), '-c', SNIPER_CONFIG, '-d', str(out_dir), '--power', '--', binary_path]
        with open(inp_file) as inp:
            r = _sub.run(cmd, stdin=inp, capture_output=True, text=True, timeout=SNIPER_TIMEOUT)
        if r.returncode != 0: return {'status': 'runtime_error'}
        cycles = instructions = 0
        sim_out = out_dir / 'sim.out'
        if sim_out.exists():
            for line in open(sim_out):
                line = line.strip()
                if line.startswith("Instructions") and "|" in line:
                    try: instructions = int(line.split("|")[1].strip())
                    except: pass
                elif line.startswith("Cycles") and "|" in line:
                    try: cycles = int(line.split("|")[1].strip())
                    except: pass
        energy = 0.0; parsing = False
        for line in r.stdout.split('\n'):
            if "Power" in line and "Energy" in line and "Energy %" in line: parsing = True; continue
            if parsing and "total" in line.lower():
                try: parts = line.split(); energy = float(parts[3]) if len(parts) >= 4 else 0; break
                except: pass
        return {'status': 'success', 'energy_joules': energy, 'cycles': cycles, 'instructions': instructions}
    except _sub.TimeoutExpired: return {'status': 'timeout'}
    except Exception as e: return {'status': 'error', 'error': str(e)[:200]}
    finally: _shu.rmtree(tdir, ignore_errors=True)

def get_test_io(pid):
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
print(f"Task {TASK_ID}: [{offset}, {offset+len(chunk)}) of {len(generations)}", flush=True)
if not chunk: print("Nothing to process"); sys.exit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / f'test_comparison_chunk_{TASK_ID}.jsonl'

completed_pids = set()
if output_file.exists():
    for line in open(output_file): completed_pids.add(json.loads(line)['problem_id'])
    print(f"Resuming: {len(completed_pids)} done", flush=True)

code_hash_to_samples = defaultdict(list)
for gen in chunk:
    if gen['problem_id'] in completed_pids: continue
    h = hashlib.md5((gen['baseline_code'] + '|||' + gen['generated_code']).encode()).hexdigest()
    code_hash_to_samples[h].append(gen)

unique_hashes = list(code_hash_to_samples.keys())
results = []; start_time = time.time(); processed = 0

for idx, code_hash in enumerate(unique_hashes, 1):
    samples_for_hash = code_hash_to_samples[code_hash]
    gen = samples_for_hash[0]
    pid, baseline_code, generated_code = gen['problem_id'], gen['baseline_code'], gen['generated_code']
    test_io = get_test_io(pid)
    if not test_io: continue

    sample = test_lookup.get(baseline_code, {})
    b_energy = sample.get('baseline_energy', 0); b_cycles = sample.get('baseline_cycles', 0)
    b_instrs = sample.get('baseline_instructions', 0)
    b_ipc = sample.get('baseline_ipc', 0) or (b_instrs / b_cycles if b_cycles else 0)
    b_runtime = b_cycles / 1.5e9 if b_cycles > 0 else 0; b_edp = b_energy * b_runtime
    opt_energy = sample.get('optimized_energy', 0); opt_cycles = sample.get('optimized_cycles', 0)
    opt_ipc = sample.get('optimized_ipc', 0); opt_edp = sample.get('optimized_edp', 0)

    tdir = _tmp.mkdtemp(prefix='tc_')
    compiled = False; compile_err = ''; tests_passed = 0; g_e, g_c, g_i = [], [], []
    try:
        cpp = Path(tdir) / 's.cpp'; cpp.write_text(generated_code)
        bn = Path(tdir) / 's.bin'
        cr = _sub.run(['g++', '-O3', '-std=c++17', '-static', str(cpp), '-o', str(bn)], capture_output=True, text=True, timeout=10)
        compiled = cr.returncode == 0
        if not compiled: compile_err = cr.stderr[:500] if cr.stderr else 'Unknown'
        else:
            for inp_txt, exp_out in test_io:
                try:
                    r = _sub.run([str(bn)], input=inp_txt, capture_output=True, text=True, timeout=5)
                    if r.returncode != 0 or r.stdout.strip() != exp_out: continue
                except: continue
                tests_passed += 1
                sim = run_sniper(str(bn), inp_txt)
                if sim['status'] == 'success' and sim.get('energy_joules', 0) > 1e-9:
                    g_e.append(sim['energy_joules']); g_c.append(sim['cycles']); g_i.append(sim['instructions'])
    except Exception as ex: print(f"  [{pid}] Error: {ex}", flush=True)
    finally: _shu.rmtree(tdir, ignore_errors=True)

    ge = float(np.mean(g_e)) if g_e else 0; gc = int(np.mean(g_c)) if g_c else 0
    g_instr = int(np.mean(g_i)) if g_i else 0
    gi_ipc = float(np.mean([i/c for i,c in zip(g_i, g_c) if c])) if g_i else 0
    g_runtime = gc / 1.5e9 if gc > 0 else 0; g_edp = ge * g_runtime

    status = 'compile_error' if not compiled else ('correctness_error' if tests_passed == 0 else ('simulation_error' if not g_e else 'success'))

    for sg in samples_for_hash:
        sd = test_lookup.get(sg['baseline_code'], {})
        sb_e = sd.get('baseline_energy', 0); sb_c = sd.get('baseline_cycles', 0)
        sb_i = sd.get('baseline_instructions', 0)
        sb_ipc = sd.get('baseline_ipc', 0) or (sb_i / sb_c if sb_c else 0)
        sb_rt = sb_c / 1.5e9 if sb_c > 0 else 0; sb_edp = sb_e * sb_rt
        so_e = sd.get('optimized_energy', 0); so_c = sd.get('optimized_cycles', 0)
        so_ipc = sd.get('optimized_ipc', 0); so_edp = sd.get('optimized_edp', 0)

        s_err = (sb_e - ge) / sb_e * 100 if (sb_e > 1e-9 and ge > 1e-9) else 0.0
        s_edp_red = (sb_edp - g_edp) / sb_edp * 100 if (sb_edp > 1e-9 and g_edp > 1e-9) else 0.0
        s_speedup = sb_c / gc if gc > 0 else 0
        s_ipc_imp = (gi_ipc - sb_ipc) / sb_ipc * 100 if sb_ipc > 0 else 0
        s_vs_gt = ((so_e - ge) / so_e * 100) if (so_e > 1e-9 and ge > 1e-9) else 0.0

        result = ComparisonResult(
            step=0, problem_id=sg['problem_id'], num_inputs=len(test_io),
            status=status, compiled=compiled, tests_passed=tests_passed,
            baseline_energy=sb_e, baseline_avg_cycles=sb_c, baseline_avg_instructions=sb_i,
            baseline_avg_ipc=sb_ipc, baseline_edp=sb_edp, generated_success_count=len(g_e),
            generated_energy=ge, generated_avg_cycles=gc, generated_avg_instructions=g_instr,
            generated_avg_ipc=gi_ipc, generated_edp=g_edp, energy_reduction=s_err,
            edp_reduction=s_edp_red, speedup=s_speedup, ipc_improvement_pct=s_ipc_imp,
            optimized_energy=so_e, optimized_cycles=so_c, optimized_ipc=so_ipc,
            optimized_edp=so_edp, vs_gt_reduction=s_vs_gt,
            baseline_code=sg['baseline_code'], generated_code=generated_code,
            optimized_code=sg.get('optimized_code', ''),
            compile_error=compile_err if not compiled else None)
        results.append(result)
        with open(output_file, 'a') as f: f.write(json.dumps(asdict(result)) + '\n')
        processed += 1

    if idx % 5 == 0 or idx == len(unique_hashes):
        elapsed = time.time() - start_time
        print(f"  [{idx}/{len(unique_hashes)}] -> {processed} total ({elapsed:.0f}s)", flush=True)

paired = [r for r in results if r.baseline_energy > 1e-9 and r.generated_energy > 1e-9]
err_vals = [r.energy_reduction for r in paired]
summary = {
    'task_id': TASK_ID, 'samples_processed': len(results),
    'compile_rate': sum(1 for r in results if r.compiled) / len(results) if results else 0,
    'mean_energy_reduction_pct': float(np.mean(err_vals)) if err_vals else 0,
    'median_energy_reduction_pct': float(np.median(err_vals)) if err_vals else 0,
}
with open(OUTPUT_DIR / f'test_summary_chunk_{TASK_ID}.json', 'w') as f: json.dump(summary, f, indent=2)
elapsed = time.time() - start_time
print(f"\nTask {TASK_ID} complete: {len(results)} samples in {elapsed:.0f}s", flush=True)
print(f"  compile={summary['compile_rate']:.0%} ERR={summary['mean_energy_reduction_pct']:.2f}%", flush=True)
PYTHON_SCRIPT

echo "Task ${TASK_ID} done: $(date)"
