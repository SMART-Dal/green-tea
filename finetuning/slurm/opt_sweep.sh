#!/bin/bash
#SBATCH --job-name=opt_sweep
#SBATCH --account=def-tusharma_gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-29
#SBATCH --output=logs/opt_sweep_%A_%a.out
#SBATCH --error=logs/opt_sweep_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=saurabh@dal.ca

export PROJECT_ROOT="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen"
export SNIPER_ROOT="${PROJECT_ROOT}/sniper/sniper"
export PIE_DATASET="${PROJECT_ROOT}/PIE_Dataset"
export CODES_FILE="${PROJECT_ROOT}/finetuning/data/opt_sweep_codes.json"
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/opt_sweep_results"
export TASK_ID=${SLURM_ARRAY_TASK_ID}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10
cd "${PROJECT_ROOT}/finetuning"
source ../config.env
mkdir -p "${OUTPUT_DIR}" logs
source "${PROJECT_ROOT}/venv/bin/activate"

echo "opt_sweep task ${TASK_ID} | $(date)"
for p in "$SNIPER_ROOT" "$PIE_DATASET" "$CODES_FILE"; do
    [ ! -e "$p" ] && echo "ERROR: $p not found" && exit 1
done

python3 -u << 'PYTHON_SCRIPT'
import json, os, sys, time, tempfile, shutil, subprocess
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
TASK_ID = int(os.environ['TASK_ID'])
SNIPER_ROOT = Path(os.environ['SNIPER_ROOT'])
PIE_ROOT = Path(os.environ['PIE_DATASET'])
CODES_FILE = os.environ['CODES_FILE']
OUTPUT_DIR = Path(os.environ['OUTPUT_DIR'])
SNIPER_CONFIG = str(SNIPER_ROOT / 'config' / 'epyc_9554p.cfg')
SNIPER_TIMEOUT = 120
OPT_LEVELS = ['-O0', '-O1', '-O2', '-O3']
N_INPUTS = 3   # use up to 3 test inputs per problem (Sniper is deterministic)

def get_inputs(pid, n):
    d = PIE_ROOT / 'extracted_testcases' / 'merged_test_cases' / pid
    if not d.exists(): return []
    pairs = []
    for f in sorted(d.glob('input.*.txt'))[:n]:
        out_f = d / f"output.{f.name.replace('input.', '')}"
        if out_f.exists():
            pairs.append((f.read_text().strip(), out_f.read_text().strip()))
    return pairs

def compile_at(code, opt, out_bin):
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False, mode='w') as f:
        cpp = Path(f.name); f.write(code)
    try:
        cmd = ['g++', opt, '-std=c++17', '-static', str(cpp), '-o', str(out_bin)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.returncode == 0, (r.stderr[:200] if r.returncode else '')
    finally:
        cpp.unlink(missing_ok=True)

def run_sniper_one(binary, inp):
    tdir = tempfile.mkdtemp(prefix='sn_')
    try:
        out_d = Path(tdir) / 'out'; out_d.mkdir()
        cmd = [str(SNIPER_ROOT / 'run-sniper'), '-c', SNIPER_CONFIG, '-d', str(out_d), '--power', '--', str(binary)]
        r = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=SNIPER_TIMEOUT)
        if r.returncode != 0: return None
        cycles = instructions = 0; energy = 0.0
        sim_out = out_d / 'sim.out'
        if sim_out.exists():
            for line in open(sim_out):
                if line.strip().startswith("Instructions") and "|" in line:
                    try: instructions = int(line.split("|")[1].strip())
                    except: pass
                elif line.strip().startswith("Cycles") and "|" in line:
                    try: cycles = int(line.split("|")[1].strip())
                    except: pass
        parsing = False
        for line in r.stdout.split('\n'):
            if "Power" in line and "Energy" in line and "Energy %" in line: parsing = True; continue
            if parsing and "total" in line.lower():
                try: energy = float(line.split()[3]); break
                except: pass
        return {'energy': energy, 'cycles': cycles, 'instructions': instructions}
    except subprocess.TimeoutExpired: return None
    except Exception: return None
    finally: shutil.rmtree(tdir, ignore_errors=True)

records = json.load(open(CODES_FILE))
if TASK_ID >= len(records):
    print(f"task {TASK_ID} >= {len(records)} records, exiting"); sys.exit(0)
rec = records[TASK_ID]
pid, cat, bcode, gcode = rec['problem_id'], rec['category'], rec['baseline_code'], rec['generated_code']
out_file = OUTPUT_DIR / f"opt_sweep_{TASK_ID:02d}_{pid}.json"
if out_file.exists():
    print(f"already done: {out_file.name}"); sys.exit(0)

inputs = get_inputs(pid, N_INPUTS)
if not inputs:
    print(f"no inputs for {pid}, skipping"); sys.exit(0)

result = {'problem_id': pid, 'category': cat, 'err_at_O3_reference': rec.get('err_at_O3', 0),
          'opts': {}, 'n_inputs_used': len(inputs)}
t0 = time.time()
for opt in OPT_LEVELS:
    cell = {'compile_b': False, 'compile_g': False, 'b_runs': [], 'g_runs': []}
    tdir = tempfile.mkdtemp(prefix='cmp_')
    try:
        bbin = Path(tdir) / 'b.bin'; gbin = Path(tdir) / 'g.bin'
        cb_ok, _ = compile_at(bcode, opt, bbin); cell['compile_b'] = cb_ok
        cg_ok, _ = compile_at(gcode, opt, gbin); cell['compile_g'] = cg_ok
        if not (cb_ok and cg_ok):
            result['opts'][opt] = cell; continue
        for inp, _ in inputs:
            for tag, bn in (('b', bbin), ('g', gbin)):
                m = run_sniper_one(bn, inp)
                if m is not None: cell[f'{tag}_runs'].append(m)
    finally:
        shutil.rmtree(tdir, ignore_errors=True)
    if cell['b_runs']:
        cell['b_energy_mean'] = sum(r['energy'] for r in cell['b_runs']) / len(cell['b_runs'])
        cell['b_cycles_mean'] = sum(r['cycles'] for r in cell['b_runs']) / len(cell['b_runs'])
        cell['b_insn_mean']   = sum(r['instructions'] for r in cell['b_runs']) / len(cell['b_runs'])
    if cell['g_runs']:
        cell['g_energy_mean'] = sum(r['energy'] for r in cell['g_runs']) / len(cell['g_runs'])
        cell['g_cycles_mean'] = sum(r['cycles'] for r in cell['g_runs']) / len(cell['g_runs'])
        cell['g_insn_mean']   = sum(r['instructions'] for r in cell['g_runs']) / len(cell['g_runs'])
    if cell.get('b_energy_mean', 0) > 0 and cell.get('g_energy_mean', 0) > 0:
        cell['err_pct'] = (cell['b_energy_mean'] - cell['g_energy_mean']) / cell['b_energy_mean'] * 100
    print(f"  {pid} {opt}: ERR={cell.get('err_pct', float('nan')):.2f}%  "
          f"b_runs={len(cell['b_runs'])} g_runs={len(cell['g_runs'])}  ({time.time()-t0:.0f}s)")
    result['opts'][opt] = cell

out_file.write_text(json.dumps(result, indent=2))
print(f"\nTask {TASK_ID} done: {pid}/{cat}  -> {out_file.name}  ({time.time()-t0:.0f}s)")
PYTHON_SCRIPT

echo "opt_sweep task ${TASK_ID} done: $(date)"
