#!/bin/bash
#SBATCH --job-name=eval-comparison
#SBATCH --account=rrg-mrdal22
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/eval_comparison_%j.out
#SBATCH --error=logs/eval_comparison_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# Val-set comparison: reads precomputed metrics from sft_generations_base/
# All metrics already written by EnergyEvaluationCallback during SFT training
# No simulation needed. Loops over all step files sequentially.

export PROJECT_ROOT="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen"
export GENERATIONS_DIR="${PROJECT_ROOT}/finetuning/data/sft_generations_base"
export OUTPUT_DIR="${PROJECT_ROOT}/finetuning/data/evaluation_results"

mkdir -p "${OUTPUT_DIR}" logs

module --force purge
module load StdEnv/2023 python/3.10

cd "${PROJECT_ROOT}/finetuning"
source ../config.env

VENV="${PROJECT_ROOT}/venv"
source "${VENV}/bin/activate"

echo "========================================="
echo "VAL EVAL COMPARISON"
echo "Generations: $(ls ${GENERATIONS_DIR}/eval_generations_step*.jsonl 2>/dev/null | wc -l) step files"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "========================================="

if [ ! -d "${GENERATIONS_DIR}" ]; then
    echo "ERROR: Generations dir not found: ${GENERATIONS_DIR}"
    exit 1
fi

export GENERATIONS_DIR OUTPUT_DIR

python3 -u << 'PYTHON_SCRIPT'
import json, os, sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

sys.stdout.reconfigure(line_buffering=True)

@dataclass
class ComparisonResult:
    step: int
    problem_id: str
    num_inputs: int
    baseline_compile: bool
    baseline_success_count: int
    baseline_avg_energy: float
    baseline_avg_cycles: int
    baseline_avg_ipc: float
    generated_compile: bool
    generated_success_count: int
    generated_avg_energy: float
    generated_avg_cycles: int
    generated_avg_ipc: float
    energy_reduction_pct: float
    speedup: float
    ipc_improvement_pct: float
    baseline_code: str = None
    generated_code: str = None
    error_msg: str = None
    optimized_energy: float = 0.0
    vs_gt_reduction: float = 0.0
    tests_passed: int = 0

def process_step(gen_file: Path, output_dir: Path):
    """Process one step file. All metrics precomputed by EnergyEvaluationCallback."""
    step = int(gen_file.stem.replace('eval_generations_step', ''))
    records = [json.loads(l) for l in open(gen_file)]
    print(f"Step {step}: {len(records)} records", flush=True)

    results = []
    for rec in records:
        compiled = rec.get('compiled', rec.get('status') != 'compile_error')
        gen_energy = rec.get('generated_energy', 0)
        gen_cycles = rec.get('generated_avg_cycles', 0)
        gen_ipc = rec.get('generated_avg_ipc', 0)
        gen_success = rec.get('generated_success_count', 1 if gen_energy > 1e-9 else 0)

        b_energy = rec.get('baseline_energy', 0)
        b_cycles = rec.get('baseline_avg_cycles', 0)
        b_ipc = rec.get('baseline_avg_ipc', 0)
        opt_energy = rec.get('optimized_energy', 0)

        err = (b_energy - gen_energy) / b_energy * 100 if b_energy > 1e-9 else 0
        speedup = b_cycles / gen_cycles if gen_cycles > 0 else 0
        ipc_imp = (gen_ipc - b_ipc) / b_ipc * 100 if b_ipc > 0 else 0
        vs_gt = ((opt_energy - gen_energy) / opt_energy * 100) if (opt_energy > 1e-9 and gen_energy > 1e-9) else 0.0

        results.append(ComparisonResult(
            step=step,
            problem_id=rec.get('problem_id', ''),
            num_inputs=rec.get('num_inputs', 0),
            baseline_compile=True,
            baseline_success_count=1 if b_energy > 1e-9 else 0,
            baseline_avg_energy=b_energy,
            baseline_avg_cycles=b_cycles,
            baseline_avg_ipc=b_ipc,
            generated_compile=compiled,
            generated_success_count=gen_success,
            generated_avg_energy=gen_energy,
            generated_avg_cycles=gen_cycles,
            generated_avg_ipc=gen_ipc,
            energy_reduction_pct=err,
            speedup=speedup,
            ipc_improvement_pct=ipc_imp,
            baseline_code=rec.get('baseline_code'),
            generated_code=rec.get('generated_code'),
            error_msg=rec.get('compile_error', rec.get('error_msg')),
            optimized_energy=opt_energy,
            vs_gt_reduction=vs_gt,
            tests_passed=rec.get('tests_passed', 0),
        ))

    # Save per-step comparison
    output_file = output_dir / f'comparison_step{step}.jsonl'
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')

    # Aggregate stats
    paired = [r for r in results if r.baseline_avg_energy > 1e-9 and r.generated_avg_energy > 1e-9]
    err_vals = [r.energy_reduction_pct for r in paired]
    vs_gt_vals = [r.vs_gt_reduction for r in results if r.optimized_energy > 1e-9 and r.generated_success_count > 0]
    total_tests = sum(r.num_inputs for r in results)
    total_passed = sum(r.tests_passed for r in results)

    summary = {
        'step': step,
        'total_samples': len(results),
        'compile_rate': sum(1 for r in results if r.generated_compile) / len(results) if results else 0,
        'success_rate': sum(1 for r in results if r.generated_success_count > 0) / len(results) if results else 0,
        'correctness_rate': total_passed / total_tests if total_tests else 0,
        'total_tests_passed': total_passed,
        'total_tests': total_tests,
        'mean_energy_reduction_pct': float(np.mean(err_vals)) if err_vals else 0,
        'median_energy_reduction_pct': float(np.median(err_vals)) if err_vals else 0,
        'num_improvements': sum(1 for e in err_vals if e > 0),
        'num_regressions': sum(1 for e in err_vals if e < -5),
        'mean_vs_gt_reduction_pct': float(np.mean(vs_gt_vals)) if vs_gt_vals else 0,
        'num_beats_gt': sum(1 for v in vs_gt_vals if v > 0),
        'num_vs_gt_compared': len(vs_gt_vals),
    }

    summary_file = output_dir / f'summary_step{step}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Step {step}: paired={len(paired)} ERR={summary['mean_energy_reduction_pct']:.2f}% "
          f"vs_GT={summary['mean_vs_gt_reduction_pct']:.2f}% beats={summary['num_beats_gt']}/{summary['num_vs_gt_compared']}", flush=True)
    return summary

def main():
    generations_dir = Path(os.environ['GENERATIONS_DIR'])
    output_dir = Path(os.environ['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_files = sorted(generations_dir.glob('eval_generations_step*.jsonl'),
                       key=lambda p: int(p.stem.replace('eval_generations_step', '')))
    if not gen_files:
        print(f"ERROR: No generation files in {generations_dir}", flush=True)
        sys.exit(1)

    print(f"Processing {len(gen_files)} step files...", flush=True)
    all_summaries = []
    for gf in gen_files:
        all_summaries.append(process_step(gf, output_dir))

    # Cross-step summary
    print(f"\n{'='*70}", flush=True)
    print(f"VAL COMPARISON SUMMARY ({len(all_summaries)} steps)", flush=True)
    print(f"{'='*70}", flush=True)
    for s in all_summaries:
        print(f"  Step {s['step']:5d}: ERR={s['mean_energy_reduction_pct']:7.2f}% "
              f"compile={s['compile_rate']:.0%} correct={s['total_tests_passed']}/{s['total_tests']} "
              f"beats_GT={s['num_beats_gt']}/{s['num_vs_gt_compared']}", flush=True)
    print(f"{'='*70}\n", flush=True)

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

echo ""
echo "========================================="
echo "EVAL COMPARISON COMPLETE"
echo "Results: ${OUTPUT_DIR}/"
echo "End: $(date)"
echo "========================================="
