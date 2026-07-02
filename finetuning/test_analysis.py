#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TestMetrics:
    total_samples: int
    compile_rate: float
    correctness_rate: float
    success_rate: float
    energy_reduction_mean: float
    energy_reduction_median: float
    edp_reduction_mean: float
    edp_reduction_median: float
    speedup_mean: float
    speedup_median: float
    num_improvements: int
    num_regressions: int
    vs_gt_mean: float
    beats_gt_count: int
    total_tests: int
    tests_passed: int

def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for chunk_file in sorted(results_dir.glob('test_comparison_chunk_*.jsonl')):
        with open(chunk_file) as f:
            results.extend(json.loads(line) for line in f)
    return results

def analyze_results(results: List[Dict]) -> TestMetrics:
    compiled = [r for r in results if r['compiled']]
    successful = [r for r in results if r['generated_success_count'] > 0]
    paired = [r for r in successful if r['baseline_energy'] > 1e-9 and r['generated_energy'] > 1e-9]

    total_tests = sum(r['num_inputs'] for r in results)
    tests_passed = sum(r['tests_passed'] for r in results)

    energy_reductions = [r['energy_reduction'] for r in paired]
    edp_reductions = [r['edp_reduction'] for r in paired]
    speedups = [r['speedup'] for r in paired]
    vs_gt = [r['vs_gt_reduction'] for r in successful if r['optimized_energy'] > 1e-9]

    return TestMetrics(
        total_samples=len(results),
        compile_rate=len(compiled) / len(results),
        correctness_rate=tests_passed / total_tests if total_tests else 0,
        success_rate=len(successful) / len(results),
        energy_reduction_mean=float(np.mean(energy_reductions)) if energy_reductions else 0,
        energy_reduction_median=float(np.median(energy_reductions)) if energy_reductions else 0,
        edp_reduction_mean=float(np.mean(edp_reductions)) if edp_reductions else 0,
        edp_reduction_median=float(np.median(edp_reductions)) if edp_reductions else 0,
        speedup_mean=float(np.mean(speedups)) if speedups else 0,
        speedup_median=float(np.median(speedups)) if speedups else 0,
        num_improvements=sum(1 for e in energy_reductions if e > 0),
        num_regressions=sum(1 for e in energy_reductions if e < -5),
        vs_gt_mean=float(np.mean(vs_gt)) if vs_gt else 0,
        beats_gt_count=sum(1 for v in vs_gt if v > 0),
        total_tests=total_tests,
        tests_passed=tests_passed,
    )

def print_summary(metrics: TestMetrics, results: List[Dict]):
    print("=" * 80)
    print("TEST SET ANALYSIS - SFT Model Performance")
    print("=" * 80)
    print(f"\nDataset: {metrics.total_samples} samples, {metrics.total_tests} test inputs")

    print(f"\n{'COMPILATION & CORRECTNESS':-^80}")
    print(f"  Compilation rate:    {metrics.compile_rate:6.1%}  ({int(metrics.compile_rate * metrics.total_samples)}/{metrics.total_samples})")
    print(f"  Correctness rate:    {metrics.correctness_rate:6.1%}  ({metrics.tests_passed}/{metrics.total_tests} tests passed)")
    print(f"  Success rate:        {metrics.success_rate:6.1%}  (compiled + correct + simulated)")

    print(f"\n{'ENERGY OPTIMIZATION':-^80}")
    print(f"  Mean reduction:      {metrics.energy_reduction_mean:6.2f}%")
    print(f"  Median reduction:    {metrics.energy_reduction_median:6.2f}%")
    print(f"  Improvements:        {metrics.num_improvements} samples")
    print(f"  Regressions (>5%):   {metrics.num_regressions} samples")

    print(f"\n{'PERFORMANCE METRICS':-^80}")
    print(f"  Mean speedup:        {metrics.speedup_mean:6.3f}x")
    print(f"  Median speedup:      {metrics.speedup_median:6.3f}x")
    print(f"  Mean EDP reduction:  {metrics.edp_reduction_mean:6.2f}%")
    print(f"  Median EDP reduction:{metrics.edp_reduction_median:6.2f}%")

    print(f"\n{'VS GROUND TRUTH (OPTIMIZED)':-^80}")
    print(f"  Mean vs GT:          {metrics.vs_gt_mean:6.2f}%  (negative = worse than GT)")
    print(f"  Beats GT:            {metrics.beats_gt_count} samples")

    successful = [r for r in results if r['generated_success_count'] > 0]
    if successful:
        paired = [r for r in successful if r['baseline_energy'] > 1e-9 and r['generated_energy'] > 1e-9]
        energy_reductions = [r['energy_reduction'] for r in paired]

        print(f"\n{'DISTRIBUTION (Energy Reduction %)':-^80}")
        if energy_reductions:
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            pct_values = np.percentile(energy_reductions, percentiles)
            for p, v in zip(percentiles, pct_values):
                print(f"  P{p:3d}:  {v:7.2f}%")

        top_5 = sorted(paired, key=lambda x: x['energy_reduction'], reverse=True)[:5]
        print(f"\n{'TOP 5 IMPROVEMENTS':-^80}")
        for i, r in enumerate(top_5, 1):
            print(f"  {i}. {r['problem_id']}: {r['energy_reduction']:.2f}% energy, {r['speedup']:.3f}x speedup")

        bottom_5 = sorted(paired, key=lambda x: x['energy_reduction'])[:5]
        print(f"\n{'TOP 5 REGRESSIONS':-^80}")
        for i, r in enumerate(bottom_5, 1):
            print(f"  {i}. {r['problem_id']}: {r['energy_reduction']:.2f}% energy, {r['speedup']:.3f}x speedup")

    print("\n" + "=" * 80)

def main():
    results_dir = Path(__file__).parent / 'data' / 'evaluation_results'

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    chunk_files = list(results_dir.glob('test_comparison_chunk_*.jsonl'))
    if not chunk_files:
        print(f"ERROR: No result files found in {results_dir}")
        return

    print(f"Loading results from {len(chunk_files)} chunk files...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} results")

    if not results:
        print("ERROR: No results loaded")
        return

    metrics = analyze_results(results)
    print_summary(metrics, results)

    output_file = results_dir / 'test_analysis_summary.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_samples': metrics.total_samples,
            'compile_rate': metrics.compile_rate,
            'correctness_rate': metrics.correctness_rate,
            'success_rate': metrics.success_rate,
            'energy_reduction_mean': metrics.energy_reduction_mean,
            'energy_reduction_median': metrics.energy_reduction_median,
            'edp_reduction_mean': metrics.edp_reduction_mean,
            'edp_reduction_median': metrics.edp_reduction_median,
            'speedup_mean': metrics.speedup_mean,
            'speedup_median': metrics.speedup_median,
            'num_improvements': metrics.num_improvements,
            'num_regressions': metrics.num_regressions,
            'vs_gt_mean': metrics.vs_gt_mean,
            'beats_gt_count': metrics.beats_gt_count,
        }, f, indent=2)
    print(f"\nSummary saved to {output_file}")

if __name__ == '__main__':
    main()
