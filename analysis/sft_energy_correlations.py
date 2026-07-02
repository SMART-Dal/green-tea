#!/usr/bin/env python3
"""Analyze correlations between code metrics and energy outcomes."""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import re

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_results():
    """Load test results."""
    results = []
    base_dir = REPO_ROOT / 'finetuning' / 'data' / 'sft_evaluation_results'
    for run in ['first_run', 'second_run']:
        run_dir = base_dir / run
        if not run_dir.exists():
            continue
        for chunk_file in sorted(run_dir.glob('test_comparison_chunk_*.jsonl')):
            with open(chunk_file) as f:
                for line in f:
                    results.append(json.loads(line))
    return results

def count_loops(code):
    """Count loop statements."""
    return len(re.findall(r'\b(for|while)\s*\(', code))

def code_length(code):
    """Count non-empty lines."""
    return len([l for l in code.split('\n') if l.strip()])

def cyclomatic_complexity(code):
    """Approximate cyclomatic complexity."""
    decision_points = len(re.findall(r'\b(if|for|while|case|catch|\?\s*:)\b', code))
    return decision_points + 1

def main():
    print("="*80)
    print("ENERGY-CODE METRICS CORRELATION ANALYSIS")
    print("="*80)

    results = load_results()
    valid = [r for r in results if r.get('baseline_energy', 0) > 1e-9 and r.get('generated_energy', 0) > 1e-9 and r.get('baseline_code') and r.get('generated_code')]

    print(f"\nAnalyzing {len(valid)} samples with valid energy measurements...")

    # Extract metrics
    data = []
    for r in valid:
        baseline_code = r['baseline_code']
        generated_code = r['generated_code']

        baseline_lines = code_length(baseline_code)
        generated_lines = code_length(generated_code)
        length_reduction = (baseline_lines - generated_lines) / baseline_lines * 100

        baseline_loops = count_loops(baseline_code)
        generated_loops = count_loops(generated_code)

        baseline_cc = cyclomatic_complexity(baseline_code)
        generated_cc = cyclomatic_complexity(generated_code)

        data.append({
            'err': r.get('energy_reduction', 0),
            'speedup': r.get('speedup', 1),
            'baseline_energy': r['baseline_energy'],
            'generated_energy': r['generated_energy'],
            'baseline_cycles': r.get('baseline_avg_cycles', 0),
            'generated_cycles': r.get('generated_avg_cycles', 0),
            'baseline_ipc': r.get('baseline_avg_ipc', 0),
            'generated_ipc': r.get('generated_avg_ipc', 0),
            'length_reduction': length_reduction,
            'baseline_lines': baseline_lines,
            'generated_lines': generated_lines,
            'baseline_loops': baseline_loops,
            'generated_loops': generated_loops,
            'loop_reduction': baseline_loops - generated_loops,
            'baseline_cc': baseline_cc,
            'generated_cc': generated_cc,
            'cc_reduction': baseline_cc - generated_cc,
        })

    # Correlation analysis
    print(f"\n{'='*80}")
    print("CORRELATION: ERR vs CODE METRICS")
    print(f"{'='*80}")

    metrics = [
        ('length_reduction', 'Length Reduction (%)'),
        ('loop_reduction', 'Loop Count Reduction'),
        ('cc_reduction', 'Cyclomatic Complexity Reduction'),
        ('speedup', 'Speedup'),
        ('baseline_energy', 'Baseline Energy'),
        ('baseline_lines', 'Baseline Code Length'),
    ]

    print(f"\n{'Metric':<40s} {'Pearson r':>12s} {'p-value':>12s} {'Spearman ρ':>12s} {'p-value':>12s}")
    print("-"*80)

    err_values = [d['err'] for d in data]

    for metric_key, metric_name in metrics:
        metric_values = [d[metric_key] for d in data]

        # Remove any NaN or inf values
        valid_pairs = [(e, m) for e, m in zip(err_values, metric_values) if np.isfinite(e) and np.isfinite(m)]
        if len(valid_pairs) < 10:
            continue

        err_clean, metric_clean = zip(*valid_pairs)

        r_pearson, p_pearson = pearsonr(err_clean, metric_clean)
        r_spearman, p_spearman = spearmanr(err_clean, metric_clean)

        sig_pearson = '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else ''
        sig_spearman = '**' if p_spearman < 0.01 else '*' if p_spearman < 0.05 else ''

        print(f"{metric_name:<40s} {r_pearson:>11.3f}{sig_pearson:<1s} {p_pearson:>12.6f} {r_spearman:>11.3f}{sig_spearman:<1s} {p_spearman:>12.6f}")

    print("\n* p<0.05, ** p<0.01")

    # ERR vs speedup scatter analysis
    print(f"\n{'='*80}")
    print("ERR VS SPEEDUP RELATIONSHIP")
    print(f"{'='*80}")

    speedup_bins = [
        (0, 0.8, 'Slowdown (< 0.8x)'),
        (0.8, 1.0, 'Slight slowdown (0.8-1.0x)'),
        (1.0, 1.2, 'Near-neutral (1.0-1.2x)'),
        (1.2, 2.0, 'Moderate speedup (1.2-2.0x)'),
        (2.0, 1000, 'Major speedup (> 2.0x)')
    ]

    print(f"\n{'Speedup Range':<30s} {'Count':>8s} {'Mean ERR':>12s} {'Median ERR':>12s}")
    print("-"*70)

    for low, high, label in speedup_bins:
        bin_data = [d for d in data if low <= d['speedup'] < high]
        if not bin_data:
            continue
        bin_errs = [d['err'] for d in bin_data]
        print(f"{label:<30s} {len(bin_data):>8d} {np.mean(bin_errs):>11.2f}% {np.median(bin_errs):>11.2f}%")

    # IPC analysis
    print(f"\n{'='*80}")
    print("IPC (INSTRUCTIONS PER CYCLE) ANALYSIS")
    print(f"{'='*80}")

    ipc_improvements = [d for d in data if d['generated_ipc'] > d['baseline_ipc']]
    ipc_regressions = [d for d in data if d['generated_ipc'] < d['baseline_ipc']]

    print(f"\nSamples with IPC improvement: {len(ipc_improvements)} ({len(ipc_improvements)/len(data)*100:.1f}%)")
    print(f"  Mean ERR: {np.mean([d['err'] for d in ipc_improvements]):.2f}%")
    print(f"  Median ERR: {np.median([d['err'] for d in ipc_improvements]):.2f}%")

    print(f"\nSamples with IPC regression: {len(ipc_regressions)} ({len(ipc_regressions)/len(data)*100:.1f}%)")
    print(f"  Mean ERR: {np.mean([d['err'] for d in ipc_regressions]):.2f}%")
    print(f"  Median ERR: {np.median([d['err'] for d in ipc_regressions]):.2f}%")

    # Energy-cycles-IPC relationship
    print(f"\n{'='*80}")
    print("ENERGY DECOMPOSITION ANALYSIS")
    print(f"{'='*80}")

    print("\nNote: Energy ≈ Power × Runtime ≈ Power × Cycles / Frequency")
    print("ERR can come from: (1) Cycle reduction (speedup), (2) Power reduction (IPC/architecture)")

    cycle_dominant = [d for d in data if abs(d['speedup'] - 1.0) > 0.1 and abs(d['generated_ipc'] / d['baseline_ipc'] - 1.0) < 0.05]
    ipc_dominant = [d for d in data if abs(d['speedup'] - 1.0) < 0.05 and abs(d['generated_ipc'] / d['baseline_ipc'] - 1.0) > 0.05]
    both = [d for d in data if abs(d['speedup'] - 1.0) > 0.1 and abs(d['generated_ipc'] / d['baseline_ipc'] - 1.0) > 0.05]

    print(f"\nCycle-dominant improvements (speedup but not IPC): {len(cycle_dominant)}")
    if cycle_dominant:
        print(f"  Mean ERR: {np.mean([d['err'] for d in cycle_dominant]):.2f}%")

    print(f"\nIPC-dominant improvements (IPC but not speedup): {len(ipc_dominant)}")
    if ipc_dominant:
        print(f"  Mean ERR: {np.mean([d['err'] for d in ipc_dominant]):.2f}%")

    print(f"\nBoth cycle and IPC improvements: {len(both)}")
    if both:
        print(f"  Mean ERR: {np.mean([d['err'] for d in both]):.2f}%")

    # Baseline energy stratification
    print(f"\n{'='*80}")
    print("ERR BY BASELINE ENERGY MAGNITUDE")
    print(f"{'='*80}")

    energy_quantiles = np.quantile([d['baseline_energy'] for d in data], [0.25, 0.5, 0.75])

    strata = [
        ([d for d in data if d['baseline_energy'] < energy_quantiles[0]], f'Very low energy (< {energy_quantiles[0]:.4f}J)'),
        ([d for d in data if energy_quantiles[0] <= d['baseline_energy'] < energy_quantiles[1]], f'Low energy ({energy_quantiles[0]:.4f}-{energy_quantiles[1]:.4f}J)'),
        ([d for d in data if energy_quantiles[1] <= d['baseline_energy'] < energy_quantiles[2]], f'Medium energy ({energy_quantiles[1]:.4f}-{energy_quantiles[2]:.4f}J)'),
        ([d for d in data if d['baseline_energy'] >= energy_quantiles[2]], f'High energy (≥ {energy_quantiles[2]:.4f}J)'),
    ]

    print(f"\n{'Energy Stratum':<50s} {'n':>6s} {'Mean ERR':>12s} {'Median ERR':>12s}")
    print("-"*80)

    for stratum_data, label in strata:
        if not stratum_data:
            continue
        errs = [d['err'] for d in stratum_data]
        print(f"{label:<50s} {len(stratum_data):>6d} {np.mean(errs):>11.2f}% {np.median(errs):>11.2f}%")

    print(f"\n{'='*80}")
    print("Analysis complete.")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
