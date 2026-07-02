#!/usr/bin/env python3
"""Statistical analysis and significance testing for SFT results."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_test_results():
    """Load all test evaluation results."""
    results = []
    base_dir = REPO_ROOT / 'finetuning' / 'data' / 'sft_evaluation_results'

    for run in ['first_run', 'second_run']:
        run_dir = base_dir / run
        if not run_dir.exists():
            continue
        for chunk_file in sorted(run_dir.glob('test_comparison_chunk_*.jsonl')):
            with open(chunk_file) as f:
                for line in f:
                    r = json.loads(line)
                    r['run'] = run
                    results.append(r)
    return results

def load_dataset(split='test'):
    """Load SFT dataset for analysis."""
    data_path = REPO_ROOT / 'finetuning' / 'data' / f'sft_pairs_{split}.jsonl'
    if not data_path.exists():
        return []
    return [json.loads(line) for line in open(data_path)]

def paired_t_test(baseline_values, generated_values):
    """Perform paired t-test."""
    t_stat, p_value = stats.ttest_rel(baseline_values, generated_values)
    return t_stat, p_value

def wilcoxon_test(baseline_values, generated_values):
    """Perform Wilcoxon signed-rank test (non-parametric)."""
    stat, p_value = stats.wilcoxon(baseline_values, generated_values, alternative='greater')
    return stat, p_value

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval."""
    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (100-ci)/2)
    upper = np.percentile(bootstrap_means, 100-(100-ci)/2)
    return lower, upper

def effect_size_cohens_d(baseline, generated):
    """Compute Cohen's d effect size."""
    diff = np.array(baseline) - np.array(generated)
    return np.mean(diff) / np.std(diff, ddof=1)

def analyze_by_problem_complexity():
    """Analyze results stratified by problem complexity."""
    results = load_test_results()
    dataset = load_dataset('test')

    # Map problem_id to complexity metrics from dataset
    complexity_map = {}
    for sample in dataset:
        pid = sample['problem_id']
        complexity_map[pid] = {
            'baseline_cycles': sample.get('baseline_cycles', 0),
            'baseline_instructions': sample.get('baseline_instructions', 0),
            'baseline_energy': sample.get('baseline_energy', 0)
        }

    # Stratify by baseline energy quartiles
    energies = [v['baseline_energy'] for v in complexity_map.values() if v['baseline_energy'] > 0]
    q25, q50, q75 = np.percentile(energies, [25, 50, 75])

    strata = {'low': [], 'medium': [], 'high': [], 'very_high': []}

    for r in results:
        if r['problem_id'] not in complexity_map:
            continue
        energy = complexity_map[r['problem_id']]['baseline_energy']

        if r.get('generated_energy', 0) > 1e-9 and r.get('baseline_energy', 0) > 1e-9:
            err = r.get('energy_reduction', 0)
            if energy < q25:
                strata['low'].append(err)
            elif energy < q50:
                strata['medium'].append(err)
            elif energy < q75:
                strata['high'].append(err)
            else:
                strata['very_high'].append(err)

    return strata, (q25, q50, q75)

def main():
    print("="*80)
    print("SFT STATISTICAL ANALYSIS")
    print("="*80)

    results = load_test_results()
    dataset = load_dataset('test')

    # Filter valid comparisons
    valid = [r for r in results if r.get('baseline_energy', 0) > 1e-9 and r.get('generated_energy', 0) > 1e-9]

    baseline_energies = [r['baseline_energy'] for r in valid]
    generated_energies = [r['generated_energy'] for r in valid]
    err_values = [r.get('energy_reduction', 0) for r in valid]

    print(f"\n1. SAMPLE STATISTICS (n={len(valid)})")
    print("-"*80)
    print(f"Baseline Energy:  {np.mean(baseline_energies):.4f} J (±{np.std(baseline_energies):.4f})")
    print(f"Generated Energy: {np.mean(generated_energies):.4f} J (±{np.std(generated_energies):.4f})")
    print(f"Mean ERR: {np.mean(err_values):.2f}% (±{np.std(err_values):.2f}%)")
    print(f"Median ERR: {np.median(err_values):.2f}%")

    # Confidence intervals
    ci_lower, ci_upper = bootstrap_ci(err_values)
    print(f"95% CI for mean ERR: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

    print(f"\n2. STATISTICAL SIGNIFICANCE TESTS")
    print("-"*80)

    # Paired t-test
    t_stat, p_value_t = paired_t_test(baseline_energies, generated_energies)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value_t:.6f}")
    print(f"  Interpretation: {'SIGNIFICANT' if p_value_t < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

    # Wilcoxon (non-parametric)
    w_stat, p_value_w = wilcoxon_test(baseline_energies, generated_energies)
    print(f"Wilcoxon signed-rank: W={w_stat:.0f}, p={p_value_w:.6f}")
    print(f"  Interpretation: {'SIGNIFICANT' if p_value_w < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

    # Effect size
    cohens_d = effect_size_cohens_d(baseline_energies, generated_energies)
    print(f"Cohen's d effect size: {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        magnitude = "negligible"
    elif abs(cohens_d) < 0.5:
        magnitude = "small"
    elif abs(cohens_d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    print(f"  Magnitude: {magnitude}")

    print(f"\n3. DISTRIBUTION ANALYSIS")
    print("-"*80)

    # Normality test
    _, p_norm = stats.shapiro(err_values[:5000])  # Shapiro-Wilk (max 5000 samples)
    print(f"Shapiro-Wilk normality test: p={p_norm:.6f}")
    print(f"  Distribution: {'Normal' if p_norm > 0.05 else 'Non-normal'}")

    # Skewness and kurtosis
    skew = stats.skew(err_values)
    kurt = stats.kurtosis(err_values)
    print(f"Skewness: {skew:.3f} ({'right-skewed' if skew > 0 else 'left-skewed'})")
    print(f"Kurtosis: {kurt:.3f} ({'heavy-tailed' if kurt > 0 else 'light-tailed'})")

    print(f"\n4. STRATIFIED ANALYSIS BY PROBLEM COMPLEXITY")
    print("-"*80)

    strata, quartiles = analyze_by_problem_complexity()
    print(f"Baseline energy quartiles: {quartiles[0]:.4f}J, {quartiles[1]:.4f}J, {quartiles[2]:.4f}J")
    print()

    for stratum, values in strata.items():
        if not values:
            continue
        print(f"{stratum.upper()} complexity (n={len(values)}):")
        print(f"  Mean ERR: {np.mean(values):.2f}%")
        print(f"  Median ERR: {np.median(values):.2f}%")
        print(f"  Std: {np.std(values):.2f}%")
        print(f"  Improvements (>0): {sum(1 for v in values if v > 0)} ({sum(1 for v in values if v > 0)/len(values)*100:.1f}%)")

    # ANOVA across strata
    if all(len(v) > 0 for v in strata.values()):
        f_stat, p_anova = stats.f_oneway(*strata.values())
        print(f"\nOne-way ANOVA across complexity strata:")
        print(f"  F={f_stat:.3f}, p={p_anova:.6f}")
        print(f"  Interpretation: {'SIGNIFICANT' if p_anova < 0.05 else 'NOT SIGNIFICANT'} difference across strata")

    print(f"\n5. SUCCESS RATE ANALYSIS")
    print("-"*80)

    total = len(results)
    compiled = sum(1 for r in results if r.get('compiled', False))
    has_gen_energy = sum(1 for r in results if r.get('generated_energy', 0) > 1e-9)

    # Binomial test for compilation rate
    p_compile = compiled / total
    ci_compile = stats.binom.interval(0.95, total, p_compile)
    print(f"Compilation rate: {p_compile:.4f} ({compiled}/{total})")
    print(f"  95% CI: [{ci_compile[0]/total:.4f}, {ci_compile[1]/total:.4f}]")

    # Test against null hypothesis (random baseline = 50%)
    result_binom = stats.binomtest(compiled, total, 0.5, alternative='greater')
    print(f"  Binomial test vs 50%: p={result_binom.pvalue:.6f} {'(significant)' if result_binom.pvalue < 0.05 else '(not significant)'}")

    print(f"\n6. VS GROUND TRUTH ANALYSIS")
    print("-"*80)

    has_gt = [r for r in valid if r.get('optimized_energy', 0) > 1e-9]
    vs_gt_values = [r.get('vs_gt_reduction', 0) for r in has_gt]
    beats_gt = sum(1 for v in vs_gt_values if v > 0)

    print(f"Comparisons with GT: {len(has_gt)}")
    print(f"Beats GT: {beats_gt} ({beats_gt/len(has_gt)*100:.2f}%)")
    print(f"Mean vs GT: {np.mean(vs_gt_values):.2f}%")
    print(f"Median vs GT: {np.median(vs_gt_values):.2f}%")

    # One-sample t-test: is mean vs_GT significantly different from 0?
    t_gt, p_gt = stats.ttest_1samp(vs_gt_values, 0)
    print(f"One-sample t-test (H0: mean=0): t={t_gt:.3f}, p={p_gt:.6f}")
    print(f"  Model {'significantly worse' if t_gt < 0 and p_gt < 0.05 else 'not significantly different from'} GT")

    print(f"\n7. PERCENTILE ANALYSIS")
    print("-"*80)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("ERR Distribution Percentiles:")
    for p in percentiles:
        val = np.percentile(err_values, p)
        print(f"  {p:2d}th: {val:8.2f}%")

    print("\n" + "="*80)
    print("Analysis complete. Results suitable for LaTeX tables and paper discussion.")
    print("="*80)

if __name__ == '__main__':
    main()
