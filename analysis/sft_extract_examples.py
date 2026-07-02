#!/usr/bin/env python3
"""Extract representative code examples from SFT dataset and evaluation results."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_test_results():
    """Load test evaluation results."""
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

def load_dataset(split='test'):
    """Load SFT dataset."""
    data_path = REPO_ROOT / 'finetuning' / 'data' / f'sft_pairs_{split}.jsonl'
    if not data_path.exists():
        return []
    return [json.loads(line) for line in open(data_path)]

def format_code_for_latex(code, max_lines=30):
    """Format code for LaTeX listing, truncate if too long."""
    lines = code.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['// ... (truncated)']
    return '\n'.join(lines)

def find_best_examples():
    """Find representative examples of different outcomes."""
    results = load_test_results()
    dataset = load_dataset('test')

    # Create problem_id -> dataset mapping
    dataset_map = {s['problem_id']: s for s in dataset}

    # Categorize results
    examples = {
        'major_improvement': [],  # >50% ERR
        'moderate_improvement': [],  # 10-50% ERR
        'minor_improvement': [],  # 0-10% ERR
        'neutral': [],  # -5 to 0% ERR
        'regression': [],  # <-5% ERR
        'compilation_failure': [],
        'correctness_failure': [],
        'beats_gt': []  # beats ground truth
    }

    for r in results:
        pid = r.get('problem_id', '')
        if not pid or pid not in dataset_map:
            continue

        err = r.get('energy_reduction', 0)
        compiled = r.get('compiled', False)
        has_energy = r.get('generated_energy', 0) > 1e-9
        vs_gt = r.get('vs_gt_reduction', 0)

        example = {
            'problem_id': pid,
            'baseline_code': r.get('baseline_code', ''),
            'generated_code': r.get('generated_code', ''),
            'optimized_code': dataset_map[pid].get('optimized_code', ''),
            'baseline_energy': r.get('baseline_energy', 0),
            'generated_energy': r.get('generated_energy', 0),
            'optimized_energy': dataset_map[pid].get('optimized_energy', 0),
            'err': err,
            'vs_gt': vs_gt,
            'speedup': r.get('speedup', 0),
            'compiled': compiled,
            'tests_passed': r.get('tests_passed', 0),
            'num_inputs': r.get('num_inputs', 0)
        }

        if not compiled:
            examples['compilation_failure'].append(example)
        elif not has_energy:
            examples['correctness_failure'].append(example)
        else:
            if err > 50:
                examples['major_improvement'].append(example)
            elif err > 10:
                examples['moderate_improvement'].append(example)
            elif err > 0:
                examples['minor_improvement'].append(example)
            elif err > -5:
                examples['neutral'].append(example)
            else:
                examples['regression'].append(example)

            if vs_gt > 0:
                examples['beats_gt'].append(example)

    # Select best representatives (prefer shorter code for paper)
    selected = {}
    for category, items in examples.items():
        if not items:
            selected[category] = None
            continue

        # Sort by code length (shorter is better for paper), then by ERR magnitude
        if category in ['major_improvement', 'moderate_improvement', 'minor_improvement', 'beats_gt']:
            items.sort(key=lambda x: (len(x['baseline_code']), -abs(x['err'])))
        else:
            items.sort(key=lambda x: len(x['baseline_code']))

        # Pick first short example
        for item in items:
            if len(item['baseline_code'].split('\n')) < 50:
                selected[category] = item
                break
        if selected.get(category) is None and items:
            selected[category] = items[0]

    return selected, examples

def generate_latex_tables():
    """Generate LaTeX table code for paper."""
    results = load_test_results()
    valid = [r for r in results if r.get('baseline_energy', 0) > 1e-9 and r.get('generated_energy', 0) > 1e-9]

    err_values = [r.get('energy_reduction', 0) for r in valid]

    print("\n" + "="*80)
    print("LATEX TABLE: ENERGY REDUCTION DISTRIBUTION")
    print("="*80)
    print(r"""
\begin{table}[htbp]
\centering
\caption{Energy Reduction Rate Distribution on Test Set (n=1,274 valid comparisons)}
\label{tab:sft_err_distribution}
\begin{tabular}{lrr}
\toprule
\textbf{ERR Range} & \textbf{Count} & \textbf{Percentage} \\
\midrule""")

    bins = [
        (float('-inf'), -50, 'Severe regression ($< -50\%$)'),
        (-50, -10, 'Major regression ($-50\%$ to $-10\%$)'),
        (-10, -5, 'Minor regression ($-10\%$ to $-5\%$)'),
        (-5, 0, 'Neutral/slight regression ($-5\%$ to $0\%$)'),
        (0, 5, 'Minor improvement ($0\%$ to $5\%$)'),
        (5, 10, 'Moderate improvement ($5\%$ to $10\%$)'),
        (10, 50, 'Major improvement ($10\%$ to $50\%$)'),
        (50, float('inf'), 'Exceptional improvement ($> 50\%$)')
    ]

    for low, high, label in bins:
        count = sum(1 for e in err_values if low <= e < high)
        pct = count / len(err_values) * 100
        print(f"{label:50s} & {count:4d} & {pct:5.1f}\\% \\\\")

    print(r"""\midrule
\textbf{Total} & """ + f"{len(err_values)} & 100.0\\% \\\\\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")

    print("\n" + "="*80)
    print("LATEX TABLE: SUMMARY STATISTICS")
    print("="*80)
    print(r"""
\begin{table}[htbp]
\centering
\caption{SFT Test Set Performance Summary}
\label{tab:sft_summary}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{95\% CI} \\
\midrule""")

    from scipy import stats

    total = len(results)
    compiled = sum(1 for r in results if r.get('compiled', False))
    has_energy = len(valid)

    # Compilation rate CI
    ci_compile = stats.binom.interval(0.95, total, compiled/total)

    print(f"Total samples & {total} & -- \\\\")
    print(f"Compilation rate & {compiled/total*100:.1f}\\% & [{ci_compile[0]/total*100:.1f}\\%, {ci_compile[1]/total*100:.1f}\\%] \\\\")
    print(f"Valid energy measurements & {has_energy} & -- \\\\")

    mean_err = np.mean(err_values)
    from scipy.stats import sem, t
    ci_err = t.interval(0.95, len(err_values)-1, loc=mean_err, scale=sem(err_values))

    print(f"Mean ERR & {mean_err:.2f}\\% & [{ci_err[0]:.2f}\\%, {ci_err[1]:.2f}\\%] \\\\")
    print(f"Median ERR & {np.median(err_values):.2f}\\% & -- \\\\")
    print(f"Std ERR & {np.std(err_values):.2f}\\% & -- \\\\")

    improved = sum(1 for e in err_values if e > 0)
    print(f"Improvements (ERR $> 0$) & {improved/len(err_values)*100:.1f}\\% & -- \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

def main():
    print("="*80)
    print("EXTRACTING CODE EXAMPLES FOR LATEX PAPER")
    print("="*80)

    selected, all_examples = find_best_examples()

    for category, example in selected.items():
        if example is None:
            print(f"\n{category.upper()}: No examples found")
            continue

        print(f"\n{'='*80}")
        print(f"{category.upper()}")
        print(f"{'='*80}")
        print(f"Problem ID: {example['problem_id']}")
        print(f"Baseline Energy: {example['baseline_energy']:.6f} J")
        print(f"Generated Energy: {example['generated_energy']:.6f} J")
        print(f"Optimized Energy: {example['optimized_energy']:.6f} J")
        print(f"ERR: {example['err']:.2f}%")
        print(f"vs GT: {example['vs_gt']:.2f}%")
        print(f"Speedup: {example['speedup']:.2f}x")
        print(f"Compiled: {example['compiled']}")
        print(f"Tests Passed: {example['tests_passed']}/{example['num_inputs']}")

        print(f"\n--- BASELINE CODE (first 20 lines) ---")
        baseline_lines = example['baseline_code'].split('\n')[:20]
        print('\n'.join(baseline_lines))
        if len(example['baseline_code'].split('\n')) > 20:
            print("... (truncated)")

        if example['generated_code']:
            print(f"\n--- GENERATED CODE (first 20 lines) ---")
            gen_lines = example['generated_code'].split('\n')[:20]
            print('\n'.join(gen_lines))
            if len(example['generated_code'].split('\n')) > 20:
                print("... (truncated)")

    # Generate category statistics
    print(f"\n{'='*80}")
    print("CATEGORY STATISTICS")
    print(f"{'='*80}")
    for category, items in all_examples.items():
        print(f"{category:30s}: {len(items):5d} examples")

    # Generate LaTeX tables
    generate_latex_tables()

    print("\n" + "="*80)
    print("Example extraction complete.")
    print("="*80)

if __name__ == '__main__':
    main()
