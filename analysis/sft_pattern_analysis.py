#!/usr/bin/env python3
"""Analyze optimization patterns and code transformations in SFT outputs."""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import difflib

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_results():
    """Load test results with code."""
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

def extract_includes(code):
    """Extract include statements."""
    return re.findall(r'#include\s*[<"]([^>"]+)[>"]', code)

def extract_macros(code):
    """Extract macro definitions."""
    return re.findall(r'#define\s+(\w+)', code)

def count_loops(code):
    """Count different loop types."""
    for_loops = len(re.findall(r'\bfor\s*\(', code))
    while_loops = len(re.findall(r'\bwhile\s*\(', code))
    do_while = len(re.findall(r'\bdo\s*\{', code))
    return for_loops, while_loops, do_while

def count_stl_containers(code):
    """Count STL container usage."""
    containers = {
        'vector': len(re.findall(r'\bvector\s*<', code)),
        'map': len(re.findall(r'\bmap\s*<', code)),
        'set': len(re.findall(r'\bset\s*<', code)),
        'queue': len(re.findall(r'\bqueue\s*<', code)),
        'stack': len(re.findall(r'\bstack\s*<', code)),
        'deque': len(re.findall(r'\bdeque\s*<', code)),
        'array': len(re.findall(r'\barray\s*<', code)),
    }
    return containers

def analyze_code_patterns(code):
    """Extract various code patterns."""
    patterns = {
        'using_namespace_std': bool(re.search(r'using\s+namespace\s+std', code)),
        'typedef_ll': bool(re.search(r'typedef\s+long\s+long\s+ll', code)),
        'fast_io': bool(re.search(r'ios_base::sync_with_stdio\(false\)', code)),
        'cin_tie': bool(re.search(r'cin\.tie\(', code)),
        'scanf_usage': bool(re.search(r'\bscanf\s*\(', code)),
        'printf_usage': bool(re.search(r'\bprintf\s*\(', code)),
        'cin_usage': bool(re.search(r'\bcin\s*>>', code)),
        'cout_usage': bool(re.search(r'\bcout\s*<<', code)),
        'memset': bool(re.search(r'\bmemset\s*\(', code)),
        'fill': bool(re.search(r'\bfill\s*\(', code)),
        'sort': bool(re.search(r'\bsort\s*\(', code)),
        'reverse': bool(re.search(r'\breverse\s*\(', code)),
        'push_back': bool(re.search(r'\.push_back\s*\(', code)),
    }
    return patterns

def code_diff_stats(baseline, generated):
    """Compute diff statistics."""
    baseline_lines = baseline.split('\n')
    generated_lines = generated.split('\n')

    diff = list(difflib.unified_diff(baseline_lines, generated_lines, lineterm=''))

    added = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

    return {
        'lines_added': added,
        'lines_removed': removed,
        'lines_changed': added + removed,
        'baseline_length': len(baseline_lines),
        'generated_length': len(generated_lines),
        'length_diff': len(generated_lines) - len(baseline_lines)
    }

def main():
    print("="*80)
    print("SFT OPTIMIZATION PATTERN ANALYSIS")
    print("="*80)

    results = load_results()
    valid = [r for r in results if r.get('baseline_code') and r.get('generated_code') and r.get('generated_energy', 0) > 1e-9]

    print(f"\nAnalyzing {len(valid)} valid code pairs...")

    # Categorize by ERR
    improvements = [r for r in valid if r.get('energy_reduction', 0) > 10]
    regressions = [r for r in valid if r.get('energy_reduction', 0) < -5]
    neutral = [r for r in valid if -5 <= r.get('energy_reduction', 0) <= 10]

    print(f"\nImprovements (>10% ERR): {len(improvements)}")
    print(f"Neutral (-5% to 10%): {len(neutral)}")
    print(f"Regressions (<-5%): {len(regressions)}")

    # Pattern analysis
    print(f"\n{'='*80}")
    print("CODE TRANSFORMATION PATTERNS")
    print(f"{'='*80}")

    baseline_patterns = defaultdict(int)
    generated_patterns = defaultdict(int)
    pattern_changes = defaultdict(int)

    for r in improvements[:200]:  # Sample for efficiency
        baseline = r['baseline_code']
        generated = r['generated_code']

        bp = analyze_code_patterns(baseline)
        gp = analyze_code_patterns(generated)

        for key, val in bp.items():
            if val:
                baseline_patterns[key] += 1
        for key, val in gp.items():
            if val:
                generated_patterns[key] += 1

        # Track changes
        for key in bp:
            if bp[key] != gp[key]:
                change_type = f"{key}: {bp[key]}->{gp[key]}"
                pattern_changes[change_type] += 1

    print("\nTop Pattern Changes in Improvements:")
    for pattern, count in sorted(pattern_changes.items(), key=lambda x: -x[1])[:15]:
        print(f"  {pattern:50s}: {count:3d}")

    # I/O pattern analysis
    print(f"\n{'='*80}")
    print("I/O PATTERN SHIFTS")
    print(f"{'='*80}")

    io_transitions = defaultdict(int)
    for r in improvements[:200]:
        baseline = r['baseline_code']
        generated = r['generated_code']

        baseline_io = 'scanf/printf' if ('scanf' in baseline or 'printf' in baseline) else 'cin/cout'
        generated_io = 'scanf/printf' if ('scanf' in generated or 'printf' in generated) else 'cin/cout'

        transition = f"{baseline_io} -> {generated_io}"
        io_transitions[transition] += 1

    for transition, count in sorted(io_transitions.items(), key=lambda x: -x[1]):
        print(f"  {transition:30s}: {count:3d}")

    # Code length analysis
    print(f"\n{'='*80}")
    print("CODE LENGTH CHANGES")
    print(f"{'='*80}")

    length_diffs_improvements = []
    length_diffs_regressions = []

    for r in improvements:
        diff_stats = code_diff_stats(r['baseline_code'], r['generated_code'])
        length_diffs_improvements.append(diff_stats['length_diff'])

    for r in regressions:
        diff_stats = code_diff_stats(r['baseline_code'], r['generated_code'])
        length_diffs_regressions.append(diff_stats['length_diff'])

    import numpy as np
    if length_diffs_improvements:
        print(f"\nImprovements (>10% ERR):")
        print(f"  Mean length change: {np.mean(length_diffs_improvements):.1f} lines")
        print(f"  Median length change: {np.median(length_diffs_improvements):.1f} lines")
        print(f"  Shorter: {sum(1 for d in length_diffs_improvements if d < 0)} ({sum(1 for d in length_diffs_improvements if d < 0)/len(length_diffs_improvements)*100:.1f}%)")
        print(f"  Same: {sum(1 for d in length_diffs_improvements if d == 0)}")
        print(f"  Longer: {sum(1 for d in length_diffs_improvements if d > 0)} ({sum(1 for d in length_diffs_improvements if d > 0)/len(length_diffs_improvements)*100:.1f}%)")

    if length_diffs_regressions:
        print(f"\nRegressions (<-5% ERR):")
        print(f"  Mean length change: {np.mean(length_diffs_regressions):.1f} lines")
        print(f"  Median length change: {np.median(length_diffs_regressions):.1f} lines")

    # Loop complexity analysis
    print(f"\n{'='*80}")
    print("LOOP PATTERN ANALYSIS")
    print(f"{'='*80}")

    loop_changes = {'reduced': 0, 'increased': 0, 'unchanged': 0}

    for r in improvements[:200]:
        baseline_loops = sum(count_loops(r['baseline_code']))
        generated_loops = sum(count_loops(r['generated_code']))

        if generated_loops < baseline_loops:
            loop_changes['reduced'] += 1
        elif generated_loops > baseline_loops:
            loop_changes['increased'] += 1
        else:
            loop_changes['unchanged'] += 1

    print(f"\nLoop count changes in improvements:")
    for change_type, count in loop_changes.items():
        print(f"  {change_type:12s}: {count:3d}")

    # Header/include analysis
    print(f"\n{'='*80}")
    print("HEADER INCLUSION PATTERNS")
    print(f"{'='*80}")

    baseline_headers = Counter()
    generated_headers = Counter()

    for r in improvements[:200]:
        baseline_headers.update(extract_includes(r['baseline_code']))
        generated_headers.update(extract_includes(r['generated_code']))

    print("\nMost common baseline headers:")
    for header, count in baseline_headers.most_common(10):
        print(f"  {header:30s}: {count:3d}")

    print("\nMost common generated headers:")
    for header, count in generated_headers.most_common(10):
        print(f"  {header:30s}: {count:3d}")

    # bits/stdc++.h usage
    bits_baseline = sum(1 for r in improvements if 'bits/stdc++.h' in r['baseline_code'])
    bits_generated = sum(1 for r in improvements if 'bits/stdc++.h' in r['generated_code'])

    print(f"\nbits/stdc++.h usage:")
    print(f"  Baseline: {bits_baseline}/{len(improvements)} ({bits_baseline/len(improvements)*100:.1f}%)")
    print(f"  Generated: {bits_generated}/{len(improvements)} ({bits_generated/len(improvements)*100:.1f}%)")

    print(f"\n{'='*80}")
    print("Analysis complete.")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
