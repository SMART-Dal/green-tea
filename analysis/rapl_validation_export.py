#!/usr/bin/env python3
"""Export program pairs for RAPL hardware energy validation.

Selects representative (baseline, optimized) C++ code pairs from SFT/GRPO
evaluation results for running on real hardware with RAPL energy measurement.

Usage:
  python3 rapl_validation_export.py [--n 50] [--out rapl_pairs.jsonl]

Output JSONL fields per record:
  pair_id, problem_id, source (sft|grpo), baseline_code, optimized_code,
  mcpat_baseline_energy_J, mcpat_optimized_energy_J, mcpat_err_pct,
  mcpat_baseline_cycles, mcpat_optimized_cycles, mcpat_speedup,
  test_inputs (list of strings for correctness checking)

On real hardware, compile both programs with: g++ -O3 -std=c++17 -static
Measure energy with RAPL: perf stat -e power/energy-pkg/ ./program < input
Report results in rapl_results.jsonl with same pair_id + rapl_baseline_J, rapl_optimized_J.
"""
import json
import argparse
import random
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "finetuning" / "data"
SFT_DIR = BASE / "sft_evaluation_results" / "second_run"
GRPO_DIR = BASE / "grpo_sim_results"
PIE_DIR = Path(__file__).parent.parent / "PIE_Dataset"
PIE_INPUTS = PIE_DIR / "extracted_testcases" / "codenet" / "generated_test_cases"


def load_records(directory: Path) -> list[dict]:
    records = []
    for f in sorted(directory.glob("test_comparison_chunk_*.jsonl")):
        for line in f.open():
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def load_test_inputs(problem_id: str, max_inputs: int = 5) -> list[str]:
    """Load up to max_inputs test inputs for a problem."""
    prob_dir = PIE_INPUTS / problem_id
    if not prob_dir.exists():
        return []
    inputs = []
    for f in sorted(prob_dir.glob("input.*.txt"))[:max_inputs]:
        try:
            inputs.append(f.read_text())
        except Exception:
            pass
    return inputs


def select_scanf_only(records: list[dict], label: str, n: int) -> list[dict]:
    """Select pairs where generated_code introduces scanf/printf not present in baseline."""
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0
             and r.get("baseline_code") and r.get("generated_code")
             and ("scanf" in r.get("generated_code", "") or "printf" in r.get("generated_code", ""))
             and not ("scanf" in r.get("baseline_code", "") or "printf" in r.get("baseline_code", ""))]
    if not valid:
        print(f"  {label}: no scanf/printf-introducing pairs found")
        return []
    errs = [r["energy_reduction"] for r in valid]
    valid.sort(key=lambda r: r["energy_reduction"], reverse=True)
    selected = valid[:n]
    for r in selected:
        r["_source"] = label
    print(f"  {label}: {len(valid)} scanf/printf pairs available, selected {len(selected)}")
    return selected


def select_stratified(records: list[dict], label: str, n: int) -> list[dict]:
    """Select n records stratified across ERR buckets for coverage."""
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0
             and r.get("baseline_code") and r.get("generated_code")]
    if not valid:
        return []
    errs = np.array([r["energy_reduction"] for r in valid])
    # 5 buckets: <0, 0-10, 10-30, 30-60, >60
    buckets = [
        [r for r, e in zip(valid, errs) if e < 0],
        [r for r, e in zip(valid, errs) if 0 <= e < 10],
        [r for r, e in zip(valid, errs) if 10 <= e < 30],
        [r for r, e in zip(valid, errs) if 30 <= e < 60],
        [r for r, e in zip(valid, errs) if e >= 60],
    ]
    per_bucket = max(1, n // 5)
    selected = []
    for bucket in buckets:
        k = min(per_bucket, len(bucket))
        selected.extend(random.sample(bucket, k))
    # fill remaining slots from largest bucket
    if len(selected) < n:
        remaining = [r for r in valid if r not in selected]
        extra = min(n - len(selected), len(remaining))
        selected.extend(random.sample(remaining, extra))
    for r in selected:
        r["_source"] = label
    return selected[:n]


def build_pair(r: dict, pair_id: int) -> dict:
    return {
        "pair_id": pair_id,
        "problem_id": r.get("problem_id", ""),
        "source": r.get("_source", "unknown"),
        "baseline_code": r.get("baseline_code", ""),
        "optimized_code": r.get("generated_code", ""),
        "mcpat_baseline_energy_J": r.get("baseline_energy", 0),
        "mcpat_optimized_energy_J": r.get("generated_energy", 0),
        "mcpat_err_pct": r.get("energy_reduction", 0),
        "mcpat_baseline_cycles": r.get("baseline_avg_cycles", 0),
        "mcpat_optimized_cycles": r.get("generated_avg_cycles", 0),
        "mcpat_speedup": r.get("speedup", 1.0),
        "mcpat_edp_reduction_pct": r.get("edp_reduction", 0),
        "num_test_inputs": r.get("num_inputs", 0),
        "test_inputs": load_test_inputs(r.get("problem_id", "")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="pairs per source (sft/grpo)")
    ap.add_argument("--out", type=str, default="rapl_validation_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scanf-only", action="store_true",
                    help="select only pairs where generated code introduces scanf/printf")
    args = ap.parse_args()
    random.seed(args.seed)

    selector = select_scanf_only if args.scanf_only else select_stratified

    sft_records = load_records(SFT_DIR)
    sft_sel = selector(sft_records, "sft", args.n)
    if not args.scanf_only:
        print(f"SFT: selected {len(sft_sel)} pairs from {sum(1 for r in sft_records if r.get('compiled') and r.get('generated_energy',0)>0)} valid")

    grpo_sel = []
    if GRPO_DIR.exists():
        grpo_records = load_records(GRPO_DIR)
        grpo_sel = selector(grpo_records, "grpo", args.n)
        if not args.scanf_only:
            print(f"GRPO: selected {len(grpo_sel)} pairs from {sum(1 for r in grpo_records if r.get('compiled') and r.get('generated_energy',0)>0)} valid")

    all_selected = sft_sel + grpo_sel
    out_path = Path(args.out)
    with out_path.open("w") as fh:
        for i, r in enumerate(all_selected):
            pair = build_pair(r, i)
            fh.write(json.dumps(pair) + "\n")

    errs = [r.get("energy_reduction", 0) for r in all_selected]
    print(f"\nExported {len(all_selected)} pairs to {out_path}")
    print(f"ERR range: [{min(errs):.1f}%, {max(errs):.1f}%]  mean={sum(errs)/len(errs):.1f}%")
    print("\nHardware instructions:")
    print("  1. For each pair, compile baseline_code and optimized_code:")
    print("       g++ -O3 -std=c++17 -static -o baseline baseline.cpp")
    print("       g++ -O3 -std=c++17 -static -o optimized optimized.cpp")
    print("  2. Run with RAPL energy measurement (10 reps each, take median):")
    print("       perf stat -r 10 -e power/energy-pkg/ ./baseline < input.txt 2>&1 | grep energy-pkg")
    print("  3. Record results in rapl_results.jsonl with fields:")
    print("       {pair_id, rapl_baseline_J, rapl_optimized_J, rapl_err_pct}")
    print("  4. Run: python3 analysis/rapl_validation_analyze.py rapl_results.jsonl")


if __name__ == "__main__":
    main()
