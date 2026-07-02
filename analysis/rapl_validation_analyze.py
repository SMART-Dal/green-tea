#!/usr/bin/env python3
"""Analyze RAPL hardware validation results vs McPAT simulation.

Usage:
  python3 rapl_validation_analyze.py rapl_results.jsonl [--pairs rapl_validation_pairs.jsonl]

RAPL results JSONL format (one record per pair):
  {"pair_id": 0, "rapl_baseline_J": 1.23, "rapl_optimized_J": 0.89}
  Optional: "rapl_err_pct" if pre-computed

Reports:
  - Pearson r(McPAT ERR, RAPL ERR)
  - Ranking agreement (% pairs where RAPL and McPAT agree on which is more efficient)
  - Mean absolute difference in ERR between McPAT and RAPL
  - Scatter of McPAT vs RAPL ERR by energy bucket
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: rapl_validation_analyze.py rapl_results.jsonl [--pairs pairs.jsonl]")
        sys.exit(1)

    rapl_path = sys.argv[1]
    pairs_path = "rapl_validation_pairs.jsonl"
    if "--pairs" in sys.argv:
        pairs_path = sys.argv[sys.argv.index("--pairs") + 1]

    rapl = {r["pair_id"]: r for r in load_jsonl(rapl_path)}
    pairs = {r["pair_id"]: r for r in load_jsonl(pairs_path)} if Path(pairs_path).exists() else {}

    results = []
    for pid, rr in rapl.items():
        rb = rr.get("rapl_baseline_J", 0)
        ro = rr.get("rapl_optimized_J", 0)
        if rb <= 0 or ro <= 0:
            continue
        rapl_err = (rb - ro) / rb * 100
        pair = pairs.get(pid, {})
        mcpat_err = pair.get("mcpat_err_pct", rr.get("mcpat_err_pct", None))
        if mcpat_err is None:
            continue
        results.append({
            "pair_id": pid,
            "problem_id": pair.get("problem_id", ""),
            "source": pair.get("source", ""),
            "mcpat_err": mcpat_err,
            "rapl_err": rapl_err,
            "mcpat_baseline_J": pair.get("mcpat_baseline_energy_J", 0),
            "rapl_baseline_J": rb,
        })

    if not results:
        print("No matched pairs found. Check pair_ids align between files.")
        sys.exit(1)

    me = np.array([r["mcpat_err"] for r in results])
    re = np.array([r["rapl_err"] for r in results])

    r_pearson, p_pearson = pearsonr(me, re)
    r_spearman, p_spearman = spearmanr(me, re)
    rank_agree = np.mean(np.sign(me) == np.sign(re)) * 100
    mae = np.mean(np.abs(me - re))
    bias = np.mean(me - re)

    print(f"RAPL vs McPAT Validation (N={len(results)} pairs)")
    print(f"  Pearson r(McPAT ERR, RAPL ERR): {r_pearson:.3f}  p={p_pearson:.4g}")
    print(f"  Spearman rho:                   {r_spearman:.3f}  p={p_spearman:.4g}")
    print(f"  Ranking agreement (sign match): {rank_agree:.1f}%")
    print(f"  Mean |McPAT - RAPL| ERR:        {mae:.2f}pp")
    print(f"  Mean bias (McPAT - RAPL):       {bias:.2f}pp  ({'McPAT overestimates' if bias>0 else 'McPAT underestimates'})")

    # Per-source breakdown
    for src in ["sft", "grpo"]:
        sub = [r for r in results if r["source"] == src]
        if not sub:
            continue
        sm = np.array([r["mcpat_err"] for r in sub])
        sr = np.array([r["rapl_err"] for r in sub])
        ra = np.mean(np.sign(sm) == np.sign(sr)) * 100
        print(f"  {src.upper()} (n={len(sub)}): ranking agree={ra:.1f}%  |err|={np.mean(np.abs(sm-sr)):.2f}pp")

    # ERR bucket agreement
    print("\n  Ranking agreement by McPAT ERR bucket:")
    for lo, hi, label in [(-100, 0, "<0% (regression)"), (0, 20, "0-20%"), (20, 50, "20-50%"), (50, 200, ">50%")]:
        sub = [r for r in results if lo <= r["mcpat_err"] < hi]
        if sub:
            sm2 = np.array([r["mcpat_err"] for r in sub])
            sr2 = np.array([r["rapl_err"] for r in sub])
            print(f"    {label:<22}: n={len(sub):3d}  agree={np.mean(np.sign(sm2)==np.sign(sr2))*100:.0f}%  |err|={np.mean(np.abs(sm2-sr2)):.2f}pp")

    print("\nManuscript numbers to use:")
    print(f"  Pearson r = {r_pearson:.3f}, ranking agreement = {rank_agree:.0f}% (N={len(results)})")


if __name__ == "__main__":
    main()
