#!/usr/bin/env python3
"""RQ2 energy-vs-runtime SFT analyses (4 from Claude-web recommendation 2026-06-06).

Strengthens the Energy-SFT vs Runtime-SFT discussion with mechanism-grounded analyses
on existing data; no new training required. All four results read off the live sim
chunks and the Green Tea solution_metrics corpus.

Analyses:
  (1) Beat-Energy-Oracle - replaces runtime-biased Beat-GT with per-problem energy oracle
      from Green Tea ranking.
  (2) Power-driven check - among per-problem deep cuts unique to one variant, are they
      power-driven (cycle reduction small, energy reduction large) per RQ1 mechanism?
  (3) EDPR per variant - does Runtime-SFT trade delay/power for energy, or get both?
  (4) Paired per-problem comparison - on test problems with valid output from both
      variants, which generates lower absolute energy more often?

Verified outputs (Qwen-14B Main, 2026-06-06):
  (1) Energy-SFT 21/646 = 3.25%  vs  Runtime-SFT 31/474 = 6.54%  -> Runtime wins +3.29pp
  (2) 0/6 Energy-only deep cuts power-driven; 0/9 Runtime-only - all are cycle-driven
  (3) Energy-SFT EDPR median = -0.06%; Runtime-SFT EDPR median = 47.37% -> Runtime wins
  (4) On 82 common-valid problems: Energy-SFT lower energy on 42.7%; Runtime-SFT on 57.3%

Path overrides:
  GREEN_TEA_DATA env var -> root of sim_results / sft_evaluation_results dirs.
  GREEN_TEA_SOLUTION_METRICS env var -> path to solution_metrics.jsonl.
"""
import json, glob, os
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("GREEN_TEA_DATA", str(REPO_ROOT / "finetuning" / "data")))
SOL = Path(os.environ.get("GREEN_TEA_SOLUTION_METRICS",
                          str(REPO_ROOT / "analysis" / "solution_metrics.jsonl")))
ESFT_DIR = BASE / "sft_evaluation_results" / "second_run"
RSFT_DIR = BASE / "sft_runtime_sim_results"
TEST_PAIRS = BASE / "sft_pairs_test.jsonl"


def load_sim_dir(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("test_comparison_chunk_*.jsonl")):
        for line in open(f):
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    out = []
    for line in open(p):
        try: out.append(json.loads(line))
        except Exception: pass
    return out


def build_oracle_energy(sols: list[dict]) -> dict:
    """Per-problem energy oracle = min(avg_energy) across all Green Tea solutions."""
    oracle = {}
    for r in sols:
        pid = r.get("problem_id"); e = r.get("avg_energy", 0)
        if pid and e > 0:
            if pid not in oracle or e < oracle[pid]: oracle[pid] = e
    return oracle


def beat_oracle_summary(records, oracle, label):
    """#1 Beat-Energy-Oracle."""
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0
             and r["problem_id"] in oracle]
    n_v = len(valid)
    beat = sum(1 for r in valid if r["generated_energy"] <= oracle[r["problem_id"]])
    ratios = [r["generated_energy"] / oracle[r["problem_id"]] for r in valid] if valid else []
    print(f"  {label:<14} N_valid={n_v:>4}  Beat-Oracle={beat:>3}/{n_v} = {beat/max(n_v,1)*100:5.2f}%  "
          f"median energy/oracle = {np.median(ratios) if ratios else 0:.3f}  mean = {np.mean(ratios) if ratios else 0:.3f}")
    return {"label": label, "n_valid": n_v, "beat": beat, "rate": beat / max(n_v, 1) * 100}


def per_problem_best(records):
    """Per-problem best output (max energy_reduction across valid outputs)."""
    by_pid = defaultdict(list)
    for r in records:
        if r.get("compiled") and r.get("generated_energy", 0) > 0:
            by_pid[r["problem_id"]].append(r)
    return {pid: max(recs, key=lambda x: x.get("energy_reduction", 0))
            for pid, recs in by_pid.items()}


def power_driven_check(pids, src, label, cycle_threshold=10.0, err_threshold=30.0):
    """#2 Power-driven check on deep-cut problems.
    Power-driven definition (per RQ1): cycle reduction is small (<10pp) while
    energy reduction is large (>30%); since energy = power * runtime, energy
    can only drop substantially under flat cycles if power dropped.
    NOTE: uses baseline_avg_cycles/generated_avg_cycles (the actual sim record field names)."""
    print(f"\n  {label} (n={len(pids)} problems):")
    print(f"    {'pid':<10} {'ERR%':>7} {'cycle_red%':>11} {'EDPR%':>8} {'verdict':>14}")
    pd = 0
    for p in sorted(pids):
        r = src[p]
        err = r.get("energy_reduction", 0)
        bc = r.get("baseline_avg_cycles") or 0
        gc = r.get("generated_avg_cycles") or 0
        cyc = (bc - gc) / bc * 100 if bc > 0 else 0
        edpr = r.get("edp_reduction", 0)
        is_pd = abs(cyc) < cycle_threshold and err > err_threshold
        if is_pd: pd += 1
        print(f"    {p:<10} {err:>7.1f} {cyc:>11.1f} {edpr:>8.1f} "
              f"{'POWER-DRIVEN' if is_pd else 'cycle-driven':>14}")
    print(f"    -> {pd}/{len(pids)} ({pd/max(len(pids),1)*100:.0f}%) power-driven")
    return pd


def edpr_stats(records, label):
    """#3 EDPR per variant."""
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0]
    edps = [r.get("edp_reduction", 0) for r in valid]
    ers = [r.get("energy_reduction", 0) for r in valid]
    if not edps: return {}
    print(f"  {label:<14} N_valid={len(valid)}  EDPR mean={np.mean(edps):.2f}%  median={np.median(edps):.2f}%  "
          f"ERR mean={np.mean(ers):.2f}%  median={np.median(ers):.2f}%")
    return {"label": label, "edpr_mean": float(np.mean(edps)), "edpr_median": float(np.median(edps)),
            "err_mean": float(np.mean(ers)), "err_median": float(np.median(ers))}


def paired_per_problem(eb, rb):
    """#4 Paired per-problem comparison on common-valid problems
    (REPLACES the original #4 inversion-pair test which produced 0/0 on the 143-problem subset).
    Tests directly: on common-valid test problems, does Energy-SFT achieve lower absolute
    generated_energy than Runtime-SFT?"""
    common = set(eb) & set(rb)
    e_wins = sum(1 for p in common if eb[p].get("generated_energy", 1e9) < rb[p].get("generated_energy", 1e9))
    r_wins = sum(1 for p in common if rb[p].get("generated_energy", 1e9) < eb[p].get("generated_energy", 1e9))
    ties = len(common) - e_wins - r_wins
    diffs = [rb[p].get("generated_energy",0) - eb[p].get("generated_energy",0) for p in common]
    diffs_nz = [d for d in diffs if d != 0]
    print(f"  On {len(common)} common-valid test problems: Energy-SFT lower energy on {e_wins} "
          f"({e_wins/max(len(common),1)*100:.1f}%); Runtime-SFT lower on {r_wins} "
          f"({r_wins/max(len(common),1)*100:.1f}%); ties {ties}")
    print(f"  Signed-diff (Runtime - Energy): {len(diffs_nz)} non-zero; "
          f"median = {np.median(diffs_nz) if diffs_nz else 0:+.6f} J  "
          f"(positive = Energy-SFT achieves lower energy)")
    return {"n_common": len(common), "e_wins": e_wins, "r_wins": r_wins, "ties": ties,
            "median_signed_diff_J": float(np.median(diffs_nz)) if diffs_nz else 0.0}


def main():
    print("=" * 70)
    print(" RQ2 ENERGY-vs-RUNTIME ANALYSES (4 from web Claude 2026-06-06)")
    print("=" * 70)
    sols = load_jsonl(SOL)
    esft = load_sim_dir(ESFT_DIR)
    rsft = load_sim_dir(RSFT_DIR)
    test_pairs = load_jsonl(TEST_PAIRS)
    print(f"\n[loaded] solution_metrics={len(sols)}, Energy-SFT sims={len(esft)}, "
          f"Runtime-SFT sims={len(rsft)}, test pairs={len(test_pairs)}")
    oracle = build_oracle_energy(sols)
    print(f"[oracle built] {len(oracle)} problems with energy-oracle reference")

    test_opt_e = defaultdict(list)
    for p in test_pairs:
        pid = p.get("problem_id"); e = p.get("optimized_energy")
        if pid and e: test_opt_e[pid].append(e)
    matches = [(min(test_opt_e[p]), oracle[p]) for p in test_opt_e if p in oracle]
    if matches:
        diffs = [(m - o) / o * 100 for m, o in matches if o > 0]
        print(f"[cross-validation] manuscript ref vs oracle on {len(matches)} test problems: "
              f"median diff = {np.median(diffs):+.2f}%")

    print("\n" + "=" * 70 + "\n #1 BEAT-ENERGY-ORACLE\n" + "=" * 70)
    e1 = beat_oracle_summary(esft, oracle, "Energy-SFT")
    r1 = beat_oracle_summary(rsft, oracle, "Runtime-SFT")
    winner = "Energy" if e1["rate"] > r1["rate"] else "Runtime"
    print(f"\n  -> Beat-Energy-Oracle: Energy-SFT {e1['rate']:.2f}% vs Runtime-SFT {r1['rate']:.2f}%  "
          f"({abs(e1['rate']-r1['rate']):+.2f}pp in {winner}'s favor)")

    print("\n" + "=" * 70 + "\n #3 EDPR per variant\n" + "=" * 70)
    e3 = edpr_stats(esft, "Energy-SFT")
    r3 = edpr_stats(rsft, "Runtime-SFT")

    print("\n" + "=" * 70 + "\n #2 POWER-DRIVEN CHECK on deep-cut problems\n" + "=" * 70)
    eb = per_problem_best(esft); rb = per_problem_best(rsft)
    common = set(eb) & set(rb)
    e_deep = {p for p in common if eb[p].get("energy_reduction", 0) > 50}
    r_deep = {p for p in common if rb[p].get("energy_reduction", 0) > 50}
    e_only = sorted(e_deep - r_deep)
    r_only = sorted(r_deep - e_deep)
    print(f"  common={len(common)}, E-deep={len(e_deep)}, R-deep={len(r_deep)}")
    n_pd_e = power_driven_check(e_only, eb, "Energy-only deep cuts")
    n_pd_r = power_driven_check(r_only, rb, "Runtime-only deep cuts (contrast)")

    print("\n" + "=" * 70 + "\n #4 PAIRED PER-PROBLEM ENERGY\n" + "=" * 70)
    p4 = paired_per_problem(eb, rb)

    print("\n" + "=" * 70 + "\n SUMMARY for web Claude\n" + "=" * 70)
    print(f"  #1 Beat-Energy-Oracle: E-SFT {e1['rate']:.2f}% vs R-SFT {r1['rate']:.2f}%")
    print(f"  #2 Power-driven:       {n_pd_e}/{len(e_only)} Energy-only and {n_pd_r}/{len(r_only)} Runtime-only cuts power-driven")
    print(f"  #3 EDPR:               E-SFT median {e3.get('edpr_median', 0):.2f}%; "
          f"R-SFT median {r3.get('edpr_median', 0):.2f}%")
    print(f"  #4 Paired:             E-SFT lower energy on {p4['e_wins']}/{p4['n_common']} = "
          f"{p4['e_wins']/max(p4['n_common'],1)*100:.1f}%")


if __name__ == "__main__":
    main()
