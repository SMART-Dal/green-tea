#!/usr/bin/env python3
"""Landscape: Energy consumption drivers in competitive C++ programs.

Reproduces all numbers in Section 3 (Energy Optimization Landscape).
Data: completed_samples.jsonl (3.4M individual Sniper/McPAT measurements),
      solution_metrics.jsonl (39,744 total; 38,743 with avg_ipc>0 and avg_power>0),
      problem_statistics.jsonl (1,474 problem-level aggregates).
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from numpy.linalg import lstsq
from scipy.stats import bootstrap as sp_bootstrap, binomtest

ANALYSIS_DIR = Path(__file__).parent.parent  # analysis/rq1/ -> analysis/
COMPLETED = ANALYSIS_DIR / "completed_samples.jsonl"
SOLUTION_METRICS = ANALYSIS_DIR / "solution_metrics.jsonl"
PROBLEM_STATS = ANALYSIS_DIR / "problem_statistics.jsonl"


def load_individual_measurements():
    """Load completed_samples.jsonl -> arrays of individual execution measurements."""
    power, energy, runtime, ipc = [], [], [], []
    n_total = n_valid = 0
    with open(COMPLETED) as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
                r = d["result"]
                p = r.get("power_watts", 0)
                e = r.get("energy_joules", 0)
                t = r.get("runtime_seconds", 0)
                c = r.get("cycles", 0)
                ins = r.get("instructions", 0)
                if p > 0 and e > 0 and t > 0 and c > 0:
                    power.append(p); energy.append(e); runtime.append(t)
                    ipc.append(ins / c)
                    n_valid += 1
            except Exception:
                pass
    print(f"Individual measurements: {n_valid:,} valid / {n_total:,} total ({n_total-n_valid:,} filtered)")
    return np.array(power), np.array(energy), np.array(runtime), np.array(ipc)


def load_solution_metrics():
    records = []
    with open(SOLUTION_METRICS) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_problem_stats():
    records = []
    with open(PROBLEM_STATS) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def print_section(title):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def power_distribution(power, energy, runtime, ipc):
    print_section("1. POWER DISTRIBUTION (individual measurements)")
    print(f"N = {len(power):,}")
    print(f"Median runtime: {np.median(runtime)*1000:.4f} ms")
    print(f"Mean:   {np.mean(power):.4f} W")
    print(f"Median: {np.median(power):.4f} W")
    print(f"Std:    {np.std(power):.4f} W  ({np.std(power)/np.mean(power)*100:.2f}%)")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pvs = np.percentile(power, pcts)
    for pct, pv in zip(pcts, pvs):
        print(f"  P{pct:2d}: {pv:.2f} W")
    p5, p95 = pvs[1], pvs[7]
    print(f"P5-P95 spread: {p95-p5:.2f} W ({(p95-p5)/np.mean(power)*100:.2f}%)")
    print(f"200-210 W: {np.mean((power>=200)&(power<=210))*100:.2f}%")


def energy_runtime_correlation(energy, runtime, power):
    print_section("2. ENERGY-RUNTIME CORRELATION")
    log_e = np.log(energy)
    log_r = np.log(runtime)
    log_p = np.log(power)
    print(f"r(log_energy, log_runtime): {np.corrcoef(log_e, log_r)[0,1]:.6f}")
    print(f"r(log_energy, log_power):   {np.corrcoef(log_e, log_p)[0,1]:.6f}")
    X_r = np.column_stack([log_r, np.ones(len(log_e))])
    c_r, _, _, _ = lstsq(X_r, log_e, rcond=None)
    ss_tot = np.sum((log_e - np.mean(log_e))**2)
    r2_r = 1 - np.sum((log_e - X_r @ c_r)**2) / ss_tot
    print(f"R2 (log_e ~ log_r):         {r2_r:.6f}  ({r2_r*100:.2f}% of energy variance)")
    print(f"OLS slope:                  {c_r[0]:.4f}")


def ipc_distribution(ipc, power):
    print_section("3. IPC DISTRIBUTION")
    print(f"Mean IPC: {np.mean(ipc):.4f}  Median: {np.median(ipc):.4f}")
    print(f"IPC < 1.0: {np.mean(ipc<1.0)*100:.2f}%")
    print(f"r(ipc, power) individual: {np.corrcoef(ipc, power)[0,1]:.4f}")
    q1_mask = ipc < np.percentile(ipc, 25)
    q3_mask = ipc > np.percentile(ipc, 75)
    diff = np.mean(power[q3_mask]) - np.mean(power[q1_mask])
    print(f"Power Q3(IPC) - Q1(IPC): {diff:.2f} W")


def within_runtime_power_variation(power, runtime):
    print_section("4. POWER VARIATION WITHIN RUNTIME BINS")
    log_rt = np.log10(runtime)
    bins = defaultdict(list)
    for p, lr in zip(power, log_rt):
        b = round(lr * 10) / 10
        bins[b].append(p)
    cvs = [np.std(bins[b]) / np.mean(bins[b]) * 100
           for b in sorted(bins.keys()) if len(bins[b]) >= 100]
    cvs = np.array(cvs)
    print(f"Within-bin CV: mean={np.mean(cvs):.2f}%  max={np.max(cvs):.2f}%")


def solution_level_ipc_power(sol_records):
    print_section("5. IPC vs POWER AT SOLUTION LEVEL")
    ipc = np.array([r["avg_ipc"] for r in sol_records
                    if r.get("avg_ipc", 0) > 0 and r.get("avg_power", 0) > 0])
    pwr = np.array([r["avg_power"] for r in sol_records
                    if r.get("avg_ipc", 0) > 0 and r.get("avg_power", 0) > 0])
    print(f"N solutions: {len(ipc):,}")
    print(f"r(ipc, power) solution-level: {np.corrcoef(ipc, pwr)[0,1]:.4f}")
    print(f"r(log_ipc, log_power):        {np.corrcoef(np.log(ipc+0.01), np.log(pwr))[0,1]:.4f}")


def ci95(arr, stat=np.median, n_resamples=2000, rng=42):
    """Bootstrap 95% CI; only for small arrays (N < 5000)."""
    res = sp_bootstrap((arr,), stat, n_resamples=n_resamples, random_state=rng,
                       confidence_level=0.95, method="percentile")
    return res.confidence_interval.low, res.confidence_interval.high


def prop_ci95(k, n):
    """Wilson 95% CI for a proportion k/n."""
    z = 1.96; p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin


def per_problem_energy_diff(sol_records):
    """Per-problem worst-vs-best stats, rank inversions, IPC direction."""
    print_section("6. PER-PROBLEM ENERGY ANALYSIS")
    FREQ = 1.5e9  # Sniper simulation frequency (Hz)
    prob = defaultdict(list)
    n_all = 0
    for r in sol_records:
        n_all += 1
        if r.get("avg_power", 0) > 190 and r.get("avg_energy", 0) > 0 and r.get("avg_cycles", 0) > 0:
            prob[r["problem_id"]].append(r)
    print(f"Solutions: {n_all:,} total; filtered to avg_power>190W: {sum(len(v) for v in prob.values()):,}")

    n_sols_per_prob = []
    energy_diffs, energy_ratios, pwr_diffs = [], [], []
    runtime_contrib, power_contrib = [], []
    best_ipcs, worst_ipcs = [], []
    srt1_total = 0; srt1_ediffs = []
    flip_ediffs = []         # all pairs where energy rank != cycle rank
    srt1_flip_ediffs = []    # same-runtime pairs where energy rank != cycle rank
    wrong_runtime_count = 0  # problems where best-cycle != best-energy solution

    for pid, sols in prob.items():
        if len(sols) < 2:
            continue
        n_sols_per_prob.append(len(sols))
        energies = [s["avg_energy"] for s in sols]
        cycles_list = [s["avg_cycles"] for s in sols]
        powers = [s["avg_power"] for s in sols]
        energy_diffs.append(max(energies) - min(energies))
        energy_ratios.append(max(energies) / min(energies))
        pwr_diffs.append(max(powers) - min(powers))

        min_e_idx = energies.index(min(energies))
        min_c_idx = cycles_list.index(min(cycles_list))
        if min_e_idx != min_c_idx and energies[min_c_idx] > energies[min_e_idx]:
            wrong_runtime_count += 1

        worst = sols[energies.index(max(energies))]
        best = sols[min_e_idx]
        best_ipcs.append(best.get("avg_ipc", 0))
        worst_ipcs.append(worst.get("avg_ipc", 0))
        ta = worst["avg_cycles"] / FREQ; tb = best["avg_cycles"] / FREQ
        Pa, Pb = worst["avg_power"], best["avg_power"]
        dE = Pa * ta - Pb * tb
        if dE > 0:
            # Laspeyres decomposition: runtime component uses P_best, power uses t_worst
            runtime_contrib.append(Pb * (ta - tb) / dE * 100)
            power_contrib.append((Pa - Pb) * ta / dE * 100)

        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                a, b = sols[i], sols[j]
                ediff_pct = (abs(a["avg_energy"] - b["avg_energy"]) /
                             max(a["avg_energy"], b["avg_energy"]) * 100)
                inverted = (a["avg_energy"] < b["avg_energy"]) != (a["avg_cycles"] < b["avg_cycles"])
                if inverted:
                    flip_ediffs.append(ediff_pct)
                ratio = max(a["avg_cycles"], b["avg_cycles"]) / min(a["avg_cycles"], b["avg_cycles"])
                if ratio <= 1.01:
                    srt1_total += 1
                    srt1_ediffs.append(ediff_pct)
                    if inverted:
                        srt1_flip_ediffs.append(ediff_pct)

    er = np.array(energy_ratios)
    rc = np.array(runtime_contrib); pc = np.array(power_contrib)
    bi = np.array([x for x in best_ipcs if x > 0])
    wi = np.array([x for x in worst_ipcs if x > 0])
    srt1_e = np.array(srt1_ediffs)
    flip_e = np.array(flip_ediffs)
    srt1_flip_e = np.array(srt1_flip_ediffs)
    ed = np.array(energy_diffs); pd_arr = np.array(pwr_diffs)
    nspp = np.array(n_sols_per_prob)

    print(f"N problems (>=2 solutions): {len(er):,}")
    print(f"Solutions per problem: mean={np.mean(nspp):.1f}  median={np.median(nspp):.0f}")
    print(f"Energy ratio (worst/best): median={np.median(er):.2f}x  mean={np.mean(er):.2f}x")
    lo, hi = ci95(er); print(f"  95% CI median: [{lo:.2f}, {hi:.2f}]x")
    print(f"  >2x: {np.mean(er>2)*100:.1f}%  >5x: {np.mean(er>5)*100:.1f}%  >10x: {np.mean(er>10)*100:.1f}%")
    print(f"Energy diff (worst-best): median={np.median(ed):.4f} J  mean={np.mean(ed):.4f} J")
    print(f"Power range (worst-best): median={np.median(pd_arr):.3f} W  mean={np.mean(pd_arr):.3f} W")
    print(f"Energy decomp - runtime: median={np.median(rc):.1f}%  power: median={np.median(pc):.1f}%")
    print(f"  (Laspeyres decomp: runtime uses P_best, power uses t_worst)")
    print(f"Power > 5%: {np.mean(pc>5)*100:.1f}% ({int(np.sum(pc>5))} problems)")
    print(f"Wrong-runtime problems (best-cycle != best-energy): {wrong_runtime_count} / {len(er)}")

    # Within-problem r(energy, cycles) — critical for within-vs-cross-problem argument
    wp_r = []
    for pid, sols in prob.items():
        if len(sols) < 3:
            continue
        e_arr = np.array([s["avg_energy"] for s in sols])
        c_arr = np.array([s["avg_cycles"] for s in sols])
        if np.std(e_arr) > 0 and np.std(c_arr) > 0:
            wp_r.append(np.corrcoef(e_arr, c_arr)[0, 1])
    wp_r = np.array(wp_r)
    print(f"Within-problem r(energy, cycles): N={len(wp_r)} problems with >=3 sols")
    print(f"  Median={np.median(wp_r):.4f}  Mean={np.mean(wp_r):.4f}  Std={np.std(wp_r):.4f}")
    lo, hi = ci95(wp_r); print(f"  95% CI median: [{lo:.4f}, {hi:.4f}]")
    print(f"  <0.90: {np.mean(wp_r<0.90)*100:.1f}%  <0.95: {np.mean(wp_r<0.95)*100:.1f}%  <0.5: {np.mean(wp_r<0.5)*100:.1f}%")
    print(f"Same-runtime pairs (<=1% cycles): {srt1_total:,}")
    print(f"  Energy diff >0.5%: {np.mean(srt1_e>0.5)*100:.1f}%  max={np.max(srt1_e):.2f}%")
    print(f"  Median energy diff: {np.median(srt1_e):.3f}%  Mean: {np.mean(srt1_e):.3f}%")
    # analytical CI for median via normal approx (N=436K, CLT applies)
    se_med = 1.2533 * np.std(srt1_e) / np.sqrt(len(srt1_e))
    m = np.median(srt1_e)
    print(f"  95% CI median (normal approx): [{m-1.96*se_med:.3f}, {m+1.96*se_med:.3f}]%")
    print(f"  Same-runtime pairs that are also rank-inverted: {len(srt1_flip_ediffs):,}")
    total_pairs = sum(len(v)*(len(v)-1)//2 for v in prob.values() if len(v) >= 2)
    print(f"Total within-problem pairs: {total_pairs:,}")
    print(f"Rank inversions (all pairs, energy rank != cycle rank):")
    for thr in [0, 1, 2, 5, 10]:
        print(f"  >{thr}% energy diff: {int(np.sum(flip_e>thr)):,} pairs  ({np.sum(flip_e>thr)/total_pairs*100:.3f}%)")
    ipc_lower_frac = np.mean(bi < wi)
    k_ipc = int(np.sum(bi < wi)); n_ipc = len(bi)
    lo, hi = prop_ci95(k_ipc, n_ipc)
    print(f"IPC direction: best has lower IPC: {ipc_lower_frac*100:.1f}% ({k_ipc}/{n_ipc})")
    print(f"  95% CI (Wilson): [{lo*100:.1f}, {hi*100:.1f}]%")
    print(f"  Best mean IPC={np.mean(bi):.3f}  Worst mean IPC={np.mean(wi):.3f}")


def optimization_potential(prob_records):
    print_section("7. OPTIMIZATION POTENTIAL")
    energy_opt = np.array([r["improvement_pct"] for r in prob_records])
    edp_opt = np.array([r["edp_improvement_pct"] for r in prob_records])
    print(f"N problems: {len(energy_opt)}")
    print(f"Energy reduction: mean={np.mean(energy_opt):.2f}%  median={np.median(energy_opt):.2f}%")
    lo, hi = ci95(energy_opt); print(f"  95% CI median: [{lo:.2f}, {hi:.2f}]%")
    n90 = int(np.sum(energy_opt > 90)); n50 = int(np.sum(energy_opt > 50))
    print(f"  >90%: {np.mean(energy_opt>90)*100:.1f}% ({n90}/{len(energy_opt)})  >50%: {np.mean(energy_opt>50)*100:.1f}% ({n50}/{len(energy_opt)})")
    print(f"EDP reduction: mean={np.mean(edp_opt):.2f}%  median={np.median(edp_opt):.2f}%")
    lo, hi = ci95(edp_opt); print(f"  95% CI median: [{lo:.2f}, {hi:.2f}]%")


def main():
    print("Landscape: Energy Optimization Landscape Analysis")
    print("Reproduces all numbers in Section 3 (landscape.tex)")
    power, energy, runtime, ipc = load_individual_measurements()
    sol_records = load_solution_metrics()
    prob_records = load_problem_stats()
    power_distribution(power, energy, runtime, ipc)
    energy_runtime_correlation(energy, runtime, power)
    ipc_distribution(ipc, power)
    within_runtime_power_variation(power, runtime)
    solution_level_ipc_power(sol_records)
    per_problem_energy_diff(sol_records)
    optimization_potential(prob_records)
    print("\nDone. Numbers from completed_samples.jsonl (individual) and solution_metrics.jsonl.")


if __name__ == "__main__":
    main()
