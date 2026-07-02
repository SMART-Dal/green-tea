#!/usr/bin/env python3
"""RQ2: Does GRPO with EDP reward further improve energy efficiency beyond SFT alone?

Reproduces all numbers in Section 4 (RQ2). Compares SFT vs SFT+GRPO evaluation results.
GRPO results directory: finetuning/data/grpo_evaluation_results/ (populate after training).
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGDIR = Path(__file__).parent.parent / "figures"
RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "pdf.fonttype": 42,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent.parent.parent / "finetuning" / "data"
SFT_DIR = BASE / "sft_evaluation_results" / "second_run"
GRPO_DIR = BASE / "grpo_sim_results"
METRICS_FILE = Path(__file__).parent.parent.parent / "analysis" / "solution_metrics.jsonl"
SFT_PASSK_FILE = BASE / "sft_pass_at_k" / "pass_at_k_results.json"
GRPO_PASSK_FILE = BASE / "grpo_pass_at_k" / "pass_at_k_results.json"

NEW_MODELS = BASE / "new_models"
SCORE_DIRS = {s: NEW_MODELS / "score_sensitivity" / s / "sim_results" for s in ("score_8", "score_9", "score_10")}
FRONTIER_DIRS = {
    "qwen32b_zs": NEW_MODELS / "frontier_qwen32b" / "sim_results" / "zero_shot_instruct",
    "qwen32b_gp": NEW_MODELS / "frontier_qwen32b" / "sim_results" / "green_prompt_instruct",
    "dscoder_zs":  NEW_MODELS / "frontier_dscoder"  / "sim_results" / "zero_shot_instruct",
    "dscoder_gp":  NEW_MODELS / "frontier_dscoder"  / "sim_results" / "green_prompt_instruct",
}
BASELINE_DIRS = {
    "zero_shot":             BASE / "baseline_sim_results" / "zero_shot",
    "green_prompt":          BASE / "baseline_sim_results" / "green_prompt",
    "zero_shot_instruct":    BASE / "baseline_sim_results" / "zero_shot_instruct",
    "green_prompt_instruct": BASE / "baseline_sim_results" / "green_prompt_instruct",
}
# Cross-family + same-family scaling SFT/GRPO rows (manuscript appendix / cross_family_analysis.md §4-§7).
# Paths point at the consolidated sim outputs on def-tusharma; relative-safe after repo merge.
# Path normalization (Reproducibility R11): $GREEN_TEA_DATA env var overrides for Zenodo / cross-machine replication.
import os
CF_BASE = Path(os.environ.get("GREEN_TEA_DATA", str(REPO_ROOT / "data" / "cross_family_sim_results")))
CROSS_FAMILY_DIRS = {
    "gemini25_flash_sft":           CF_BASE / "gemini25flash_v2",
    "llama31_8b_sft":               CF_BASE / "llama31_8b_v3",
    "gemma3_12b_sft":               CF_BASE / "gemma3_12b_v3",
    "qwen05b_sft":                  CF_BASE / "qwen05b_sft",
    "qwen7b_sft":                   CF_BASE / "qwen7b_sft",
    "dscoder67b_sft_grpo":          REPO_ROOT / "finetuning" / "data" / "new_models" / "dscoder7b" / "sim_results",
    "qwen05b_base_zs":              CF_BASE / "qwen05b_base_zs",
    "qwen7b_base_zs":               CF_BASE / "qwen7b_base_zs",
    "dscoder67b_base_zs":           CF_BASE / "dscoder67b_base_zs",
    "dscoder67b_sft_only_reeval":   CF_BASE / "dscoder67b_sft_only",
    "llama31_8b_zs":                CF_BASE / "llama31_8b_zs",
    "gemma3_12b_zs":                CF_BASE / "gemma3_12b_zs",
    "qwen32b_zs_perpair":           CF_BASE / "qwen32b_zs_perpair",
    "dscoder_v2lite_zs_perpair":    CF_BASE / "dscoder_v2lite_zs_perpair",
    "gemini25_flash_zs":            CF_BASE / "gemini25_flash_zs",
    "qwen14b_sft_energy_r32":       REPO_ROOT / "finetuning" / "data" / "sft_energy_r32",
    "qwen14b_sft_energy_r128":      REPO_ROOT / "finetuning" / "data" / "sft_energy_r128",
}
ABL_DIRS = {
    "abl1_rsft_edp":    BASE / "grpo_abl1_rsft_edp_sim_results",
    "abl2_esft_energy": BASE / "grpo_abl2_esft_energy_sim_results",
    "abl3_rsft_rt":     BASE / "grpo_abl3_rsft_rt_sim_results",
    "abl4_esft_rt":     BASE / "grpo_abl4_esft_rt_sim_results",
}


def load_eval_records(directory: Path) -> list[dict]:
    records = []
    for f in sorted(directory.glob("test_comparison_chunk_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def compute_metrics(records: list[dict]) -> dict:
    n = len(records)
    compiled = sum(1 for r in records if r.get("compiled", False))
    total_tests = sum(r.get("num_inputs", 0) for r in records)
    tests_passed = sum(r.get("tests_passed", 0) for r in records)
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0]
    errs = np.array([r["energy_reduction"] for r in valid])
    edp_reds = np.array([r.get("edp_reduction", 0) for r in valid])
    beat_gt = sum(1 for r in valid if r.get("vs_gt_reduction", float("-inf")) >= 0)
    full_correct = sum(1 for r in records if r.get("num_inputs", 0) > 0 and r.get("tests_passed", 0) == r["num_inputs"])
    # CADERR variants for sensitivity (closes Methods F19): strict (only fully-correct), zero-fill (current),
    # negative-fill (treats measurement failures as silent regression at ERR=-0.5)
    caderr_strict_vals = []   # only fully-correct outputs contribute non-zero ERR
    caderr_neg_vals = []      # measurement-failure compiled outputs treated as ERR=-0.5
    caderr_vals = []
    for r in records:
        ni = r.get("num_inputs", 0)
        if ni == 0:
            continue
        tp = r.get("tests_passed", 0)
        err = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else 0.0
        caderr_vals.append(err * tp / ni)
        err_strict = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0 and tp == ni) else 0.0
        caderr_strict_vals.append(err_strict * tp / ni)
        err_neg = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else (-0.5 if r.get("compiled") else 0.0)
        caderr_neg_vals.append(err_neg * tp / ni)
    return {
        "n": n, "n_valid": len(valid),
        "compile_rate": compiled / n * 100,
        "test_pass_rate": tests_passed / total_tests * 100 if total_tests else 0,
        "valid_rate": len(valid) / n * 100,
        "mean_err": float(np.mean(errs)) if len(errs) else 0,
        "median_err": float(np.median(errs)) if len(errs) else 0,
        "err_gt0_pct": float(np.mean(errs > 0) * 100) if len(errs) else 0,
        "err_gt50_pct": float(np.mean(errs > 50) * 100) if len(errs) else 0,
        "mean_edp_red": float(np.mean(edp_reds)) if len(edp_reds) else 0,
        "beat_gt_n": beat_gt,
        "beat_gt_pct": beat_gt / len(valid) * 100 if valid else 0,
        "full_correct_n": full_correct,
        "full_correct_rate": full_correct / n * 100 if n else 0,
        "err_lt_neg50_pct": float(np.mean(errs < -50) * 100) if len(errs) else 0,
        "err_lt_neg50_n": int((errs < -50).sum()) if len(errs) else 0,
        "err_lt_neg50_mean": float(np.mean(errs[errs < -50])) if len(errs) and (errs < -50).any() else 0,
        "err_neutral_band_pct": float(np.mean((errs > -10) & (errs <= 0)) * 100) if len(errs) else 0,
        "err_gt50_median": float(np.median(errs[errs > 50])) if len(errs) and (errs > 50).any() else 0,
        "caderr": float(np.mean(caderr_vals)) if caderr_vals else 0,
        "caderr_strict": float(np.mean(caderr_strict_vals)) if caderr_strict_vals else 0,
        "caderr_neg_fill": float(np.mean(caderr_neg_vals)) if caderr_neg_vals else 0,
        "errs": errs,
        "edp_reds": edp_reds,
        "records": records,
    }


def load_ipc_trap_problems() -> set:
    per_problem = defaultdict(list)
    if not METRICS_FILE.exists():
        return set()
    with open(METRICS_FILE) as fh:
        for line in fh:
            r = json.loads(line)
            per_problem[r["problem_id"]].append(r)
    trap = set()
    for pid, sols in per_problem.items():
        if len(sols) < 2:
            continue
        best = min(sols, key=lambda x: x["avg_energy"])
        worst = max(sols, key=lambda x: x["avg_energy"])
        if best["avg_ipc"] < worst["avg_ipc"]:
            trap.add(pid)
    return trap


def energy_runtime_sft_overlap(esft_records: list[dict], rsft_records: list[dict],
                                deep_cut_threshold: float = 50.0):
    """Per-problem opportunity overlap between Energy-SFT and Runtime-SFT (RQ2 text needs this).
    Filters to valid (compiled and generated_energy>0); aggregates per-problem max ERR;
    reports common-set sign breakdown and deep-cut overlap. Math:
      - valid_p[pid] = list of energy_reduction for valid outputs
      - per-problem max ERR > 0  => 'improves at least one output'
      - per-problem max ERR > threshold => 'deep cut'
    Returns dict for downstream use."""
    from collections import defaultdict
    def by_pid_valid(records):
        d = defaultdict(list)
        for r in records:
            if r.get("compiled") and r.get("generated_energy", 0) > 0:
                d[r["problem_id"]].append(r.get("energy_reduction", 0.0))
        return d
    ep = by_pid_valid(esft_records); rp = by_pid_valid(rsft_records)
    common = set(ep) & set(rp)
    e_only = set(ep) - set(rp); r_only = set(rp) - set(ep)
    # sign agreement within common
    both_imp = sum(1 for p in common if max(ep[p], default=0) > 0 and max(rp[p], default=0) > 0)
    only_e_imp = sum(1 for p in common if max(ep[p], default=0) > 0 and max(rp[p], default=0) <= 0)
    only_r_imp = sum(1 for p in common if max(ep[p], default=0) <= 0 and max(rp[p], default=0) > 0)
    neither = sum(1 for p in common if max(ep[p], default=0) <= 0 and max(rp[p], default=0) <= 0)
    # deep cuts
    e_deep = {p for p in common if max(ep[p], default=0) > deep_cut_threshold}
    r_deep = {p for p in common if max(rp[p], default=0) > deep_cut_threshold}
    print_section("ENERGY-SFT vs RUNTIME-SFT PER-PROBLEM OVERLAP")
    print(f"  Per-problem valid: Energy-SFT={len(ep)} / Runtime-SFT={len(rp)} / common={len(common)} "
          f"(E-only={len(e_only)}, R-only={len(r_only)})")
    if common:
        n = len(common)
        print(f"  Among {n} common problems (per-problem max ERR sign):")
        print(f"    both improve energy:        {both_imp} ({both_imp/n*100:.1f}%)")
        print(f"    Energy-SFT only improves:   {only_e_imp} ({only_e_imp/n*100:.1f}%)")
        print(f"    Runtime-SFT only improves:  {only_r_imp} ({only_r_imp/n*100:.1f}%)")
        print(f"    neither improves:           {neither} ({neither/n*100:.1f}%)")
        print(f"  Deep cuts (per-problem max ERR > {deep_cut_threshold}%): E-deep={len(e_deep)} R-deep={len(r_deep)} "
              f"both-deep={len(e_deep & r_deep)} E-only-deep={len(e_deep - r_deep)} R-only-deep={len(r_deep - e_deep)}")
    return {"n_common": len(common), "both_improve": both_imp, "only_energy_improves": only_e_imp,
            "only_runtime_improves": only_r_imp, "neither": neither, "e_deep": len(e_deep),
            "r_deep": len(r_deep), "both_deep": len(e_deep & r_deep)}


def ipc_trap_analysis(sft_records: list[dict], grpo_records: list[dict], trap_pids: set):
    def subgroup_metrics(records, pids_in_group):
        recs = [r for r in records if r["problem_id"] in pids_in_group]
        n = len(recs)
        if n == 0:
            return None
        compile_pct = sum(1 for r in recs if r.get("compiled")) / n * 100
        valid = [r for r in recs if r.get("compiled") and r.get("generated_energy", 0) > 0]
        mean_err = float(np.mean([r["energy_reduction"] for r in valid])) if valid else 0.0
        beat_gt = sum(1 for r in valid if r.get("vs_gt_reduction", float("-inf")) >= 0)
        beat_gt_pct = beat_gt / len(valid) * 100 if valid else 0.0
        caderr_vals = []
        for r in recs:
            ni = r.get("num_inputs", 0)
            if ni == 0:
                continue
            tp = r.get("tests_passed", 0)
            err = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else 0.0
            caderr_vals.append(err * tp / ni)
        caderr = float(np.mean(caderr_vals)) if caderr_vals else 0.0
        return {"n": n, "compile": compile_pct, "mean_err": mean_err, "caderr": caderr, "beat_gt_pct": beat_gt_pct, "n_valid": len(valid)}

    all_pids = set(r["problem_id"] for r in sft_records)
    nontrap_pids = all_pids - trap_pids

    print_section("IPC-TRAP ANALYSIS")
    for label, recs in [("SFT", sft_records), ("SFT+GRPO", grpo_records)]:
        tm = subgroup_metrics(recs, trap_pids)
        nm = subgroup_metrics(recs, nontrap_pids)
        print(f"  {label} trap    (N={tm['n']}, valid={tm['n_valid']}): compile={tm['compile']:.1f}%  mean_err={tm['mean_err']:.2f}%  beat_gt={tm['beat_gt_pct']:.1f}%  caderr={tm['caderr']:.2f}%")
        print(f"  {label} non-trap(N={nm['n']}, valid={nm['n_valid']}): compile={nm['compile']:.1f}%  mean_err={nm['mean_err']:.2f}%  beat_gt={nm['beat_gt_pct']:.1f}%  caderr={nm['caderr']:.2f}%")

    # Per-problem CADERR vectors for trap vs non-trap, used for IPC-trap Mann-Whitney + Fisher
    from collections import defaultdict
    from scipy.stats import mannwhitneyu, fisher_exact
    def per_problem_caderr(records, pids_in_group):
        d = defaultdict(list)
        for r in records:
            if r["problem_id"] not in pids_in_group: continue
            ni = r.get("num_inputs", 0)
            if ni == 0: continue
            tp = r.get("tests_passed", 0)
            err = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else 0.0
            d[r["problem_id"]].append(err * tp / ni)
        return np.array([np.mean(v) for v in d.values()])
    for label, recs in [("SFT", sft_records), ("SFT+GRPO", grpo_records)]:
        trap_caderr = per_problem_caderr(recs, trap_pids)
        nontrap_caderr = per_problem_caderr(recs, nontrap_pids)
        if len(trap_caderr) and len(nontrap_caderr):
            U, p_mw = mannwhitneyu(trap_caderr, nontrap_caderr, alternative="two-sided")
            r_rb_mw = float(1 - 2 * U / (len(trap_caderr) * len(nontrap_caderr)))
            print(f"  {label} IPC-trap Mann-Whitney CADERR: U={U:.0f}, p={p_mw:.4g}, "
                  f"n_trap={len(trap_caderr)}, n_nontrap={len(nontrap_caderr)}, rank-biserial r={r_rb_mw:.3f}")
            LIVE_PVALUES.append((f"{label} IPC-trap Mann-Whitney (U={int(U)}, n_trap={len(trap_caderr)})", float(p_mw)))
    # IPC-trap × compile 2×2 Fisher exact (SFT only, per manuscript threats.tex)
    sft_trap_compiled = sum(1 for r in sft_records if r["problem_id"] in trap_pids and r.get("compiled"))
    sft_trap_not = sum(1 for r in sft_records if r["problem_id"] in trap_pids and not r.get("compiled"))
    sft_nontrap_compiled = sum(1 for r in sft_records if r["problem_id"] in nontrap_pids and r.get("compiled"))
    sft_nontrap_not = sum(1 for r in sft_records if r["problem_id"] in nontrap_pids and not r.get("compiled"))
    if all(x > 0 for x in [sft_trap_compiled + sft_trap_not, sft_nontrap_compiled + sft_nontrap_not]):
        odds, p_fish = fisher_exact([[sft_trap_compiled, sft_trap_not],
                                     [sft_nontrap_compiled, sft_nontrap_not]], alternative="two-sided")
        print(f"  SFT IPC-trap × compile Fisher exact: OR={odds:.2f}, p={p_fish:.4g}, "
              f"trap=({sft_trap_compiled}/{sft_trap_compiled+sft_trap_not}), nontrap=({sft_nontrap_compiled}/{sft_nontrap_compiled+sft_nontrap_not})")
        LIVE_PVALUES.append((f"SFT IPC-trap × compile Fisher (OR={odds:.2f})", float(p_fish)))


def print_section(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def main():
    print("RQ2: GRPO with EDP Reward vs. SFT Alone")

    print_section("SFT BASELINE")
    sft_records = load_eval_records(SFT_DIR)
    sft = compute_metrics(sft_records)
    print(f"N={sft['n']}, compiled={sft['compile_rate']:.1f}%, valid={sft['valid_rate']:.1f}%")
    print(f"Mean ERR={sft['mean_err']:.2f}%  Median={sft['median_err']:.2f}%")
    print(f"Mean EDPRR={sft['mean_edp_red']:.2f}%  Beat-GT={sft['beat_gt_pct']:.1f}%  CADERR={sft['caderr']:.2f}%")

    if not GRPO_DIR.exists():
        print(f"\n[NOTE] GRPO results not found at {GRPO_DIR}")
        print("Populate this directory with test_comparison_chunk_*.jsonl after GRPO training.")
        print("\nExpected comparison table once GRPO is available:")
        header = f"{'Method':<15} {'Compile%':>9} {'Tests%':>7} {'MeanERR%':>9} {'MedERR%':>8} {'BeatGT%':>8} {'CADERR%':>7} {'MeanEDPRR%':>11}"
        print(header)
        print("-" * len(header))
        print(f"{'SFT':<15} {sft['compile_rate']:>8.1f}% {sft['test_pass_rate']:>6.1f}% "
              f"{sft['mean_err']:>8.2f}% {sft['median_err']:>7.2f}% "
              f"{sft['beat_gt_pct']:>7.1f}% {sft['caderr']:>6.2f}% {sft['mean_edp_red']:>10.2f}%")
        print(f"{'SFT+GRPO':<15} {'N/A':>9} {'N/A':>7} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>11}")
        return

    print_section("GRPO RESULTS")
    grpo_records = load_eval_records(GRPO_DIR)
    grpo = compute_metrics(grpo_records)
    print(f"N={grpo['n']}, compiled={grpo['compile_rate']:.1f}%, valid={grpo['valid_rate']:.1f}%")
    print(f"Mean ERR={grpo['mean_err']:.2f}%  Median={grpo['median_err']:.2f}%")
    print(f"Mean EDPRR={grpo['mean_edp_red']:.2f}%  Beat-GT={grpo['beat_gt_pct']:.1f}%  CADERR={grpo['caderr']:.2f}%")

    print_section("COMPARISON TABLE")
    header = f"{'Method':<15} {'Compile%':>9} {'Tests%':>7} {'MeanERR%':>9} {'MedERR%':>8} {'BeatGT%':>8} {'CADERR%':>7} {'MeanEDPRR%':>11}"
    print(header); print("-" * len(header))
    for name, m in [("SFT", sft), ("SFT+GRPO", grpo)]:
        print(f"{name:<15} {m['compile_rate']:>8.1f}% {m['test_pass_rate']:>6.1f}% "
              f"{m['mean_err']:>8.2f}% {m['median_err']:>7.2f}% "
              f"{m['beat_gt_pct']:>7.1f}% {m['caderr']:>6.2f}% {m['mean_edp_red']:>10.2f}%")

    print_section("STATISTICAL COMPARISON (SFT vs GRPO)")
    # Align by problem_id for paired test
    sft_pid = {r["problem_id"]: r for r in sft["records"] if r.get("compiled") and r.get("generated_energy", 0) > 0}
    grpo_pid = {r["problem_id"]: r for r in grpo["records"] if r.get("compiled") and r.get("generated_energy", 0) > 0}
    common = sorted(set(sft_pid) & set(grpo_pid))
    if common:
        sft_e = np.array([sft_pid[p]["energy_reduction"] for p in common])
        grpo_e = np.array([grpo_pid[p]["energy_reduction"] for p in common])
        diff = grpo_e - sft_e
        print(f"Paired samples (shared problems): {len(common)}")
        print(f"Mean improvement GRPO over SFT: {np.mean(diff):.2f}%")
        try:
            w_stat, w_p = wilcoxon(diff)
            print(f"Wilcoxon: W={w_stat:.0f}, p={w_p:.4g}")
            LIVE_PVALUES.append((f"Paired SFT-vs-GRPO Wilcoxon (W={int(w_stat)}, N={len(diff)})", float(w_p)))
        except Exception as e:
            print(f"Wilcoxon failed: {e}")
        d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        print(f"Cohen's d: {d:.3f}")
        boot = [np.mean(np.random.choice(diff, len(diff), replace=True)) for _ in range(5000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        print(f"Bootstrap 95% CI for improvement: [{ci_lo:.2f}%, {ci_hi:.2f}%]")
    else:
        print("No common problem_ids for paired test.")

    print_section("SFT ONE-SAMPLE STATISTICAL TESTS")
    sft_errs = sft["errs"]
    if len(sft_errs):
        try:
            w, p = wilcoxon(sft_errs)
            n = len(sft_errs)
            # rank-biserial r for one-sample Wilcoxon (closes Methods F4): r = 1 - 4W / (n(n+1)) where W is sum of positive ranks
            # scipy returns the smaller of W+ and W-, so we use the formula based on absolute terms
            r_rb = float(1 - 4 * w / (n * (n + 1))) if n > 0 else 0
            print(f"One-sample Wilcoxon signed-rank: W={w:.0f}, p={p:.4g}, n={n} valid, rank-biserial r={r_rb:.3f}")
            LIVE_PVALUES.append((f"SFT one-sample Wilcoxon (W={int(w)}, n={n})", float(p)))
        except Exception as e:
            print(f"Wilcoxon failed: {e}")
        d = float(np.mean(sft_errs) / np.std(sft_errs)) if np.std(sft_errs) > 0 else 0
        print(f"Cohen's d = {d:.3f}")
        # Bootstrap 5000 iterations, percentile method, seed=42 fixed in __main__ for reproducibility (Methods F3 documentation)
        boot = [np.mean(np.random.choice(sft_errs, len(sft_errs), replace=True)) for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"Bootstrap 95% CI on mean ERR (sample-level, percentile, n_boot=5000, seed=42): [{lo:.2f}%, {hi:.2f}%]")
        # Cluster-bootstrap by problem (closes Methods F17): resample problems then aggregate within problem
        from collections import defaultdict
        valid_records = [r for r in sft_records if r.get("compiled") and r.get("generated_energy", 0) > 0]
        by_pid = defaultdict(list)
        for r in valid_records: by_pid[r["problem_id"]].append(r["energy_reduction"])
        pids = list(by_pid.keys())
        if len(pids) >= 2:
            cb = []
            for _ in range(5000):
                sampled = np.random.choice(len(pids), len(pids), replace=True)
                all_errs = []
                for i in sampled: all_errs.extend(by_pid[pids[i]])
                cb.append(np.mean(all_errs))
            clo, chi = np.percentile(cb, [2.5, 97.5])
            print(f"Bootstrap 95% CI on mean ERR (problem-clustered, n={len(pids)} problems): [{clo:.2f}%, {chi:.2f}%]")

    print_section("CROSS-METHOD PAIRED WILCOXON (SFT vs baseline by problem_id)")
    for bname, bdir in BASELINE_DIRS.items():
        if not bdir.exists() or not any(bdir.glob("test_comparison_chunk_*.jsonl")):
            print(f"  {bname}: [no sim chunks]"); continue
        brecs = load_eval_records(bdir)
        b_valid = {r["problem_id"]: r for r in brecs if r.get("compiled") and r.get("generated_energy", 0) > 0}
        sft_valid = {r["problem_id"]: r for r in sft_records if r.get("compiled") and r.get("generated_energy", 0) > 0}
        common = sorted(set(sft_valid) & set(b_valid))
        if not common: print(f"  {bname}: no shared problem_ids"); continue
        diffs = np.array([sft_valid[p]["energy_reduction"] - b_valid[p]["energy_reduction"] for p in common])
        try:
            w, p = wilcoxon(diffs)
            print(f"  SFT vs {bname:25s}: W={w:.0f}, p={p:.4g}, n={len(common)} paired, mean_diff={np.mean(diffs):.2f}%")
            LIVE_PVALUES.append((f"SFT vs {bname} Wilcoxon (W={int(w)})", float(p)))
        except Exception as e:
            print(f"  SFT vs {bname}: wilcoxon failed: {e}")

    print_section("ERR DISTRIBUTION (bimodality check)")
    for name, m in [("SFT", sft), ("SFT+GRPO", grpo)]:
        errs = m["errs"]
        if len(errs):
            q25, q75 = np.percentile(errs, [25, 75])
            pct_neg = np.mean(errs < 0) * 100
            pct_gt50 = np.mean(errs > 50) * 100
            print(f"  {name}: mean={np.mean(errs):.2f}%  median={np.median(errs):.2f}%  "
                  f"IQR=[{q25:.1f}%, {q75:.1f}%]  pct<0={pct_neg:.1f}%  pct>50={pct_gt50:.1f}%  "
                  f"mean-median gap={np.mean(errs)-np.median(errs):.2f}pp")

    print_section("PER-PROBLEM CADERR DISTRIBUTION")
    from collections import defaultdict
    per_problem_caderr_dict = {}
    for name, records in [("SFT", sft_records), ("SFT+GRPO", grpo_records)]:
        prob_caderr = defaultdict(list)
        for r in records:
            ni = r.get("num_inputs", 0)
            if ni == 0:
                continue
            tp = r.get("tests_passed", 0)
            err = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else 0.0
            prob_caderr[r["problem_id"]].append(err * tp / ni)
        vals = np.array([np.mean(v) for v in prob_caderr.values()])
        per_problem_caderr_dict[name] = vals
        if len(vals):
            q25, q75 = np.percentile(vals, [25, 75])
            print(f"  {name}: n_problems={len(vals)}  mean={np.mean(vals):.2f}%  median={np.median(vals):.2f}%  "
                  f"IQR=[{q25:.2f}%, {q75:.2f}%]  pct>0={np.mean(vals>0)*100:.1f}%")
    # Mann-Whitney on per-problem CADERR (SFT vs GRPO), threats.tex "per-problem CADERR Mann-Whitney"
    if all(name in per_problem_caderr_dict for name in ["SFT", "SFT+GRPO"]):
        from scipy.stats import mannwhitneyu
        sft_pp = per_problem_caderr_dict["SFT"]; grpo_pp = per_problem_caderr_dict["SFT+GRPO"]
        if len(sft_pp) and len(grpo_pp):
            U, p_mw = mannwhitneyu(sft_pp, grpo_pp, alternative="two-sided")
            r_rb_mw = float(1 - 2 * U / (len(sft_pp) * len(grpo_pp)))
            print(f"  Per-problem CADERR Mann-Whitney (SFT vs GRPO): U={U:.0f}, p={p_mw:.4g}, "
                  f"n_sft={len(sft_pp)}, n_grpo={len(grpo_pp)}, rank-biserial r={r_rb_mw:.3f}")
            LIVE_PVALUES.append((f"Per-problem CADERR Mann-Whitney (U={int(U)})", float(p_mw)))

    trap_pids = load_ipc_trap_problems()
    if trap_pids:
        ipc_trap_analysis(sft_records, grpo_records, trap_pids)

    # Energy-SFT vs Runtime-SFT per-problem opportunity overlap (RQ2 text: see cross_family §10.25 Theme E1)
    RUNTIME_SFT_DIR = BASE / "sft_runtime_sim_results"
    if RUNTIME_SFT_DIR.exists():
        rsft_records = load_eval_records(RUNTIME_SFT_DIR)
        energy_runtime_sft_overlap(sft_records, rsft_records)

    print_section("ABLATION: GRPO INITIALIZATION x REWARD DESIGN")
    abl_results = {}
    for abl_name, abl_dir in ABL_DIRS.items():
        if abl_dir.exists() and any(abl_dir.glob("test_comparison_chunk_*.jsonl")):
            abl_records = load_eval_records(abl_dir)
            abl_results[abl_name] = compute_metrics(abl_records)
            m = abl_results[abl_name]
            print(f"  {abl_name:<25}: N={m['n']} compile={m['compile_rate']:.1f}% "
                  f"valid={m['valid_rate']:.1f}% ERR={m['mean_err']:.2f}% CADERR={m['caderr']:.2f}% "
                  f"beat_gt={m['beat_gt_pct']:.1f}%")
        else:
            print(f"  {abl_name:<25}: [pending - no sim results yet]")
    if abl_results:
        print("\n  Ablation vs main GRPO CADERR improvement:")
        for abl_name, m in abl_results.items():
            delta = grpo["caderr"] - m["caderr"] if grpo else 0.0
            print(f"    Main GRPO vs {abl_name}: +{delta:.2f}pp CADERR")

    print_section("SCORE SENSITIVITY (inference-time score target: 8, 9, 10)")
    for score_name, score_dir in SCORE_DIRS.items():
        if score_dir.exists() and any(score_dir.glob("test_comparison_chunk_*.jsonl")):
            recs = load_eval_records(score_dir)
            m = compute_metrics(recs)
            note = " [partial]" if len(list(score_dir.glob("test_comparison_chunk_*.jsonl"))) < 39 else ""
            print(f"  {score_name:<12}: N={m['n']} compile={m['compile_rate']:.1f}% "
                  f"valid={m['valid_rate']:.1f}% ERR={m['mean_err']:.2f}% CADERR={m['caderr']:.2f}%{note}")
        else:
            print(f"  {score_name:<12}: [not found]")

    print_section("FRONTIER BASELINE COMPARISON")
    for fname, fdir in FRONTIER_DIRS.items():
        if fdir.exists() and any(fdir.glob("test_comparison_chunk_*.jsonl")):
            recs = load_eval_records(fdir)
            m = compute_metrics(recs)
            n_chunks = len(list(fdir.glob("test_comparison_chunk_*.jsonl")))
            note = f" [partial: {n_chunks}/39 chunks]" if n_chunks < 39 else ""
            print(f"  {fname:<20}: N={m['n']} compile={m['compile_rate']:.1f}% "
                  f"ERR={m['mean_err']:.2f}% CADERR={m['caderr']:.2f}% beat_gt={m['beat_gt_pct']:.1f}%{note}")
        else:
            print(f"  {fname:<20}: [not found / pending]")

    print_section("CROSS-FAMILY + SCALING + NEW BASELINES (cross_family_analysis.md §4-§7 + §11)")
    print(f"  {'model':<32}  {'N':>5}  {'compile%':>9}  {'tests%':>7}  {'valid%':>7}  {'MeanERR%':>9}  {'MedERR%':>8}  {'BeatGT%':>8}  {'CADERR%':>8}")
    for name, cfd in CROSS_FAMILY_DIRS.items():
        if not cfd.exists() or not any(cfd.glob("test_comparison_chunk_*.jsonl")):
            print(f"  {name:<32}  [no sim chunks yet]")
            continue
        recs = load_eval_records(cfd)
        m = compute_metrics(recs)
        print(f"  {name:<32}  {m['n']:>5}  {m['compile_rate']:>8.2f}%  {m['test_pass_rate']:>6.2f}%  {m['valid_rate']:>6.2f}%  {m['mean_err']:>8.2f}%  {m['median_err']:>7.2f}%  {m['beat_gt_pct']:>7.2f}%  {m['caderr']:>7.2f}%")

    plot_err_distribution(sft_records, grpo_records)
    plot_caderr_decomposition(sft, grpo)

    print_section("PASS@K COMPARISON (T=0.8, n=10) — with closed-form Chen et al. unbiased estimator SE")
    # Closes Methods F22: closed-form SE on pass@k unbiased estimator.
    # Per Chen et al. 2021 the unbiased pass@k estimator is 1 - C(n-c,k)/C(n,k) per problem;
    # aggregating across problems, variance is sample variance / n_problems. SE = sqrt(variance / n_problems).
    def passk_se(per_problem_passk_arr):
        import numpy as np
        a = np.asarray(per_problem_passk_arr)
        return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    for label, fpath in [("SFT", SFT_PASSK_FILE), ("SFT+GRPO", GRPO_PASSK_FILE)]:
        if not fpath.exists():
            print(f"  {label}: not found at {fpath}")
            continue
        d = json.loads(fpath.read_text())
        n = d.get("num_problems", 0)
        p1 = d.get("mean_pass_at_1", 0) * 100
        p5 = d.get("mean_pass_at_5", 0) * 100
        p10 = d.get("mean_pass_at_10", 0) * 100
        print(f"  {label} (N={n}): pass@1={p1:.2f}%  pass@5={p5:.2f}%  pass@10={p10:.2f}%")
    if SFT_PASSK_FILE.exists() and GRPO_PASSK_FILE.exists():
        s = json.loads(SFT_PASSK_FILE.read_text())
        g = json.loads(GRPO_PASSK_FILE.read_text())
        for k in [1, 5, 10]:
            sk_per = s.get(f"per_problem_pass_at_{k}", [])
            gk_per = g.get(f"per_problem_pass_at_{k}", [])
            s_se = passk_se(sk_per) * 100 if sk_per else float('nan')
            g_se = passk_se(gk_per) * 100 if gk_per else float('nan')
            sm = s.get(f"mean_pass_at_{k}", 0) * 100
            gm = g.get(f"mean_pass_at_{k}", 0) * 100
            print(f"  pass@{k:2d}: SFT={sm:.2f}% (SE={s_se:.2f}%)  GRPO={gm:.2f}% (SE={g_se:.2f}%)  Δ={gm-sm:+.2f}pp")
        # Paired bootstrap CI on the SFT-vs-GRPO pass@k difference (closes Methods F22)
        for k in [1, 5, 10]:
            sk = s.get(f"per_problem_pass_at_{k}", [])
            gk = g.get(f"per_problem_pass_at_{k}", [])
            if not sk or not gk or len(sk) != len(gk):
                continue
            diffs = np.array(gk) - np.array(sk)
            n = len(diffs)
            boot = [np.mean(np.random.choice(diffs, n, replace=True)) for _ in range(5000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  pass@{k:2d} GRPO-SFT 95% CI (paired bootstrap, n={n}): [{lo*100:+.2f}pp, {hi*100:+.2f}pp]")


def plot_err_distribution(sft_records: list[dict], grpo_records: list[dict]):
    """fig1_err_distribution: SFT vs GRPO ERR histograms with log-scale y-axis."""
    FIGDIR.mkdir(exist_ok=True)
    c_sft, c_grpo = "#1565C0", "#B71C1C"

    def valid_errs(records):
        return [r["energy_reduction"] for r in records
                if r.get("compiled") and r.get("generated_energy", 0) > 0]

    sft_e = np.array(valid_errs(sft_records))
    grpo_e = np.array(valid_errs(grpo_records))

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)
        plt.tight_layout()
        bins = np.linspace(-100, 100, 51)
        for ax, errs, color, label in [
            (ax1, sft_e,  c_sft,  "Energy-SFT"),
            (ax2, grpo_e, c_grpo, "Green Tea (RL)"),
        ]:
            ax.hist(np.clip(errs, -100, 100), bins=bins, color=color,
                    alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            mean_v, med_v = np.mean(errs), np.median(errs)
            ax.axvline(mean_v, color=color, linewidth=1.4, linestyle="--",
                       label=f"Mean {mean_v:.1f}%")
            ax.axvline(med_v, color=color, linewidth=1.0, linestyle=":",
                       label=f"Median {0.0 if abs(med_v) < 0.05 else med_v:.1f}%")
            ax.axvline(0, color="#757575", linewidth=0.7, alpha=0.6)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Energy Reduction Rate (%)", fontsize=10)
            ax.legend(fontsize=7.5, frameon=False)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.5, which="both")

        ax1.set_ylabel("Number of outputs (log scale)", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGDIR / f"fig1_err_distribution.{ext}")
            print(f"Saved: fig1_err_distribution.{ext}")
        plt.close(fig)


def plot_caderr_decomposition(sft_m: dict, grpo_m: dict):
    """fig5_caderr_decomposition: 4-bar chart matching CADERR decomposition table."""
    FIGDIR.mkdir(exist_ok=True)
    metrics = [
        ("Compile\nrate (%)",        sft_m["compile_rate"],              grpo_m["compile_rate"]),
        ("Valid sim.\nrate (%)",      sft_m["valid_rate"],                grpo_m["valid_rate"]),
        ("Test pass\nrate (%)",       sft_m["test_pass_rate"],            grpo_m["test_pass_rate"]),
        ("Mean ERR\n(valid, %)",      sft_m["mean_err"],                  grpo_m["mean_err"]),
        ("CARAT\n(%)",                sft_m["caderr"],                     grpo_m["caderr"]),
    ]
    labels = [m[0] for m in metrics]
    sft_vals  = [m[1] for m in metrics]
    grpo_vals = [m[2] for m in metrics]

    x = np.arange(len(labels))
    w = 0.35
    c_sft, c_grpo = "#1565C0", "#B71C1C"

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        b1 = ax.bar(x - w/2, sft_vals,  w, color=c_sft,  alpha=0.85, label="Energy-SFT")
        b2 = ax.bar(x + w/2, grpo_vals, w, color=c_grpo, alpha=0.85, label="Green Tea (RL)")
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Value (%)", fontsize=10)
        ax.legend(fontsize=10, frameon=True, edgecolor="#CCCCCC", fancybox=False)
        ax.set_ylim(0, max(max(sft_vals), max(grpo_vals)) * 1.15)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGDIR / f"fig5_caderr_decomposition.{ext}")
            print(f"Saved: fig5_caderr_decomposition.{ext}")
        plt.close(fig)


def holm_bonferroni(p_value_list, alpha=0.05):
    """Holm-Bonferroni step-down correction across a family of p-values.
    Input: list of (name, raw_p) tuples. Output: sorted list of (name, raw_p, adj_p, significant)."""
    sorted_p = sorted(p_value_list, key=lambda x: x[1])
    m = len(sorted_p)
    out, max_adj = [], 0.0
    for i, (name, p) in enumerate(sorted_p):
        adj = min(1.0, p * (m - i))
        adj = max(adj, max_adj)
        max_adj = adj
        out.append((name, p, adj, adj < alpha))
    return out


# Global registry — tests append (name, p) when run; printed at __main__ exit.
LIVE_PVALUES = []


if __name__ == "__main__":
    np.random.seed(42)
    main()
    # Closes Methods F2: Holm-Bonferroni table — LIVE p-values collected during main().
    # If a test wasn't run in this invocation, it's absent (we don't fabricate).
    print_section("HOLM-BONFERRONI ACROSS LIVE-COLLECTED p-VALUES (Methods F2)")
    if not LIVE_PVALUES:
        print("  No live p-values collected. (Each statistical test must call LIVE_PVALUES.append((name, p)) when computed.)")
    else:
        print(f"  Collected {len(LIVE_PVALUES)} live p-values; applying Holm-Bonferroni at α=0.05.")
        print(f"  {'#':>2}  {'test':<55}  {'raw p':>10}  {'adj p':>10}  {'sig α=.05':>10}")
        for i, (name, raw_p, adj_p, sig) in enumerate(holm_bonferroni(LIVE_PVALUES), 1):
            print(f"  {i:>2}  {name:<55}  {raw_p:>10.4g}  {adj_p:>10.4g}  {'YES' if sig else 'no':>10}")
