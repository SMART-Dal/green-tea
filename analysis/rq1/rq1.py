#!/usr/bin/env python3
"""RQ1: Does score-conditioned SFT improve energy efficiency over zero-shot and green-prompting baselines?

Reproduces all numbers in Section 4 (RQ1). Data from second_run evaluation results.
"""
import json
import glob
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel

BASE = Path(__file__).parent.parent.parent / "finetuning" / "data"
EVAL_DIR = BASE / "sft_evaluation_results" / "second_run"
BASELINE_DIR = BASE / "baseline_sim_results"
INSTRUCT_SFT_DIR = BASE / "sft_instruct_sim_results"


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

    beat_gt = sum(1 for r in valid if r.get("vs_gt_reduction", float("-inf")) >= 0)

    # CAERR = mean over all samples of ERR_i * (tests_passed_i / num_inputs_i)
    caerr_vals = []
    for r in records:
        ni = r.get("num_inputs", 0)
        if ni == 0:
            continue
        tp = r.get("tests_passed", 0)
        err = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else 0.0
        caerr_vals.append(err * tp / ni)

    return {
        "n": n,
        "n_valid": len(valid),
        "compile_rate": compiled / n * 100,
        "test_pass_rate": tests_passed / total_tests * 100 if total_tests else 0,
        "valid_rate": len(valid) / n * 100,
        "mean_err": float(np.mean(errs)) if len(errs) else 0,
        "median_err": float(np.median(errs)) if len(errs) else 0,
        "std_err": float(np.std(errs)) if len(errs) else 0,
        "err_gt0_pct": float(np.mean(errs > 0) * 100) if len(errs) else 0,
        "err_gt50_pct": float(np.mean(errs > 50) * 100) if len(errs) else 0,
        "beat_gt_pct": beat_gt / len(valid) * 100 if valid else 0,
        "caerr": float(np.mean(caerr_vals)) if caerr_vals else 0,
        "errs": errs,
    }


def print_section(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def statistical_tests(errs: np.ndarray):
    """Wilcoxon signed-rank test and t-test vs zero improvement (H0: median/mean ERR = 0)."""
    if len(errs) < 10:
        print("  Insufficient data for statistical tests.")
        return
    try:
        w_stat, w_p = wilcoxon(errs)
        print(f"  Wilcoxon: W={w_stat:.0f}, p={w_p:.4g}")
    except Exception as e:
        print(f"  Wilcoxon failed: {e}")
    # t-test vs 0
    from scipy.stats import ttest_1samp
    t, p = ttest_1samp(errs, 0)
    print(f"  t-test vs 0: t={t:.3f}, p={p:.4g}")
    # Cohen's d
    d = np.mean(errs) / np.std(errs) if np.std(errs) > 0 else 0
    print(f"  Cohen's d: {d:.3f}")
    # 95% CI via bootstrap
    boot = [np.mean(np.random.choice(errs, len(errs), replace=True)) for _ in range(5000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI for mean ERR: [{ci_lo:.2f}%, {ci_hi:.2f}%]")


def compare_methods(methods: dict[str, dict]):
    print_section("COMPARISON TABLE")
    header = f"{'Method':<24} {'N':>5} {'Compile%':>9} {'Tests%':>7} {'Valid%':>7} {'MeanERR%':>9} {'MedERR%':>8} {'BeatGT%':>8} {'CAERR%':>7}"
    print(header)
    print("-" * len(header))
    for name, m in methods.items():
        if m is None:
            print(f"{name:<24} {'N/A':>5} {'N/A':>9} {'N/A':>7} {'N/A':>7} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>7}")
        else:
            print(f"{name:<24} {m['n']:>5} {m['compile_rate']:>8.1f}% {m['test_pass_rate']:>6.1f}% "
                  f"{m['valid_rate']:>6.1f}% {m['mean_err']:>8.2f}% {m['median_err']:>7.2f}% "
                  f"{m['beat_gt_pct']:>7.1f}% {m['caerr']:>6.2f}%")


def main():
    print("RQ1: SFT vs Zero-Shot and Green-Prompting Baselines")

    print_section("SFT RESULTS (second_run)")
    sft_records = load_eval_records(EVAL_DIR)
    sft = compute_metrics(sft_records)
    print(f"N samples: {sft['n']} across {len(set(r['problem_id'] for r in sft_records))} problems")
    print(f"Compiled: {sft['compile_rate']:.1f}% ({int(sft['n']*sft['compile_rate']/100)}/{sft['n']})")
    print(f"Tests passed: {sft['test_pass_rate']:.1f}%")
    print(f"Valid for energy: {sft['valid_rate']:.1f}% ({sft['n_valid']}/{sft['n']})")
    print(f"Mean ERR: {sft['mean_err']:.2f}%  Median ERR: {sft['median_err']:.2f}%  Std: {sft['std_err']:.2f}%")
    print(f"ERR > 0: {sft['err_gt0_pct']:.1f}%   ERR > 50%: {sft['err_gt50_pct']:.1f}%")
    print(f"Beat GT: {sft['beat_gt_pct']:.1f}%")
    print(f"CAERR: {sft['caerr']:.2f}%")
    print("\nStatistical tests:")
    statistical_tests(sft["errs"])

    # ERR distribution breakdown
    print_section("ERR DISTRIBUTION BREAKDOWN")
    errs = sft["errs"]
    brackets = [(-float("inf"), -10), (-10, 0), (0, 5), (5, 50), (50, float("inf"))]
    labels = ["<-10% (regression)", "-10..0% (near-neutral)", "0..5%", "5..50%", ">50% (major win)"]
    for (lo, hi), lab in zip(brackets, labels):
        mask = (errs >= lo) & (errs < hi)
        n_b = np.sum(mask)
        med_b = np.median(errs[mask]) if n_b > 0 else 0
        print(f"  {lab:<30}: {n_b:4d} ({n_b/len(errs)*100:5.1f}%)  median={med_b:.1f}%")

    # Baseline results
    methods = {"SFT (ours)": sft}
    all_baselines = [
        ("Zero-shot (base)", "zero_shot"),
        ("Green-prompt (base)", "green_prompt"),
        ("Zero-shot (instruct)", "zero_shot_instruct"),
        ("Green-prompt (instruct)", "green_prompt_instruct"),
    ]
    for name, subdir in all_baselines:
        d = BASELINE_DIR / subdir
        if d.exists():
            rec = load_eval_records(d)
            methods[name] = compute_metrics(rec) if rec else None
        else:
            methods[name] = None

    compare_methods(methods)

    # W1 ablation: runtime-SFT comparison
    RUNTIME_SFT_DIR = BASE / "sft_runtime_sim_results"
    GRPO_DIR = BASE / "grpo_sim_results"
    if RUNTIME_SFT_DIR.exists() or GRPO_DIR.exists():
        print_section("W1 ABLATION + GRPO: DISTRIBUTIONAL COMPARISON")
        datasets = {"energy-SFT": sft_records}
        if RUNTIME_SFT_DIR.exists():
            datasets["runtime-SFT"] = load_eval_records(RUNTIME_SFT_DIR)
        if GRPO_DIR.exists():
            datasets["grpo"] = load_eval_records(GRPO_DIR)
        for name, recs in datasets.items():
            m = compute_metrics(recs)
            v = [r for r in recs if r.get('baseline_energy',0)>1e-9 and r.get('generated_energy',0)>1e-9]
            errs = np.array([r['energy_reduction'] for r in v])
            upper = errs[errs > 10]
            fc_count = sum(1 for r in recs if r.get('tests_passed',0)>=r.get('num_inputs',999) and r.get('num_inputs',0)>0)
            safe = sum(1 for r in recs if r.get('compiled') and r.get('tests_passed',0)>0 and r.get('energy_reduction',-999)>-10)
            cat_neg = sum(1 for r in v if r['energy_reduction'] < -10)
            n80 = np.sum(errs > 80); n90 = np.sum(errs > 90)
            print(f"\n  {name} (N={m['n']}, valid={m['n_valid']}):")
            print(f"    compile={m['compile_rate']:.1f}%  valid={m['valid_rate']:.1f}%  FC={fc_count/m['n']*100:.1f}%  deploy-safe={safe/m['n']*100:.1f}%")
            print(f"    CAERR={m['caerr']:.2f}%  mean_ERR={m['mean_err']:.2f}%  median_ERR={m['median_err']:.2f}%")
            print(f"    cat_neg(ERR<-10%)={cat_neg/m['n_valid']*100:.2f}%  ERR>80%={n80}  ERR>90%={n90}")
            if len(upper) > 0:
                print(f"    upper_mode(ERR>10%): n={len(upper)} ({len(upper)/m['n_valid']*100:.1f}%), mean={np.mean(upper):.1f}%, median={np.median(upper):.1f}%, p25={np.percentile(upper,25):.1f}%, p75={np.percentile(upper,75):.1f}%")
            if len(errs) > 0:
                print(f"    ERR percentiles: p10={np.percentile(errs,10):.1f}%  p25={np.percentile(errs,25):.1f}%  p50={np.percentile(errs,50):.1f}%  p75={np.percentile(errs,75):.1f}%  p90={np.percentile(errs,90):.1f}%")

    instruct_baselines = {k: v for k, v in methods.items() if "(instruct)" in k and v is not None}
    if instruct_baselines:
        print_section("WILCOXON COMPARISON: SFT vs INSTRUCT BASELINES")
        for name, bsl in instruct_baselines.items():
            if bsl is None or len(bsl["errs"]) == 0:
                continue
            n_min = min(len(sft["errs"]), len(bsl["errs"]))
            w_stat, w_p = wilcoxon(sft["errs"][:n_min] - bsl["errs"][:n_min])
            print(f"  SFT vs {name}: W={w_stat:.0f}, p={w_p:.4g}")

    if INSTRUCT_SFT_DIR.exists():
        print_section("INSTRUCT SFT ABLATION")
        instruct_sft_records = load_eval_records(INSTRUCT_SFT_DIR)
        isft = compute_metrics(instruct_sft_records)
        print(f"N={isft['n']}, compile={isft['compile_rate']:.1f}%, valid={isft['valid_rate']:.1f}%")
        print(f"Mean ERR={isft['mean_err']:.2f}%  Median={isft['median_err']:.2f}%")
        print(f"Beat-GT={isft['beat_gt_pct']:.1f}%  CAERR={isft['caerr']:.2f}%")
        print(f"vs SFT base: compile {isft['compile_rate']:.1f}% vs {sft['compile_rate']:.1f}%  "
              f"CAERR {isft['caerr']:.2f}% vs {sft['caerr']:.2f}% "
              f"({sft['caerr']/isft['caerr']:.0f}x higher base SFT)" if isft['caerr'] > 0 else "")
    else:
        print(f"\n[NOTE] Instruct SFT sim results not found at {INSTRUCT_SFT_DIR}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
