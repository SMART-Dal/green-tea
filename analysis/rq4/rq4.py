#!/usr/bin/env python3
"""RQ4: Does CAERR better reflect practical energy optimization effectiveness than raw ERR?

Computes CAERR (Correctness-Adjusted Energy Reduction Rate) for SFT and GRPO.
CAERR_i = ERR_i * (tests_passed_i / num_inputs_i), averaged over ALL outputs.
Decomposition: compile_rate * pass_rate|compiled * ERR|correct = CAERR.

Reproduces all numbers in Section 4 (RQ4).
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent.parent / "finetuning" / "data"
SFT_DIR = BASE / "sft_evaluation_results" / "second_run"
GRPO_DIR = BASE / "grpo_sim_results"
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


def caerr_analysis(records: list[dict], label: str) -> dict:
    n = len(records)
    compiled = [r for r in records if r.get("compiled", False)]
    n_compiled = len(compiled)

    total_tests = sum(r.get("num_inputs", 0) for r in records)
    tests_passed = sum(r.get("tests_passed", 0) for r in records)

    # Valid = compiled AND produced valid Sniper output (generated_energy > 0)
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0]
    n_valid = len(valid)

    errs_valid = np.array([r["energy_reduction"] for r in valid])

    # CAERR: ERR_i * (tests_passed_i / num_inputs_i), summed over ALL samples
    caerr_vals = []
    for r in records:
        ni = r.get("num_inputs", 0)
        if ni == 0:
            continue
        tp = r.get("tests_passed", 0)
        # ERR is 0 for non-compiled or non-valid outputs
        err = r["energy_reduction"] if (r.get("compiled") and r.get("generated_energy", 0) > 0) else 0.0
        caerr_vals.append(err * tp / ni)

    caerr = float(np.mean(caerr_vals)) if caerr_vals else 0.0
    mean_err_valid = float(np.mean(errs_valid)) if len(errs_valid) else 0.0

    # Decomposition
    compile_rate = n_compiled / n if n else 0
    pass_rate_given_compiled = tests_passed / (compile_rate * n * 102) if (compile_rate * n) > 0 else 0
    # Simpler: pass_rate = total tests passed / (compiled * avg_inputs)
    avg_inputs_compiled = sum(r.get("num_inputs", 0) for r in compiled) / n_compiled if n_compiled > 0 else 0
    pass_rate_given_compiled2 = tests_passed / (n_compiled * avg_inputs_compiled) if (n_compiled * avg_inputs_compiled) > 0 else 0
    err_given_correct = mean_err_valid

    decomp_caerr = compile_rate * pass_rate_given_compiled2 * err_given_correct / 100

    print(f"\n--- {label} ---")
    print(f"N samples: {n}")
    print(f"Compile rate:          {compile_rate*100:.1f}%  ({n_compiled}/{n})")
    print(f"Test pass rate (total tests): {tests_passed/total_tests*100 if total_tests else 0:.1f}%  ({tests_passed}/{total_tests})")
    print(f"Pass rate | compiled:   {pass_rate_given_compiled2*100:.1f}%")
    print(f"Valid for energy:       {n_valid/n*100:.1f}%  ({n_valid}/{n})")
    print(f"Mean ERR (valid only):  {mean_err_valid:.2f}%")
    print(f"CAERR (direct):        {caerr:.2f}%")
    print(f"CAERR (decomposition): {decomp_caerr*100:.2f}%  "
          f"= {compile_rate*100:.1f}% x {pass_rate_given_compiled2*100:.1f}% x {err_given_correct:.2f}%")
    print(f"Overstatement ratio:   {mean_err_valid/caerr:.2f}x  (raw ERR / CAERR)")

    # Pareto: per-problem correctness vs ERR
    prob_errs = defaultdict(list)
    prob_correct = defaultdict(list)
    for r in records:
        pid = r.get("problem_id", "")
        ni = r.get("num_inputs", 1)
        tp = r.get("tests_passed", 0)
        prob_correct[pid].append(tp / ni)
        if r.get("compiled") and r.get("generated_energy", 0) > 0:
            prob_errs[pid].append(r["energy_reduction"])

    all_pids = sorted(set(r["problem_id"] for r in records))
    correctness_rates = [np.mean(prob_correct[p]) for p in all_pids]
    mean_errs = [np.mean(prob_errs[p]) if prob_errs[p] else 0 for p in all_pids]

    corr = np.corrcoef(correctness_rates, mean_errs)[0, 1] if len(all_pids) > 1 else 0
    print(f"Per-problem correlation (correctness, ERR): r={corr:.3f}")

    return {"n": n, "caerr": caerr, "mean_err_valid": mean_err_valid,
            "compile_rate": compile_rate * 100,
            "pass_rate_given_compiled": pass_rate_given_compiled2 * 100,
            "n_valid": n_valid}


def simulation_completeness(records: list[dict], label: str):
    """M5: Report no-energy-after-compile fraction (timeout/wrong-answer indicator)."""
    n = len(records)
    compiled = sum(1 for r in records if r.get("compiled"))
    valid = sum(1 for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0)
    no_en = compiled - valid
    print(f"  {label}: compiled={compiled/n*100:.1f}% ({compiled}/{n}), "
          f"no energy after compile={no_en/compiled*100:.1f}% ({no_en}/{compiled}) = "
          f"{no_en/n*100:.1f}% of total")


def matched_cycle_analysis(records: list[dict], label: str, cycle_tol_pct: float = 1.0):
    """W2: Fraction of valid outputs in matched-cycle regime (cycle diff < tol%) where EDP diverges from runtime."""
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0
             and r.get("baseline_avg_cycles", 0) > 0 and r.get("generated_avg_cycles", 0) > 0]
    if not valid:
        print(f"  {label}: no cycle data available")
        return
    cycle_reds = np.array([
        (r["baseline_avg_cycles"] - r["generated_avg_cycles"]) / r["baseline_avg_cycles"] * 100
        for r in valid
    ])
    energy_reds = np.array([r["energy_reduction"] for r in valid])
    matched = np.abs(cycle_reds) < cycle_tol_pct
    n_matched = np.sum(matched)
    if n_matched > 0:
        en_diff_in_matched = np.abs(energy_reds[matched])
        diverge_05 = np.sum(en_diff_in_matched > 0.5)
        print_section(f"MATCHED-CYCLE REGIME (|cycle diff|<{cycle_tol_pct}%) -- {label} (W2)")
        print(f"  Valid outputs: {len(valid)}")
        print(f"  In matched-cycle regime: {n_matched} ({n_matched/len(valid)*100:.1f}%)")
        print(f"  Of those, |ERR|>0.5pp: {diverge_05} ({diverge_05/n_matched*100:.1f}%) -- EDP diverges from runtime")
        print(f"  Mean |ERR| in matched regime: {np.mean(en_diff_in_matched):.2f}%")
    else:
        print(f"  {label}: no outputs in matched-cycle regime (tol={cycle_tol_pct}%)")


def print_section(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def main():
    print("RQ4: CAERR vs Raw ERR for Practical Assessment")

    print_section("CAERR ANALYSIS")
    sft_records = load_eval_records(SFT_DIR)
    sft = caerr_analysis(sft_records, "SFT")

    grpo = None
    if GRPO_DIR.exists():
        grpo_records = load_eval_records(GRPO_DIR)
        grpo = caerr_analysis(grpo_records, "SFT+GRPO")
    else:
        print(f"\n[NOTE] GRPO results not found at {GRPO_DIR}")

    print_section("SUMMARY TABLE")
    header = f"{'Method':<12} {'MeanERR(valid)%':>15} {'CompileRate%':>13} {'PassRate|comp%':>15} {'CAERR%':>7} {'Overstate':>10}"
    print(header); print("-" * len(header))
    row_sft = (f"{'SFT':<12} {sft['mean_err_valid']:>14.2f}% {sft['compile_rate']:>12.1f}% "
               f"{sft['pass_rate_given_compiled']:>14.1f}% {sft['caerr']:>6.2f}% "
               f"{sft['mean_err_valid']/sft['caerr']:>9.2f}x")
    print(row_sft)
    if grpo:
        row_grpo = (f"{'SFT+GRPO':<12} {grpo['mean_err_valid']:>14.2f}% {grpo['compile_rate']:>12.1f}% "
                    f"{grpo['pass_rate_given_compiled']:>14.1f}% {grpo['caerr']:>6.2f}% "
                    f"{grpo['mean_err_valid']/grpo['caerr'] if grpo['caerr']>0 else 0:>9.2f}x")
        print(row_grpo)
    else:
        print(f"{'SFT+GRPO':<12} {'N/A':>15} {'N/A':>13} {'N/A':>15} {'N/A':>7} {'N/A':>10}")

    print_section("SIMULATION COMPLETENESS (M5: timeout/no-energy fraction)")
    simulation_completeness(sft_records, "SFT")
    if GRPO_DIR.exists():
        simulation_completeness(grpo_records, "SFT+GRPO")

    print_section("MATCHED-CYCLE REGIME (W2: EDP vs runtime divergence)")
    matched_cycle_analysis(sft_records, "SFT")
    if GRPO_DIR.exists():
        matched_cycle_analysis(grpo_records, "SFT+GRPO")

    print_section("ABLATION CAERR COMPARISON")
    for abl_name, abl_dir in ABL_DIRS.items():
        if abl_dir.exists() and any(abl_dir.glob("test_comparison_chunk_*.jsonl")):
            abl_records = load_eval_records(abl_dir)
            caerr_analysis(abl_records, abl_name)
        else:
            print(f"  {abl_name}: [pending]")

    print_section("KEY FINDING")
    print(f"Raw ERR ({sft['mean_err_valid']:.2f}%) overstates practical utility by "
          f"{sft['mean_err_valid']/sft['caerr']:.2f}x vs CAERR ({sft['caerr']:.2f}%).")
    print(f"{100-sft['n_valid']/sft['n']*100:.1f}% of outputs are unusable (compilation/correctness failures).")
    print("CAERR captures this gap in a single number, enabling fair comparison across model variants.")


if __name__ == "__main__":
    main()
