#!/usr/bin/env python3
"""Robustness analyses added in the TOSEM revision (Section 5.3, Table 7; Sections 4.1, 4.4, 8).

Reproduces: CARET under graded / matched-input / strict / full-coverage / per-baseline / common-population
conventions for every Qwen2.5-Coder-14B configuration; per-baseline and per-problem paired tests; ghost-execution
sensitivity of the pair labels (writes analysis/matched_labels.jsonl); fidelity of the single-input reward.

Usage: python3 analysis/robustness_audit.py --dataset <zenodo green-tea-dataset dir> --testcases <.../merged_test_cases>
Matched-input CARET re-executes partially correct outputs natively (g++ -O3 -static); run it on a compute node.
"""
import argparse, hashlib, json, os, subprocess, tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
D, CF = ROOT / "finetuning/data", Path(os.environ.get("GREEN_TEA_DATA", ROOT / "data/cross_family_sim_results"))
CONFIGS = {"Energy-SFT": D / "sft_evaluation_results/second_run", "Runtime-SFT": D / "sft_runtime_sim_results",
           "EDP-SFT": CF / "qwen14b_sft_edp", "E-SFT+EDP": D / "grpo_sim_results", "R-SFT+EDP": D / "grpo_abl1_rsft_edp_sim_results",
           "E-SFT+energy": D / "grpo_abl2_esft_energy_sim_results", "R-SFT+runtime": D / "grpo_abl3_rsft_rt_sim_results",
           "E-SFT+runtime": D / "grpo_abl4_esft_rt_sim_results"}
h12 = lambda s: hashlib.md5(s.encode()).hexdigest()[:12]
h8 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()[:8]
TC = None


def load(d):
    return [json.loads(l) for f in sorted(d.glob("test_comparison_chunk_*.jsonl")) for l in open(f) if l.strip()]


def valid(r):
    return r.get("compiled") and r.get("generated_energy", 0) > 0


def contrib(r, err, rule):
    return (err if valid(r) and rule(r) else 0.0) * r["tests_passed"] / r["num_inputs"]


def passed_inputs(job):
    """Inputs the program passes under the evaluator protocol (sorted input.*.txt with an output file, 5 s timeout)."""
    key, code, pid = job
    with tempfile.TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR")) as t:
        cpp, b = Path(t) / "s.cpp", Path(t) / "s.bin"
        cpp.write_text(code)
        if subprocess.run(["g++", "-O3", "-std=c++17", "-static", str(cpp), "-o", str(b)], capture_output=True, timeout=60).returncode:
            return key, None
        out = []
        for inf in sorted((TC / pid).glob("input.*.txt")):
            outf = TC / pid / f"output.{inf.name[6:]}"
            if not outf.exists():
                continue
            try:
                with open(inf) as fi:
                    r = subprocess.run([str(b)], stdin=fi, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip() == outf.read_text().strip():
                    out.append(h8(inf))
            except subprocess.TimeoutExpired:
                pass
        return key, out


def rrb(d):
    d = d[d != 0]; rk = rankdata(np.abs(d)); tp, tn = rk[d > 0].sum(), rk[d < 0].sum()
    return (tp - tn) / (tp + tn)


def main():
    global TC
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, required=True); ap.add_argument("--testcases", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))); a = ap.parse_args(); TC = a.testcases
    pairs = {s: [json.loads(l) for l in open(a.dataset / f"training/sft_pairs_{s}.jsonl")] for s in ("train", "val", "test")}
    recs = {n: load(d) for n, d in CONFIGS.items()}
    need = {(p["problem_id"], p[k]) for P in pairs.values() for p in P for k in ("baseline_code_hash", "optimized_code_hash")}
    E, n_all, n_ghost, PW, RT = defaultdict(dict), 0, 0, [], []  # (pid, code) -> {input: (energy, ghost)}
    for l in open(a.dataset / "energy_measurements/completed_samples.jsonl"):
        r = json.loads(l); pid, ch, ih = r["execution_id"].rsplit("_", 2); res = r.get("result") or {}
        e, g = res.get("energy_joules") or 0, (res.get("energy_joules") or 0) <= 0 or (res.get("instructions") or 0) <= 0
        n_all += 1; n_ghost += g
        if (res.get("power_watts") or 0) > 0 and (res.get("runtime_seconds") or 0) > 0:
            PW.append(res["power_watts"]); RT.append(res["runtime_seconds"])
        if (pid, ch) in need:
            E[(pid, ch)][ih] = (e, g)
    print(f"ghost executions: {n_ghost} of {n_all} records ({n_ghost / n_all:.2%})")
    PW, RT = np.array(PW), np.array(RT); bins = np.round(np.log10(RT) * 10)  # Section 5.1 variability figures
    sel = [k for k in np.unique(bins) if (bins == k).sum() >= 100]
    cv = lambda x: x.std() / x.mean() * 100
    print(f"power CV over all executions {cv(PW):.2f}%; mean within-runtime-bin CV: power {np.mean([cv(PW[bins == k]) for k in sel]):.2f}%, "
          f"energy {np.mean([cv((PW * RT)[bins == k]) for k in sel]):.2f}% ({len(sel)} bins of 0.1 decade with >= 100 executions)")

    # Ghost-execution sensitivity of pair labels: reduction on inputs where both programs have a valid record.
    mean = lambda c, ks=None: np.mean([E[c][k][0] for k in (ks if ks is not None else E[c])])
    with open(ROOT / "analysis/matched_labels.jsonl", "w") as fo:
        for s, P in pairs.items():
            ro, rm = [], []
            for p in P:
                b, t = (p["problem_id"], p["baseline_code_hash"]), (p["problem_id"], p["optimized_code_hash"])
                ks = [k for k in E[b] if k in E[t] and not E[b][k][1] and not E[t][k][1]]
                o, m = 1 - mean(t) / mean(b), (1 - mean(t, ks) / mean(b, ks)) if ks else float("nan")
                ro.append(o); rm.append(m)
                fo.write(json.dumps(dict(split=s, problem_id=b[0], baseline_code_hash=b[1], optimized_code_hash=t[1], reduction=o,
                                         reduction_matched=m, target_has_ghost=any(v[1] for v in E[t].values()), below_threshold_matched=bool(m < 0.10))) + "\n")
            ro, rm = np.array(ro), np.array(rm)
            print(f"labels {s}: n={len(P)} mean reduction {np.nanmean(ro):.1%} -> {np.nanmean(rm):.1%} matched; below 10% {(rm < .1).sum()}, "
                  f"reversed {(rm <= 0).sum()}; Spearman {spearmanr(ro, rm, nan_policy='omit')[0]:.2f}")

    # Single median-size input vs all-input mean (reward input protocol, Section 4.4).
    x, y = [], []
    for p in pairs["train"]:
        ins = sorted((TC / p["problem_id"]).glob("input.*.txt"), key=lambda q: q.stat().st_size)
        b, t = (p["problem_id"], p["baseline_code_hash"]), (p["problem_id"], p["optimized_code_hash"])
        k = h8(ins[len(ins) // 2]) if ins else None
        if k in E[b] and k in E[t] and not E[b][k][1] and not E[t][k][1]:
            ng = lambda c: np.mean([v[0] for v in E[c].values() if not v[1]])
            x.append(E[t][k][0] / E[b][k][0]); y.append(ng(t) / ng(b))
    x, y = np.array(x), np.array(y)
    print(f"reward input: n={len(x)} sign agreement {np.mean((x < 1) == (y < 1)):.1%}, Spearman {spearmanr(x, y)[0]:.2f}, median deviation {np.median(np.abs(np.log(x / y))):.1%}")

    # Matched-input CARET: recover the passing inputs of every partially correct output.
    jobs = {(r["problem_id"], h12(r["generated_code"])): ((r["problem_id"], h12(r["generated_code"])), r["generated_code"], r["problem_id"])
            for R in recs.values() for r in R if valid(r) and r["tests_passed"] < r["num_inputs"]}
    with Pool(a.jobs) as pool:
        P_IN = dict(pool.imap_unordered(passed_inputs, jobs.values(), chunksize=4))

    def matched_err(r):
        b = (r["problem_id"], h12(r["baseline_code"]))
        ks = [k for k, v in E[b].items() if not v[1]] if r["tests_passed"] == r["num_inputs"] else \
             [k for k in (P_IN.get((r["problem_id"], h12(r["generated_code"]))) or []) if k in E[b] and not E[b][k][1]]
        return (1 - r["generated_energy"] / mean(b, ks)) * 100 if ks else 0.0

    key = lambda r: (r["problem_id"], h12(r["baseline_code"]), h12(r["optimized_code"]))
    common = set.intersection(*[{key(r) for r in R} for R in recs.values()])
    ALL, STRICT = (lambda r: True), (lambda r: r["tests_passed"] == r["num_inputs"])
    COV = lambda r: STRICT(r) and r["generated_success_count"] == r["num_inputs"]
    perbase = lambda R, rule: {k: np.mean(v) for k, v in _group(R, rule).items()}
    print(f"\n{'config':15s} {'N':>5} {'Comp':>5} {'Tests':>5} {'Full':>5} | {'Graded':>6} {'Matched':>7} {'Strict':>6} {'Cov':>5} {'Base':>6} {'Problem':>7} {'Common':>6}")
    for n, R in recs.items():
        g = lambda rule, err=lambda r: r["energy_reduction"], RR=R: np.mean([contrib(r, err(r), rule) for r in RR])
        print(f"{n:15s} {len(R):5d} {np.mean([r['compiled'] for r in R]) * 100:5.1f} {sum(r['tests_passed'] for r in R) / sum(r['num_inputs'] for r in R) * 100:5.1f} "
              f"{np.mean([STRICT(r) for r in R]) * 100:5.1f} | {g(ALL):6.2f} {g(ALL, matched_err):7.2f} {g(STRICT):6.2f} {g(COV):5.2f} "
              f"{np.mean(list(perbase(R, ALL).values())):6.2f} {np.mean([np.mean(v) for v in _byprob(R).values()]):7.2f} {g(ALL, RR=[r for r in R if key(r) in common]):6.2f}")
    print(f"common population: {len(common)} pairs")

    # Paired tests: distinct baseline as pairing unit, problem as resampling unit (Section 5.3).
    for rule_n, rule in (("graded", ALL), ("strict", STRICT)):
        for ft in ("Energy-SFT", "Runtime-SFT", "EDP-SFT"):
            a_, b_ = perbase(recs[ft], rule), perbase(recs["E-SFT+EDP"], rule); ks = sorted(set(a_) & set(b_))
            d = np.array([b_[k] - a_[k] for k in ks]); pp = defaultdict(list)
            for k, v in zip(ks, d): pp[k[0]].append(v)
            dp = np.array([np.mean(v) for v in pp.values()]); keys = list(pp); rng = np.random.default_rng(0)
            boot = [np.mean(np.concatenate([pp[keys[j]] for j in rng.integers(0, len(keys), len(keys))])) for _ in range(5000)]
            print(f"{rule_n} E-SFT+EDP vs {ft}: baselines n={len(ks)} p={wilcoxon(d).pvalue:.1e} r_rb={rrb(d):.2f}; problems better/worse "
                  f"{(dp > 0).sum()}/{(dp < 0).sum()} p={wilcoxon(dp).pvalue:.1e} r_rb={rrb(dp):.2f}; mean diff {d.mean():.2f}pp CI {np.percentile(boot, [2.5, 97.5]).round(2)}")


def _byprob(R):
    g = defaultdict(list)
    for r in R:
        g[r["problem_id"]].append(contrib(r, r["energy_reduction"], lambda r: True))
    return g


def _group(R, rule):
    g = defaultdict(list)
    for r in R:
        g[(r["problem_id"], h12(r["baseline_code"]))].append(contrib(r, r["energy_reduction"], rule))
    return g


if __name__ == "__main__":
    main()
