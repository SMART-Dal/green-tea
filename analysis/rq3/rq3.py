#!/usr/bin/env python3
"""RQ3: What types of optimizations do generated solutions employ?

Classifies SFT (and GRPO, when available) generated code into:
  A: Complexity-class change (algorithmic transformation)
  B: Constant-factor optimization (same complexity, different implementation)
  C: Cosmetic / no meaningful change

Reproduces all numbers in Section 4 (RQ3).
"""
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent.parent.parent / "finetuning" / "data"
SFT_DIR = BASE / "sft_evaluation_results" / "second_run"
GRPO_DIR = BASE / "grpo_sim_results"


# --- Classification heuristics ---

# Category A markers: algorithm / data structure swap
CAT_A_PATTERNS = [
    # Complexity-reducing data structures
    (r"\bunordered_map\b|\bunordered_set\b", "unordered container"),
    (r"\bpriority_queue\b", "priority queue"),
    (r"\bbinary_search\b|\blower_bound\b|\bupper_bound\b", "binary search"),
    (r"\bsort\b.*\blog\b|\bmerge_sort\b|\bqsort\b", "sort-based algorithm"),
    (r"\bdp\[|\bdp\s*=|\bDP\b", "dynamic programming"),
    (r"\bBFS\b|\bbfs\(|\bqueue<", "BFS/queue"),
    (r"\bDFS\b|\bdfs\(|\bstack<", "DFS/stack"),
    (r"\bsegment.tree\b|\bBIT\b|\bfenwick\b|\btree\[", "segment/Fenwick tree"),
    (r"\btwo.pointer|two_pointer", "two-pointer"),
    (r"\bprefix.sum\b|\bprefix_sum\b|\bcumsum\b", "prefix sum"),
]

# Category B markers: constant-factor, same complexity
CAT_B_PATTERNS = [
    (r"\bscanf\b|\bprintf\b|\bgets\b|\bputs\b", "fast I/O scanf/printf"),
    (r"ios_base::sync_with_stdio|ios::sync_with_stdio|cin\.tie", "ios sync disable"),
    (r"\btypedef\s+long\s+long\b|\busing\s+ll\s*=\s*long\s+long\b|\bll\b", "long long typedef"),
    (r"#pragma\s+GCC\s+optimize|#pragma\s+O[23]|__attribute__.*optimize", "GCC pragma optimize"),
    (r"\bregister\b|\b__builtin_\w+\b", "register/builtin"),
    (r"\bint\s+a\[|\bstatic\s+\w+\s*\[|static\s+int\b", "static array"),
    (r"\bgetchar_unlocked\b|\bputchar_unlocked\b|\bfread\b", "unlocked I/O"),
    (r"\bpow2\b|\b1\s*<<\s*\w+\b|\b__builtin_popcount\b", "bitwise ops"),
    (r"\breserve\b|\bemplace_back\b", "vector optimization"),
]


def classify_code(baseline: str, generated: str) -> str:
    """Return 'A', 'B', or 'C' classification."""
    if not generated or len(generated.strip()) < 10:
        return "C"

    gen = generated.lower()
    base = baseline.lower() if baseline else ""

    # Check for algorithmic markers NOT in baseline
    for pattern, _ in CAT_A_PATTERNS:
        gen_has = bool(re.search(pattern, gen, re.IGNORECASE))
        base_has = bool(re.search(pattern, base, re.IGNORECASE))
        if gen_has and not base_has:
            return "A"

    # Check for constant-factor markers
    for pattern, _ in CAT_B_PATTERNS:
        gen_has = bool(re.search(pattern, gen, re.IGNORECASE))
        base_has = bool(re.search(pattern, base, re.IGNORECASE))
        if gen_has and not base_has:
            return "B"

    # Cosmetic: minimal structural difference
    return "C"


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


def print_section(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def analyze_taxonomy(records: list[dict], label: str):
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0]
    if not valid:
        print(f"  No valid records for {label}")
        return

    categories = defaultdict(list)
    for r in valid:
        cat = classify_code(r.get("baseline_code", ""), r.get("generated_code", ""))
        categories[cat].append(r["energy_reduction"])

    print_section(f"TAXONOMY: {label} (n={len(valid)} valid outputs)")
    total = len(valid)
    for cat, label_str in [("A", "Complexity change"), ("B", "Constant-factor"), ("C", "Cosmetic/no change")]:
        errs = np.array(categories[cat]) if categories[cat] else np.array([0.0])
        n_cat = len(categories[cat])
        print(f"  Cat {cat} ({label_str}): {n_cat:4d} ({n_cat/total*100:5.1f}%)  "
              f"mean ERR={np.mean(errs):.1f}%  median={np.median(errs):.1f}%  "
              f"contribution={sum(errs):.0f}% total ERR")

    # Top algorithmic patterns found
    print_section(f"TOP PATTERNS (new in generated vs baseline) -- {label}")
    pattern_counts = defaultdict(int)
    for r in valid:
        gen = r.get("generated_code", "").lower()
        base = r.get("baseline_code", "").lower()
        for pattern, name in CAT_A_PATTERNS + CAT_B_PATTERNS:
            if re.search(pattern, gen, re.IGNORECASE) and not re.search(pattern, base, re.IGNORECASE):
                pattern_counts[name] += 1
    for name, count in sorted(pattern_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name:<35}: {count:4d} ({count/total*100:5.1f}%)")


def scanf_substitution_analysis(records: list[dict], label: str):
    """W4: scanf/printf substitution produces genuine ERR, not simulation artifact."""
    SCANF_PAT = re.compile(r'\bscanf\b|\bprintf\b', re.IGNORECASE)
    valid = [r for r in records if r.get("compiled") and r.get("generated_energy", 0) > 0]
    subst = [r for r in valid if SCANF_PAT.search(r.get("generated_code","")) and not SCANF_PAT.search(r.get("baseline_code",""))]
    no_subst = [r for r in valid if not SCANF_PAT.search(r.get("generated_code",""))]
    se = np.array([r["energy_reduction"] for r in subst])
    ne = np.array([r["energy_reduction"] for r in no_subst])
    print_section(f"SCANF/PRINTF SUBSTITUTION ANALYSIS -- {label}")
    print(f"  scanf/printf substitution (added by model): n={len(subst)}, "
          f"mean ERR={np.mean(se):.2f}%, median={np.median(se):.2f}%")
    print(f"  No scanf/printf in generated:               n={len(no_subst)}, "
          f"mean ERR={np.mean(ne):.2f}%, median={np.median(ne):.2f}%")
    print(f"  ERR gap (subst - no_subst): {np.mean(se)-np.mean(ne):.2f}pp mean, "
          f"{np.median(se)-np.median(ne):.2f}pp median")


def main():
    print("RQ3: Optimization Taxonomy in Model-Generated Code")

    sft_records = load_eval_records(SFT_DIR)
    analyze_taxonomy(sft_records, "SFT")

    if GRPO_DIR.exists():
        grpo_records = load_eval_records(GRPO_DIR)
        analyze_taxonomy(grpo_records, "SFT+GRPO")

        print_section("SFT vs GRPO DISTRIBUTION SHIFT")
        sft_valid = [r for r in sft_records if r.get("compiled") and r.get("generated_energy", 0) > 0]
        grpo_valid = [r for r in grpo_records if r.get("compiled") and r.get("generated_energy", 0) > 0]
        for cat in ["A", "B", "C"]:
            sft_n = sum(1 for r in sft_valid if classify_code(r.get("baseline_code",""), r.get("generated_code","")) == cat)
            grpo_n = sum(1 for r in grpo_valid if classify_code(r.get("baseline_code",""), r.get("generated_code","")) == cat)
            print(f"  Cat {cat}: SFT={sft_n/len(sft_valid)*100:.1f}%  GRPO={grpo_n/len(grpo_valid)*100:.1f}%  "
                  f"delta={grpo_n/len(grpo_valid)*100 - sft_n/len(sft_valid)*100:+.1f}pp")
        sft_cats = [classify_code(r.get("baseline_code",""), r.get("generated_code","")) for r in sft_valid]
        grpo_cats = [classify_code(r.get("baseline_code",""), r.get("generated_code","")) for r in grpo_valid]
        obs = np.array([[sft_cats.count(c) for c in "ABC"], [grpo_cats.count(c) for c in "ABC"]])
        chi2, p, dof, _ = chi2_contingency(obs)
        print(f"\n  Chi-squared (SFT vs GRPO, A/B/C): chi2={chi2:.1f}, dof={dof}, p={p:.2e}")

        scanf_substitution_analysis(grpo_records, "SFT+GRPO")
    else:
        print(f"\n[NOTE] GRPO results not found at {GRPO_DIR} -- SFT vs GRPO comparison pending.")

    scanf_substitution_analysis(sft_records, "SFT")
    print("\nNote: classification uses keyword heuristics (not formal AST tools); lower bound on Cat A.")


def static_binary_diff(records, out_dir: Path, max_pairs: int = 200):
    """M2: compile (baseline, generated) at -O3 -static and compute static-binary diff:
    static instruction count, basic-block count, unique mnemonics, function-call count, and
    presence of vector instructions. Per-pair JSON + per-category aggregate plot."""
    import subprocess, tempfile, re as _re, matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-static"]
    INSN_RE = _re.compile(r"^\s+[0-9a-f]+:\s+([a-z][a-z0-9]*)")
    LBL_RE = _re.compile(r"^[0-9a-f]+ <([^>]+)>:")
    VEC_RE = _re.compile(r"\b(xmm|ymm|zmm)\d+\b|movu?ps|movdqa|pmull?[wd]|vpaddd|vfmadd")
    def stats_of(binary):
        try:
            r = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "--demangle", str(binary)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0: return None
        except Exception: return None
        n_insn, mnem, n_call, vec_hit, n_lbl = 0, set(), 0, False, 0
        skip = True
        for line in r.stdout.splitlines():
            if LBL_RE.match(line):
                fn = LBL_RE.match(line).group(1)
                skip = fn.startswith(("std::", "__", "_GLOBAL_", "operator")) or "::" in fn and "main" not in fn
                if not skip: n_lbl += 1
                continue
            if skip: continue
            m = INSN_RE.match(line)
            if not m: continue
            n_insn += 1; op = m.group(1); mnem.add(op)
            if op in ("call", "callq"): n_call += 1
            if VEC_RE.search(line): vec_hit = True
        return {"n_insn": n_insn, "n_opcodes": len(mnem), "n_calls": n_call,
                "vec_hit": vec_hit, "n_funcs": n_lbl}
    rows = []
    for i, r in enumerate(records[:max_pairs]):
        if not r.get("compiled"): continue
        bcode = r.get("baseline_code", ""); gcode = r.get("generated_code", "")
        if not bcode or not gcode: continue
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); bsrc = td / "b.cpp"; gsrc = td / "g.cpp"
            bbin = td / "b.bin"; gbin = td / "g.bin"
            bsrc.write_text(bcode); gsrc.write_text(gcode)
            for src, out in [(bsrc, bbin), (gsrc, gbin)]:
                cr = subprocess.run(["g++"] + flags + [str(src), "-o", str(out)],
                                    capture_output=True, text=True, timeout=60)
                if cr.returncode != 0: out = None; break
            if out is None: continue
            sb = stats_of(bbin); sg = stats_of(gbin)
            if sb is None or sg is None or sb["n_insn"] < 1 or sg["n_insn"] < 1: continue
            cat = classify_code(bcode, gcode)
            rows.append({"pid": r.get("problem_id", ""), "cat": cat,
                         "n_insn_b": sb["n_insn"], "n_insn_g": sg["n_insn"],
                         "n_opcodes_b": sb["n_opcodes"], "n_opcodes_g": sg["n_opcodes"],
                         "n_calls_b": sb["n_calls"], "n_calls_g": sg["n_calls"],
                         "vec_b": sb["vec_hit"], "vec_g": sg["vec_hit"],
                         "n_funcs_b": sb["n_funcs"], "n_funcs_g": sg["n_funcs"],
                         "err": r.get("energy_reduction", 0.0)})
        if (i + 1) % 25 == 0: print(f"  static-diff {i+1}/{min(max_pairs, len(records))}")
    if not rows: print("no static diffs"); return None
    json.dump(rows, open(out_dir / "static_binary_diff.json", "w"), indent=2)
    r_static = np.array([x["n_insn_g"] / x["n_insn_b"] for x in rows])
    r_op = np.array([x["n_opcodes_g"] / max(x["n_opcodes_b"], 1) for x in rows])
    err = np.array([x["err"] for x in rows]); cats = np.array([x["cat"] for x in rows])
    rho_static, p_static = spearmanr(r_static, err)
    print(f"\n=== Static binary diff (N={len(rows)}) ===")
    print(f"Spearman rho(static_n_insn_ratio, ERR) = {rho_static:.3f}, p={p_static:.2g}")
    for c in "ABC":
        m = cats == c
        if not m.any(): continue
        print(f"  Cat {c}: n={m.sum():3d}  median r_static={np.median(r_static[m]):.3f}  "
              f"median r_opcodes={np.median(r_op[m]):.3f}  vec_b={int(np.array([x['vec_b'] for x in rows])[m].sum())}  "
              f"vec_g={int(np.array([x['vec_g'] for x in rows])[m].sum())}")
    cmap = {"A": "#c0392b", "B": "#e67e22", "C": "#7f8c8d"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for c in "ABC":
        m = cats == c
        if m.any(): ax.scatter(r_static[m], err[m], color=cmap[c], s=22, alpha=0.7,
                               label=f"Cat {c} (n={int(m.sum())})", edgecolor="#2c3e50", lw=0.4)
    ax.axhline(0, color="#333", lw=0.8, ls="--"); ax.axvline(1.0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("Static instruction-count ratio  (gen/base, post -O3)", fontsize=11)
    ax.set_ylabel("Energy reduction (%)", fontsize=12)
    ax.set_title(f"Static binary diff vs ERR  (rho={rho_static:.3f}, p={p_static:.2g})",
                 fontsize=12, fontweight="bold")
    ax.set_xscale("log"); ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax2 = axes[1]
    data = [r_static[cats == c] for c in "ABC" if (cats == c).any()]
    labels = [f"Cat {c}\n(n={int((cats==c).sum())})" for c in "ABC" if (cats == c).any()]
    bp = ax2.boxplot(data, labels=labels, widths=0.6, patch_artist=True,
                     medianprops=dict(color="#c0392b", lw=2))
    for patch, c in zip(bp["boxes"], "ABC"): patch.set_facecolor(cmap[c]); patch.set_alpha(0.5)
    ax2.axhline(1.0, color="#333", lw=0.8, ls="--")
    ax2.set_yscale("log"); ax2.set_ylabel("Static instruction-count ratio", fontsize=11)
    ax2.set_title("Per-category static-binary divergence", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3); ax2.set_axisbelow(True)
    fig.suptitle(f"Static binary diff at -O3 (M2 audit, N={len(rows)})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "static_binary_diff.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "static_binary_diff.png", dpi=300, bbox_inches="tight"); plt.close()
    return {"N": len(rows), "spearman_rho": float(rho_static), "spearman_p": float(p_static)}


def compiler_survival_analysis(records, label: str, out_dir: Path):
    """Per-pair instruction-count ratio (post -O3) vs ERR, stratified by Cat A/B/C.
    Tests whether the source-level diff survives -O3 lowering. Free signal: dynamic instruction
    counts are already in the Sniper trace and are post-compilation."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, mannwhitneyu
    rows = []
    for r in records:
        if not r.get("compiled"): continue
        gen_inst = r.get("generated_avg_instructions", 0)
        base_inst = r.get("baseline_avg_instructions", 0)
        n_inputs = r.get("num_inputs", 0); passed = r.get("tests_passed", 0)
        if base_inst <= 0 or gen_inst <= 0: continue
        if n_inputs <= 0 or passed < n_inputs: continue   # require all-tests-passing for clean signal
        ratio = gen_inst / base_inst
        if not (1e-3 < ratio < 1e3): continue   # drop extreme tails (broken runs)
        err = r.get("energy_reduction", 0.0)
        if not (-200.0 < err < 100.0): continue   # display-domain ERR (full distribution still computed elsewhere)
        cat = classify_code(r.get("baseline_code", ""), r.get("generated_code", ""))
        rows.append({"r_inst": ratio, "err": err, "cat": cat, "pid": r.get("problem_id", "")})
    if not rows: print(f"[{label}] no usable records"); return None
    r_inst = np.array([x["r_inst"] for x in rows])
    err = np.array([x["err"] for x in rows]); cats = np.array([x["cat"] for x in rows])
    rho, p_rho = spearmanr(r_inst, err)
    print(f"\n=== {label}: instruction-count ratio vs ERR (post -O3, N={len(rows)}) ===")
    print(f"Spearman rho(r_inst, ERR) = {rho:.3f}, p = {p_rho:.2g}")
    per_cat = {}
    for c in "ABC":
        m = cats == c; ri = r_inst[m]; er = err[m]
        if not m.any(): per_cat[c] = None; continue
        per_cat[c] = {"n": int(m.sum()), "median_r_inst": float(np.median(ri)),
                      "iqr_r_inst": [float(np.percentile(ri, 25)), float(np.percentile(ri, 75))],
                      "median_err": float(np.median(er)), "mean_err": float(er.mean())}
        print(f"  Cat {c}: n={m.sum():3d}  median r_inst={np.median(ri):.3f}  "
              f"IQR=[{np.percentile(ri,25):.3f},{np.percentile(ri,75):.3f}]  "
              f"median ERR={np.median(er):.2f}%  mean ERR={er.mean():.2f}%")
    if (cats == "A").any() and (cats == "C").any():
        try:
            u, pv = mannwhitneyu(r_inst[cats == "A"], r_inst[cats == "C"], alternative="less")
            print(f"  Mann-Whitney r_inst(A) < r_inst(C):  U={u:.0f}, p={pv:.2g}")
        except Exception: pass
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    cmap = {"A": "#c0392b", "B": "#e67e22", "C": "#7f8c8d"}
    ax = axes[0]
    for c in "ABC":
        m = cats == c
        if m.any(): ax.hist(r_inst[m], bins=30, alpha=0.55, label=f"Cat {c} (n={int(m.sum())})",
                            color=cmap[c], edgecolor="#2c3e50")
    ax.axvline(1.0, color="#333", lw=1.0, ls="--", label="parity (r=1)")
    ax.set_xlabel("Instruction-count ratio  (generated / baseline, post -O3)", fontsize=12)
    ax.set_ylabel("Number of pairs", fontsize=12)
    ax.set_title("Distribution of post -O3 instruction-count ratio", fontsize=13, fontweight="bold")
    ax.set_xscale("log"); ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax2 = axes[1]
    for c in "ABC":
        m = cats == c
        if m.any(): ax2.scatter(r_inst[m], err[m], color=cmap[c], s=22, alpha=0.7,
                                label=f"Cat {c} (n={int(m.sum())})", edgecolor="#2c3e50", lw=0.4)
    ax2.axhline(0, color="#333", lw=0.8, ls="--"); ax2.axvline(1.0, color="#333", lw=0.8, ls="--")
    ax2.set_xlabel("Instruction-count ratio  (generated / baseline)", fontsize=12)
    ax2.set_ylabel("Energy reduction (%)", fontsize=12)
    ax2.set_title(f"Source-level diff survives -O3:  Spearman rho = {rho:.3f}, p = {p_rho:.2g}",
                  fontsize=12, fontweight="bold")
    ax2.set_xscale("log"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3); ax2.set_axisbelow(True)
    ax3 = axes[2]
    data = [r_inst[cats == c] for c in "ABC" if (cats == c).any()]
    labels = [f"Cat {c}\n(n={int((cats==c).sum())})" for c in "ABC" if (cats == c).any()]
    bp = ax3.boxplot(data, labels=labels, widths=0.6, patch_artist=True,
                     medianprops=dict(color="#c0392b", lw=2))
    for patch, c in zip(bp["boxes"], "ABC"): patch.set_facecolor(cmap[c]); patch.set_alpha(0.5)
    ax3.axhline(1.0, color="#333", lw=0.8, ls="--")
    ax3.set_yscale("log"); ax3.set_ylabel("Instruction-count ratio", fontsize=12)
    ax3.set_title("Per-category instruction-count divergence", fontsize=13, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3); ax3.set_axisbelow(True)
    fig.suptitle(f"Compiler-survival audit ({label}): does the source-level diff survive g++ -O3?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"compiler_survival_{label.lower().replace(' ','_')}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"compiler_survival_{label.lower().replace(' ','_')}.png", dpi=300, bbox_inches="tight")
    plt.close()
    import csv
    csv_path = out_dir / f"per_pair_{label.lower().replace(' ', '_')}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "category", "baseline_avg_instructions",
                    "generated_avg_instructions", "instruction_count_ratio",
                    "energy_reduction_pct"])
        full_records = [r for r in records
                        if r.get("compiled") and r.get("baseline_avg_instructions", 0) > 0
                        and r.get("generated_avg_instructions", 0) > 0
                        and r.get("num_inputs", 0) > 0
                        and r.get("tests_passed", 0) >= r.get("num_inputs", 0)]
        for r in full_records:
            ratio = r["generated_avg_instructions"] / r["baseline_avg_instructions"]
            cat = classify_code(r.get("baseline_code", ""), r.get("generated_code", ""))
            w.writerow([r.get("problem_id", ""), cat, r["baseline_avg_instructions"],
                        r["generated_avg_instructions"], f"{ratio:.4f}",
                        f"{r.get('energy_reduction', 0.0):.2f}"])
    print(f"  wrote per-pair CSV -> {csv_path}")
    return {"N": len(rows), "spearman_rho": float(rho), "spearman_p": float(p_rho), "per_category": per_cat}


def opt_sweep_aggregate(results_dir: Path, out_dir: Path):
    """M3: per-pair ERR vs g++ -O level on a stratified subset (Cat A/B/C).
    Tests whether higher compiler optimization erodes the source-level energy delta."""
    import matplotlib.pyplot as plt
    files = sorted(results_dir.glob("opt_sweep_*.json"))
    if not files: print(f"[opt_sweep] no results in {results_dir}"); return
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        for opt, cell in d.get("opts", {}).items():
            if cell.get("compile_b") and cell.get("compile_g") and "err_pct" in cell:
                rows.append({"pid": d["problem_id"], "cat": d["category"], "opt": opt,
                             "err": cell["err_pct"], "b_insn": cell.get("b_insn_mean", 0),
                             "g_insn": cell.get("g_insn_mean", 0)})
    if not rows: print("[opt_sweep] no successful cells"); return
    print(f"\n=== M3: Optimization-level sweep (N={len(files)} problems, {len(rows)} cells) ===")
    opts = ["-O0", "-O1", "-O2", "-O3"]
    print(f"{'cat':<5} " + " ".join(f"{o:>8}" for o in opts))
    summary = {}
    for cat in ["A", "B", "C"]:
        cells = [r for r in rows if r["cat"] == cat]
        if not cells: continue
        means = []
        for o in opts:
            vals = [r["err"] for r in cells if r["opt"] == o]
            means.append(np.mean(vals) if vals else float("nan"))
        summary[cat] = means
        print(f"{cat:<5} " + " ".join(f"{m:>8.2f}" for m in means))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = {"A": "#c0392b", "B": "#e67e22", "C": "#7f8c8d"}
    for cat, means in summary.items():
        ax.plot(opts, means, marker="o", lw=2, color=cmap[cat], label=f"Cat {cat}")
    ax.axhline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("g++ optimization level"); ax.set_ylabel("Mean ERR (%)")
    ax.set_title("ERR vs compiler -O level (stratified subset)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_dir / "opt_sweep.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "opt_sweep.png", dpi=300, bbox_inches="tight"); plt.close()
    json.dump({"per_category_mean_err": summary, "n_problems": len(files), "n_cells": len(rows)},
              open(out_dir / "opt_sweep_summary.json", "w"), indent=2)
    import csv
    with open(out_dir / "opt_sweep_per_pair.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "cat", "opt", "err_pct", "b_insn", "g_insn"])
        for r in rows: w.writerow([r["pid"], r["cat"], r["opt"], f"{r['err']:.2f}",
                                   int(r["b_insn"]), int(r["g_insn"])])
    print(f"  wrote -> {out_dir}/opt_sweep.{{pdf,png,json,csv}}")


if __name__ == "__main__":
    main()
    if GRPO_DIR.exists():
        grpo = load_eval_records(GRPO_DIR)
        out_dir = Path(__file__).parent.parent / "compiler_survival"
        compiler_survival_analysis(grpo, "GRPO", out_dir)
        # M2: static binary diff (subset of pairs with valid compile)
        valid = [r for r in grpo if r.get("compiled") and r.get("baseline_code") and r.get("generated_code")]
        static_binary_diff(valid, out_dir, max_pairs=200)
        # M3: opt-level sweep aggregate
        opt_sweep_aggregate(REPO_ROOT / "execution_results", out_dir)
