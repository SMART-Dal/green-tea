#!/usr/bin/env python3
"""
Hero figure: Base -> SFT -> GRPO energy optimization progression.
Output: analysis/figures/fig_hero_progression.{pdf,png}
Usage:  python3 analysis/training_plots.py
"""

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, "finetuning", "data")
OUTDIR = os.path.join(ROOT, "analysis", "figures")
BDIR   = os.path.join(DATA, "baseline_sim_results")

def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)]

def ecdf(vals, clip_lo=-30, clip_hi=105):
    s = np.sort(np.clip(vals, clip_lo, clip_hi))
    y = np.arange(1, len(s) + 1) / len(s)
    # prepend left anchor
    s = np.concatenate([[clip_lo], s])
    y = np.concatenate([[0.0], y])
    return s, y

# ERR on all outputs that compiled AND produced a valid sim (whether correct or not)
# This captures the full bimodal shape of SFT.
def err_all_valid(recs):
    return [r["energy_reduction"] for r in recs
            if r.get("compiled") and r.get("generated_energy", 0) > 0]

def outcome_frac(recs):
    n       = len(recs)
    comp    = sum(1 for r in recs if r.get("compiled"))
    valid   = sum(1 for r in recs if r.get("compiled") and r.get("generated_energy", 0) > 0)
    correct = sum(1 for r in recs if r.get("compiled") and r.get("generated_energy", 0) > 0
                  and r.get("tests_passed", 0) == r.get("num_inputs", 1))
    return {
        "fail":    (n - comp)      / n * 100,
        "norun":   (comp - valid)  / n * 100,
        "wrong":   (valid - correct) / n * 100,
        "correct": correct / n * 100,
    }

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

STAGES = [
    ("Zero-shot",       "#9E9E9E", (4, 2),   1.2,
     f"{BDIR}/zero_shot_instruct/test_comparison_chunk_*.jsonl"),
    ("Green-prompt",    "#455A64", (2, 2),   1.2,
     f"{BDIR}/green_prompt_instruct/test_comparison_chunk_*.jsonl"),
    ("SFT",             "#1565C0", "solid",  2.0,
     f"{DATA}/sft_evaluation_results/second_run/test_comparison_chunk_*.jsonl"),
    ("GRPO (ours)",     "#B71C1C", "solid",  2.8,
     f"{DATA}/grpo_sim_results/test_comparison_chunk_*.jsonl"),
]

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    stage_data = []
    for label, color, ls, lw, pattern in STAGES:
        recs = load(pattern)
        if recs:
            stage_data.append((label, color, ls, lw, recs))
            errs = err_all_valid(recs)
            oc   = outcome_frac(recs)
            caerr = np.mean([r["energy_reduction"] * r.get("tests_passed", 0)
                             / max(r.get("num_inputs", 1), 1) for r in recs])
            print(f"{label:<20} n={len(recs)}  valid={len(errs)}  "
                  f"mean={np.mean(errs):.1f}%  median={np.median(errs):.1f}%  "
                  f"CAERR={caerr:.2f}%  correct={oc['correct']:.1f}%")

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(5.2, 3.8))

        # Main axes: leave room right for inset, top for title
        ax = fig.add_axes([0.12, 0.13, 0.57, 0.77])

        # --- Shading: positive-ERR region fills for SFT and GRPO ---
        for label, color, ls, lw, recs in stage_data:
            if label not in ("SFT", "GRPO (ours)"):
                continue
            errs = err_all_valid(recs)
            xs, ys = ecdf(errs)
            alpha  = 0.20 if label == "SFT" else 0.18
            fcolor = "#BBDEFB" if label == "SFT" else "#FFCDD2"
            mask   = xs >= 0
            if mask.any():
                ax.fill_betweenx(ys, np.where(mask, xs, 0), 0,
                                 where=mask, color=fcolor, alpha=alpha, zorder=0)

        # --- CDF curves ---
        for label, color, ls, lw, recs in stage_data:
            errs = err_all_valid(recs)
            xs, ys = ecdf(errs)
            ax.step(xs, ys, where="post", color=color,
                    linestyle=ls if isinstance(ls, str) else (0, ls),
                    linewidth=lw, zorder=3, label=label, solid_capstyle="round")

        # --- ERR=0 reference line ---
        ax.axvline(0, color="#424242", linewidth=0.65, linestyle="--",
                   alpha=0.55, zorder=1)
        ax.text(0.3, 0.025, "no change", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=6, color="#616161", style="italic")

        # --- Median dots + clean annotations ---
        annot_cfg = {
            "SFT":       dict(dx=-4, dy=0.13, ha="right"),
            "GRPO (ours)": dict(dx=4,  dy=-0.12, ha="left"),
        }
        for label, color, ls, lw, recs in stage_data:
            if label not in annot_cfg:
                continue
            errs = np.array(err_all_valid(recs))
            med  = float(np.median(errs))
            frac = float(np.mean(errs <= med))
            cfg  = annot_cfg[label]
            ax.plot(med, frac, "o", color=color, markersize=4, zorder=5)
            ax.annotate(f"median = {med:.1f}%",
                        xy=(med, frac),
                        xytext=(med + cfg["dx"], frac + cfg["dy"]),
                        fontsize=6.5, color=color, ha=cfg["ha"],
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=0.7, shrinkA=3, shrinkB=3))

        # --- Axes cosmetics ---
        ax.set_xlim(-28, 100)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Energy Reduction Rate, ERR  (%)\n"
                      r"(positive $\Rightarrow$ energy saved; all compiled outputs with valid simulation)",
                      labelpad=4)
        ax.set_ylabel("Cumulative fraction of outputs", labelpad=4)
        ax.set_title("Energy Optimization Across Training Stages",
                     pad=6, fontsize=9, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5, zorder=0)

        # --- Legend (inside main axes, top-left) ---
        handles = [Line2D([0], [0], color=c,
                          linestyle=ls if isinstance(ls, str) else (0, ls),
                          linewidth=lw, label=lab)
                   for lab, c, ls, lw, _ in stage_data]
        ax.legend(handles=handles, loc="upper left",
                  frameon=True, edgecolor="#CCCCCC", fancybox=False,
                  borderpad=0.5, labelspacing=0.3)

        # ---- Inset: outcome stacked bar (right panel) ----
        ax_in = fig.add_axes([0.73, 0.13, 0.24, 0.77])
        ax_in.set_title("Outputs\nbreakdown", fontsize=7, pad=4)

        bar_labels = ["Zero-shot", "Green-\nprompt", "SFT", "GRPO"]
        segs = ["fail", "norun", "wrong", "correct"]
        seg_colors = ["#EF9A9A", "#FFE082", "#FFCC80", "#A5D6A7"]
        seg_names  = ["Compile\nfail", "No sim\nresult", "Failed\ntests", "ERR\nmeasured"]

        oc_list = [outcome_frac(recs) for _, _, _, _, recs in stage_data]
        stage_colors = [c for _, c, *_ in stage_data]

        y   = np.arange(len(bar_labels))
        h   = 0.6
        left = np.zeros(len(bar_labels))

        for seg, col, sname in zip(segs, seg_colors, seg_names):
            vals = np.array([oc[seg] for oc in oc_list])
            ax_in.barh(y, vals, h, left=left, color=col,
                       edgecolor="white", linewidth=0.4)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v >= 8:
                    ax_in.text(l + v / 2, i, f"{v:.0f}",
                               ha="center", va="center",
                               fontsize=5.5, color="#212121")
            left += vals

        ax_in.set_yticks(y)
        ax_in.set_yticklabels(bar_labels, fontsize=6.5)
        ax_in.set_xlabel("% of outputs", fontsize=6.5, labelpad=3)
        ax_in.set_xlim(0, 100)
        ax_in.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax_in.tick_params(axis="x", labelsize=6)
        ax_in.spines["top"].set_visible(False)
        ax_in.spines["right"].set_visible(False)
        ax_in.spines["left"].set_visible(False)
        ax_in.tick_params(left=False)

        # Inset legend at bottom
        patches = [mpatches.Patch(color=c, label=n)
                   for c, n in zip(seg_colors, seg_names)]
        ax_in.legend(handles=patches, loc="lower center",
                     bbox_to_anchor=(0.5, -0.38),
                     fontsize=5.2, frameon=False, ncol=2,
                     columnspacing=0.5, handlelength=0.9)

        # ---- CAERR headline annotation ----
        ax.text(0.98, 0.04,
                "CAERR: GRPO 12.6% vs SFT 4.5%  (2.84\u00d7)",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E1", ec="#FFC107",
                          alpha=0.9, linewidth=0.8))

        # ---- Divider line between panels ----
        fig.add_artist(
            plt.Line2D([0.715, 0.715], [0.05, 0.97],
                       transform=fig.transFigure,
                       color="#CCCCCC", linewidth=0.6))

        for ext in ("pdf", "png"):
            p = os.path.join(OUTDIR, f"fig_hero_progression.{ext}")
            fig.savefig(p)
            print(f"Saved: {p}")

GRPO_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "finetuning", "logs",
                        "qwen-coder-base-14b_grpo_20260227_033015_grpo_vllm_9400877.log")
SFT_STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "finetuning", "checkpoints",
                         "qwen-coder-base-14b_sft_20260208_214200",
                         "checkpoint-933", "trainer_state.json")

def _smooth(vals, w=150):
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i-w):i+1] if v is not None]
        out.append(float(np.mean(chunk)) if chunk else 0.0)
    return np.array(out)

def training_dynamics():
    sft_log = json.load(open(SFT_STATE))["log_history"]
    sft_steps = [e["step"] for e in sft_log if "loss" in e]
    sft_loss  = [e["loss"] for e in sft_log if "loss" in e]
    eval_pts  = [(e["step"], e["eval_loss"]) for e in sft_log if "eval_loss" in e]

    records = []
    with open(GRPO_LOG) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{'loss'"):
                try:
                    records.append(eval(line))
                except Exception:
                    pass
    if not records:
        print("No GRPO training records found")
        return

    g_steps    = np.array(range(1, len(records) + 1)) / 1000.0
    reward     = np.array([r.get("reward", 0.0) for r in records])
    energy_imp = np.array([r.get("total/energy_improvement_rate", 0.0) for r in records])
    compile_ok = 1.0 - np.array([r.get("total/compile_error_rate", 1.0) for r in records])
    kl         = np.array([r.get("kl", 0.0) for r in records])

    sm_rew = _smooth(reward.tolist(), 150)
    sm_ei  = _smooth(energy_imp.tolist(), 150)
    sm_c   = _smooth(compile_ok.tolist(), 150)
    sm_k   = _smooth(kl.tolist(), 150)

    c_loss, c_compile, c_reward, c_ei, c_kl = "#37474F", "#1565C0", "#B71C1C", "#2E7D32", "#7B1FA2"

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.0, 5.4))
        gs = fig.add_gridspec(4, 2, width_ratios=[1, 2.2], hspace=0.09,
                              wspace=0.35, left=0.09, right=0.97, top=0.94, bottom=0.08)

        # -- Panel (a): SFT loss (spans all 4 rows on left) --
        ax_sft = fig.add_subplot(gs[:, 0])
        ax_sft.plot(sft_steps, sft_loss, color=c_loss, linewidth=1.5, label="Train loss")
        if eval_pts:
            ex, ey = zip(*eval_pts)
            ax_sft.scatter(ex, ey, color="#B71C1C", s=18, zorder=4, label="Eval loss", marker="D")
        ax_sft.set_xlabel("Training step")
        ax_sft.set_ylabel("Cross-entropy loss")
        ax_sft.set_title("(a) SFT training", fontsize=8.5, fontweight="bold", pad=5)
        ax_sft.set_xlim(0, max(sft_steps))
        ax_sft.set_ylim(bottom=0)
        ax_sft.legend(fontsize=6, frameon=True, edgecolor="#CCCCCC", fancybox=False)
        ax_sft.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax_sft.annotate(f"{sft_loss[0]:.3f}", xy=(sft_steps[0], sft_loss[0]),
                        xytext=(sft_steps[0]+50, sft_loss[0]+0.003),
                        fontsize=6, color=c_loss,
                        arrowprops=dict(arrowstyle="-", color=c_loss, lw=0.5))
        ax_sft.annotate(f"{sft_loss[-1]:.4f}", xy=(sft_steps[-1], sft_loss[-1]),
                        xytext=(sft_steps[-1]-200, sft_loss[-1]+0.008),
                        fontsize=6, color=c_loss,
                        arrowprops=dict(arrowstyle="-", color=c_loss, lw=0.5))

        _lbbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88)

        # -- Panel (b): Compile success rate --
        ax_comp = fig.add_subplot(gs[0, 1])
        ax_comp.fill_between(g_steps, 0.5, sm_c, alpha=0.08, color=c_compile)
        ax_comp.plot(g_steps, sm_c, color=c_compile, linewidth=1.6, zorder=3)
        ax_comp.set_ylabel("Compile\nsuccess")
        ax_comp.set_ylim(0.50, 0.92)
        ax_comp.set_xlim(0, g_steps[-1])
        ax_comp.set_title("(b) GRPO training dynamics", fontsize=8.5, fontweight="bold", pad=5)
        ax_comp.tick_params(labelbottom=False)
        ax_comp.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax_comp.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 0))
        ax_comp.annotate(f"{sm_c[80]:.0%}",
                         xy=(g_steps[80], sm_c[80]), xytext=(0.9, 0.565),
                         fontsize=6, color=c_compile, ha="center",
                         bbox=_lbbox, zorder=5,
                         arrowprops=dict(arrowstyle="->", color=c_compile, lw=0.7,
                                         connectionstyle="arc3,rad=-0.25"))
        ax_comp.annotate(f"{sm_c[-1]:.0%}",
                         xy=(g_steps[-1], sm_c[-1]), xytext=(6.2, 0.885),
                         fontsize=6, color=c_compile, ha="center",
                         bbox=_lbbox, zorder=5,
                         arrowprops=dict(arrowstyle="->", color=c_compile, lw=0.7,
                                         connectionstyle="arc3,rad=0.25"))
        _dc = round((sm_c[-1] - sm_c[80]) * 100)
        ax_comp.annotate("", xy=(3.0, sm_c[-1]-0.004), xytext=(3.0, sm_c[80]+0.004),
                         arrowprops=dict(arrowstyle="<->", color=c_compile, lw=1.0))
        ax_comp.text(3.12, (sm_c[-1] + sm_c[80]) / 2, f"+{_dc}pp",
                     fontsize=6.5, color=c_compile, va="center", fontweight="bold",
                     bbox=_lbbox)

        # -- Panel (c): Mean EDP reward (shrinking gap to 0 = training signal improving) --
        ax_rew = fig.add_subplot(gs[1, 1], sharex=ax_comp)
        ax_rew.fill_between(g_steps, sm_rew, 0, alpha=0.13, color=c_reward, interpolate=True)
        ax_rew.plot(g_steps, sm_rew, color=c_reward, linewidth=1.6, zorder=3)
        ax_rew.axhline(0, color="#9E9E9E", linewidth=0.7, linestyle="--", alpha=0.7)
        ax_rew.text(0.02, 0.93, r"$\uparrow$ better (closer to 0)",
                    transform=ax_rew.transAxes, fontsize=5.5, color="#757575",
                    ha="left", va="top")
        ax_rew.set_ylabel("Mean EDP\nreward")
        ax_rew.set_title("(c)", fontsize=8, fontweight="bold", loc="left", pad=3)
        ax_rew.set_ylim(-0.22, 0.04)
        ax_rew.tick_params(labelbottom=False)
        ax_rew.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax_rew.annotate(f"{sm_rew[80]:.2f}",
                        xy=(g_steps[80], sm_rew[80]), xytext=(0.9, -0.205),
                        fontsize=6, color=c_reward, ha="center",
                        bbox=_lbbox, zorder=5,
                        arrowprops=dict(arrowstyle="->", color=c_reward, lw=0.7,
                                        connectionstyle="arc3,rad=-0.25"))
        ax_rew.annotate(f"{sm_rew[-1]:.2f}",
                        xy=(g_steps[-1], sm_rew[-1]), xytext=(6.1, -0.04),
                        fontsize=6, color=c_reward, ha="center",
                        bbox=_lbbox, zorder=5,
                        arrowprops=dict(arrowstyle="->", color=c_reward, lw=0.7,
                                        connectionstyle="arc3,rad=0.25"))
        _pct_r = round((sm_rew[-1] - sm_rew[80]) / abs(sm_rew[80]) * 100)
        ax_rew.annotate("", xy=(4.0, sm_rew[-1]+0.003), xytext=(4.0, sm_rew[80]-0.003),
                        arrowprops=dict(arrowstyle="<->", color=c_reward, lw=1.0))
        ax_rew.text(4.12, (sm_rew[-1] + sm_rew[80]) / 2, f"+{_pct_r}%",
                    fontsize=6.5, color=c_reward, va="center", fontweight="bold",
                    bbox=_lbbox)

        # -- Panel (d): Energy improvement rate --
        ax_ei = fig.add_subplot(gs[2, 1], sharex=ax_comp)
        ax_ei.fill_between(g_steps, sm_ei, alpha=0.10, color=c_ei)
        ax_ei.plot(g_steps, sm_ei, color=c_ei, linewidth=1.6, zorder=3)
        ax_ei.set_ylabel("Energy\nimproved (%)")
        ax_ei.set_title("(d)", fontsize=8, fontweight="bold", loc="left", pad=3)
        ax_ei.set_ylim(0.0, 0.60)
        ax_ei.tick_params(labelbottom=False)
        ax_ei.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax_ei.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 0))
        ax_ei.annotate(f"{sm_ei[80]:.0%}",
                       xy=(g_steps[80], sm_ei[80]), xytext=(0.9, 0.08),
                       fontsize=6, color=c_ei, ha="center",
                       bbox=_lbbox, zorder=5,
                       arrowprops=dict(arrowstyle="->", color=c_ei, lw=0.7,
                                       connectionstyle="arc3,rad=-0.25"))
        ax_ei.annotate(f"{sm_ei[-1]:.0%}",
                       xy=(g_steps[-1], sm_ei[-1]), xytext=(6.1, 0.53),
                       fontsize=6, color=c_ei, ha="center",
                       bbox=_lbbox, zorder=5,
                       arrowprops=dict(arrowstyle="->", color=c_ei, lw=0.7,
                                       connectionstyle="arc3,rad=0.25"))
        _pct_ei = round((sm_ei[-1] - sm_ei[80]) / sm_ei[80] * 100)
        ax_ei.annotate("", xy=(3.5, sm_ei[-1]-0.004), xytext=(3.5, sm_ei[80]+0.004),
                       arrowprops=dict(arrowstyle="<->", color=c_ei, lw=1.0))
        ax_ei.text(3.62, (sm_ei[-1] + sm_ei[80]) / 2, f"+{_pct_ei}%",
                   fontsize=6.5, color=c_ei, va="center", fontweight="bold",
                   bbox=_lbbox)

        # -- Panel (e): KL divergence (log scale to show transient spike at ~1.4k) --
        ax_kl = fig.add_subplot(gs[3, 1], sharex=ax_comp)
        ax_kl.fill_between(g_steps, np.maximum(sm_k, 0.005), 0.005,
                           alpha=0.06, color=c_kl)
        ax_kl.plot(g_steps, sm_k, color=c_kl, linewidth=1.6, zorder=3)
        ax_kl.set_yscale("log")
        ax_kl.set_ylim(0.005, 5.0)
        ax_kl.set_ylabel("KL div.\n(log)")
        ax_kl.set_title("(e)", fontsize=8, fontweight="bold", loc="left", pad=3)
        ax_kl.set_xlabel("Training step (x1,000)")
        ax_kl.yaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.4)))
        ax_kl.grid(axis="y", color="#EEEEEE", linewidth=0.5, which="both")
        ax_kl.axhline(sm_k[-1], color=c_kl, linewidth=0.5, linestyle=":", alpha=0.5)
        ax_kl.text(0.97, 0.12, f"stable at {sm_k[-1]:.3f}", transform=ax_kl.transAxes,
                   fontsize=6, color=c_kl, ha="right", va="bottom", style="italic")

        for ext in ("pdf", "png"):
            p = os.path.join(OUTDIR, f"fig_grpo_dynamics.{ext}")
            fig.savefig(p, bbox_inches="tight")
            print(f"Saved: {p}")
        plt.close(fig)

def wandb_training_dynamics():
    """SFT->GRPO unified progression plot from wandb-exported jsonl files."""
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    adir = os.path.dirname(os.path.abspath(__file__))
    sft_f  = os.path.join(adir, "sft_training_history.jsonl")
    grpo_f = os.path.join(adir, "grpo_training_history.jsonl")
    if not os.path.exists(sft_f) or not os.path.exists(grpo_f):
        print("wandb history jsonl files not found; run fetch first")
        return

    sft  = pd.read_json(sft_f,  lines=True)
    grpo = pd.read_json(grpo_f, lines=True)

    # SFT: train loss and eval loss
    sft_tr = sft[sft['train/loss'].notna()].sort_values('train/global_step')
    sft_ev = sft[sft['eval/loss'].notna()].sort_values('train/global_step')

    # GRPO: key metrics
    gcols = ['train/global_step','train/step/reward_mean','train/step/compile_error_rate',
             'train/step/energy_improvement_rate','train/kl','train/energy/mean_reduction_pct']
    gd = grpo[gcols].dropna().sort_values('train/global_step').reset_index(drop=True)

    def smooth(arr, sigma=4):
        return gaussian_filter1d(arr.astype(float), sigma=sigma, mode='nearest')

    gs  = gd['train/global_step'].values / 1000.0
    rew = smooth(gd['train/step/reward_mean'].values)
    cer = smooth(1.0 - gd['train/step/compile_error_rate'].values)  # compile SUCCESS
    eir = smooth(gd['train/step/energy_improvement_rate'].values)
    kl  = smooth(gd['train/kl'].values)
    emr = smooth(gd['train/energy/mean_reduction_pct'].values)

    c_sft  = "#1565C0"  # blue for SFT
    c_grpo = "#B71C1C"  # red for GRPO
    c_rew  = "#B71C1C"
    c_comp = "#1565C0"
    c_ei   = "#2E7D32"
    c_emr  = "#6A1B9A"
    _bg_sft  = "#E3F2FD"   # light blue
    _bg_grpo = "#FFF3E0"   # light orange

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.2, 5.0))
        # 4 rows: SFT loss | GRPO reward / compile / energy-improvement / energy-mean
        gs_layout = fig.add_gridspec(4, 2, width_ratios=[1, 2.0], hspace=0.10,
                                     wspace=0.38, left=0.09, right=0.97, top=0.96, bottom=0.09)

        # Shared background color fills to indicate phases
        def shade(ax, color, label=None, loc='left'):
            ax.set_facecolor(color)
            if label:
                xp = 0.05 if loc == 'left' else 0.95
                ha = 'left' if loc == 'left' else 'right'
                ax.text(xp, 0.96, label, transform=ax.transAxes, fontsize=6.5,
                        ha=ha, va='top', color='#37474F', style='italic', alpha=0.8)

        # ---- SFT panel (left column, all 4 rows) ----
        ax_sft = fig.add_subplot(gs_layout[:, 0])
        shade(ax_sft, _bg_sft, 'SFT phase')
        ax_sft.plot(sft_tr['train/global_step'], sft_tr['train/loss'],
                    color=c_sft, linewidth=1.6, label='Train loss', zorder=3)
        if len(sft_ev):
            ax_sft.scatter(sft_ev['train/global_step'], sft_ev['eval/loss'],
                           color=c_grpo, s=22, zorder=4, label='Eval loss', marker='D')
        ax_sft.set_xlabel('Training step')
        ax_sft.set_ylabel('Cross-entropy loss')
        ax_sft.set_title('(a) SFT training', fontsize=8.5, fontweight='bold', pad=5)
        ax_sft.set_xlim(0, sft_tr['train/global_step'].max())
        ax_sft.set_ylim(bottom=0)
        ax_sft.legend(fontsize=6.5, frameon=True, edgecolor='#CCCCCC', fancybox=False)
        ax_sft.grid(axis='y', color='#DDDDDD', linewidth=0.5)
        # Annotate start/end loss
        ax_sft.text(0.97, 0.95, f"start: {sft_tr['train/loss'].iloc[0]:.3f}",
                    transform=ax_sft.transAxes, fontsize=6, ha='right', va='top', color=c_sft)
        ax_sft.text(0.97, 0.08, f"end: {sft_tr['train/loss'].iloc[-1]:.4f}",
                    transform=ax_sft.transAxes, fontsize=6, ha='right', va='bottom', color=c_sft)

        # Vertical divider between SFT and GRPO columns
        fig.add_artist(plt.Line2D([0.52, 0.52], [0.09, 0.93], transform=fig.transFigure,
                                  color='#BDBDBD', linewidth=0.7, linestyle='--'))
        fig.text(0.52, 0.05, 'SFT init.', fontsize=5.5, ha='center', va='top',
                 color='#546E7A', style='italic', transform=fig.transFigure)

        _lbbox = dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85)

        def annotate_ends(ax, xs, ys, color, fmt='{:.2f}', offset_y=0.05):
            ax.annotate(fmt.format(ys[0]), xy=(xs[0], ys[0]),
                        xytext=(xs[0] + (xs[-1]-xs[0])*0.08, ys[0]),
                        fontsize=6, color=color, bbox=_lbbox, zorder=5,
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.5))
            ax.annotate(fmt.format(ys[-1]), xy=(xs[-1], ys[-1]),
                        xytext=(xs[-1] - (xs[-1]-xs[0])*0.08, ys[-1] + offset_y*(ax.get_ylim()[1]-ax.get_ylim()[0])),
                        fontsize=6, color=color, bbox=_lbbox, zorder=5,
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.5))

        # ---- GRPO: reward mean ----
        ax_r = fig.add_subplot(gs_layout[0, 1])
        shade(ax_r, _bg_grpo)
        ax_r.fill_between(gs, rew, 0, alpha=0.12, color=c_rew)
        ax_r.plot(gs, rew, color=c_rew, linewidth=1.6, zorder=3)
        ax_r.axhline(0, color='#9E9E9E', linewidth=0.6, linestyle='--', alpha=0.6)
        ax_r.set_ylabel('Reward')
        ax_r.set_title('(b) GRPO training dynamics', fontsize=8.5, fontweight='bold', pad=5)
        ax_r.tick_params(labelbottom=False)
        ax_r.grid(axis='y', color='#DDDDDD', linewidth=0.5)
        ax_r.text(0.97, 0.08, 'higher = better', transform=ax_r.transAxes,
                  fontsize=5.5, color='#757575', ha='right', va='bottom', style='italic')

        # ---- GRPO: compile success ----
        ax_c = fig.add_subplot(gs_layout[1, 1], sharex=ax_r)
        shade(ax_c, _bg_grpo)
        ax_c.fill_between(gs, cer, alpha=0.10, color=c_comp)
        ax_c.plot(gs, cer, color=c_comp, linewidth=1.6, zorder=3)
        ax_c.set_ylabel('Compile (%)')
        ax_c.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 0))
        ax_c.tick_params(labelbottom=False)
        ax_c.set_ylim(0.70, 0.92)
        ax_c.grid(axis='y', color='#DDDDDD', linewidth=0.5)
        ax_c.text(0.50, 0.08, f"+{(cer[-1]-cer[0])*100:.0f}pp ({cer[0]:.0%} to {cer[-1]:.0%})",
                  transform=ax_c.transAxes, fontsize=6, ha='center', va='bottom',
                  color=c_comp, fontweight='bold', bbox=_lbbox)

        # ---- GRPO: energy improvement rate ----
        ax_e = fig.add_subplot(gs_layout[2, 1], sharex=ax_r)
        shade(ax_e, _bg_grpo)
        ax_e.fill_between(gs, eir, alpha=0.10, color=c_ei)
        ax_e.plot(gs, eir, color=c_ei, linewidth=1.6, zorder=3)
        ax_e.set_ylabel('Energy imp. (%)')
        ax_e.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 0))
        ax_e.tick_params(labelbottom=False)
        ax_e.grid(axis='y', color='#DDDDDD', linewidth=0.5)
        ax_e.text(0.97, 0.08, f"{eir[0]:.0%}→{eir[-1]:.0%}",
                  transform=ax_e.transAxes, fontsize=6, ha='right', va='bottom',
                  color=c_ei, fontweight='bold', bbox=_lbbox)

        # ---- GRPO: KL divergence ----
        c_kl = "#E65100"
        ax_m = fig.add_subplot(gs_layout[3, 1], sharex=ax_r)
        shade(ax_m, _bg_grpo)
        ax_m.fill_between(gs, kl, alpha=0.10, color=c_kl)
        ax_m.plot(gs, kl, color=c_kl, linewidth=1.6, zorder=3)
        ax_m.set_ylabel('KL divergence')
        ax_m.set_xlabel('Training step (x1,000)')
        ax_m.grid(axis='y', color='#DDDDDD', linewidth=0.5)
        ax_m.text(0.97, 0.92, f"{kl[0]:.2f} to {kl[-1]:.2f}",
                  transform=ax_m.transAxes, fontsize=6, ha='right', va='top',
                  color=c_kl, fontweight='bold', bbox=_lbbox)

        # Remove top/right spines on all GRPO axes
        for ax in [ax_r, ax_c, ax_e, ax_m]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # suptitle removed: panel titles (a)/(b) are sufficient for ICSE format

        os.makedirs(OUTDIR, exist_ok=True)
        for ext in ('pdf', 'png'):
            p = os.path.join(OUTDIR, f'fig_training_dynamics.{ext}')
            fig.savefig(p, bbox_inches='tight')
            print(f'Saved: {p}')
        plt.close(fig)


def fetch_wandb_history(api_key=None):
    """Fetch training history from wandb and save to analysis/*.jsonl for replication."""
    import pandas as pd
    try:
        import wandb
    except ImportError:
        print("wandb not installed"); return

    key = api_key or os.environ.get('WANDB_API_KEY', '')
    if key:
        import wandb as _w; _w.login(key=key, relogin=True)

    api = wandb.Api()
    project = 'saurabhsinghrajput/energy-code-generation'
    runs = {
        'sft':         'mn28avjl',
        'grpo':        'grpo-qwen-coder-base-14b-20260225230520',
        'runtime_sft': '4vvw680m',
    }
    adir = os.path.dirname(os.path.abspath(__file__))
    for name, run_id in runs.items():
        run = api.run(f'{project}/{run_id}')
        hist = run.history(samples=2000)
        out = os.path.join(adir, f'{name}_training_history.jsonl')
        hist.to_json(out, orient='records', lines=True)
        print(f"Saved {name}: {len(hist)} rows -> {out}")


if __name__ == "__main__":
    main()
    training_dynamics()
    wandb_training_dynamics()
