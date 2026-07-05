import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Classes based on rq4.tex
classes = [
    "Container\ntypes",
    "Input-output\ncalls",
    "Loop\nkeywords",
    "Numeric\nliterals",
    "Identifiers",
    "Syntactic\ntokens"
]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def stars(pval):
    if pval < 0.001: return "***"
    if pval < 0.01: return "**"
    if pval < 0.05: return "*"
    return ""

def main():
    stats = load_json(REPO_ROOT / "analysis" / "attention_plots_compare_paper" / "aggregate" / "threeway_stats.json")
    
    # Energy-SFT + EDP (main)
    mean_main = np.array(stats["main"]["mean"]) * 100 # convert to percentage points
    p_main = stats["main"]["p"]
    n_main = stats["main"]["N"]
    
    # Runtime-SFT + EDP (abl1)
    mean_abl1 = np.array(stats["abl1"]["mean"]) * 100
    p_abl1 = stats["abl1"]["p"]
    n_abl1 = stats["abl1"]["N"]

    fig, ax = plt.subplots(figsize=(6.5, 4))

    x = np.arange(len(classes))
    width = 0.35

    color_main = "#1565C0" # Blue
    color_abl1 = "#B71C1C" # Red

    ax.bar(x - width/2, mean_main, width, label='Energy-SFT + EDP', color=color_main, edgecolor='white')
    ax.bar(x + width/2, mean_abl1, width, label='Runtime-SFT + EDP', color=color_abl1, edgecolor='white')

    # Add stars with a cleaner look (subtle white background to prevent gridline clash)
    bbox_props = dict(boxstyle="square,pad=0", fc="white", ec="none", alpha=0.7)
    for i in range(len(classes)):
        # Main
        if p_main[i] < 0.05:
            y = mean_main[i]
            offset = 0.2 if y > 0 else -0.7
            va = 'bottom' if y > 0 else 'top'
            ax.text(i - width/2, y + offset, stars(p_main[i]), ha='center', va=va, color=color_main, fontweight='bold', fontsize=10, bbox=bbox_props)
        
        # Abl1
        if p_abl1[i] < 0.05:
            y = mean_abl1[i]
            offset = 0.2 if y > 0 else -0.7
            va = 'bottom' if y > 0 else 'top'
            ax.text(i + width/2, y + offset, stars(p_abl1[i]), ha='center', va=va, color=color_abl1, fontweight='bold', fontsize=10, bbox=bbox_props)

    ax.set_ylabel("Attention shift vs base (percentage points)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper left', fancybox=False, edgecolor='#CCCCCC')
    
    ax.axhline(0, color='black', linewidth=0.8, zorder=0)
    ax.grid(axis='y', color='#EEEEEE', linewidth=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    min_y = min(np.min(mean_main), np.min(mean_abl1))
    max_y = max(np.max(mean_main), np.max(mean_abl1))
    ax.set_ylim(min_y - 1.5, max_y + 1.0)

    plt.tight_layout()
    
    outdir = REPO_ROOT / "analysis" / "figures"
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, "fig_attn_class_shift_threeway.pdf")
    png_path = os.path.join(outdir, "fig_attn_class_shift_threeway.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    main()
