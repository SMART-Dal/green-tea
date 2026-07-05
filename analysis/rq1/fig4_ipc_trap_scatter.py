#!/usr/bin/env python3
import json
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

ANALYSIS_DIR = Path(__file__).parent.parent
SOLUTION_METRICS = ANALYSIS_DIR / "solution_metrics.jsonl"
OUTPUT_DIR = ANALYSIS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
}
mpl.rcParams.update(RC)

def generate_ipc_trap_plot():
    problems = collections.defaultdict(list)
    with open(SOLUTION_METRICS) as f:
        for line in f:
            d = json.loads(line)
            if d.get("avg_ipc", 0) > 0 and d.get("avg_power", 0) > 0:
                problems[d["problem_id"]].append(d)

    traps_x, traps_y = [], []
    non_traps_x, non_traps_y = [], []
    
    for pid, sols in problems.items():
        if len(sols) < 2: continue
        sols.sort(key=lambda x: x["avg_energy"])
        best = sols[0]
        worst = sols[-1]
        if best["avg_energy"] == worst["avg_energy"]: continue
        
        x = worst["avg_ipc"]
        y = best["avg_ipc"]
        
        if y < x:
            traps_x.append(x)
            traps_y.append(y)
        else:
            non_traps_x.append(x)
            non_traps_y.append(y)

    total_valid = len(traps_x) + len(non_traps_x)
    trap_pct = len(traps_x) / total_valid * 100
    non_trap_pct = len(non_traps_x) / total_valid * 100

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(traps_x, traps_y, color='#D32F2F', alpha=0.5, s=8, label=f'IPC-trap (67.8%)')
    ax.scatter(non_traps_x, non_traps_y, color='#1976D2', alpha=0.5, s=8, label=f'Non-trap (32.2%)')
    
    # Diagonal line
    ax.plot([0, 3.5], [0, 3.5], 'k--', alpha=0.6, linewidth=1, label='IPC equal line')
    
    ax.set_xlim([0, 3.6])
    ax.set_ylim([0, 3.6])
    
    ax.set_xlabel("IPC of worst-energy solution", fontsize=11)
    ax.set_ylabel("IPC of best-energy solution", fontsize=11)
    
    # Grid and legend
    ax.grid(True)
    ax.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=1.0)
    
    # Add text boxes
    box_props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
    
    text1 = f"IPC Inversion: 67.8% of problems\nhave lower IPC in best-energy solution"
    ax.text(0.03, 0.96, text1, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=box_props)
    

    plt.tight_layout()
    
    out_pdf = OUTPUT_DIR / "fig4_ipc_trap_scatter.pdf"
    out_png = OUTPUT_DIR / "fig4_ipc_trap_scatter.png"
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")

if __name__ == "__main__":
    generate_ipc_trap_plot()
