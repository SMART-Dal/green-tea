#!/usr/bin/env python3
"""Build EDP-sorted SFT pairs from solution_metrics.jsonl.

Closes the SFT-vs-GRPO axis asymmetry: GRPO has 3 reward axes (EDP/Energy/Runtime);
SFT had 2 (Energy/Runtime). This script adds the 3rd SFT axis by selecting pairs
ranked by EDP reduction rather than energy reduction.

Output: data/edp/sft_pairs_{train,val,test}.jsonl matching the schema of data/sft_pairs_*.jsonl.

Reads: analysis/solution_metrics.jsonl
       (39,744 records; has avg_edp + avg_energy + avg_runtime per (problem, code_hash))

Pair construction parallel to dataset_preprocessing.py (energy-sorted), but sort key is avg_edp.
"""
import json, os, random
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "finetuning" / "data"
SOL = REPO_ROOT / "analysis" / "solution_metrics.jsonl"
OUT_DIR = Path(os.environ.get("GREEN_TEA_EDP_OUT", str(DATA / "edp")))
SPLIT_FILE = DATA / "problem_split.json"
MIN_EDP_REDUCTION = 0.20  # 20% — matches the energy-SFT threshold
SEED = 42

random.seed(SEED)

# Load solution metrics, group by problem
print(f"Loading {SOL}...")
by_problem = defaultdict(list)
with open(SOL) as f:
    for line in f:
        r = json.loads(line)
        if r.get("avg_edp", 0) > 0 and r.get("avg_energy", 0) > 0:
            by_problem[r["problem_id"]].append(r)
print(f"  {len(by_problem)} problems with valid EDP measurements")

# Load existing problem split (train/val/test)
if SPLIT_FILE.exists():
    split = json.loads(SPLIT_FILE.read_text())
    train_pids = set(split["train"]); val_pids = set(split["val"]); test_pids = set(split["test"])
    print(f"  Existing split: train={len(train_pids)} val={len(val_pids)} test={len(test_pids)}")
else:
    print(f"  WARNING: {SPLIT_FILE} not found; using random 80/10/10 split.")
    all_pids = sorted(by_problem.keys())
    random.shuffle(all_pids)
    n = len(all_pids)
    train_pids = set(all_pids[:int(n*0.8)])
    val_pids = set(all_pids[int(n*0.8):int(n*0.9)])
    test_pids = set(all_pids[int(n*0.9):])

# Need solution code as well (solution_metrics doesn't include code, only code_hash).
# Pull code from the existing sft_pairs file's baseline_code / optimized_code keyed by code_hash.
CODE_BY_HASH = {}
for src in [DATA / "sft_pairs_train.jsonl",
            DATA / "sft_pairs_val.jsonl",
            DATA / "sft_pairs_test.jsonl"]:
    if not src.exists(): continue
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            CODE_BY_HASH[r["baseline_code_hash"]] = r.get("inefficient_code", "")
            CODE_BY_HASH[r["optimized_code_hash"]] = r.get("optimized_code", "")
print(f"  {len(CODE_BY_HASH)} unique code snippets loaded by hash")

def build_pairs(pids):
    """For each problem, sort solutions by avg_edp ascending (lowest EDP = best),
    then pair each non-best solution with the best solution if EDP reduction >= threshold."""
    pairs = []
    for pid in pids:
        sols = by_problem.get(pid, [])
        if len(sols) < 2: continue
        sols_sorted = sorted(sols, key=lambda x: x["avg_edp"])
        best = sols_sorted[0]
        for baseline in sols_sorted[1:]:
            edp_red = (baseline["avg_edp"] - best["avg_edp"]) / baseline["avg_edp"]
            if edp_red < MIN_EDP_REDUCTION: continue
            base_code = CODE_BY_HASH.get(baseline["code_hash"], "")
            opt_code = CODE_BY_HASH.get(best["code_hash"], "")
            if not base_code or not opt_code: continue
            en_red = (baseline["avg_energy"] - best["avg_energy"]) / max(baseline["avg_energy"], 1e-9)
            pairs.append({
                "problem_id": pid,
                "inefficient_code": base_code, "optimized_code": opt_code,
                "baseline_code_hash": baseline["code_hash"], "optimized_code_hash": best["code_hash"],
                "baseline_energy": baseline["avg_energy"], "optimized_energy": best["avg_energy"],
                "baseline_cycles": baseline.get("avg_cycles", 0), "optimized_cycles": best.get("avg_cycles", 0),
                "baseline_ipc": baseline.get("avg_ipc", 0), "optimized_ipc": best.get("avg_ipc", 0),
                "baseline_power": baseline.get("avg_power", 0), "optimized_power": best.get("avg_power", 0),
                "baseline_runtime": baseline.get("avg_runtime", 0), "optimized_runtime": best.get("avg_runtime", 0),
                "baseline_instructions": baseline.get("avg_instructions", 0), "optimized_instructions": best.get("avg_instructions", 0),
                "baseline_edp": baseline["avg_edp"], "optimized_edp": best["avg_edp"],
                "baseline_score": 1, "optimized_score": 10,
                "edp_reduction_pct": edp_red * 100, "energy_reduction_pct": en_red * 100,
                "speedup": baseline.get("avg_runtime", 0) / max(best.get("avg_runtime", 1e-9), 1e-9),
                "ipc_improvement": best.get("avg_ipc", 0) - baseline.get("avg_ipc", 0),
                "baseline_execution_id": "", "optimized_execution_id": "",
            })
    return pairs

OUT_DIR.mkdir(parents=True, exist_ok=True)
for split_name, pids in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
    pairs = build_pairs(pids)
    out_file = OUT_DIR / f"sft_pairs_{split_name}.jsonl"
    with open(out_file, "w") as f:
        for p in pairs: f.write(json.dumps(p) + "\n")
    edp_reds = [p["edp_reduction_pct"] for p in pairs]
    en_reds = [p["energy_reduction_pct"] for p in pairs]
    import numpy as np
    print(f"  {split_name}: {len(pairs)} pairs, EDP red mean={np.mean(edp_reds):.1f}% median={np.median(edp_reds):.1f}%, "
          f"energy red mean={np.mean(en_reds):.1f}% median={np.median(en_reds):.1f}%")
    print(f"    -> {out_file}")
print("\nEDP-sorted pair construction complete.")
