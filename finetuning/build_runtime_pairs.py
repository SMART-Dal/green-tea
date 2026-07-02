#!/usr/bin/env python3
"""Build runtime-contrastive SFT training pairs for W1 ablation.

Reads sft_pairs_{train,val}.jsonl, re-scores solutions by per-problem
runtime percentile (1=slowest, 10=fastest), filters for speedup>=10%,
and writes to data/runtime/sft_pairs_{train,val}.jsonl.

The 'optimized_score' field is replaced with the runtime-percentile score
so sft_train_trl.py picks it up unchanged (--template runtime changes
only the prompt wording, not the score field).
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA = Path(__file__).parent / "data"
OUT  = DATA / "runtime"
OUT.mkdir(exist_ok=True)

MIN_SPEEDUP = 0.10   # 10% minimum speedup (mirrors energy 10% threshold)


def runtime_score(rt: float, all_rts: list[float]) -> int:
    """Percentile rank 1-10: 10 = fastest (lowest runtime)."""
    arr = np.array(all_rts)
    pct = np.mean(arr >= rt)   # fraction with runtime >= this (higher = slower)
    return max(1, min(10, int(pct * 10) + 1))


def build(split: str):
    src = DATA / f"sft_pairs_{split}.jsonl"
    if not src.exists():
        print(f"  {src} not found, skipping")
        return

    pairs = [json.loads(l) for l in src.open()]

    # Collect all runtimes per problem
    prob_rts: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        pid = p["problem_id"]
        prob_rts[pid].append(p["baseline_runtime"])
        prob_rts[pid].append(p["optimized_runtime"])

    out_pairs = []
    skipped = 0
    for p in pairs:
        pid = p["problem_id"]
        speedup = p["speedup"]
        if speedup < 1 + MIN_SPEEDUP:   # <10% speedup
            skipped += 1
            continue
        all_rts = prob_rts[pid]
        base_score = runtime_score(p["baseline_runtime"], all_rts)   # slow = low score
        opt_score  = runtime_score(p["optimized_runtime"], all_rts)  # fast = high score
        if opt_score <= base_score:
            skipped += 1
            continue
        out = dict(p)
        out["optimized_score"] = opt_score      # runtime percentile (replaces energy score)
        out["baseline_score"]  = base_score
        out_pairs.append(out)

    dst = OUT / f"sft_pairs_{split}.jsonl"
    with dst.open("w") as fh:
        for p in out_pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"  {split}: {len(pairs)} -> {len(out_pairs)} pairs (skipped {skipped}) -> {dst}")


for split in ["train", "val"]:
    build(split)
