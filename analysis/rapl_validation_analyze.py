#!/usr/bin/env python3
"""Hardware cross-check of simulated energy orderings with RAPL (Section 4.2, Table 1).

Usage (from analysis/):
  python3 rapl_validation_analyze.py rapl_results.jsonl --pairs rapl_validation_pairs.jsonl \
      --rapl-scanf rapl_scanf_results.jsonl --pairs-scanf rapl_scanf_pairs.jsonl

A pair is measured when both programs returned a positive RAPL reading, tied when both readings are equal
(an unresolved ordering), and agrees when the simulated and measured energies order the two programs alike.
"""
import argparse, json
import numpy as np


def load(path):
    return [json.loads(l) for l in open(path)]


def counts(meas):
    res = [r for r in meas if r["rapl_baseline_J"] != r["rapl_optimized_J"]]
    agree = int(sum(np.sign(r["mcpat_baseline_energy_J"] - r["mcpat_optimized_energy_J"]) == np.sign(r["rapl_baseline_J"] - r["rapl_optimized_J"]) for r in res))
    return len(meas) - len(res), len(res), agree


def show(name, n_pairs, meas):
    tied, res, agree = counts(meas)
    print(f"{name:26s} pairs={n_pairs:4d} measured={len(meas):4d} tied={tied:3d} resolved={res:3d} agree={agree:3d} ({agree / max(res, 1):.0%}) disagree={res - agree:3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rapl"); ap.add_argument("--pairs", required=True)
    ap.add_argument("--rapl-scanf"); ap.add_argument("--pairs-scanf")
    a = ap.parse_args()
    sets = [("Stratified sample", load(a.rapl), {p["pair_id"]: p for p in load(a.pairs)})]
    if a.rapl_scanf:
        sets.append(("scanf/printf", load(a.rapl_scanf), {p["pair_id"]: p for p in load(a.pairs_scanf)}))
    ok = lambda r: r["rapl_baseline_J"] and r["rapl_optimized_J"]
    key = lambda r, P: (P[r["pair_id"]]["baseline_code"], P[r["pair_id"]]["optimized_code"])
    for name, R, P in sets:
        show(name, len(R), [r for r in R if ok(r)])
        for r in R:
            if not ok(r):
                cause = "no exported test input" if not P[r["pair_id"]]["test_inputs"] else f"no valid reading ({r.get('error')})"
                print(f"    excluded pair {r['pair_id']} ({r['problem_id']}): {cause}")
    show("Total", sum(len(R) for _, R, _ in sets), [r for _, R, _ in sets for r in R if ok(r)])
    seen, distinct = set(), []  # each distinct (baseline, generated) program pair once, at its first measurement
    for _, R, P in sets:
        for r in R:
            if ok(r) and key(r, P) not in seen:
                seen.add(key(r, P)); distinct.append(r)
    show("Distinct program pairs", len({key(r, P) for _, R, P in sets for r in R}), distinct)
    meas = [r for _, R, _ in sets for r in R if ok(r)]
    print(f"measured pairs with ten valid readings for both programs: {sum(r['n_reps_base'] == 10 and r['n_reps_opt'] == 10 for r in meas)} of {len(meas)}")


if __name__ == "__main__":
    main()
