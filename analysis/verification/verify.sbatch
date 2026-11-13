#!/bin/bash
# Re-runs the revision analyses from the released data (Table 7, all Reviewer 2 statistics, label and reward
# checks) and re-evaluates the case-study run behind Section 6, then compares it with the packaged results.
# From the repository root: export VENV, DATASET (Zenodo green-tea-dataset), TESTCASES (PIE merged_test_cases),
# SNIPER_ROOT, and WORK (scratch folder), then run: sbatch --account=<account> analysis/verification/verify.sbatch
#SBATCH --job-name=pkg_verify
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=verify_%j.log
source "$VENV/bin/activate"
export TMPDIR=${SLURM_TMPDIR:-/tmp} SNIPER_ROOT
cd "${SLURM_SUBMIT_DIR:-.}"
echo "=== robustness_audit"
python3 -u analysis/robustness_audit.py --dataset "$DATASET" --testcases "$TESTCASES"
echo "=== case study re-evaluation"
mkdir -p "$WORK/bm" && python3 analysis/library_case_study/best_models/make_cases.py
cp analysis/library_case_study/best_models/{cases.json,generations.jsonl} "$WORK/bm/"
CASE_SET="$WORK/bm" python3 -u analysis/library_case_study/evaluate.py > "$WORK/bm_eval.txt"
python3 - "$WORK/bm/results.json" analysis/library_case_study/best_models/results.json <<'PY'
import json, sys
load = lambda f: {(r["model"], r["case"], r["sample"]): r for r in json.load(open(f))["results"]}
a, b = load(sys.argv[1]), load(sys.argv[2])
errs = [abs(a[k].get("err", 0) - b[k].get("err", 0)) for k in b if "err" in a[k] or "err" in b[k]]
print(f"records {len(b)}; identical pass counts {sum(a[k]['passed'] == b[k]['passed'] for k in b)}; outputs with energy {len(errs)}; max ERR difference {max(errs):.3f} pp")
for k in [("R-SFT+EDP", "alg/kactl_dinic", 3), ("E-SFT+energy", "alg/acl_dsu", 0), ("R-SFT+EDP", "stdin/ngx_murmurhash2", 0)]:
    print(k, "package", round(b[k].get("err", float("nan")), 2), "re-run", round(a[k].get("err", float("nan")), 2))
PY
