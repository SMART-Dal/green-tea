import json, os, random, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "finetuning"))
from evaluation import SniperEvaluator as EnergyEvaluator
S = os.environ.get("SNIPER_ROOT", str(ROOT / "sniper" / "sniper"))
CFG = {"1.5": f"{S}/config/epyc_9554p.cfg", "3.1": str(Path(__file__).resolve().parent / "epyc_9554p_3p1.cfg")}
TC = Path(os.environ.get("PIE_TESTCASES", str(ROOT / "PIE_Dataset/extracted_testcases/merged_test_cases")))
pairs = [json.loads(l) for l in open(ROOT / "finetuning/data/sft_pairs_test.jsonl")]
random.seed(0)
by_pid = {}
for p in pairs: by_pid.setdefault(p["problem_id"], []).append(p)
sample = [random.choice(v) for k, v in sorted(by_pid.items())]
sample = random.sample(sample, 40)  # one pair per problem, 40 problems
def inputs(pid):
    return [f.read_text() for f in sorted(TC.glob(f"{pid}/input.*.txt"))[:3]]
jobs = [(f, i, role, j, code, inp) for i, p in enumerate(sample) for role, code in (("base", p["inefficient_code"]), ("opt", p["optimized_code"]))
        for j, inp in enumerate(inputs(p["problem_id"])) for f in CFG]
def run(job):
    f, i, role, j, code, inp = job
    r = EnergyEvaluator(S, sniper_config=CFG[f], timeout=600).measure_energy(code, inp)
    return (f, i, role, j), (r.get("energy_joules") if r.get("status") == "success" else None, r.get("cycles"))
if __name__ == "__main__":
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as pool:
        E = dict(pool.map(run, jobs, chunksize=1))
    err = {f: [] for f in CFG}; keep = []
    for i, p in enumerate(sample):
        js = sorted({k[3] for k in E if k[1] == i})
        good = [j for j in js if all(E.get((f, i, r, j), (None,))[0] for f in CFG for r in ("base", "opt"))]
        if not good: continue
        keep.append(p["problem_id"])
        for f in CFG:
            b = np.mean([E[(f, i, "base", j)][0] for j in good]); o = np.mean([E[(f, i, "opt", j)][0] for j in good])
            err[f].append((1 - o / b) * 100)
    a, b = np.array(err["1.5"]), np.array(err["3.1"])
    print(f"pairs={len(a)} spearman={spearmanr(a, b)[0]:.3f} sign_agree={np.mean(np.sign(a) == np.sign(b)):.3f} "
          f"median|diff|={np.median(np.abs(a - b)):.2f}pp mean_err_1.5={a.mean():.2f} mean_err_3.1={b.mean():.2f} "
          f"below10_at_3.1_of_ge10_at_1.5={int(((a >= 10) & (b < 10)).sum())}/{int((a >= 10).sum())}")
    json.dump(dict(problems=keep, err_1p5=list(a), err_3p1=list(b)), open(Path(__file__).resolve().parent / "freq_results.json", "w"))
