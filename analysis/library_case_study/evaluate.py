import json, os, subprocess, sys, tempfile
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "finetuning"))
from evaluation import SniperEvaluator as EnergyEvaluator
EV = EnergyEvaluator(os.environ.get("SNIPER_ROOT", str(ROOT / "sniper" / "sniper")), timeout=600)
cases = json.load(open(str(HERE / os.environ.get("CASE_SET", "systems_stdin") / "cases.json")))
gens = [json.loads(l) for l in open(str(HERE / os.environ.get("CASE_SET", "systems_stdin") / "generations.jsonl"))]

def correctness(g):
    c = cases[g["case"]]
    with tempfile.TemporaryDirectory() as t:
        src, b = Path(t) / "s.cpp", Path(t) / "s.bin"; src.write_text(g["generated_code"])
        try:
            if subprocess.run(["g++", "-O3", "-std=c++17", "-static", str(src), "-o", str(b)], capture_output=True, timeout=60).returncode:
                return dict(g, compiled=False, passed=0)
        except subprocess.TimeoutExpired:
            return dict(g, compiled=False, passed=0)
        ok = 0
        for inp, ref in zip(c["test_inputs"], c["test_outputs"]):
            try:
                r = subprocess.run([str(b)], input=inp, capture_output=True, text=True, timeout=10)
                ok += r.returncode == 0 and r.stdout.strip() == ref
            except subprocess.TimeoutExpired:
                pass
        return dict(g, compiled=True, passed=ok)

def energy(job):
    key, code, inp = job
    r = EV.measure_energy(code, inp)
    return key, (r.get("energy_joules") if r.get("status") == "success" else None, r.get("cycles"))

if __name__ == "__main__":
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    with Pool(n) as p:
        res = p.map(correctness, gens)
    jobs = [(("base", c, i), cases[c]["code"], inp) for c in cases for i, inp in enumerate(cases[c]["energy_inputs"])]
    jobs += [((j, r["case"], i), r["generated_code"], inp) for j, r in enumerate(res) if r["passed"] == 10 for i, inp in enumerate(cases[r["case"]]["energy_inputs"])]
    with Pool(n) as p:
        E = dict(p.map(energy, jobs, chunksize=1))
    for j, r in enumerate(res):
        if r["passed"] == 10:
            b = [E[("base", r["case"], i)][0] for i in range(3)]; g = [E[(j, r["case"], i)][0] for i in range(3)]
            if None not in b and None not in g:
                r["err"] = (1 - np.mean(g) / np.mean(b)) * 100
                r["cycle_ratio"] = float(np.mean([E[(j, r["case"], i)][1] for i in range(3)]) / np.mean([E[("base", r["case"], i)][1] for i in range(3)]))
    json.dump(dict(results=res, base={f"{k[1]}:{k[2]}": v for k, v in E.items() if k[0] == "base"}), open(str(HERE / os.environ.get("CASE_SET", "systems_stdin") / "results.json"), "w"), indent=1)
    for r in res:
        print(r["model"], r["case"], r["sample"], "compiled" if r["compiled"] else "no-compile", f"{r['passed']}/10", f"ERR={r['err']:.2f}%" if "err" in r else "")
