#!/usr/bin/env python3
"""Hardware RAPL energy measurement for McPAT validation.

Reads rapl_validation_pairs.jsonl, compiles each (baseline, optimized) pair,
measures energy via RAPL using perf stat, and writes rapl_results.jsonl in the
same format ready for rapl_validation_analyze.py.

Usage:
  python3 rapl_measure.py --pairs rapl_validation_pairs.jsonl [options]

Requirements:
  - Linux with RAPL support (Intel Sandy Bridge+ or AMD Zen+)
  - perf with energy-pkg event: perf stat -e power/energy-pkg/ true
  - g++ for compilation
  - Sudo for perf_event_paranoid if needed:
      echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

Best-practice measurement protocol:
  - CPU pinned to core 0 via taskset (avoids migration overhead)
  - N_WARMUP=2 discard runs before measurement
  - N_REPS=10 measurement runs, median taken (robust to OS jitter)
  - SLEEP_BETWEEN=3s between all runs (thermal recovery + capacitor discharge)
  - SLEEP_PAIR=8s between pairs (full thermal settling)
  - Programs run interleaved: baseline/optimized/baseline/optimized/...
    so thermal drift cancels symmetrically
  - Compile with -O3 -std=c++17 -static (same as Sniper evaluation)
  - Each input run is timed to detect timeout (120s hard kill)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

# ── tuneable constants ────────────────────────────────────────────────────────
N_WARMUP = 2          # discarded runs before measurement
N_REPS = 10           # measurement runs per (program, input) combo
SLEEP_BETWEEN = 3.0   # seconds between individual runs (thermal + RAPL reset)
SLEEP_PAIR = 8.0      # seconds between pairs (full settle)
COMPILE_TIMEOUT = 60  # seconds
RUN_TIMEOUT = 120     # seconds per single execution
CPU_CORE = "0"        # taskset core; change if core 0 is noisy
CXX = "g++"
CXXFLAGS = ["-O3", "-std=c++17", "-static"]
PERF_EVENT = "power/energy-pkg/"
# ─────────────────────────────────────────────────────────────────────────────


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def compile_code(src: str, out: Path, tmpdir: Path) -> tuple[bool, str]:
    src_path = tmpdir / "prog.cpp"
    src_path.write_text(src)
    try:
        r = subprocess.run(
            [CXX] + CXXFLAGS + ["-o", str(out), str(src_path)],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT
        )
        return r.returncode == 0, r.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "compile timeout"


def parse_energy_joules(perf_output: str) -> float | None:
    """Extract Joules from perf stat output. Handles comma-separated numbers."""
    for line in perf_output.splitlines():
        if "energy-pkg" in line or "energy_pkg" in line:
            # e.g. "         3.14 Joules power/energy-pkg/"
            # or   "         3,142.56 Joules power/energy-pkg/"
            m = re.search(r"([\d,]+\.?\d*)\s+Joules", line)
            if m:
                return float(m.group(1).replace(",", ""))
    return None


def measure_energy(binary: Path, stdin_data: str | None, label: str) -> list[float]:
    """Run binary N_WARMUP+N_REPS times, return N_REPS energy readings (Joules)."""
    cmd = ["taskset", "-c", CPU_CORE,
           "perf", "stat", "-e", PERF_EVENT,
           str(binary)]
    all_readings = []
    total_runs = N_WARMUP + N_REPS
    for i in range(total_runs):
        phase = "warmup" if i < N_WARMUP else f"rep {i - N_WARMUP + 1}/{N_REPS}"
        try:
            result = subprocess.run(
                cmd,
                input=stdin_data, capture_output=True, text=True,
                timeout=RUN_TIMEOUT
            )
            # perf writes to stderr
            joules = parse_energy_joules(result.stderr)
            if joules is None or joules <= 0:
                log(f"    {label} [{phase}] energy parse failed -- stderr: {result.stderr[:200]}")
            else:
                if i >= N_WARMUP:
                    all_readings.append(joules)
                    log(f"    {label} [{phase}] {joules:.4f} J")
                else:
                    log(f"    {label} [{phase}] {joules:.4f} J (discarded)")
        except subprocess.TimeoutExpired:
            log(f"    {label} [{phase}] TIMEOUT after {RUN_TIMEOUT}s -- skipped")
        except Exception as e:
            log(f"    {label} [{phase}] ERROR: {e}")
        if i < total_runs - 1:
            time.sleep(SLEEP_BETWEEN)
    return all_readings


def measure_pair_interleaved(base_bin: Path, opt_bin: Path,
                              stdin_data: str | None) -> tuple[list[float], list[float]]:
    """Interleave baseline/optimized runs so thermal drift cancels."""
    cmd_base = ["taskset", "-c", CPU_CORE, "perf", "stat", "-e", PERF_EVENT, str(base_bin)]
    cmd_opt  = ["taskset", "-c", CPU_CORE, "perf", "stat", "-e", PERF_EVENT, str(opt_bin)]
    base_readings, opt_readings = [], []
    total_runs = N_WARMUP + N_REPS

    # warmup both
    log("    Warmup runs (discarded)...")
    for _ in range(N_WARMUP):
        for cmd, lbl in [(cmd_base, "base"), (cmd_opt, "opt")]:
            try:
                subprocess.run(cmd, input=stdin_data, capture_output=True,
                                text=True, timeout=RUN_TIMEOUT)
            except Exception:
                pass
            time.sleep(SLEEP_BETWEEN)

    # interleaved measurement
    log(f"    Measuring {N_REPS} reps (interleaved)...")
    for rep in range(N_REPS):
        for cmd, readings, lbl in [
            (cmd_base, base_readings, "base"),
            (cmd_opt,  opt_readings,  "opt"),
        ]:
            try:
                r = subprocess.run(cmd, input=stdin_data, capture_output=True,
                                   text=True, timeout=RUN_TIMEOUT)
                j = parse_energy_joules(r.stderr)
                if j and j > 0:
                    readings.append(j)
                    log(f"    rep {rep+1}/{N_REPS} {lbl}: {j:.4f} J")
                else:
                    log(f"    rep {rep+1}/{N_REPS} {lbl}: parse fail")
            except subprocess.TimeoutExpired:
                log(f"    rep {rep+1}/{N_REPS} {lbl}: TIMEOUT")
            except Exception as e:
                log(f"    rep {rep+1}/{N_REPS} {lbl}: ERROR {e}")
            if not (rep == N_REPS - 1 and lbl == "opt"):
                time.sleep(SLEEP_BETWEEN)

    return base_readings, opt_readings


def check_perf_available() -> bool:
    try:
        r = subprocess.run(
            ["perf", "stat", "-e", PERF_EVENT, "true"],
            capture_output=True, text=True, timeout=10
        )
        return parse_energy_joules(r.stderr) is not None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="rapl_validation_pairs.jsonl")
    ap.add_argument("--out", default="rapl_results.jsonl")
    ap.add_argument("--start", type=int, default=0, help="resume from pair index")
    ap.add_argument("--end", type=int, default=None, help="stop after this pair index (exclusive)")
    ap.add_argument("--input-idx", type=int, default=0, help="which test input index to use (0=first)")
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs file not found: {pairs_path}")
        sys.exit(1)

    pairs = []
    with pairs_path.open() as fh:
        for line in fh:
            try:
                pairs.append(json.loads(line))
            except Exception:
                pass
    log(f"Loaded {len(pairs)} pairs from {pairs_path}")

    if not check_perf_available():
        log("WARNING: perf energy-pkg event not readable. Try:")
        log("  echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid")
        log("Continuing anyway -- measurements may fail.")

    out_path = Path(args.out)
    # load already-completed pair_ids for resume
    done_ids = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["pair_id"])
                except Exception:
                    pass
        log(f"Resuming: {len(done_ids)} pairs already done")

    end = args.end if args.end is not None else len(pairs)
    pairs_to_run = [p for p in pairs[args.start:end] if p["pair_id"] not in done_ids]
    log(f"Running {len(pairs_to_run)} pairs (index {args.start}..{end-1})")

    with out_path.open("a") as out_fh, tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for idx, pair in enumerate(pairs_to_run):
            pid = pair["pair_id"]
            problem = pair.get("problem_id", "")
            source = pair.get("source", "")
            log(f"Pair {idx+1}/{len(pairs_to_run)} | pair_id={pid} | {problem} | {source}")

            # pick test input
            inputs = pair.get("test_inputs", [])
            stdin_data = inputs[args.input_idx] if inputs and args.input_idx < len(inputs) else None
            if stdin_data is None:
                log("  No test input available -- running without stdin")

            # compile
            base_bin = tmpdir / f"base_{pid}"
            opt_bin  = tmpdir / f"opt_{pid}"
            base_ok, base_err = compile_code(pair["baseline_code"], base_bin, tmpdir)
            opt_ok,  opt_err  = compile_code(pair["optimized_code"], opt_bin, tmpdir)

            record = {
                "pair_id": pid,
                "problem_id": problem,
                "source": source,
                "mcpat_err_pct": pair.get("mcpat_err_pct", 0),
                "mcpat_baseline_energy_J": pair.get("mcpat_baseline_energy_J", 0),
                "mcpat_optimized_energy_J": pair.get("mcpat_optimized_energy_J", 0),
                "compile_ok_base": base_ok,
                "compile_ok_opt": opt_ok,
                "rapl_baseline_J": None,
                "rapl_optimized_J": None,
                "rapl_err_pct": None,
                "n_reps_base": 0,
                "n_reps_opt": 0,
                "error": None,
            }

            if not base_ok:
                log(f"  Baseline compile FAILED: {base_err[:100]}")
                record["error"] = f"base compile: {base_err[:100]}"
            elif not opt_ok:
                log(f"  Optimized compile FAILED: {opt_err[:100]}")
                record["error"] = f"opt compile: {opt_err[:100]}"
            else:
                base_readings, opt_readings = measure_pair_interleaved(
                    base_bin, opt_bin, stdin_data
                )
                if base_readings and opt_readings:
                    base_med = median(base_readings)
                    opt_med  = median(opt_readings)
                    rapl_err = (base_med - opt_med) / base_med * 100 if base_med > 0 else 0
                    record["rapl_baseline_J"]  = round(base_med, 6)
                    record["rapl_optimized_J"] = round(opt_med, 6)
                    record["rapl_err_pct"]     = round(rapl_err, 4)
                    record["n_reps_base"] = len(base_readings)
                    record["n_reps_opt"]  = len(opt_readings)
                    log(f"  RESULT: base={base_med:.4f}J  opt={opt_med:.4f}J  "
                        f"RAPL ERR={rapl_err:.2f}%  McPAT ERR={pair.get('mcpat_err_pct',0):.2f}%")
                else:
                    record["error"] = f"measurement failed: base={len(base_readings)} opt={len(opt_readings)} readings"
                    log(f"  FAILED: insufficient readings")

            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

            if idx < len(pairs_to_run) - 1:
                log(f"  Sleeping {SLEEP_PAIR}s before next pair...")
                time.sleep(SLEEP_PAIR)

    # summary
    results = []
    with out_path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r.get("rapl_err_pct") is not None:
                    results.append(r)
            except Exception:
                pass
    log(f"\nDone. {len(results)} pairs with valid measurements saved to {out_path}")
    if results:
        agree = sum(1 for r in results
                    if (r["rapl_err_pct"] >= 0) == (r["mcpat_err_pct"] >= 0))
        log(f"Ranking agreement (sign match): {agree}/{len(results)} ({agree/len(results)*100:.1f}%)")
        log(f"Next step: python3 rapl_validation_analyze.py {out_path}")


if __name__ == "__main__":
    main()
