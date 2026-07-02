# Energy Data Collection Scripts

Production scripts for PIE dataset energy collection using Sniper+McPAT simulation.

---

## Scripts Overview

### 1. `generate_execution_master.py`

Creates execution master from PIE train.jsonl with code-hash deduplication.

```bash
python3 generate_execution_master.py \
    --input ../PIE_Dataset/train.jsonl \
    --output ../PIE_Dataset/execution_master.jsonl
```

**Output:** 3,546,816 unique (problem, code, test_input) executions

### 2. `create_batch_splits.py`

Splits execution master into 440 balanced batches.

```bash
python3 create_batch_splits.py \
    --execution-master ../PIE_Dataset/execution_master.jsonl \
    --output-dir ../PIE_Dataset/batches \
    --num-batches 440
```

**Output:** 440 files (~8,061 executions each), balanced by problem

### 3. `sniper_execution_runner.py` ⭐ Main Runner

Processes execution batches with Sniper simulation.

```bash
python3 sniper_execution_runner.py \
    --batch-file ../PIE_Dataset/batches/execution_master_batch_000.jsonl \
    --cache-dir ../pie_energy_cache \
    --sniper-path ../sniper/sniper \
    --sniper-config ../sniper/sniper/config/epyc_9554p.cfg
```

**Features:**
- Automatic resumability via SharedExecutionCache
- 2-minute timeout per execution
- High-precision energy (6 decimals)
- Progress tracking

### 4. `slurm_execution_master.sh` ⭐ Production Script

SLURM array job script (440 parallel jobs).

```bash
sbatch slurm_execution_master.sh
# Or specific batch:
sbatch --array=0 slurm_execution_master.sh
```

**Features:**
- Sources `config.env` for paths
- Trap logic for data safety (SIGTERM/SIGUSR1)
- Background Python process with PID tracking
- Metadata saved to `logs/`

### 5. `check_incomplete_jobs.py`

Monitor batch progress.

```bash
python3 check_incomplete_jobs.py --batch-id 0
```

**Output:**
```
Batch 0: 6,500 / 8,061 (80.6% complete)
  Completed: 6,500
  Failed: 12
  Remaining: 1,549
```

### 6. `shared_cache.py`

SharedExecutionCache implementation (imported by runner).

**Features:**
- Atomic file-based cache
- Thread-safe operations
- Tracks completed/failed executions

---

## Configuration

All scripts use `../config.env` for paths:

```bash
source ../config.env
```

**Key variables:**
- `PROJECT_ROOT` - Project directory
- `SNIPER_ROOT` - Sniper installation
- `SNIPER_CONFIG` - Config file (epyc_9554p.cfg)
- `CACHE_DIR` - Shared cache location
- `BATCH_DIR` - Batch files location

---

## Workflow

```
1. generate_execution_master.py   → execution_master.jsonl
2. create_batch_splits.py         → 440 batch files
3. slurm_execution_master.sh      → Submit 440 jobs
   └─► sniper_execution_runner.py → Process batches
       └─► shared_cache.py        → Save results
4. check_incomplete_jobs.py       → Monitor progress
```

---

## Cache Structure

```
pie_energy_cache/
├── completed/
│   ├── p00000_23684839615f_68b329da.done
│   └── ... (execution results)
└── failed/
    ├── p00123_xyz.failed
    └── ... (error messages)
```

**Completed format (`.done`):**
```json
{
  "execution_id": "p00000_23684839615f_68b329da",
  "completed_at": 1760964255.026678,
  "result": {
    "energy_joules": 0.033674,
    "power_watts": 204.56,
    "runtime_seconds": 0.0001652,
    "cycles": 247800,
    "instructions": 260109
  }
}
```

---

## Deprecated Scripts

Moved to `depr/`:
- `sniper_parallel_runner.py` - Old sample-based runner
- `analyze_*.py` - One-time analysis scripts
- `calculate_unique_simulations.py` - Migration helper
- `export_problem_statistics.py` - Statistics helper

---

## Monitoring & Debugging

**SLURM logs:**
```
logs/pie_energy_exec_<job_id>_<array_id>.out
logs/pie_energy_exec_<job_id>_<array_id>.err
logs/job_metadata_<job_id>_<array_id>.json
logs/summary_<job_id>_<array_id>.txt
```

**Check cache stats:**
```bash
find ../pie_energy_cache/completed -name "*.done" | wc -l
find ../pie_energy_cache/failed -name "*.failed" | wc -l
```

**Test runner locally:**
```bash
head -10 ../PIE_Dataset/execution_master.jsonl > test.jsonl
python3 sniper_execution_runner.py \
    --batch-file test.jsonl \
    --cache-dir /tmp/test \
    --sniper-config ../sniper/sniper/config/epyc_9554p.cfg
```

---

## Performance

**Typical:** 25s per execution
**Budget:** 75s per execution (3× safety margin)
**Cache hits:** ~74,000 executions/second (instant skip)

---

**Last Updated:** 2025-10-20
