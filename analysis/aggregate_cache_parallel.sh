#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --account=rrg-mrdal22
#SBATCH --job-name=aggregate_cache
#SBATCH --output=../logs/aggregate_cache_%j.out
#SBATCH --error=../logs/aggregate_cache_%j.err
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

echo "Cache Aggregation - Job $SLURM_JOB_ID - $(date)"
echo "Node: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"

module load StdEnv/2023
module load python/3.10

export CACHE_DIR="../pie_energy_cache"
export OUTPUT_DIR="."
export WORKERS=$SLURM_CPUS_PER_TASK

python3 << 'EOF'
import os
import json
import time
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime, timedelta

CACHE_DIR = Path(os.environ['CACHE_DIR'])
OUTPUT_DIR = Path(os.environ['OUTPUT_DIR'])
WORKERS = int(os.environ['WORKERS'])
BATCH_SIZE = 1000

def process_file_batch(files):
    results = []
    for fpath in files:
        try:
            with open(fpath) as f:
                results.append(json.load(f))
        except:
            pass
    return results

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def aggregate_files(directory, suffix, output_file, workers):
    print(f"\nProcessing {directory}...")
    print(f"  Scanning directory...")

    files = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.endswith(suffix):
                files.append(entry.path)

    total = len(files)
    print(f"  Found {total:,} files")

    batches = [files[i:i+BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"  Split into {total_batches:,} batches of {BATCH_SIZE} files")
    print(f"  Processing with {workers} workers...\n")

    start_time = time.time()
    processed = 0
    all_results = []

    with Pool(workers) as pool:
        for batch_results in pool.imap_unordered(process_file_batch, batches):
            all_results.extend(batch_results)
            processed += BATCH_SIZE
            processed = min(processed, total)

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0

            pct = 100 * processed / total
            print(f"  [{pct:5.1f}%] {processed:>9,}/{total:,} files | "
                  f"{rate:>7,.0f} files/s | Elapsed: {format_time(elapsed)} | "
                  f"ETA: {format_time(remaining)}", flush=True)

    elapsed_total = time.time() - start_time
    print(f"\n  Completed in {format_time(elapsed_total)}")
    print(f"  Average rate: {total/elapsed_total:,.0f} files/s")

    print(f"  Writing {len(all_results):,} records to {output_file}...")
    write_start = time.time()
    with open(output_file, 'w') as out:
        for record in all_results:
            out.write(json.dumps(record) + '\n')

    write_time = time.time() - write_start
    file_size_mb = os.path.getsize(output_file) / 1024**2
    print(f"  Written {file_size_mb:.1f} MB in {format_time(write_time)}\n")

print("="*80)
print(f"AGGREGATING PIE ENERGY CACHE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

completed_dir = CACHE_DIR / "completed"
failed_dir = CACHE_DIR / "failed"

if completed_dir.exists():
    aggregate_files(completed_dir, '.done', OUTPUT_DIR / 'completed_samples.jsonl', WORKERS)

if failed_dir.exists():
    aggregate_files(failed_dir, '.failed', OUTPUT_DIR / 'failed_samples.jsonl', WORKERS)

print("="*80)
print(f"AGGREGATION COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
EOF

echo "Aggregation job completed at $(date)"
