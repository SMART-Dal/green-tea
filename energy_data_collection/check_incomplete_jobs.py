#!/usr/bin/env python3
"""
Check which batch jobs are incomplete by comparing batch files with cache.

Usage:
    python3 check_incomplete_jobs.py \
        --batch-dir PIE_Dataset/batches/ \
        --cache-dir /scratch/shared/pie_energy_cache

Output:
    - List of incomplete job IDs
    - Progress statistics per batch
    - Command to rerun incomplete jobs
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_batch_executions(batch_file):
    """Load execution IDs from a batch file."""
    executions = []
    with open(batch_file) as f:
        for line in f:
            record = json.loads(line)
            executions.append(record['execution_id'])
    return executions

def check_batch_completion(batch_id, batch_dir, cache_dir):
    """
    Check completion status of a batch.

    Returns:
        (total, completed, remaining, progress_pct)
    """
    # Load batch file
    batch_file = batch_dir / f"execution_master_batch_{batch_id:03d}.jsonl"
    if not batch_file.exists():
        return None

    execution_ids = load_batch_executions(batch_file)
    total = len(execution_ids)

    # Check how many are completed
    completed_dir = cache_dir / "completed"
    completed = 0

    for exec_id in execution_ids:
        marker_file = completed_dir / f"{exec_id}.done"
        if marker_file.exists():
            completed += 1

    remaining = total - completed
    progress_pct = (completed / total * 100) if total > 0 else 0

    return (total, completed, remaining, progress_pct)

def main():
    parser = argparse.ArgumentParser(
        description='Check which batch jobs are incomplete'
    )
    parser.add_argument(
        '--batch-dir',
        default='PIE_Dataset/batches/',
        help='Directory containing batch files'
    )
    parser.add_argument(
        '--cache-dir',
        default='/scratch/shared/pie_energy_cache',
        help='Cache directory'
    )
    parser.add_argument(
        '--num-batches',
        type=int,
        default=440,
        help='Total number of batches'
    )
    parser.add_argument(
        '--incomplete-threshold',
        type=float,
        default=99.0,
        help='Consider batch incomplete if less than this percent complete (default: 99.0)'
    )

    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    cache_dir = Path(args.cache_dir)

    if not batch_dir.exists():
        print(f"❌ Error: Batch directory not found: {batch_dir}")
        return 1

    if not cache_dir.exists():
        print(f"❌ Error: Cache directory not found: {cache_dir}")
        return 1

    print("=" * 70)
    print("CHECKING BATCH COMPLETION STATUS")
    print("=" * 70)
    print(f"Batch directory: {batch_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Total batches: {args.num_batches}")
    print()

    incomplete_jobs = []
    complete_jobs = []
    in_progress_jobs = []

    total_executions = 0
    total_completed = 0

    # Check each batch
    for batch_id in range(args.num_batches):
        result = check_batch_completion(batch_id, batch_dir, cache_dir)

        if result is None:
            print(f"⚠️  Batch {batch_id:03d}: File not found")
            continue

        total, completed, remaining, progress_pct = result

        total_executions += total
        total_completed += completed

        if progress_pct < 1.0:
            # Essentially nothing done
            incomplete_jobs.append(batch_id)
            status = "❌ NOT STARTED"
        elif progress_pct < args.incomplete_threshold:
            # Partially done
            in_progress_jobs.append(batch_id)
            status = f"⏳ IN PROGRESS ({progress_pct:.1f}%)"
        else:
            # Complete or nearly complete
            complete_jobs.append(batch_id)
            status = f"✅ COMPLETE ({progress_pct:.1f}%)"

        # Show details for incomplete/in-progress batches
        if progress_pct < args.incomplete_threshold:
            print(f"Batch {batch_id:03d}: {status} - {completed:,}/{total:,} done, {remaining:,} remaining")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total executions across all batches: {total_executions:,}")
    print(f"Completed executions: {total_completed:,}")
    print(f"Remaining executions: {total_executions - total_completed:,}")
    print(f"Overall progress: {total_completed / total_executions * 100:.2f}%")
    print()
    print(f"✅ Complete batches: {len(complete_jobs)}")
    print(f"⏳ In-progress batches: {len(in_progress_jobs)}")
    print(f"❌ Not started batches: {len(incomplete_jobs)}")
    print()

    # Incomplete jobs list
    needs_rerun = in_progress_jobs + incomplete_jobs

    if needs_rerun:
        print("=" * 70)
        print("JOBS THAT NEED TO BE RERUN")
        print("=" * 70)
        print(f"Number of jobs: {len(needs_rerun)}")
        print()

        # Format for SLURM --array parameter
        if len(needs_rerun) <= 10:
            # List individually
            array_spec = ",".join(str(j) for j in needs_rerun)
        else:
            # Show first 10 and indicate there are more
            array_spec = ",".join(str(j) for j in needs_rerun[:10]) + ",..."

        print("Jobs to rerun (batch IDs):")
        print(needs_rerun)
        print()

        print("SLURM command to rerun these jobs:")
        print(f"sbatch --array={array_spec} energy_data_collection/slurm_pie_energy_parallel.sh")
        print()

        # Save to file for easy rerun
        rerun_file = Path("rerun_jobs.txt")
        with open(rerun_file, 'w') as f:
            f.write(",".join(str(j) for j in needs_rerun))

        print(f"✅ Job list saved to: {rerun_file}")
        print(f"   Use: sbatch --array=$(cat {rerun_file}) slurm_pie_energy_parallel.sh")
    else:
        print("=" * 70)
        print("🎉 ALL BATCHES COMPLETE!")
        print("=" * 70)
        print("No jobs need to be rerun. All executions are complete.")

    print()

    return 0

if __name__ == "__main__":
    exit(main())
