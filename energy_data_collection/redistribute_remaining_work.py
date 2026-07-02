#!/usr/bin/env python3
"""
Redistribute Remaining Work Across All Jobs

This script identifies uncompleted executions from all batch files and
redistributes them evenly across 440 new batch files for better parallelization.

Safety Features:
- Does NOT modify original batch files
- Does NOT delete or modify cache
- Creates new batch files in separate directory
- Preserves all execution metadata

Usage:
    python3 redistribute_remaining_work.py \
        --batch-dir PIE_Dataset/batches \
        --cache-dir pie_energy_cache \
        --output-dir PIE_Dataset/batches_remaining \
        --num-jobs 440

Author: Energy Analysis Team
Date: 2025-12-05
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Any
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_completed_execution_ids(cache_dir: Path) -> Set[str]:
    """
    Load set of completed execution IDs from cache.

    Args:
        cache_dir: Path to shared cache directory

    Returns:
        Set of completed execution IDs
    """
    completed_dir = cache_dir / "completed"
    if not completed_dir.exists():
        logger.warning(f"Completed cache directory not found: {completed_dir}")
        return set()

    completed_ids = set()
    for marker_file in completed_dir.glob("*.done"):
        execution_id = marker_file.stem
        completed_ids.add(execution_id)

    logger.info(f"Loaded {len(completed_ids):,} completed execution IDs from cache")
    return completed_ids


def collect_remaining_executions(batch_dir: Path,
                                 completed_ids: Set[str],
                                 max_batches: int = None) -> List[Dict[str, Any]]:
    """
    Collect all remaining (uncompleted) executions from batch files.

    Args:
        batch_dir: Directory containing original batch files
        completed_ids: Set of completed execution IDs
        max_batches: Optional limit on number of batches to scan

    Returns:
        List of execution records that are not yet completed
    """
    logger.info(f"Scanning batch files in: {batch_dir}")

    batch_files = sorted(batch_dir.glob("execution_master_batch_*.jsonl"))

    if max_batches:
        batch_files = batch_files[:max_batches]

    logger.info(f"Found {len(batch_files)} batch files to scan")

    remaining_executions = []
    total_executions = 0
    skipped_count = 0

    start_time = time.time()

    for i, batch_file in enumerate(batch_files, 1):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            logger.info(f"  Progress: {i}/{len(batch_files)} batches ({i/len(batch_files)*100:.1f}%) - {rate:.1f} batches/s")

        with open(batch_file) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    execution = json.loads(line)
                    execution_id = execution['execution_id']
                    total_executions += 1

                    # Skip if already completed
                    if execution_id in completed_ids:
                        skipped_count += 1
                        continue

                    # Add to remaining work
                    remaining_executions.append(execution)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {batch_file.name}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"\nScan complete in {elapsed:.1f} seconds")
    logger.info(f"  Total executions scanned: {total_executions:,}")
    logger.info(f"  Already completed (skipped): {skipped_count:,}")
    logger.info(f"  Remaining to process: {len(remaining_executions):,}")
    logger.info(f"  Completion rate: {skipped_count/total_executions*100:.2f}%")

    return remaining_executions


def write_redistributed_batches(executions: List[Dict[str, Any]],
                                output_dir: Path,
                                num_jobs: int):
    """
    Write remaining executions into evenly distributed batch files.

    Args:
        executions: List of execution records to redistribute
        output_dir: Directory to write new batch files
        num_jobs: Number of batch files to create (usually 440)
    """
    if not executions:
        logger.error("No executions to redistribute!")
        return

    logger.info(f"\nRedistributing {len(executions):,} executions across {num_jobs} batch files")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate batch size
    batch_size = len(executions) // num_jobs
    remainder = len(executions) % num_jobs

    logger.info(f"  Base batch size: {batch_size}")
    logger.info(f"  Batches with extra execution: {remainder}")

    # Write batch files
    start_idx = 0
    written_files = []

    for job_id in range(num_jobs):
        # Calculate size for this batch
        current_batch_size = batch_size + (1 if job_id < remainder else 0)
        end_idx = start_idx + current_batch_size

        # Get executions for this batch
        batch_executions = executions[start_idx:end_idx]

        # Write batch file
        batch_file = output_dir / f"execution_master_batch_{job_id:03d}.jsonl"
        with open(batch_file, 'w') as f:
            for execution in batch_executions:
                f.write(json.dumps(execution) + '\n')

        written_files.append((batch_file, len(batch_executions)))
        start_idx = end_idx

        if (job_id + 1) % 50 == 0 or (job_id + 1) == num_jobs:
            logger.info(f"  Written: {job_id + 1}/{num_jobs} batch files")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("REDISTRIBUTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total batch files: {len(written_files)}")
    logger.info(f"Total executions: {sum(count for _, count in written_files):,}")

    # Show distribution
    sizes = [count for _, count in written_files]
    logger.info(f"\nBatch size distribution:")
    logger.info(f"  Min: {min(sizes):,}")
    logger.info(f"  Max: {max(sizes):,}")
    logger.info(f"  Average: {sum(sizes)/len(sizes):.1f}")

    # Show examples
    logger.info(f"\nExample batch files:")
    for batch_file, count in written_files[:5]:
        logger.info(f"  {batch_file.name}: {count:,} executions")
    if len(written_files) > 5:
        logger.info(f"  ... ({len(written_files) - 5} more)")


def main():
    parser = argparse.ArgumentParser(
        description='Redistribute remaining work across all SLURM jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 redistribute_remaining_work.py \\
      --batch-dir PIE_Dataset/batches \\
      --cache-dir pie_energy_cache \\
      --output-dir PIE_Dataset/batches_remaining \\
      --num-jobs 440

This will:
1. Scan all 440 original batch files
2. Check cache to identify completed executions
3. Create 440 new batch files with only remaining work
4. Distribute remaining work evenly for maximum parallelization
        """
    )

    parser.add_argument('--batch-dir', type=Path, required=True,
                        help='Directory containing original batch files')
    parser.add_argument('--cache-dir', type=Path, required=True,
                        help='Path to shared cache directory')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory to write redistributed batch files')
    parser.add_argument('--num-jobs', type=int, default=440,
                        help='Number of batch files to create (default: 440)')
    parser.add_argument('--max-batches', type=int,
                        help='Optional limit on batches to scan (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show statistics without writing batch files')

    args = parser.parse_args()

    # Validate inputs
    if not args.batch_dir.exists():
        logger.error(f"Batch directory not found: {args.batch_dir}")
        return 1

    if not args.cache_dir.exists():
        logger.error(f"Cache directory not found: {args.cache_dir}")
        return 1

    logger.info("="*70)
    logger.info("REDISTRIBUTING REMAINING WORK")
    logger.info("="*70)
    logger.info(f"Original batches: {args.batch_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target jobs: {args.num_jobs}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*70)

    # Step 1: Load completed IDs from cache
    logger.info("\n[1/3] Loading completed execution IDs from cache...")
    completed_ids = load_completed_execution_ids(args.cache_dir)

    # Step 2: Collect remaining executions
    logger.info("\n[2/3] Scanning batch files for remaining work...")
    remaining_executions = collect_remaining_executions(
        args.batch_dir,
        completed_ids,
        args.max_batches
    )

    if not remaining_executions:
        logger.warning("\n⚠️  No remaining executions found - all work is completed!")
        return 0

    # Step 3: Write redistributed batches
    if not args.dry_run:
        logger.info("\n[3/3] Writing redistributed batch files...")
        write_redistributed_batches(
            remaining_executions,
            args.output_dir,
            args.num_jobs
        )

        logger.info(f"\n✓ SUCCESS!")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Update config.env:")
        logger.info(f"   BATCH_DIR=\"{args.output_dir}\"")
        logger.info(f"2. Submit SLURM jobs:")
        logger.info(f"   sbatch energy_data_collection/slurm_execution_master.sh")
    else:
        logger.info("\n[3/3] Dry run - no files written")
        logger.info(f"\nWould create {args.num_jobs} batch files with {len(remaining_executions):,} executions")

    return 0


if __name__ == '__main__':
    exit(main())
