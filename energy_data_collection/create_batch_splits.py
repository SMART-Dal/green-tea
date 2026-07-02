#!/usr/bin/env python3
"""
Create balanced batch splits from execution_master.jsonl for SLURM job arrays.

Features:
- Conservative splitting (50s/simulation estimate)
- Problem-wise grouping (for cache locality)
- Greedy bin packing (balanced load)
- Generates batch files for SLURM job arrays

Usage:
    python3 create_batch_splits.py \\
        --input PIE_Dataset/execution_master.jsonl \\
        --output-dir PIE_Dataset/batches/ \\
        --num-jobs 300 \\
        --time-per-sim 50
"""

import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Problem:
    """Represents a problem with all its executions."""
    problem_id: str
    executions: List[Dict[str, Any]]
    total_count: int
    estimated_time: float  # seconds


class BatchSplitter:
    """Create balanced batches using greedy bin packing."""

    def __init__(self, num_batches: int, time_per_execution: float, max_time_seconds: float = 7 * 24 * 3600):
        """
        Initialize batch splitter.

        Args:
            num_batches: Number of batches to create
            time_per_execution: Estimated time per execution (seconds)
            max_time_seconds: Maximum time per batch (default: 7 days)
        """
        self.num_batches = num_batches
        self.time_per_execution = time_per_execution
        self.max_time_seconds = max_time_seconds
        self.batches: List[List[Dict[str, Any]]] = [[] for _ in range(num_batches)]
        self.batch_times: List[float] = [0.0] * num_batches
        self.batch_counts: List[int] = [0] * num_batches

    def add_problem(self, problem: Problem):
        """
        Add problem to batches (may split across multiple if too large).

        Args:
            problem: Problem to add
        """
        # Calculate max executions per batch
        max_executions_per_batch = int(self.max_time_seconds / self.time_per_execution)

        # If problem fits in one batch, use greedy assignment
        if problem.total_count <= max_executions_per_batch:
            # Find batch with minimum time
            min_batch_idx = min(range(self.num_batches), key=lambda i: self.batch_times[i])

            # Add all executions from this problem to the batch
            self.batches[min_batch_idx].extend(problem.executions)
            self.batch_times[min_batch_idx] += problem.estimated_time
            self.batch_counts[min_batch_idx] += problem.total_count

            logger.debug(
                f"Added {problem.problem_id} ({problem.total_count} executions, "
                f"{problem.estimated_time:.1f}s) to batch {min_batch_idx}"
            )
        else:
            # Problem is too large - split across multiple batches
            logger.warning(
                f"Problem {problem.problem_id} has {problem.total_count} executions "
                f"(max {max_executions_per_batch} per batch). Splitting across multiple batches."
            )

            # Split executions into chunks
            executions_list = problem.executions
            chunk_size = max_executions_per_batch
            num_chunks = (len(executions_list) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(executions_list))
                chunk_executions = executions_list[start_idx:end_idx]
                chunk_count = len(chunk_executions)
                chunk_time = chunk_count * self.time_per_execution

                # Find batch with minimum time
                min_batch_idx = min(range(self.num_batches), key=lambda i: self.batch_times[i])

                # Add this chunk
                self.batches[min_batch_idx].extend(chunk_executions)
                self.batch_times[min_batch_idx] += chunk_time
                self.batch_counts[min_batch_idx] += chunk_count

                logger.info(
                    f"Added {problem.problem_id} chunk {chunk_idx + 1}/{num_chunks} "
                    f"({chunk_count} executions) to batch {min_batch_idx}"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get batch statistics."""
        return {
            'num_batches': self.num_batches,
            'batch_counts': self.batch_counts,
            'batch_times': self.batch_times,
            'min_count': min(self.batch_counts),
            'max_count': max(self.batch_counts),
            'avg_count': sum(self.batch_counts) / self.num_batches,
            'min_time_hours': min(self.batch_times) / 3600,
            'max_time_hours': max(self.batch_times) / 3600,
            'avg_time_hours': sum(self.batch_times) / self.num_batches / 3600,
        }


def load_and_group_by_problem(execution_master_file: Path, time_per_execution: float) -> List[Problem]:
    """
    Load execution master and group by problem.

    Args:
        execution_master_file: Path to execution_master.jsonl
        time_per_execution: Estimated time per execution (seconds)

    Returns:
        List of Problem objects
    """
    logger.info(f"Loading execution master from {execution_master_file}")

    problem_map = defaultdict(list)

    with open(execution_master_file) as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                logger.info(f"Loaded {line_num:,} executions...")

            execution = json.loads(line)
            problem_id = execution['problem_id']
            problem_map[problem_id].append(execution)

    logger.info(f"Loaded {line_num:,} total executions")
    logger.info(f"Found {len(problem_map)} unique problems")

    # Convert to Problem objects
    problems = []
    for problem_id, executions in problem_map.items():
        count = len(executions)
        estimated_time = count * time_per_execution
        problems.append(Problem(
            problem_id=problem_id,
            executions=executions,
            total_count=count,
            estimated_time=estimated_time
        ))

    # Sort by count (descending) for better bin packing
    problems.sort(key=lambda p: p.total_count, reverse=True)

    logger.info(f"Problem statistics:")
    logger.info(f"  Largest problem: {problems[0].problem_id} with {problems[0].total_count:,} executions")
    logger.info(f"  Smallest problem: {problems[-1].problem_id} with {problems[-1].total_count:,} executions")
    logger.info(f"  Average per problem: {sum(p.total_count for p in problems) / len(problems):.1f} executions")

    return problems


def create_batches(problems: List[Problem], num_batches: int, time_per_execution: float) -> BatchSplitter:
    """
    Create balanced batches using greedy bin packing.

    Args:
        problems: List of problems
        num_batches: Number of batches to create
        time_per_execution: Estimated time per execution (seconds)

    Returns:
        BatchSplitter with all problems assigned
    """
    logger.info(f"Creating {num_batches} balanced batches...")

    splitter = BatchSplitter(num_batches, time_per_execution)

    # Add problems using greedy bin packing
    for i, problem in enumerate(problems):
        if (i + 1) % 100 == 0:
            logger.info(f"Assigned {i + 1}/{len(problems)} problems...")
        splitter.add_problem(problem)

    logger.info("Batch creation complete!")

    return splitter


def write_batch_files(splitter: BatchSplitter, output_dir: Path):
    """
    Write batch files to disk.

    Args:
        splitter: BatchSplitter with assigned batches
        output_dir: Output directory for batch files
    """
    logger.info(f"Writing batch files to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, executions in enumerate(splitter.batches):
        output_file = output_dir / f"execution_master_batch_{batch_idx:03d}.jsonl"

        with open(output_file, 'w') as f:
            for execution in executions:
                f.write(json.dumps(execution) + '\n')

        logger.info(
            f"Wrote batch {batch_idx:03d}: {len(executions):,} executions "
            f"(~{splitter.batch_times[batch_idx] / 3600:.1f} hours)"
        )

    logger.info(f"All {splitter.num_batches} batch files written")


def write_batch_stats(splitter: BatchSplitter, output_file: Path):
    """
    Write batch statistics to JSON file.

    Args:
        splitter: BatchSplitter with assigned batches
        output_file: Output statistics file
    """
    stats = splitter.get_stats()

    # Add per-batch details
    stats['batches'] = []
    for i in range(splitter.num_batches):
        stats['batches'].append({
            'batch_id': i,
            'execution_count': splitter.batch_counts[i],
            'estimated_time_hours': splitter.batch_times[i] / 3600,
            'estimated_time_days': splitter.batch_times[i] / 3600 / 24,
        })

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Batch statistics written to {output_file}")

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH STATISTICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Number of batches: {stats['num_batches']}")
    logger.info(f"Total executions: {sum(stats['batch_counts']):,}")
    logger.info(f"\nExecution counts per batch:")
    logger.info(f"  Min: {stats['min_count']:,}")
    logger.info(f"  Max: {stats['max_count']:,}")
    logger.info(f"  Avg: {stats['avg_count']:,.0f}")
    logger.info(f"\nEstimated time per batch:")
    logger.info(f"  Min: {stats['min_time_hours']:.1f} hours ({stats['min_time_hours'] / 24:.2f} days)")
    logger.info(f"  Max: {stats['max_time_hours']:.1f} hours ({stats['max_time_hours'] / 24:.2f} days)")
    logger.info(f"  Avg: {stats['avg_time_hours']:.1f} hours ({stats['avg_time_hours'] / 24:.2f} days)")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced batch splits from execution_master.jsonl'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to execution_master.jsonl'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for batch files'
    )
    parser.add_argument(
        '--num-jobs',
        type=int,
        default=300,
        help='Number of batches to create (default: 300)'
    )
    parser.add_argument(
        '--time-per-sim',
        type=float,
        default=50.0,
        help='Estimated time per simulation in seconds (default: 50.0)'
    )

    args = parser.parse_args()

    # Convert to paths
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    # Load and group by problem
    problems = load_and_group_by_problem(input_file, args.time_per_sim)

    # Create balanced batches
    splitter = create_batches(problems, args.num_jobs, args.time_per_sim)

    # Write batch files
    write_batch_files(splitter, output_dir)

    # Write statistics
    stats_file = output_dir / "batch_stats.json"
    write_batch_stats(splitter, stats_file)

    logger.info("\n✅ Batch splitting complete!")
    logger.info(f"Batch files: {output_dir}/execution_master_batch_*.jsonl")
    logger.info(f"Statistics: {stats_file}")

    return 0


if __name__ == "__main__":
    exit(main())
