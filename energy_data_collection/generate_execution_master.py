#!/usr/bin/env python3
"""
Generate Execution Master File

This script creates a deduplicated master execution file from train.jsonl.
Each line represents one unique (problem_id, code_id, test_input) simulation.

Key Features:
- Eliminates 74% of redundant simulations through deduplication
- Preserves traceability via source_sample_indices
- Enables efficient batch splitting for SLURM jobs
- Supports incremental generation and resumability

Usage:
    python3 generate_execution_master.py --input PIE_Dataset/train.jsonl \
                                          --output PIE_Dataset/execution_master.jsonl \
                                          --max-test-inputs 10

Author: Energy Analysis Team
Date: 2025-10-20
"""

import json
import sys
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Represents a single unique execution in the master file."""
    execution_id: str
    problem_id: str
    code_hash: str
    code: str
    test_input_file: str
    test_input_hash: str
    test_input_size: int
    associated_code_ids: List[str]  # All code_ids with this hash (can be src/tgt in different samples)
    code_id_to_samples: Dict[str, List[int]]  # Map each code_id to its sample indices
    first_seen_index: int
    total_references: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def hash_string(text: str) -> str:
    """Create MD5 hash of string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def hash_file(file_path: Path) -> str:
    """Create MD5 hash of file contents."""
    if not file_path.exists():
        return "file_not_found"

    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return "hash_error"


def select_test_inputs(problem_id: str,
                       pie_dataset_path: Path,
                       max_inputs: Optional[int] = None) -> List[Path]:
    """
    Select test inputs for a problem.

    Args:
        problem_id: Problem identifier
        pie_dataset_path: Path to PIE dataset root
        max_inputs: Maximum number of inputs to select (None = all inputs)

    Returns:
        List of Path objects to test input files
    """
    # Find test input directory - handle both absolute and relative paths
    if pie_dataset_path.name == "train.jsonl":
        pie_root = pie_dataset_path.parent
    else:
        pie_root = pie_dataset_path

    test_case_dir = pie_root / "extracted_testcases" / "merged_test_cases" / problem_id

    if not test_case_dir.exists():
        logger.warning(f"Test case directory not found: {test_case_dir}")
        return []

    # Get all input files
    input_files = sorted(test_case_dir.glob("input.*.txt"))

    if not input_files:
        logger.warning(f"No input files found for problem {problem_id}")
        return []

    # If no limit, return all inputs
    if max_inputs is None or len(input_files) <= max_inputs:
        return input_files

    # If max_inputs specified, use stratified sampling
    # Sort by file size for representative distribution
    input_files_with_size = [(f, f.stat().st_size) for f in input_files]
    input_files_with_size.sort(key=lambda x: x[1])

    selected = []

    # Small inputs (first 3)
    small_count = min(3, max_inputs // 3)
    selected.extend([f for f, _ in input_files_with_size[:small_count]])

    # Large inputs (last 3)
    large_count = min(3, max_inputs // 3)
    selected.extend([f for f, _ in input_files_with_size[-large_count:]])

    # Middle inputs (evenly distributed)
    remaining_count = max_inputs - len(selected)
    if remaining_count > 0:
        middle_files = input_files_with_size[small_count:-large_count]
        if middle_files:
            step = max(1, len(middle_files) // remaining_count)
            for i in range(0, len(middle_files), step):
                if len(selected) >= max_inputs:
                    break
                selected.append(middle_files[i][0])

    return selected[:max_inputs]


def generate_execution_master(input_file: Path,
                               output_file: Path,
                               max_test_inputs: Optional[int] = None,
                               max_samples: Optional[int] = None,
                               dry_run: bool = False,
                               resume: bool = False) -> Dict:
    """
    Generate deduplicated execution master file from train.jsonl.

    Args:
        input_file: Path to train.jsonl
        output_file: Path to output execution_master.jsonl
        max_test_inputs: Maximum test inputs per problem (None = all inputs)
        max_samples: Optional limit for testing (None = process all)
        dry_run: If True, only show statistics without writing output
        resume: If True, resume from existing output file

    Returns:
        Dictionary with generation statistics
    """
    logger.info("="*80)
    logger.info("EXECUTION MASTER FILE GENERATION")
    logger.info("="*80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Max test inputs: {'ALL (no limit)' if max_test_inputs is None else max_test_inputs}")
    logger.info(f"Dry run: {dry_run}")

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    # Track execution records
    execution_map: Dict[str, ExecutionRecord] = {}

    # Track existing executions if resuming
    existing_execution_ids: Set[str] = set()
    if resume and output_file.exists():
        logger.info(f"Resuming from existing output: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    existing_execution_ids.add(record['execution_id'])
                except json.JSONDecodeError:
                    pass
        logger.info(f"Found {len(existing_execution_ids)} existing executions")

    # Statistics
    stats = {
        'total_samples': 0,
        'total_codes': 0,
        'unique_problems': set(),
        'unique_code_hashes': set(),
        'unique_execution_ids': 0,
        'total_simulations_without_dedup': 0,
        'total_simulations_with_dedup': 0,
        'deduplication_rate': 0.0,
        'problems_without_test_inputs': set(),
        'malformed_samples': 0,
        'processing_time_seconds': 0
    }

    # Problem test input cache
    problem_test_inputs: Dict[str, List[Path]] = {}

    start_time = time.time()

    logger.info("\nScanning dataset...")

    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            # Progress indicator
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                logger.info(f"  Processed {idx + 1:,} samples... ({rate:.1f} samples/sec)")

            # Optional limit for testing
            if max_samples and idx >= max_samples:
                break

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {idx}: {e}")
                stats['malformed_samples'] += 1
                continue

            # Extract sample fields
            problem_id = sample.get('problem_id')
            src_id = sample.get('src_id')
            tgt_id = sample.get('tgt_id')
            src_code = sample.get('src_code', '')
            tgt_code = sample.get('tgt_code', '')

            if not problem_id or not src_id or not src_code:
                logger.warning(f"Sample {idx} missing required fields")
                stats['malformed_samples'] += 1
                continue

            stats['total_samples'] += 1
            stats['unique_problems'].add(problem_id)

            # Get test inputs for this problem (cached)
            if problem_id not in problem_test_inputs:
                test_inputs = select_test_inputs(
                    problem_id, input_file, max_test_inputs
                )
                problem_test_inputs[problem_id] = test_inputs

                if not test_inputs:
                    stats['problems_without_test_inputs'].add(problem_id)
            else:
                test_inputs = problem_test_inputs[problem_id]

            # Process source code
            src_hash = hash_string(src_code)
            stats['unique_code_hashes'].add(src_hash)
            stats['total_codes'] += 1

            for test_input_file in test_inputs:
                test_input_hash = hash_file(test_input_file)
                test_input_size = test_input_file.stat().st_size if test_input_file.exists() else 0

                # Create execution ID using code_hash (NOT code_id) to deduplicate across IDs
                execution_id = f"{problem_id}_{src_hash[:12]}_{test_input_hash[:8]}"

                stats['total_simulations_without_dedup'] += 1

                # Skip if already processed (resume mode)
                if execution_id in existing_execution_ids:
                    continue

                if execution_id not in execution_map:
                    execution_map[execution_id] = ExecutionRecord(
                        execution_id=execution_id,
                        problem_id=problem_id,
                        code_hash=src_hash,
                        code=src_code,
                        test_input_file=str(test_input_file),
                        test_input_hash=test_input_hash,
                        test_input_size=test_input_size,
                        associated_code_ids=[],
                        code_id_to_samples={},
                        first_seen_index=idx,
                        total_references=0
                    )

                # Track this code_id for this execution
                if src_id not in execution_map[execution_id].associated_code_ids:
                    execution_map[execution_id].associated_code_ids.append(src_id)

                if src_id not in execution_map[execution_id].code_id_to_samples:
                    execution_map[execution_id].code_id_to_samples[src_id] = []

                execution_map[execution_id].code_id_to_samples[src_id].append(idx)
                execution_map[execution_id].total_references += 1

            # Process target code (if exists)
            if tgt_id and tgt_code:
                tgt_hash = hash_string(tgt_code)
                stats['unique_code_hashes'].add(tgt_hash)
                stats['total_codes'] += 1

                for test_input_file in test_inputs:
                    test_input_hash = hash_file(test_input_file)
                    test_input_size = test_input_file.stat().st_size if test_input_file.exists() else 0

                    # Create execution ID using code_hash (NOT code_id) to deduplicate across IDs
                    execution_id = f"{problem_id}_{tgt_hash[:12]}_{test_input_hash[:8]}"

                    stats['total_simulations_without_dedup'] += 1

                    # Skip if already processed (resume mode)
                    if execution_id in existing_execution_ids:
                        continue

                    if execution_id not in execution_map:
                        execution_map[execution_id] = ExecutionRecord(
                            execution_id=execution_id,
                            problem_id=problem_id,
                            code_hash=tgt_hash,
                            code=tgt_code,
                            test_input_file=str(test_input_file),
                            test_input_hash=test_input_hash,
                            test_input_size=test_input_size,
                            associated_code_ids=[],
                            code_id_to_samples={},
                            first_seen_index=idx,
                            total_references=0
                        )

                    # Track this code_id for this execution
                    if tgt_id not in execution_map[execution_id].associated_code_ids:
                        execution_map[execution_id].associated_code_ids.append(tgt_id)

                    if tgt_id not in execution_map[execution_id].code_id_to_samples:
                        execution_map[execution_id].code_id_to_samples[tgt_id] = []

                    execution_map[execution_id].code_id_to_samples[tgt_id].append(idx)
                    execution_map[execution_id].total_references += 1

    # Calculate final statistics
    stats['unique_execution_ids'] = len(execution_map)
    stats['total_simulations_with_dedup'] = len(execution_map)
    stats['simulations_avoided'] = stats['total_simulations_without_dedup'] - stats['total_simulations_with_dedup']
    stats['deduplication_rate'] = (stats['simulations_avoided'] / stats['total_simulations_without_dedup'] * 100) if stats['total_simulations_without_dedup'] > 0 else 0
    stats['processing_time_seconds'] = time.time() - start_time

    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("GENERATION STATISTICS")
    logger.info("="*80)
    logger.info(f"\nInput Dataset:")
    logger.info(f"  Total samples: {stats['total_samples']:,}")
    logger.info(f"  Unique problems: {len(stats['unique_problems']):,}")
    logger.info(f"  Total codes (src + tgt): {stats['total_codes']:,}")
    logger.info(f"  Unique code hashes: {len(stats['unique_code_hashes']):,}")

    logger.info(f"\nSimulation Requirements:")
    logger.info(f"  Without deduplication: {stats['total_simulations_without_dedup']:,}")
    logger.info(f"  With deduplication: {stats['total_simulations_with_dedup']:,}")
    logger.info(f"  Simulations avoided: {stats['simulations_avoided']:,}")
    logger.info(f"  Deduplication rate: {stats['deduplication_rate']:.2f}%")

    logger.info(f"\nOutput:")
    logger.info(f"  Unique executions: {stats['unique_execution_ids']:,}")
    if stats['unique_execution_ids'] > 0:
        logger.info(f"  Average references per execution: {stats['total_simulations_without_dedup'] / stats['unique_execution_ids']:.1f}")
    else:
        logger.warning(f"  No executions generated (likely missing test input files)")

    if stats['problems_without_test_inputs']:
        logger.warning(f"\nProblems without test inputs: {len(stats['problems_without_test_inputs'])}")
        logger.warning(f"  Examples: {list(stats['problems_without_test_inputs'])[:5]}")

    if stats['malformed_samples'] > 0:
        logger.warning(f"\nMalformed samples: {stats['malformed_samples']}")

    logger.info(f"\nProcessing time: {stats['processing_time_seconds']:.1f} seconds")

    # Write output file (unless dry run)
    if not dry_run:
        logger.info(f"\nWriting execution master file: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        write_mode = 'a' if resume else 'w'
        with open(output_file, write_mode) as out:
            for execution_id in sorted(execution_map.keys()):
                exec_record = execution_map[execution_id]
                out.write(json.dumps(exec_record.to_dict()) + '\n')

        logger.info(f"✓ Execution master file written: {output_file}")
        logger.info(f"  Lines written: {len(execution_map):,}")
        logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        logger.info("\nDry run mode - no output file written")

    # Convert sets to lists for JSON serialization
    stats['unique_problems'] = list(stats['unique_problems'])
    stats['unique_code_hashes'] = list(stats['unique_code_hashes'])
    stats['problems_without_test_inputs'] = list(stats['problems_without_test_inputs'])

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate deduplicated execution master file from train.jsonl',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate execution master from full dataset
  python3 generate_execution_master.py --input PIE_Dataset/train.jsonl \\
                                        --output PIE_Dataset/execution_master.jsonl

  # Dry run to see statistics without writing
  python3 generate_execution_master.py --input PIE_Dataset/train.jsonl --dry-run

  # Test with first 1000 samples
  python3 generate_execution_master.py --input PIE_Dataset/train.jsonl \\
                                        --output test_master.jsonl \\
                                        --max-samples 1000

  # Resume interrupted generation
  python3 generate_execution_master.py --input PIE_Dataset/train.jsonl \\
                                        --output PIE_Dataset/execution_master.jsonl \\
                                        --resume
        """
    )

    parser.add_argument('--input', type=Path, required=True,
                        help='Path to input train.jsonl file')
    parser.add_argument('--output', type=Path,
                        help='Path to output execution_master.jsonl file')
    parser.add_argument('--max-test-inputs', type=int, default=None,
                        help='Maximum test inputs per problem (default: None = all inputs)')
    parser.add_argument('--max-samples', type=int,
                        help='Limit number of samples to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show statistics without writing output file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file (append new executions)')
    parser.add_argument('--stats-output', type=Path,
                        help='Optional path to save statistics as JSON')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.dry_run and not args.output:
        logger.error("Error: --output is required unless --dry-run is specified")
        sys.exit(1)

    # Generate execution master
    stats = generate_execution_master(
        input_file=args.input,
        output_file=args.output if args.output else Path('/dev/null'),
        max_test_inputs=args.max_test_inputs,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        resume=args.resume
    )

    # Save statistics if requested
    if args.stats_output:
        with open(args.stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\n✓ Statistics saved: {args.stats_output}")

    logger.info("\n" + "="*80)
    logger.info("GENERATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
