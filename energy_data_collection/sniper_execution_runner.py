#!/usr/bin/env python3
"""
Sniper Execution Runner - Processes execution_master.jsonl batches

This is a simplified runner that:
- Processes ONE execution at a time (no sample-level logic)
- Uses SharedExecutionCache for resumability
- Saves results per-execution (not per-sample)
- Much simpler than the old sample-based runner

Usage:
    python3 sniper_execution_runner.py \
        --batch-file PIE_Dataset/batches/execution_master_batch_000.jsonl \
        --cache-dir /scratch/shared/pie_energy_cache \
        --output-dir execution_results
"""

import os
import sys
import json
import argparse
import logging
import time
import tempfile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import shared cache
from shared_cache import SharedExecutionCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single execution."""
    execution_id: str
    success: bool
    energy_joules: float = 0.0
    power_watts: float = 0.0
    runtime_seconds: float = 0.0
    cycles: int = 0
    instructions: int = 0
    error_message: str = ""
    timestamp: str = ""


class SniperExecutionRunner:
    """Runs Sniper simulations for individual executions."""

    def __init__(self, sniper_path: str, sniper_config: str, cache_dir: str, output_dir: Optional[str] = None):
        """
        Initialize runner.

        Args:
            sniper_path: Path to Sniper installation
            sniper_config: Path to Sniper configuration file
            cache_dir: Path to shared cache directory
            output_dir: Optional directory to save detailed results
        """
        self.sniper_path = Path(sniper_path)
        self.sniper_config = Path(sniper_config)
        self.cache = SharedExecutionCache(cache_dir)
        self.output_dir = Path(output_dir) if output_dir else None

        # Validate paths
        if not self.sniper_path.exists():
            raise FileNotFoundError(f"Sniper path not found: {sniper_path}")
        if not self.sniper_config.exists():
            raise FileNotFoundError(f"Sniper config not found: {sniper_config}")

        # Create output dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SniperExecutionRunner initialized")
        logger.info(f"  Sniper: {self.sniper_path}")
        logger.info(f"  Config: {self.sniper_config}")
        logger.info(f"  Cache: {cache_dir}")

    def compile_code(self, code: str, output_binary: Path) -> bool:
        """
        Compile C++ code to binary.

        Args:
            code: C++ source code
            output_binary: Path where binary should be saved

        Returns:
            True if compilation succeeded, False otherwise
        """
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                temp_cpp = Path(f.name)
                f.write(code)

            # Compile with g++
            compile_cmd = [
                'g++',
                '-std=c++17',
                '-O3',
                '-static',
                str(temp_cpp),
                '-o', str(output_binary)
            ]

            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Clean up temp file
            temp_cpp.unlink()

            if result.returncode != 0:
                logger.error(f"Compilation failed: {result.stderr}")
                return False

            return True

        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return False

    def run_sniper(self, binary: Path, test_input: Path, output_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Run Sniper simulation.

        Args:
            binary: Path to compiled binary
            test_input: Path to test input file
            output_dir: Directory for Sniper output

        Returns:
            Dict with energy/performance metrics, or None if failed
        """
        try:
            # Prepare Sniper command with power modeling
            sniper_cmd = [
                str(self.sniper_path / 'run-sniper'),
                '-c', str(self.sniper_config),
                '-d', str(output_dir),
                '--power',  # Enable built-in McPAT energy analysis
                '--', str(binary)
            ]

            # Run Sniper with test input
            with open(test_input, 'r') as input_file:
                result = subprocess.run(
                    sniper_cmd,
                    stdin=input_file,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

            if result.returncode != 0:
                logger.error(f"Sniper failed with code {result.returncode}")
                logger.error(f"stderr: {result.stderr[:500]}")
                return None

            # Parse results
            sim_out = output_dir / 'sim.out'
            if not sim_out.exists():
                logger.error(f"sim.out not found in {output_dir}")
                return None

            return self._parse_sniper_output(sim_out, result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Sniper simulation timed out after 300 seconds (5 minutes)")
            return None
        except Exception as e:
            logger.error(f"Sniper execution error: {e}")
            return None

    def _parse_sniper_output(self, sim_out: Path, stdout: str) -> Optional[Dict[str, Any]]:
        """
        Parse Sniper output for energy and performance metrics.

        Args:
            sim_out: Path to sim.out file
            stdout: Stdout from Sniper

        Returns:
            Dict with parsed metrics
        """
        try:
            result = {
                'energy_joules': 0.0,
                'power_watts': 0.0,
                'runtime_seconds': 0.0,
                'cycles': 0,
                'instructions': 0
            }

            # Read sim.out
            with open(sim_out) as f:
                lines = f.readlines()

            # Parse table format
            for line in lines:
                line = line.strip()

                # Instructions: "  Instructions                       |    3818368 |          0 | ..."
                if line.startswith("Instructions") and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) > 1:
                        total_instructions = 0
                        for part in parts[1:]:
                            try:
                                total_instructions += int(part)
                            except ValueError:
                                pass
                        result['instructions'] = total_instructions

                # Cycles: "  Cycles                             |    2621700 |    2621700 | ..."
                elif line.startswith("Cycles") and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) > 1:
                        try:
                            result['cycles'] = int(parts[1])
                        except ValueError:
                            pass

            # Parse energy from stdout
            stdout_lines = stdout.split('\n')
            parsing_energy = False

            for line in stdout_lines:
                line = line.strip()

                # Look for energy table header
                if "Power" in line and "Energy" in line and "Energy %" in line:
                    parsing_energy = True
                    continue

                # Parse total energy line
                if parsing_energy and "total" in line.lower():
                    parts = line.split()
                    try:
                        if len(parts) >= 4:
                            result['power_watts'] = float(parts[1])
                            result['energy_joules'] = float(parts[3])
                    except (ValueError, IndexError):
                        pass
                    break

            # Calculate runtime from cycles (assuming 1.5 GHz clock)
            if result['cycles'] > 0:
                clock_freq_hz = 1.5e9  # 1.5 GHz
                result['runtime_seconds'] = result['cycles'] / clock_freq_hz

            # Validation
            if result['runtime_seconds'] > 0 and result['cycles'] == 0:
                logger.warning("⚠️  Suspicious: runtime > 0 but cycles = 0")

            if result['runtime_seconds'] > 0 and result['instructions'] == 0:
                logger.warning("⚠️  Suspicious: runtime > 0 but instructions = 0")

            return result

        except Exception as e:
            logger.error(f"Error parsing Sniper output: {e}")
            return None

    def run_execution(self, execution: Dict[str, Any]) -> ExecutionResult:
        """
        Run a single execution.

        Args:
            execution: Execution record from execution_master.jsonl

        Returns:
            ExecutionResult with metrics
        """
        execution_id = execution['execution_id']
        start_time = time.time()

        try:
            # Create temp directory for this execution
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Compile code
                binary = temp_path / 'program'
                logger.info(f"  Compiling...")
                if not self.compile_code(execution['code'], binary):
                    return ExecutionResult(
                        execution_id=execution_id,
                        success=False,
                        error_message="Compilation failed",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                    )

                # Run Sniper
                sniper_output_dir = temp_path / 'sniper_output'
                sniper_output_dir.mkdir()

                test_input = Path(execution['test_input_file'])
                if not test_input.exists():
                    return ExecutionResult(
                        execution_id=execution_id,
                        success=False,
                        error_message=f"Test input not found: {test_input}",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                    )

                logger.info(f"  Running Sniper...")
                metrics = self.run_sniper(binary, test_input, sniper_output_dir)

                if metrics is None:
                    return ExecutionResult(
                        execution_id=execution_id,
                        success=False,
                        error_message="Sniper execution failed",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                    )

                # Create result
                elapsed = time.time() - start_time
                logger.info(f"  ✅ Success in {elapsed:.1f}s: {metrics['energy_joules']:.6f}J, {metrics['cycles']:,} cycles")

                return ExecutionResult(
                    execution_id=execution_id,
                    success=True,
                    energy_joules=metrics['energy_joules'],
                    power_watts=metrics['power_watts'],
                    runtime_seconds=metrics['runtime_seconds'],
                    cycles=metrics['cycles'],
                    instructions=metrics['instructions'],
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
                )

        except Exception as e:
            logger.error(f"  ❌ Execution error: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                error_message=str(e),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
            )

    def process_batch(self, batch_file: Path):
        """
        Process a batch file of executions.

        Args:
            batch_file: Path to execution_master_batch_*.jsonl
        """
        logger.info(f"=" * 70)
        logger.info(f"Processing batch: {batch_file.name}")
        logger.info(f"=" * 70)

        # Load batch
        executions = []
        with open(batch_file) as f:
            for line in f:
                if line.strip():
                    executions.append(json.loads(line))

        total = len(executions)
        logger.info(f"Loaded {total:,} executions")

        # Check cache stats
        cache_stats = self.cache.get_stats()
        logger.info(f"Cache: {cache_stats['completed']:,} completed, {cache_stats['failed']:,} failed")

        # Process each execution
        completed = 0
        skipped = 0
        failed = 0

        start_time = time.time()

        for i, execution in enumerate(executions, 1):
            execution_id = execution['execution_id']

            # Progress
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%) - {rate:.1f} exec/s - ETA: {remaining/3600:.1f}h")

            # Check cache
            if self.cache.is_completed(execution_id):
                skipped += 1
                if i <= 5:  # Log first few skips
                    logger.info(f"[{i}/{total}] ✓ Skipping {execution_id} (already completed)")
                continue

            # Run execution
            logger.info(f"[{i}/{total}] ▶ Running {execution_id}")
            result = self.run_execution(execution)

            if result.success:
                # Save to cache
                self.cache.mark_completed(execution_id, {
                    'energy_joules': result.energy_joules,
                    'power_watts': result.power_watts,
                    'runtime_seconds': result.runtime_seconds,
                    'cycles': result.cycles,
                    'instructions': result.instructions
                })
                completed += 1
            else:
                # Mark as failed
                self.cache.mark_failed(execution_id, {
                    'error': result.error_message
                })
                failed += 1
                logger.error(f"  Failed: {result.error_message}")

        # Summary
        total_time = time.time() - start_time
        logger.info(f"")
        logger.info(f"=" * 70)
        logger.info(f"BATCH COMPLETE")
        logger.info(f"=" * 70)
        logger.info(f"Total executions: {total:,}")
        logger.info(f"Completed: {completed:,}")
        logger.info(f"Skipped (cached): {skipped:,}")
        logger.info(f"Failed: {failed:,}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time: {total_time/completed:.1f}s per execution" if completed > 0 else "")
        logger.info(f"=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Run Sniper simulations for execution master batches')
    parser.add_argument('--batch-file', required=True, help='Path to execution_master_batch_*.jsonl')
    parser.add_argument('--cache-dir', required=True, help='Path to shared cache directory')
    parser.add_argument('--sniper-path', default=str(REPO_ROOT / 'sniper' / 'sniper'), help='Path to Sniper installation')
    parser.add_argument('--sniper-config', default=str(REPO_ROOT / 'sniper' / 'sniper' / 'config' / 'epyc_9554p.cfg'), help='Path to Sniper config')
    parser.add_argument('--output-dir', help='Optional directory for detailed output')

    args = parser.parse_args()

    # Validate inputs
    batch_file = Path(args.batch_file)
    if not batch_file.exists():
        logger.error(f"Batch file not found: {batch_file}")
        return 1

    # Create runner
    runner = SniperExecutionRunner(
        sniper_path=args.sniper_path,
        sniper_config=args.sniper_config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )

    # Process batch
    runner.process_batch(batch_file)

    return 0


if __name__ == "__main__":
    exit(main())
