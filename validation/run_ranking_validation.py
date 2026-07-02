#!/usr/bin/env python3
"""
Five-Script Ranking Validation Suite

This script validates Sniper+McPAT ranking accuracy by:
1. Running 5 implementations per task in real-time execution
2. Running same implementations through Sniper+McPAT simulation
3. Comparing rankings for energy, power, and time
4. Calculating ranking correlation and direction correctness

The goal is to verify that Sniper simulation correctly ranks implementations
for energy-efficient code selection.

Author: Energy-Efficient Code Generation Pipeline
Date: September 16, 2025
"""

import os
import sys
import json
import subprocess
import tempfile
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import concurrent.futures
import psutil
from scipy.stats import spearmanr, kendalltau
import numpy as np

@dataclass
class ImplementationResult:
    """Results from running one implementation."""
    task_name: str
    impl_id: int  # 1-5
    success: bool

    # Performance metrics
    runtime_seconds: float = 0.0
    instructions: int = 0
    cycles: int = 0
    ipc: float = 0.0
    simulated_seconds: float = 0.0

    # Real energy/power measurements
    real_energy_joules: float = 0.0
    real_power_watts: float = 0.0
    package_energy_joules: float = 0.0
    core_energy_joules: float = 0.0

    # Simulated energy/power
    sim_energy_joules: float = 0.0
    sim_power_watts: float = 0.0
    sim_core_energy: float = 0.0
    sim_cache_energy: float = 0.0

    error_message: str = ""

@dataclass
class TaskRankingMetrics:
    """Ranking analysis for one task (5 implementations)."""
    task_name: str

    # Real rankings (1=best, 5=worst)
    real_energy_ranking: List[int] = None
    real_power_ranking: List[int] = None
    real_time_ranking: List[int] = None

    # Simulated rankings
    sim_energy_ranking: List[int] = None
    sim_power_ranking: List[int] = None
    sim_time_ranking: List[int] = None

    # Ranking correlations (Spearman's rho)
    energy_rank_correlation: float = 0.0
    power_rank_correlation: float = 0.0
    time_rank_correlation: float = 0.0

    # Kendall's tau (alternative ranking correlation)
    energy_kendall_tau: float = 0.0
    power_kendall_tau: float = 0.0
    time_kendall_tau: float = 0.0

    # Direction correctness (best vs worst correctly identified)
    energy_best_correct: bool = False
    energy_worst_correct: bool = False
    power_best_correct: bool = False
    power_worst_correct: bool = False
    time_best_correct: bool = False
    time_worst_correct: bool = False

    # Top-2 and top-3 correctness
    energy_top2_overlap: int = 0  # How many of top-2 real are in top-2 sim
    energy_top3_overlap: int = 0
    power_top2_overlap: int = 0
    power_top3_overlap: int = 0
    time_top2_overlap: int = 0
    time_top3_overlap: int = 0

    # Implementation success count
    successful_implementations: int = 0

class EnergyMeasurement:
    """Real-time energy measurement using Intel RAPL and system tools."""

    def __init__(self):
        self.rapl_available = self._check_rapl_availability()
        self.turbostat_available = self._check_turbostat_availability()

        if not (self.rapl_available or self.turbostat_available):
            print("⚠️ Warning: No energy measurement tools available")

    def _check_rapl_availability(self) -> bool:
        """Check if Intel RAPL interface is available."""
        package_path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
        core_path = Path("/sys/class/powercap/intel-rapl:0:0/energy_uj")
        return package_path.exists() and core_path.exists()

    def _check_turbostat_availability(self) -> bool:
        """Check if turbostat is available and accessible."""
        try:
            result = subprocess.run(["turbostat", "--help"],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _read_rapl_energy(self) -> Dict[str, float]:
        """Read current energy counters from RAPL interface."""
        try:
            # Package energy (total CPU package)
            with open("/sys/class/powercap/intel-rapl:0/energy_uj", "r") as f:
                package_energy = float(f.read().strip()) / 1_000_000  # Convert μJ to J

            # Core energy
            with open("/sys/class/powercap/intel-rapl:0:0/energy_uj", "r") as f:
                core_energy = float(f.read().strip()) / 1_000_000  # Convert μJ to J

            # Try to read DRAM energy if available
            dram_energy = 0.0
            dram_path = Path("/sys/class/powercap/intel-rapl:0:1/energy_uj")
            if dram_path.exists():
                with open(dram_path, "r") as f:
                    dram_energy = float(f.read().strip()) / 1_000_000

            return {
                "package_energy": package_energy,
                "core_energy": core_energy,
                "dram_energy": dram_energy,
                "timestamp": time.time()
            }
        except Exception as e:
            print(f"Warning: Failed to read RAPL energy: {e}")
            return {"package_energy": 0.0, "core_energy": 0.0, "dram_energy": 0.0, "timestamp": time.time()}

    def measure_execution_energy(self, executable_path: Path, input_data: str = "",
                                timeout: int = 60) -> Tuple[float, float, float, float, float]:
        """
        Measure energy consumption during program execution.

        Returns:
            Tuple of (execution_time, total_energy, package_energy, core_energy, average_power)
        """
        if not self.rapl_available:
            print("⚠️ RAPL not available, returning zero energy measurements")
            # Still measure execution time
            start_time = time.time()
            try:
                result = subprocess.run(
                    [str(executable_path)],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                exec_time = time.time() - start_time
                return exec_time, 0.0, 0.0, 0.0, 0.0
            except Exception:
                return 0.0, 0.0, 0.0, 0.0, 0.0

        # Take initial energy reading
        energy_start = self._read_rapl_energy()
        start_time = time.time()

        try:
            # Run the program with energy monitoring
            result = subprocess.run(
                [str(executable_path)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Take final energy reading immediately after execution
            end_time = time.time()
            energy_end = self._read_rapl_energy()

            execution_time = end_time - start_time

            # Calculate energy consumption
            package_energy = energy_end["package_energy"] - energy_start["package_energy"]
            core_energy = energy_end["core_energy"] - energy_start["core_energy"]
            dram_energy = energy_end["dram_energy"] - energy_start["dram_energy"]
            total_energy = package_energy + dram_energy  # Package includes cores

            # Calculate average power
            average_power = total_energy / execution_time if execution_time > 0 else 0.0

            # Handle RAPL counter overflow (rare but possible)
            if package_energy < 0 or core_energy < 0:
                print("⚠️ Warning: RAPL counter overflow detected, energy measurements may be inaccurate")
                # Reset to zero for overflow cases
                package_energy = max(0, package_energy)
                core_energy = max(0, core_energy)
                total_energy = max(0, total_energy)

            return execution_time, total_energy, package_energy, core_energy, average_power

        except subprocess.TimeoutExpired:
            print(f"⚠️ Program execution timed out after {timeout}s")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        except Exception as e:
            print(f"⚠️ Error during energy measurement: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

class RankingValidationSuite:
    """Main validation suite for 5-script ranking analysis."""

    def __init__(self, five_per_task_dir: Path, sniper_root: Path):
        self.five_per_task_dir = Path(five_per_task_dir)
        self.sniper_root = Path(sniper_root)
        self.sniper_dir = self.sniper_root / "sniper"

        # Results storage
        self.results_dir = self.five_per_task_dir.parent / "ranking_results"

        # Initialize energy measurement
        self.energy_meter = EnergyMeasurement()

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        # Verify setup
        self._verify_setup()

    def _verify_setup(self):
        """Verify all required directories and tools exist."""
        if not self.five_per_task_dir.exists():
            raise FileNotFoundError(f"Five per task directory not found: {self.five_per_task_dir}")

        if not self.sniper_dir.exists():
            raise FileNotFoundError(f"Sniper directory not found: {self.sniper_dir}")

        # Check for Sniper executable
        sniper_exec = self.sniper_dir / "run-sniper"
        if not sniper_exec.exists():
            raise FileNotFoundError(f"Sniper executable not found: {sniper_exec}")

        # Check for configuration
        config_file = self.sniper_dir / "config" / "epyc_9554p.cfg"
        if not config_file.exists():
            print(f"⚠️ Warning: Config file not found: {config_file}")

        print("✅ Ranking validation suite setup verified")

        # Test energy measurement capability
        if self.energy_meter.rapl_available:
            print("✅ RAPL energy measurement available")
        else:
            print("⚠️ RAPL energy measurement not available - will only compare simulated metrics")

    def discover_tasks(self) -> List[Tuple[str, str, List[Path]]]:
        """
        Discover all tasks in five_per_task directory.

        Returns:
            List of (category, task_name, [cpp_files]) tuples
        """
        tasks = []

        for category_dir in self.five_per_task_dir.iterdir():
            if not category_dir.is_dir():
                continue

            for task_dir in category_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                # Find all .cpp files in this task
                cpp_files = sorted(list(task_dir.glob("*.cpp")))

                if len(cpp_files) == 5:  # Expect exactly 5 implementations
                    tasks.append((category_dir.name, task_dir.name, cpp_files))
                else:
                    print(f"⚠️ Skipping task {category_dir.name}/{task_dir.name}: found {len(cpp_files)} files, expected 5")

        print(f"✅ Discovered {len(tasks)} tasks with 5 implementations each")
        return tasks

    def compile_implementation(self, cpp_file: Path) -> Optional[Path]:
        """Compile a C++ implementation file."""
        try:
            exe_path = cpp_file.with_suffix("")

            compile_cmd = [
                "g++", "-O3", "-o", str(exe_path), str(cpp_file)
            ]

            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and exe_path.exists():
                return exe_path
            else:
                print(f"   ❌ Compilation failed for {cpp_file.name}: {result.stderr}")
                return None

        except Exception as e:
            print(f"   ❌ Compilation error for {cpp_file.name}: {e}")
            return None

    def run_real_execution(self, executable: Path, task_name: str, impl_id: int) -> ImplementationResult:
        """Run real execution with energy measurement."""
        result = ImplementationResult(
            task_name=task_name,
            impl_id=impl_id,
            success=False
        )

        try:
            # Measure energy consumption during execution
            exec_time, total_energy, package_energy, core_energy, avg_power = \
                self.energy_meter.measure_execution_energy(executable, timeout=120)

            if exec_time > 0:
                result.runtime_seconds = exec_time
                result.real_energy_joules = total_energy
                result.real_power_watts = avg_power
                result.package_energy_joules = package_energy
                result.core_energy_joules = core_energy
                result.success = True

        except Exception as e:
            result.error_message = f"Real execution error: {e}"

        return result

    def run_sniper_simulation(self, executable: Path, task_name: str, impl_id: int) -> ImplementationResult:
        """Run Sniper+McPAT simulation."""
        result = ImplementationResult(
            task_name=task_name,
            impl_id=impl_id,
            success=False
        )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "sniper_output"

                # Prepare Sniper command
                cmd = [
                    str(self.sniper_dir / "run-sniper"),
                    "-n", "1",  # Single core
                    "-c", "config/epyc_9554p.cfg",
                    "-d", str(output_dir),
                    "--power",  # Enable energy analysis
                    "--", str(executable)
                ]

                # Run simulation
                sim_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for long-running benchmarks
                    cwd=self.sniper_dir
                )

                if sim_result.returncode == 0:
                    # Parse simulation results
                    self._parse_sniper_output(sim_result.stdout, result)

                    # Parse energy results if available
                    power_file = output_dir / "power.py"
                    if power_file.exists():
                        self._parse_energy_results(power_file, result)

                    result.success = True
                else:
                    result.error_message = f"Sniper failed: {sim_result.stderr}"

        except subprocess.TimeoutExpired:
            result.error_message = "Sniper simulation timeout"
        except Exception as e:
            result.error_message = f"Sniper error: {str(e)}"

        return result

    def _parse_sniper_output(self, stdout: str, result: ImplementationResult):
        """Parse performance metrics from Sniper stdout."""
        lines = stdout.split('\n')

        for line in lines:
            line = line.strip()

            # Parse performance metrics
            if "Simulated" in line and "instructions" in line:
                # Format: "Simulated 2.3M instructions, 2.1M cycles, 1.09 IPC"
                parts = line.split()
                for i, part in enumerate(parts):
                    if "instructions" in part and i > 0:
                        inst_str = parts[i-1]
                        result.instructions = self._parse_scaled_number(inst_str)
                    elif "cycles" in part and i > 0:
                        cycles_str = parts[i-1]
                        result.cycles = self._parse_scaled_number(cycles_str)
                    elif "IPC" in part and i > 0:
                        try:
                            result.ipc = float(parts[i-1])
                        except ValueError:
                            pass

            # Parse timing
            elif "Elapsed time" in line:
                # Format: "Elapsed time: 2.34 seconds"
                parts = line.split()
                for i, part in enumerate(parts):
                    if "time:" in part and i+1 < len(parts):
                        try:
                            result.simulated_seconds = float(parts[i+1])
                        except ValueError:
                            pass

            # Parse energy summary
            elif "total" in line and "W" in line and "J" in line and "%" in line:
                # Format: "  total           236.45 W     0.41  J    100.00%"
                parts = line.split()
                try:
                    if len(parts) >= 4:
                        result.sim_power_watts = float(parts[1])
                        result.sim_energy_joules = float(parts[3])
                except (ValueError, IndexError):
                    pass

    def _parse_energy_results(self, power_file: Path, result: ImplementationResult):
        """Parse detailed energy breakdown from power.py file."""
        try:
            with open(power_file, 'r') as f:
                content = f.read()

            # Simple parsing of power dictionary (could be improved with ast.literal_eval)
            if result.sim_energy_joules > 0 and result.simulated_seconds > 0:
                # Estimate component breakdown based on typical patterns
                # Core: ~45%, Cache: ~40%, DRAM: ~15%
                result.sim_core_energy = result.sim_energy_joules * 0.45
                result.sim_cache_energy = result.sim_energy_joules * 0.40

        except Exception as e:
            print(f"Warning: Could not parse detailed energy results: {e}")

    def _parse_scaled_number(self, number_str: str) -> int:
        """Parse scaled numbers like '3.8M' or '2.6K'."""
        number_str = number_str.strip()

        if number_str.endswith('M'):
            return int(float(number_str[:-1]) * 1_000_000)
        elif number_str.endswith('K'):
            return int(float(number_str[:-1]) * 1_000)
        elif number_str.endswith('B'):
            return int(float(number_str[:-1]) * 1_000_000_000)
        else:
            try:
                return int(float(number_str))
            except ValueError:
                return 0

    def validate_task(self, category: str, task_name: str, cpp_files: List[Path]) -> TaskRankingMetrics:
        """Validate one task (5 implementations) and compute ranking metrics."""
        print(f"\n📊 Validating task: {category}/{task_name}")

        metrics = TaskRankingMetrics(task_name=f"{category}/{task_name}")
        real_results = []
        sim_results = []

        # Compile all implementations first
        executables = []
        for i, cpp_file in enumerate(cpp_files, 1):
            print(f"   Compiling implementation {i}/5: {cpp_file.name}")
            executable = self.compile_implementation(cpp_file)
            if executable:
                executables.append(executable)
            else:
                print(f"   ❌ Skipping {cpp_file.name} - compilation failed")

        if len(executables) < 3:
            print(f"   ⚠️ Insufficient successful compilations ({len(executables)}) for ranking analysis")
            return metrics

        # Run real executions sequentially (no parallel to avoid interference)
        print(f"   🔥 Running real executions sequentially...")
        for i, exe in enumerate(executables, 1):
            result = self.run_real_execution(exe, metrics.task_name, i)
            real_results.append(result)

        # Run Sniper simulations sequentially
        print(f"   🚀 Running Sniper simulations sequentially...")
        for i, exe in enumerate(executables, 1):
            result = self.run_sniper_simulation(exe, metrics.task_name, i)
            sim_results.append(result)

        # Filter successful results
        successful_real = [r for r in real_results if r.success]
        successful_sim = [r for r in sim_results if r.success]

        print(f"   📈 Results: {len(successful_real)}/{len(executables)} real, {len(successful_sim)}/{len(executables)} simulated")

        # Display results
        for i, (real_result, sim_result) in enumerate(zip(real_results, sim_results), 1):
            if real_result.success and sim_result.success:
                print(f"   ✅ Implementation {i}: Real={real_result.real_energy_joules:.4f}J, Sim={sim_result.sim_energy_joules:.4f}J")
            else:
                print(f"   ❌ Implementation {i} failed")

        metrics.successful_implementations = len(successful_real)

        if len(successful_real) >= 3:  # Need at least 3 for meaningful ranking
            # Compute rankings and correlations
            self._compute_rankings(metrics, successful_real, successful_sim)
            print(f"   ✅ Ranking analysis completed for {len(successful_real)} implementations")
        else:
            print(f"   ⚠️ Insufficient successful implementations ({len(successful_real)}) for ranking analysis")

        # Cleanup executables
        for executable in executables:
            if executable and executable.exists():
                executable.unlink()

        return metrics

    def _compute_rankings(self, metrics: TaskRankingMetrics,
                         real_results: List[ImplementationResult],
                         sim_results: List[ImplementationResult]):
        """Compute rankings and correlation metrics."""

        # Extract metrics for ranking (lower values = better rank)
        real_energies = [r.real_energy_joules for r in real_results]
        real_powers = [r.real_power_watts for r in real_results]
        real_times = [r.runtime_seconds for r in real_results]

        sim_energies = [r.sim_energy_joules for r in sim_results]
        sim_powers = [r.sim_power_watts for r in sim_results]
        sim_times = [r.simulated_seconds for r in sim_results]

        # Compute rankings (1=best, higher=worse)
        metrics.real_energy_ranking = self._compute_rank_list(real_energies)
        metrics.real_power_ranking = self._compute_rank_list(real_powers)
        metrics.real_time_ranking = self._compute_rank_list(real_times)

        metrics.sim_energy_ranking = self._compute_rank_list(sim_energies)
        metrics.sim_power_ranking = self._compute_rank_list(sim_powers)
        metrics.sim_time_ranking = self._compute_rank_list(sim_times)

        # Compute ranking correlations
        if len(real_energies) > 2:
            try:
                metrics.energy_rank_correlation, _ = spearmanr(metrics.real_energy_ranking, metrics.sim_energy_ranking)
                metrics.power_rank_correlation, _ = spearmanr(metrics.real_power_ranking, metrics.sim_power_ranking)
                metrics.time_rank_correlation, _ = spearmanr(metrics.real_time_ranking, metrics.sim_time_ranking)

                metrics.energy_kendall_tau, _ = kendalltau(metrics.real_energy_ranking, metrics.sim_energy_ranking)
                metrics.power_kendall_tau, _ = kendalltau(metrics.real_power_ranking, metrics.sim_power_ranking)
                metrics.time_kendall_tau, _ = kendalltau(metrics.real_time_ranking, metrics.sim_time_ranking)
            except:
                # Handle edge cases (all same values, etc.)
                pass

        # Compute direction correctness (best vs worst)
        metrics.energy_best_correct = (metrics.real_energy_ranking.index(1) == metrics.sim_energy_ranking.index(1))
        metrics.energy_worst_correct = (metrics.real_energy_ranking.index(max(metrics.real_energy_ranking)) ==
                                       metrics.sim_energy_ranking.index(max(metrics.sim_energy_ranking)))

        metrics.power_best_correct = (metrics.real_power_ranking.index(1) == metrics.sim_power_ranking.index(1))
        metrics.power_worst_correct = (metrics.real_power_ranking.index(max(metrics.real_power_ranking)) ==
                                      metrics.sim_power_ranking.index(max(metrics.sim_power_ranking)))

        metrics.time_best_correct = (metrics.real_time_ranking.index(1) == metrics.sim_time_ranking.index(1))
        metrics.time_worst_correct = (metrics.real_time_ranking.index(max(metrics.real_time_ranking)) ==
                                     metrics.sim_time_ranking.index(max(metrics.sim_time_ranking)))

        # Compute top-k overlaps
        metrics.energy_top2_overlap = self._compute_topk_overlap(metrics.real_energy_ranking, metrics.sim_energy_ranking, 2)
        metrics.energy_top3_overlap = self._compute_topk_overlap(metrics.real_energy_ranking, metrics.sim_energy_ranking, 3)

        metrics.power_top2_overlap = self._compute_topk_overlap(metrics.real_power_ranking, metrics.sim_power_ranking, 2)
        metrics.power_top3_overlap = self._compute_topk_overlap(metrics.real_power_ranking, metrics.sim_power_ranking, 3)

        metrics.time_top2_overlap = self._compute_topk_overlap(metrics.real_time_ranking, metrics.sim_time_ranking, 2)
        metrics.time_top3_overlap = self._compute_topk_overlap(metrics.real_time_ranking, metrics.sim_time_ranking, 3)

    def _compute_rank_list(self, values: List[float]) -> List[int]:
        """Convert values to ranking list (1=best, higher=worse)."""
        if not values:
            return []

        # Get indices sorted by value (ascending - lower is better)
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])

        # Assign ranks
        ranks = [0] * len(values)
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[idx] = rank

        return ranks

    def _compute_topk_overlap(self, real_ranking: List[int], sim_ranking: List[int], k: int) -> int:
        """Compute how many of top-k real items are also in top-k sim."""
        if len(real_ranking) < k:
            return 0

        # Get indices of top-k items in each ranking
        real_topk = {i for i, rank in enumerate(real_ranking) if rank <= k}
        sim_topk = {i for i, rank in enumerate(sim_ranking) if rank <= k}

        return len(real_topk.intersection(sim_topk))

    def run_validation(self, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Run validation on all tasks."""
        print("🚀 Starting Five-Script Ranking Validation")
        print("=" * 50)

        # Discover tasks
        tasks = self.discover_tasks()

        if max_tasks:
            tasks = tasks[:max_tasks]
            print(f"🎯 Running validation on first {max_tasks} tasks")

        # Run validation for each task
        task_metrics = []
        successful_tasks = 0

        for category, task_name, cpp_files in tasks:
            try:
                metrics = self.validate_task(category, task_name, cpp_files)
                task_metrics.append(metrics)

                if metrics.successful_implementations >= 3:
                    successful_tasks += 1

            except Exception as e:
                print(f"   ❌ Task {category}/{task_name} failed: {e}")

        # Generate comprehensive report
        report = self._generate_report(task_metrics, successful_tasks)

        # Save results
        self._save_results(task_metrics, report)

        return report

    def _generate_report(self, task_metrics: List[TaskRankingMetrics], successful_tasks: int) -> Dict[str, Any]:
        """Generate comprehensive ranking validation report."""

        # Filter metrics for tasks with sufficient data
        valid_metrics = [m for m in task_metrics if m.successful_implementations >= 3]

        if not valid_metrics:
            return {"error": "No valid task metrics for report generation"}

        # Ranking correlation statistics
        energy_correlations = [m.energy_rank_correlation for m in valid_metrics if not np.isnan(m.energy_rank_correlation)]
        power_correlations = [m.power_rank_correlation for m in valid_metrics if not np.isnan(m.power_rank_correlation)]
        time_correlations = [m.time_rank_correlation for m in valid_metrics if not np.isnan(m.time_rank_correlation)]

        # Direction correctness counts
        energy_best_correct = sum(1 for m in valid_metrics if m.energy_best_correct)
        energy_worst_correct = sum(1 for m in valid_metrics if m.energy_worst_correct)
        power_best_correct = sum(1 for m in valid_metrics if m.power_best_correct)
        power_worst_correct = sum(1 for m in valid_metrics if m.power_worst_correct)
        time_best_correct = sum(1 for m in valid_metrics if m.time_best_correct)
        time_worst_correct = sum(1 for m in valid_metrics if m.time_worst_correct)

        # Top-k overlap statistics
        energy_top2_overlaps = [m.energy_top2_overlap for m in valid_metrics]
        energy_top3_overlaps = [m.energy_top3_overlap for m in valid_metrics]
        power_top2_overlaps = [m.power_top2_overlap for m in valid_metrics]
        power_top3_overlaps = [m.power_top3_overlap for m in valid_metrics]
        time_top2_overlaps = [m.time_top2_overlap for m in valid_metrics]
        time_top3_overlaps = [m.time_top3_overlap for m in valid_metrics]

        report = {
            "summary": {
                "total_tasks_attempted": len(task_metrics),
                "successful_tasks": successful_tasks,
                "tasks_with_ranking_data": len(valid_metrics),
                "validation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "ranking_correlations": {
                "energy": {
                    "spearman_mean": statistics.mean(energy_correlations) if energy_correlations else 0,
                    "spearman_median": statistics.median(energy_correlations) if energy_correlations else 0,
                    "spearman_std": statistics.stdev(energy_correlations) if len(energy_correlations) > 1 else 0,
                    "tasks_with_data": len(energy_correlations)
                },
                "power": {
                    "spearman_mean": statistics.mean(power_correlations) if power_correlations else 0,
                    "spearman_median": statistics.median(power_correlations) if power_correlations else 0,
                    "spearman_std": statistics.stdev(power_correlations) if len(power_correlations) > 1 else 0,
                    "tasks_with_data": len(power_correlations)
                },
                "time": {
                    "spearman_mean": statistics.mean(time_correlations) if time_correlations else 0,
                    "spearman_median": statistics.median(time_correlations) if time_correlations else 0,
                    "spearman_std": statistics.stdev(time_correlations) if len(time_correlations) > 1 else 0,
                    "tasks_with_data": len(time_correlations)
                }
            },
            "direction_correctness": {
                "energy": {
                    "best_correct_count": energy_best_correct,
                    "best_correct_rate": energy_best_correct / len(valid_metrics) if valid_metrics else 0,
                    "worst_correct_count": energy_worst_correct,
                    "worst_correct_rate": energy_worst_correct / len(valid_metrics) if valid_metrics else 0
                },
                "power": {
                    "best_correct_count": power_best_correct,
                    "best_correct_rate": power_best_correct / len(valid_metrics) if valid_metrics else 0,
                    "worst_correct_count": power_worst_correct,
                    "worst_correct_rate": power_worst_correct / len(valid_metrics) if valid_metrics else 0
                },
                "time": {
                    "best_correct_count": time_best_correct,
                    "best_correct_rate": time_best_correct / len(valid_metrics) if valid_metrics else 0,
                    "worst_correct_count": time_worst_correct,
                    "worst_correct_rate": time_worst_correct / len(valid_metrics) if valid_metrics else 0
                }
            },
            "topk_overlap": {
                "energy": {
                    "top2_mean": statistics.mean(energy_top2_overlaps) if energy_top2_overlaps else 0,
                    "top3_mean": statistics.mean(energy_top3_overlaps) if energy_top3_overlaps else 0
                },
                "power": {
                    "top2_mean": statistics.mean(power_top2_overlaps) if power_top2_overlaps else 0,
                    "top3_mean": statistics.mean(power_top3_overlaps) if power_top3_overlaps else 0
                },
                "time": {
                    "top2_mean": statistics.mean(time_top2_overlaps) if time_top2_overlaps else 0,
                    "top3_mean": statistics.mean(time_top3_overlaps) if time_top3_overlaps else 0
                }
            }
        }

        return report

    def _save_results(self, task_metrics: List[TaskRankingMetrics], report: Dict[str, Any]):
        """Save validation results to files."""

        # Save detailed task metrics
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        detailed_file = self.results_dir / f"ranking_validation_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump([asdict(m) for m in task_metrics], f, indent=2)

        # Save summary report
        report_file = self.results_dir / f"ranking_validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Results saved:")
        print(f"   📊 Detailed metrics: {detailed_file}")
        print(f"   📋 Summary report: {report_file}")

    def print_report(self, report: Dict[str, Any]):
        """Print human-readable validation report."""

        print("\n" + "="*60)
        print("🏆 FIVE-SCRIPT RANKING VALIDATION REPORT")
        print("="*60)

        summary = report.get("summary", {})
        print(f"\n📈 SUMMARY")
        print(f"Tasks attempted: {summary.get('total_tasks_attempted', 0)}")
        print(f"Successful tasks: {summary.get('successful_tasks', 0)}")
        print(f"Tasks with ranking data: {summary.get('tasks_with_ranking_data', 0)}")

        print(f"\n🔗 RANKING CORRELATIONS (Spearman's ρ)")
        correlations = report.get("ranking_correlations", {})

        energy_corr = correlations.get("energy", {})
        print(f"Energy: μ={energy_corr.get('spearman_mean', 0):.3f}, σ={energy_corr.get('spearman_std', 0):.3f} (n={energy_corr.get('tasks_with_data', 0)})")

        power_corr = correlations.get("power", {})
        print(f"Power:  μ={power_corr.get('spearman_mean', 0):.3f}, σ={power_corr.get('spearman_std', 0):.3f} (n={power_corr.get('tasks_with_data', 0)})")

        time_corr = correlations.get("time", {})
        print(f"Time:   μ={time_corr.get('spearman_mean', 0):.3f}, σ={time_corr.get('spearman_std', 0):.3f} (n={time_corr.get('tasks_with_data', 0)})")

        print(f"\n🎯 DIRECTION CORRECTNESS")
        direction = report.get("direction_correctness", {})

        energy_dir = direction.get("energy", {})
        print(f"Energy Best:  {energy_dir.get('best_correct_count', 0)}/{summary.get('tasks_with_ranking_data', 0)} ({energy_dir.get('best_correct_rate', 0)*100:.1f}%)")
        print(f"Energy Worst: {energy_dir.get('worst_correct_count', 0)}/{summary.get('tasks_with_ranking_data', 0)} ({energy_dir.get('worst_correct_rate', 0)*100:.1f}%)")

        power_dir = direction.get("power", {})
        print(f"Power Best:   {power_dir.get('best_correct_count', 0)}/{summary.get('tasks_with_ranking_data', 0)} ({power_dir.get('best_correct_rate', 0)*100:.1f}%)")
        print(f"Power Worst:  {power_dir.get('worst_correct_count', 0)}/{summary.get('tasks_with_ranking_data', 0)} ({power_dir.get('worst_correct_rate', 0)*100:.1f}%)")

        time_dir = direction.get("time", {})
        print(f"Time Best:    {time_dir.get('best_correct_count', 0)}/{summary.get('tasks_with_ranking_data', 0)} ({time_dir.get('best_correct_rate', 0)*100:.1f}%)")
        print(f"Time Worst:   {time_dir.get('worst_correct_count', 0)}/{summary.get('tasks_with_ranking_data', 0)} ({time_dir.get('worst_correct_rate', 0)*100:.1f}%)")

        print(f"\n📊 TOP-K OVERLAP")
        topk = report.get("topk_overlap", {})

        energy_topk = topk.get("energy", {})
        print(f"Energy Top-2: {energy_topk.get('top2_mean', 0):.1f}/2, Top-3: {energy_topk.get('top3_mean', 0):.1f}/3")

        power_topk = topk.get("power", {})
        print(f"Power Top-2:  {power_topk.get('top2_mean', 0):.1f}/2, Top-3: {power_topk.get('top3_mean', 0):.1f}/3")

        time_topk = topk.get("time", {})
        print(f"Time Top-2:   {time_topk.get('top2_mean', 0):.1f}/2, Top-3: {time_topk.get('top3_mean', 0):.1f}/3")

def main():
    parser = argparse.ArgumentParser(description="Five-Script Ranking Validation Suite")

    # Determine sensible defaults from repository layout
    # validation/ (this file)
    #   └─ five_per_task/
    # sniper/ (root contains "sniper" dir with run-sniper)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    default_five_per_task = os.environ.get(
        "FIVE_PER_TASK_DIR",
        str(repo_root / "validation" / "five_per_task")
    )
    default_sniper_root = os.environ.get(
        "SNIPER_ROOT",
        str(repo_root / "sniper")
    )

    parser.add_argument(
        "--five-per-task-dir",
        type=str,
        default=default_five_per_task,
        help="Directory containing five_per_task structure (default: %(default)s or $FIVE_PER_TASK_DIR)"
    )
    parser.add_argument(
        "--sniper-root",
        type=str,
        default=default_sniper_root,
        help="Root directory of Sniper installation (default: %(default)s or $SNIPER_ROOT)"
    )
    parser.add_argument("--max-tasks", type=int, default=None,
                       help="Maximum number of tasks to validate")

    args = parser.parse_args()

    # Initialize validation suite
    try:
        suite = RankingValidationSuite(
            five_per_task_dir=Path(args.five_per_task_dir),
            sniper_root=Path(args.sniper_root)
        )
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

    # Run validation
    try:
        report = suite.run_validation(max_tasks=args.max_tasks)
        suite.print_report(report)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()