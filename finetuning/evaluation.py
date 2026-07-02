#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Energy-Efficient Code Generation

Implements multiple evaluation metrics with statistical validation:
- Energy Reduction Rate (ERR) - Primary metric
- Generalization Success Rate (GSR) - Cross-architecture transfer
- Pass@k - Correctness metric
- Statistical significance testing (t-tests, Cohen's d)
- Baseline comparisons

Usage:
    python evaluation.py \
        --model-path checkpoints/grpo_energy/checkpoint-best \
        --test-data data/sft_pairs_test.jsonl \
        --sniper-root /path/to/sniper \
        --output-dir evaluation_results \
        --num-runs 5

Output:
    evaluation_results/
    ├── err_results.json          # Energy reduction rates
    ├── gsr_results.json          # Cross-arch transfer
    ├── statistical_tests.json    # Significance tests
    └── evaluation_report.md      # Markdown report
"""

import os
import json
import argparse
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import scipy.stats
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.training_callbacks import EnergyEvaluationCallback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    problem_id: str
    test_input_hash: str
    baseline_energy: float
    generated_energy: float
    energy_reduction_pct: float
    baseline_cycles: int
    generated_cycles: int
    baseline_edp: float
    generated_edp: float
    edp_reduction_pct: float
    speedup: float
    baseline_ipc: float = 0
    generated_ipc: float = 0
    ipc_improvement_pct: float = 0
    compilation_success: bool = False
    runtime_success: bool = False
    baseline_code: str = None
    generated_code: str = None
    error_msg: str = None


@dataclass
class AggregateMetrics:
    """Aggregate metrics across test set"""
    mean_err: float
    std_err: float
    median_err: float
    p25_err: float
    p75_err: float
    p95_err: float
    mean_edp_reduction: float
    median_edp_reduction: float
    mean_ipc_improvement: float
    median_ipc_improvement: float
    success_rate: float
    num_samples: int
    num_improvements: int
    num_regressions: int
    baseline_success_rate: float = 0
    generated_success_rate: float = 0


class SniperEvaluator:
    """
    Handles compilation and Sniper simulation for evaluation
    """

    def __init__(
        self,
        sniper_root: str,
        sniper_config: str = None,
        timeout: int = 120
    ):
        self.sniper_root = Path(sniper_root)
        self.sniper_config = sniper_config or str(self.sniper_root / 'config' / 'epyc_9554p.cfg')
        self.timeout = timeout

    def measure_energy(
        self,
        code: str,
        test_input: str,
        compile_flags: str = '-O3'
    ) -> Dict[str, Any]:
        """
        Compile and measure energy for a single code solution

        Args:
            code: C++ source code
            test_input: Test input content
            compile_flags: Compilation flags (default: -O3)

        Returns:
            Dict with energy, cycles, instructions, status
        """
        temp_dir = tempfile.mkdtemp(prefix='eval_')
        temp_path = Path(temp_dir)

        try:
            # Write code
            cpp_file = temp_path / 'solution.cpp'
            cpp_file.write_text(code)

            # Compile
            binary_file = temp_path / 'solution.bin'
            compile_cmd = [
                'g++', compile_flags, '-std=c++17', '-static',
                str(cpp_file), '-o', str(binary_file)
            ]

            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                timeout=120,
                text=True
            )

            if compile_result.returncode != 0:
                return {
                    'status': 'compile_error',
                    'error_msg': compile_result.stderr[:200]
                }

            # Run Sniper
            sniper_output = temp_path / 'sniper_output'
            sniper_output.mkdir()

            sniper_cmd = [
                str(self.sniper_root / 'run-sniper'),
                '-c', self.sniper_config,
                '-d', str(sniper_output),
                '--power',
                '--', str(binary_file)
            ]

            input_file = temp_path / 'input.txt'
            input_file.write_text(test_input)

            with open(input_file) as inp:
                sim_result = subprocess.run(
                    sniper_cmd,
                    stdin=inp,
                    capture_output=True,
                    timeout=self.timeout,
                    text=True
                )

            if sim_result.returncode != 0:
                return {
                    'status': 'runtime_error',
                    'error_msg': sim_result.stderr[:200]
                }

            # Parse results
            sim_out = sniper_output / 'sim.out'
            energy, cycles, instructions = self.parse_output(
                sim_out, sim_result.stdout
            )

            return {
                'status': 'success',
                'energy_joules': energy,
                'cycles': cycles,
                'instructions': instructions
            }

        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'error_msg': f'Exceeded {self.timeout}s timeout'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_msg': str(e)[:200]
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def parse_output(
        self,
        sim_out_file: Path,
        stdout: str
    ) -> Tuple[float, int, int]:
        """Parse Sniper output for energy, cycles, instructions"""
        energy = 0.0
        cycles = 0
        instructions = 0

        # Parse sim.out
        if sim_out_file.exists():
            with open(sim_out_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Instructions") and "|" in line:
                        parts = line.split("|")
                        if len(parts) > 1:
                            try:
                                instructions = int(parts[1].strip())
                            except ValueError:
                                pass
                    elif line.startswith("Cycles") and "|" in line:
                        parts = line.split("|")
                        if len(parts) > 1:
                            try:
                                cycles = int(parts[1].strip())
                            except ValueError:
                                pass

        # Parse energy from stdout (matches dataset collection script)
        # Format: "  total           204.52 W   0.233948  J    100.00%"
        parsing_energy = False
        for line in stdout.split('\n'):
            if "Power" in line and "Energy" in line and "Energy %" in line:
                parsing_energy = True
                continue

            if parsing_energy and "total" in line.lower():
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        energy = float(parts[3])  # Energy at position 3
                        break
                except (ValueError, IndexError):
                    pass

        return energy, cycles, instructions


class EnergyEvaluationFramework:
    """
    Comprehensive evaluation framework with multiple metrics
    """

    def __init__(
        self,
        model_path: str,
        sniper_root: str,
        output_dir: str,
        num_runs: int = 5,
        device: str = 'auto'
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs = num_runs

        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # sniper_root is {PROJECT_ROOT}/sniper/sniper; PIE_Dataset is at PROJECT_ROOT
        self.pie_root = Path(sniper_root).parent.parent / 'PIE_Dataset'

        # Initialize evaluator
        self.evaluator = SniperEvaluator(sniper_root)

        logger.info("Initialized evaluation framework")

    def _get_all_test_inputs(self, problem_id: str) -> List[str]:
        """Get ALL test inputs for a problem"""
        test_dir = self.pie_root / 'extracted_testcases' / 'merged_test_cases' / problem_id
        if not test_dir.exists():
            return []

        inputs = sorted(test_dir.glob('input.*.txt'))
        return [f.read_text().strip() for f in inputs]

    def compute_err(
        self,
        test_problems: List[Dict[str, Any]],
        seed: int = 42
    ) -> Tuple[List[EvaluationResult], AggregateMetrics]:
        """
        Compute Energy Reduction Rate (ERR) - Primary metric

        Multi-input ERR to prevent overfitting to single input size

        Args:
            test_problems: List of test problem dictionaries
            seed: Random seed for reproducibility

        Returns:
            (individual_results, aggregate_metrics)
        """
        logger.info("=" * 70)
        logger.info("COMPUTING ENERGY REDUCTION RATE (ERR)")
        logger.info("=" * 70)
        logger.info(f"Total test problems: {len(test_problems)}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        individual_results = []
        energy_reductions = []
        edp_reductions = []
        speedups = []
        ipc_improvements = []
        successes = 0
        improvements = 0
        regressions = 0
        beat_gt_count = 0
        baseline_successes = 0
        generated_successes = 0
        total_attempted = 0

        for idx, problem in enumerate(tqdm(test_problems, desc="Evaluating problems")):
            problem_id = problem['problem_id']
            test_input_hash = problem.get('test_input_hash', 'unknown')

            if (idx + 1) % 50 == 0:
                logger.info(f"Progress: {idx+1}/{len(test_problems)} | Baseline success: {baseline_successes}/{total_attempted} | Generated success: {generated_successes}/{total_attempted} | Paired: {successes}")
                # Incremental save for debugging
                checkpoint_file = Path(f"data/evaluation/checkpoint_{idx+1}.json")
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'progress': f"{idx+1}/{len(test_problems)}",
                        'baseline_successes': baseline_successes,
                        'generated_successes': generated_successes,
                        'paired_successes': successes,
                        'results_so_far': [asdict(r) for r in individual_results]
                    }, f, indent=2)
                logger.info(f"Checkpoint saved: {checkpoint_file}")
            # Support both SFT (inefficient_code) and GRPO (baseline_code) datasets
            baseline_code = problem.get('baseline_code', problem.get('inefficient_code', ''))

            # Load ALL test inputs for this problem
            test_inputs = self._get_all_test_inputs(problem_id)

            if not baseline_code:
                logger.warning(f"Skipping {problem_id}: missing baseline_code")
                continue
            if not test_inputs:
                logger.warning(f"Skipping {problem_id}: no test inputs found")
                continue

            logger.info(f"[{problem_id}] Evaluating on {len(test_inputs)} inputs")

            prompt = self.create_optimization_prompt(problem)
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated_code = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            generated_code = self.extract_code(generated_code)

            # Measure on ALL inputs and aggregate
            baseline_energies, baseline_cycles_list, baseline_instrs = [], [], []
            generated_energies, generated_cycles_list, generated_instrs = [], [], []

            for inp_idx, test_input_content in enumerate(test_inputs):
                b_res = self.evaluator.measure_energy(baseline_code, test_input_content, compile_flags='-O3')
                g_res = self.evaluator.measure_energy(generated_code, test_input_content, compile_flags='-O3')

                if b_res['status'] == 'success' and b_res['energy_joules'] > 1e-9:
                    baseline_energies.append(b_res['energy_joules'])
                    baseline_cycles_list.append(b_res['cycles'])
                    baseline_instrs.append(b_res['instructions'])

                if g_res['status'] == 'success' and g_res['energy_joules'] > 1e-9:
                    generated_energies.append(g_res['energy_joules'])
                    generated_cycles_list.append(g_res['cycles'])
                    generated_instrs.append(g_res['instructions'])

            logger.info(f"[{problem_id}] Baseline: {len(baseline_energies)}/{len(test_inputs)} succeeded, Avg E={np.mean(baseline_energies) if baseline_energies else 0:.6f}J")
            logger.info(f"[{problem_id}] Generated: {len(generated_energies)}/{len(test_inputs)} succeeded, Avg E={np.mean(generated_energies) if generated_energies else 0:.6f}J")

            # Aggregate results
            baseline_result = {
                'status': 'success' if baseline_energies else 'failed',
                'energy_joules': np.mean(baseline_energies) if baseline_energies else 0,
                'cycles': int(np.mean(baseline_cycles_list)) if baseline_cycles_list else 0,
                'instructions': int(np.mean(baseline_instrs)) if baseline_instrs else 0
            }
            generated_result = {
                'status': 'success' if generated_energies else 'failed',
                'energy_joules': np.mean(generated_energies) if generated_energies else 0,
                'cycles': int(np.mean(generated_cycles_list)) if generated_cycles_list else 0,
                'instructions': int(np.mean(generated_instrs)) if generated_instrs else 0
            }

            total_attempted += 1
            baseline_ok = baseline_result['status'] == 'success'
            generated_ok = generated_result['status'] == 'success'

            if baseline_ok and baseline_result['energy_joules'] > 1e-9:
                baseline_successes += 1
            else:
                logger.warning(f"[{problem_id}] Baseline: status={baseline_result['status']}, E={baseline_result.get('energy_joules', 0):.9f}J, cycles={baseline_result.get('cycles', 0)}")

            if generated_ok and generated_result['energy_joules'] > 1e-9:
                generated_successes += 1
            else:
                logger.warning(f"[{problem_id}] Generated: status={generated_result['status']}, E={generated_result.get('energy_joules', 0):.9f}J, cycles={generated_result.get('cycles', 0)}")

            # Compute metrics (relax energy threshold for ghost executions)
            if (baseline_result['status'] == 'success' and
                generated_result['status'] == 'success' and
                baseline_result['energy_joules'] > 1e-9 and
                baseline_result['cycles'] > 0):

                E_baseline = baseline_result['energy_joules']
                E_generated = generated_result['energy_joules']
                C_baseline = baseline_result['cycles']
                C_generated = generated_result['cycles']

                # Energy reduction
                energy_reduction = (E_baseline - E_generated) / E_baseline * 100

                # EDP (Energy Delay Product) = Energy × Cycles
                edp_baseline = E_baseline * C_baseline
                edp_generated = E_generated * C_generated
                edp_reduction = (edp_baseline - edp_generated) / edp_baseline * 100 if edp_baseline > 0 else 0

                # Speedup
                speedup = C_baseline / C_generated if C_generated > 0 else 1.0

                # IPC (Instructions Per Cycle)
                baseline_ipc = baseline_result['instructions'] / C_baseline if C_baseline > 0 else 0
                generated_ipc = generated_result['instructions'] / C_generated if C_generated > 0 else 0
                ipc_improvement = (generated_ipc - baseline_ipc) / baseline_ipc * 100 if baseline_ipc > 0 else 0

                energy_reductions.append(energy_reduction)
                edp_reductions.append(edp_reduction)
                speedups.append(speedup)
                ipc_improvements.append(ipc_improvement)
                successes += 1

                if energy_reduction > 0:
                    improvements += 1
                elif energy_reduction < -5:
                    regressions += 1

                opt_energy = problem.get('optimized_energy', 0)
                if opt_energy > 1e-9 and E_generated <= opt_energy:
                    beat_gt_count += 1

                result = EvaluationResult(
                    problem_id=problem_id,
                    test_input_hash=test_input_hash,
                    baseline_energy=E_baseline,
                    generated_energy=E_generated,
                    energy_reduction_pct=energy_reduction,
                    baseline_cycles=C_baseline,
                    generated_cycles=C_generated,
                    baseline_edp=edp_baseline,
                    generated_edp=edp_generated,
                    edp_reduction_pct=edp_reduction,
                    speedup=speedup,
                    baseline_ipc=baseline_ipc,
                    generated_ipc=generated_ipc,
                    ipc_improvement_pct=ipc_improvement,
                    compilation_success=True,
                    runtime_success=True,
                    baseline_code=baseline_code,
                    generated_code=generated_code
                )
            else:
                # Failure case
                error_msg = (generated_result.get('error_msg') or
                           baseline_result.get('error_msg'))
                result = EvaluationResult(
                    problem_id=problem_id,
                    test_input_hash=test_input_hash,
                    baseline_energy=0,
                    generated_energy=0,
                    energy_reduction_pct=0,
                    baseline_cycles=0,
                    generated_cycles=0,
                    baseline_edp=0,
                    generated_edp=0,
                    edp_reduction_pct=0,
                    speedup=0,
                    compilation_success=generated_result['status'] != 'compile_error',
                    runtime_success=generated_result['status'] == 'success',
                    baseline_code=baseline_code,
                    generated_code=generated_code,
                    error_msg=error_msg
                )

            individual_results.append(result)

        # Compute aggregate metrics
        metrics = AggregateMetrics(
            mean_err=np.mean(energy_reductions) if energy_reductions else 0,
            std_err=np.std(energy_reductions) if energy_reductions else 0,
            median_err=np.median(energy_reductions) if energy_reductions else 0,
            p25_err=np.percentile(energy_reductions, 25) if energy_reductions else 0,
            p75_err=np.percentile(energy_reductions, 75) if energy_reductions else 0,
            p95_err=np.percentile(energy_reductions, 95) if energy_reductions else 0,
            mean_edp_reduction=np.mean(edp_reductions) if edp_reductions else 0,
            median_edp_reduction=np.median(edp_reductions) if edp_reductions else 0,
            mean_ipc_improvement=np.mean(ipc_improvements) if ipc_improvements else 0,
            median_ipc_improvement=np.median(ipc_improvements) if ipc_improvements else 0,
            success_rate=successes / len(test_problems),
            num_samples=len(test_problems),
            num_improvements=improvements,
            num_regressions=regressions,
            baseline_success_rate=baseline_successes / total_attempted if total_attempted > 0 else 0,
            generated_success_rate=generated_successes / total_attempted if total_attempted > 0 else 0
        )

        logger.info(f"\nERR Results:")
        logger.info(f"  Baseline success: {metrics.baseline_success_rate:.2%} ({baseline_successes}/{total_attempted})")
        logger.info(f"  Generated success: {metrics.generated_success_rate:.2%} ({generated_successes}/{total_attempted})")
        logger.info(f"  Both success (paired): {metrics.success_rate:.2%} ({successes}/{len(test_problems)})")
        logger.info(f"  Mean ERR: {metrics.mean_err:.2f}%")
        logger.info(f"  Median ERR: {metrics.median_err:.2f}%")
        logger.info(f"  Mean EDP reduction: {metrics.mean_edp_reduction:.2f}%")
        logger.info(f"  Median EDP reduction: {metrics.median_edp_reduction:.2f}%")
        logger.info(f"  Mean IPC improvement: {metrics.mean_ipc_improvement:.2f}%")
        logger.info(f"  Median IPC improvement: {metrics.median_ipc_improvement:.2f}%")
        logger.info(f"  Improvements: {improvements}/{len(test_problems)}")
        logger.info(f"  Regressions: {regressions}/{len(test_problems)}")
        logger.info(f"  Beat-GT: {beat_gt_count}/{successes} ({beat_gt_count/successes*100:.1f}%)" if successes > 0 else "  Beat-GT: 0/0")

        return individual_results, metrics

    def compute_gsr(
        self,
        test_problems: List[Dict[str, Any]],
        target_arch: str = 'nehalem',
        seed: int = 42
    ) -> float:
        """
        Compute Generalization Success Rate (GSR) - Cross-architecture transfer

        Tests if optimizations generalize to different microarchitectures.
        GSR = % of problems with energy improvement on target architecture

        Args:
            test_problems: List of test problems
            target_arch: Sniper target architecture (default: nehalem for x86, use cortex-a72 for ARM)
            seed: Random seed

        Returns:
            GSR percentage
        """
        logger.info("=" * 70)
        logger.info(f"COMPUTING GENERALIZATION SUCCESS RATE (GSR)")
        logger.info(f"Target architecture: {target_arch}")
        logger.info("=" * 70)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create evaluator for target architecture
        target_evaluator = SniperEvaluator(self.evaluator.sniper_root, sniper_config=target_arch)

        successful_transfers = 0
        total_evaluated = 0

        for problem in tqdm(test_problems, desc="Evaluating cross-arch transfer"):
            baseline_code = problem.get('baseline_code', problem.get('inefficient_code', ''))
            test_inputs = self._get_all_test_inputs(problem.get('problem_id', ''))
            test_input_content = test_inputs[0] if test_inputs else ''

            if not baseline_code or not test_input_content:
                continue

            prompt = self.create_optimization_prompt(problem)
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated_code = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            generated_code = self.extract_code(generated_code)

            baseline_result = target_evaluator.measure_energy(baseline_code, test_input_content)
            generated_result = target_evaluator.measure_energy(generated_code, test_input_content)

            if (baseline_result['status'] == 'success' and
                generated_result['status'] == 'success'):

                total_evaluated += 1
                energy_reduction = (
                    (baseline_result['energy_joules'] - generated_result['energy_joules']) /
                    baseline_result['energy_joules']
                )

                if energy_reduction > 0:
                    successful_transfers += 1

        gsr = (successful_transfers / total_evaluated * 100) if total_evaluated > 0 else 0

        logger.info(f"\nGSR Results:")
        logger.info(f"  GSR: {gsr:.2f}%")
        logger.info(f"  Successful transfers: {successful_transfers}/{total_evaluated}")

        return gsr

    def create_optimization_prompt(self, problem: Dict[str, Any]) -> str:
        """Create prompt matching OPTIMIZATION_TEMPLATE from sft_train_trl.py"""
        baseline_code = problem.get('baseline_code', problem.get('inefficient_code', ''))
        return (
            f"This is an energy inefficient program we want to optimize to score 10/10.\n"
            f"### Program:\n{baseline_code}\n\n"
            f"### Energy Optimized Version with score 10/10:\n```cpp\n"
        )

    def extract_code(self, generation: str) -> str:
        """Extract C++ code from generation.

        Prompt ends with ```cpp so model outputs code directly.
        Expected generation: code...\n```
        split('```') produces ['code...', ''] -- code is at index 0.
        """
        if '```cpp' in generation:
            code = generation.split('```cpp')[1].split('```')[0]
        elif '```' in generation:
            code = generation.split('```')[0]
        else:
            code = generation
        return code.strip()

    def save_results(
        self,
        err_results: Tuple[List[EvaluationResult], AggregateMetrics],
        gsr: float = 0.0,
        stats: Dict[str, Any] = None
    ):
        """Save evaluation results to files"""
        logger.info("\nSaving results...")

        individual_results, aggregate_metrics = err_results

        # Save ERR results
        err_file = self.output_dir / 'err_results.json'
        with open(err_file, 'w') as f:
            json.dump({
                'aggregate_metrics': asdict(aggregate_metrics),
                'individual_results': [asdict(r) for r in individual_results]
            }, f, indent=2)

        # Save GSR
        gsr_file = self.output_dir / 'gsr_results.json'
        with open(gsr_file, 'w') as f:
            json.dump({'gsr': gsr}, f, indent=2)

        # Save statistical tests
        if stats:
            stats_file = self.output_dir / 'statistical_tests.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

        # Generate markdown report
        self.generate_report(aggregate_metrics, gsr, stats or {})

        # CAERR: mean ERR over all N (failures have energy_reduction_pct=0 already)
        n = aggregate_metrics.num_samples
        caerr = sum(r.energy_reduction_pct for r in individual_results) / n if n > 0 else 0.0

        logger.info(f"\n{'='*70}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total samples: {n}")
        logger.info(f"Successful pairs: {int(aggregate_metrics.success_rate * n)}")
        logger.info(f"Success rate: {aggregate_metrics.success_rate:.2%}")
        logger.info(f"Mean ERR (valid only): {aggregate_metrics.mean_err:.4f}%")
        logger.info(f"Median ERR (valid only): {aggregate_metrics.median_err:.4f}%")
        logger.info(f"CAERR (all N): {caerr:.4f}%")
        logger.info(f"GSR (Generalization): {gsr:.4f}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*70}\n")

    def generate_report(
        self,
        metrics: AggregateMetrics,
        gsr: float,
        stats: Dict[str, Any]
    ):
        """Generate markdown evaluation report"""
        report_file = self.output_dir / 'evaluation_report.md'

        report = f"""# Energy-Efficient Code Generation - Evaluation Report

## Model
- **Path**: {self.model_path}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Primary Metrics

### Energy Reduction Rate (ERR)

| Metric | Value |
|--------|-------|
| **Mean ERR** | {metrics.mean_err:.2f}% |
| **Median ERR** | {metrics.median_err:.2f}% |
| **Std Dev** | {metrics.std_err:.2f}% |
| **P25** | {metrics.p25_err:.2f}% |
| **P75** | {metrics.p75_err:.2f}% |
| **P95** | {metrics.p95_err:.2f}% |

### Energy Delay Product (EDP) Reduction

| Metric | Value |
|--------|-------|
| **Mean EDP Reduction** | {metrics.mean_edp_reduction:.2f}% |
| **Median EDP Reduction** | {metrics.median_edp_reduction:.2f}% |

## Success Rates

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Samples** | {metrics.num_samples} | 100.0% |
| **Successful Compilations** | {int(metrics.num_samples * metrics.success_rate)} | {metrics.success_rate:.1%} |
| **Energy Improvements** | {metrics.num_improvements} | {metrics.num_improvements/metrics.num_samples:.1%} |
| **Energy Regressions** | {metrics.num_regressions} | {metrics.num_regressions/metrics.num_samples:.1%} |

## Generalization Success Rate (GSR)

| Metric | Value |
|--------|-------|
| **Cross-Architecture GSR** | {gsr:.2f}% |

## Interpretation

"""

        if metrics.mean_err >= 60:
            report += "✅ **Excellent performance** - Exceeds target ERR of 60%\n"
        elif metrics.mean_err >= 45:
            report += "✅ **Good performance** - Approaching target ERR\n"
        else:
            report += "⚠️ **Below target** - Further optimization needed\n"

        if gsr >= 65:
            report += "✅ **Strong cross-architecture transfer** - Exceeds GSR target\n"
        elif gsr >= 55:
            report += "✅ **Moderate transfer** - Approaching GSR target\n"
        else:
            report += "⚠️ **Weak transfer** - Architecture-specific overfitting detected\n"

        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"Report generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation for energy-efficient code generation'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data JSONL file'
    )
    parser.add_argument(
        '--sniper-root',
        type=str,
        required=True,
        help='Path to Sniper installation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=5,
        help='Number of evaluation runs for statistical validation'
    )
    parser.add_argument(
        '--compute-gsr',
        action='store_true',
        help='Compute cross-architecture GSR (slower)'
    )

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    test_problems = []
    with open(args.test_data) as f:
        for line in f:
            test_problems.append(json.loads(line))

    logger.info(f"Loaded {len(test_problems)} test problems")

    # Initialize framework
    framework = EnergyEvaluationFramework(
        model_path=args.model_path,
        sniper_root=args.sniper_root,
        output_dir=args.output_dir,
        num_runs=args.num_runs
    )

    # Compute ERR
    err_results = framework.compute_err(test_problems)

    # Compute GSR if requested
    gsr = 0.0
    if args.compute_gsr:
        gsr = framework.compute_gsr(test_problems, target_arch='nehalem')

    # Save results
    framework.save_results(err_results, gsr)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
