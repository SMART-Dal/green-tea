#!/usr/bin/env python3
"""
GRPO Training with vLLM Generation and Multi-Objective Rewards

Key innovations:
1. vLLM generation: 10-20x faster than standard HF .generate()
2. Reward: R = R_correct + R_energy (correctness gate + tanh energy reduction)
3. Parallel Sniper simulation: Utilizes 48 CPUs for throughput

Based on:
- https://huggingface.co/docs/trl/main/en/grpo_trainer
- DeepSeek-Math GRPO implementation

Usage:
    python grpo_trainer_vllm.py \
        --model checkpoints/qwen_dpo_final/final \
        --train-data data/grpo_train.jsonl \
        --sniper-root ../sniper \
        --output-dir checkpoints/qwen_grpo_final
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import time
import threading
from dataclasses import dataclass

from unsloth import FastLanguageModel
from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MULTI-OBJECTIVE REWARD FUNCTION
# ==============================================================================

@dataclass
class ExecutionMetrics:
    """Metrics from Sniper simulation"""
    status: str  # 'success', 'compile_error', 'runtime_error', 'timeout'
    energy_joules: float = 0.0
    runtime_seconds: float = 0.0
    cycles: int = 0
    instructions: int = 0
    power_watts: float = 0.0
    ipc: float = 0.0


class MultiObjectiveRewardFunction:
    """
    FINAL REWARD: Energy-Optimized GRPO

    R_final = R_correct + R_energy

    R_correct (Hierarchical 5-level):
      -1.0:  Doesn't compile (WORST)
      -0.8 to +0.29: Compiles, partial correctness (scaled by pass_rate)
        -0.8:  0% tests pass
        -0.25: 50% tests pass
        +0.19: 90% tests pass
        +0.29: 99% tests pass
      +0.5:  100% tests pass (CORRECT)

    R_energy (only if 100% correct):
      tanh((EDP_baseline - EDP_generated) / EDP_baseline)
      where EDP = energy_joules * runtime_seconds (runtime derived from cycles / 1.5GHz)
      Range: (-1, 1); practical max tanh(1.0) ~ 0.76 when EDP reduced to near-zero

    Final Range: [-1.0, 1.5)
      Worst: -1.0 (compile fail)
      Good: +1.26 (100% tests, EDP reduced to near-zero; tanh(1.0)=0.76)
      Theoretical bound: 1.5 as EDP gain -> inf (not achievable in practice)
    """

    def __init__(
        self,
        sniper_root: str,
        pie_dataset_root: str,
        sniper_config: str = 'epyc_9554p',
        compile_timeout: int = 5,
        simulation_timeout: int = 10,
        log_file: str = "grpo_generations.jsonl",
        reward_type: str = 'edp'
    ):
        self.reward_type = reward_type
        self.sniper_root = Path(sniper_root)
        self.pie_dataset_root = Path(pie_dataset_root)
        cfg_path = self.sniper_root / 'config' / f'{sniper_config}.cfg'
        self.sniper_config = str(cfg_path) if cfg_path.exists() else sniper_config
        self.compile_timeout = compile_timeout
        self.simulation_timeout = simulation_timeout
        self.log_file = Path(log_file)
        self.job_id = os.environ.get('SLURM_JOB_ID', 'local')
        self.__name__ = "multi_objective_reward"

        # Initialize log file
        if not self.log_file.parent.exists():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Append mode is safer if restarting, but for new run we might want to clear?
        # Let's just append.

        self.stats = {
            'total_evaluations': 0,
            'compile_errors': 0,
            'runtime_errors': 0,
            'wrong_answers': 0,
            'timeouts': 0,
            'successes': 0,
            'partial_correctness': 0,
            'energy_improvements': 0,
            'energy_reductions': [],
            'rewards': []
        }
        self._stats_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._sniper_sema = threading.Semaphore(16)

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        baseline_energy: List[float],
        baseline_runtime: List[float],
        baseline_ipc: List[float],
        test_input_path: List[str],
        problem_id: List[str],
        **kwargs
    ) -> List[float]:
        n = len(completions)

        def _eval_one(i):
            code = completions[i]
            if isinstance(code, list):
                try:
                    code = code[-1]['content']
                except (IndexError, KeyError, TypeError):
                    code = str(code)
            clean_code = self._extract_code(code)
            b_metrics = {
                'energy': baseline_energy[i] if i < len(baseline_energy) else 1.0,
                'runtime': baseline_runtime[i] if i < len(baseline_runtime) else 1.0,
                'ipc': baseline_ipc[i] if i < len(baseline_ipc) else 1.0
            }
            return i, self._compute_single_reward(
                clean_code, b_metrics,
                test_input_path[i] if i < len(test_input_path) else "",
                problem_id[i] if i < len(problem_id) else "unknown"
            )

        from concurrent.futures import ThreadPoolExecutor
        rewards = [0.0] * n
        with ThreadPoolExecutor(max_workers=min(48, n)) as pool:
            for idx, reward in pool.map(_eval_one, range(n)):
                rewards[idx] = reward

        with self._stats_lock:
            total = self.stats['total_evaluations']
        if total % 50 < n:
            stats = self.get_stats()
            logger.info(f"GRPO Stats @ {total}: success={stats['success_rate']:.2%}, compile_err={stats['compile_error_rate']:.2%}, wrong_ans={stats['wrong_answer_rate']:.2%}, energy_impr={stats['energy_improvement_rate']:.2%}")

        return rewards

    def _extract_code(self, generation: str) -> str:
        if '```cpp' in generation:
            return generation.split('```cpp')[1].split('```')[0].strip()
        # Completion starts after opening ```cpp in prompt; closing ``` ends the block
        return generation.split('```')[0].strip()

    def _compute_single_reward(
        self,
        solution_code: str,
        baseline_metrics: Dict[str, float],
        test_input_path: str = "",
        problem_id: str = "unknown"
    ) -> float:
        metrics = ExecutionMetrics(status='pending')
        r_correct = 0.0
        r_energy = 0.0
        pass_rate = 0.0
        stat_delta = {}

        with tempfile.TemporaryDirectory(prefix='grpo_eval_') as temp_dir:
            temp_path = Path(temp_dir)
            binary_path = temp_path / 'solution.bin'

            if not self._compile_code(solution_code, temp_path, binary_path):
                metrics.status = 'compile_error'
                r_correct = -1.0
                stat_delta['compile_errors'] = 1
                logger.debug(f"[{problem_id}] Compile failed -> R_correct={r_correct}")
            else:
                pass_rate = self._check_correctness_rate(binary_path, problem_id)
                if pass_rate < 1.0:
                    if pass_rate == 0.0:
                        stat_delta['wrong_answers'] = 1
                        metrics.status = 'all_tests_fail'
                    else:
                        stat_delta['partial_correctness'] = 1
                        metrics.status = 'partial_correctness'
                    r_correct = -0.8 + (pass_rate * 1.1)
                    logger.debug(f"[{problem_id}] Partial correctness {pass_rate:.1%} -> R_correct={r_correct:.3f}")
                else:
                    metrics = self._measure_energy(binary_path, problem_id, test_input_path)

                    if metrics.status != 'success':
                        if metrics.status == 'timeout':
                            stat_delta['timeouts'] = 1
                            r_correct = 0.0
                        else:
                            stat_delta['runtime_errors'] = 1
                            r_correct = -1.0
                        logger.debug(f"[{problem_id}] Sniper {metrics.status} -> R_correct={r_correct}")
                    elif metrics.energy_joules < 1e-6:
                        logger.warning(f"[{problem_id}] Ghost execution")
                        stat_delta['runtime_errors'] = 1
                        metrics.status = 'ghost_execution'
                        r_correct = -1.0
                    else:
                        b_energy = baseline_metrics.get('energy', 1.0)
                        b_runtime = baseline_metrics.get('runtime', 1.0)
                        if self.reward_type == 'energy':
                            gain = (b_energy - metrics.energy_joules) / (b_energy + 1e-6)
                        elif self.reward_type == 'runtime':
                            gain = (b_runtime - metrics.runtime_seconds) / (b_runtime + 1e-6)
                        else:  # edp
                            gain = (b_energy * b_runtime - metrics.energy_joules * metrics.runtime_seconds) / (b_energy * b_runtime + 1e-6)
                        r_correct = 0.5
                        r_energy = np.tanh(gain)
                        stat_delta['successes'] = 1
                        if gain > 0:
                            stat_delta['energy_improvements'] = 1
                            stat_delta['energy_reductions_val'] = gain * 100
                        logger.debug(f"[{problem_id}] {self.reward_type}_gain={gain:.3f}, R={r_correct+r_energy:.3f}")

        reward = r_correct + r_energy

        with self._stats_lock:
            self.stats['total_evaluations'] += 1
            for k, v in stat_delta.items():
                if k == 'energy_reductions_val':
                    self.stats['energy_reductions'].append(v)
                else:
                    self.stats[k] = self.stats.get(k, 0) + v
            self.stats['rewards'].append(reward)

        log_entry = {
            'timestamp': time.time(),
            'job_id': self.job_id,
            'problem_id': problem_id,
            'status': metrics.status,
            'reward': reward,
            'r_correct': r_correct,
            'r_energy': r_energy,
            'tests_passed_rate': pass_rate,
            'energy_joules': metrics.energy_joules,
            'runtime_seconds': metrics.runtime_seconds,
            'edp': metrics.energy_joules * metrics.runtime_seconds,
            'baseline_energy': baseline_metrics.get('energy'),
            'baseline_runtime': baseline_metrics.get('runtime'),
            'baseline_edp': baseline_metrics.get('energy', 1.0) * baseline_metrics.get('runtime', 1.0),
            'code': solution_code[:2000] if len(solution_code) > 2000 else solution_code
        }
        try:
            with self._log_lock:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log entry: {e}")

        return reward

    def _compile_code(self, code: str, temp_path: Path, output_path: Path) -> bool:
        cpp_file = temp_path / 'solution.cpp'
        cpp_file.write_text(code)
        
        compile_cmd = [
            'g++', '-O3', '-std=c++17', '-static',
            str(cpp_file), '-o', str(output_path)
        ]
        
        try:
            result = subprocess.run(
                compile_cmd, capture_output=True, timeout=self.compile_timeout
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_correctness(self, binary_path: Path, problem_id: str) -> bool:
        """Run binary against all input.*.txt files for the problem"""
        test_dir = self.pie_dataset_root / "extracted_testcases" / "merged_test_cases" / problem_id
        if not test_dir.exists():
            logger.warning(f"No tests found for {problem_id}")
            return False # Fail safe

        inputs = list(test_dir.glob("input.*.txt"))
        if not inputs:
            return False

        for input_file in inputs:
            # Match output file: input.X.txt -> output.X.txt
            suffix = input_file.name.replace("input.", "")
            output_file = test_dir / f"output.{suffix}"
            
            if not output_file.exists():
                continue

            try:
                # Run binary
                with open(input_file, 'r') as infile:
                    result = subprocess.run(
                        [str(binary_path)],
                        stdin=infile,
                        capture_output=True,
                        text=True,
                        timeout=2 # Fast timeout for functional tests
                    )
                
                if result.returncode != 0:
                    return False
                
                # Check output (trim whitespace)
                expected = output_file.read_text().strip()
                actual = result.stdout.strip()
                
                if expected != actual:
                    return False

            except Exception:
                return False
        
        return True

    def _check_correctness_rate(self, binary_path: Path, problem_id: str) -> float:
        """Return fraction of tests passed (0.0 to 1.0)"""
        test_dir = self.pie_dataset_root / "extracted_testcases" / "merged_test_cases" / problem_id
        if not test_dir.exists():
            return 0.0

        inputs = list(test_dir.glob("input.*.txt"))
        if not inputs:
            return 0.0

        passed = 0
        total = 0

        for input_file in inputs:
            suffix = input_file.name.replace("input.", "")
            output_file = test_dir / f"output.{suffix}"

            if not output_file.exists():
                continue

            total += 1

            try:
                with open(input_file, 'r') as infile:
                    result = subprocess.run(
                        [str(binary_path)],
                        stdin=infile,
                        capture_output=True,
                        text=True,
                        timeout=2
                    )

                if result.returncode != 0:
                    continue

                expected = output_file.read_text().strip()
                actual = result.stdout.strip()

                if expected == actual:
                    passed += 1

            except Exception:
                continue

        return passed / total if total > 0 else 0.0

    def _measure_energy(self, binary_path: Path, problem_id: str, test_input_path: str = "") -> ExecutionMetrics:
        """Run Sniper on pre-selected canonical input or fallback to median"""
        
        input_file = None
        if test_input_path:
            # Try direct path
            p = Path(test_input_path)
            if p.exists():
                input_file = p
            else:
                # Try relative to parent (project root)
                p = Path("..") / test_input_path
                if p.exists():
                    input_file = p

        if not input_file:
            # Fallback logic: Median sized input
            test_dir = self.pie_dataset_root / "extracted_testcases" / "merged_test_cases" / problem_id
            inputs = list(test_dir.glob("input.*.txt"))
            if not inputs:
                return ExecutionMetrics(status='runtime_error')
            inputs.sort(key=lambda p: p.stat().st_size)
            input_file = inputs[len(inputs) // 2]

        if not input_file.exists():
            logger.warning(f"Test input file not found for {problem_id}")
            return ExecutionMetrics(status='runtime_error')

        with self._sniper_sema, tempfile.TemporaryDirectory() as sniper_out_dir:
            sniper_cmd = [
                str(self.sniper_root / 'run-sniper'),
                '-c', self.sniper_config,
                '-d', sniper_out_dir,
                '--power',
                '--', str(binary_path)
            ]
            
            try:
                with open(input_file, 'r') as infile:
                    result = subprocess.run(
                        sniper_cmd,
                        stdin=infile,
                        capture_output=True,
                        text=True,
                        timeout=self.simulation_timeout,
                        cwd=str(self.sniper_root) # Run from sniper root
                    )
                
                if result.returncode != 0:
                    if result.stderr:
                        logger.debug(f"[{problem_id}] Sniper stderr: {result.stderr[:500]}")
                    return ExecutionMetrics(status='runtime_error')

                sim_out = Path(sniper_out_dir) / 'sim.out'
                if sim_out.exists():
                    return self._parse_sniper_output(sim_out, result.stdout)
                return ExecutionMetrics(status='runtime_error')

            except subprocess.TimeoutExpired:
                return ExecutionMetrics(status='timeout')
            except Exception as e:
                logger.debug(f"[{problem_id}] Sniper exception: {e}")
                return ExecutionMetrics(status='runtime_error')

    def _parse_sniper_output(self, sim_out_file: Path, stdout: str = "") -> ExecutionMetrics:
        """Parse Sniper output: cycles/instructions from sim.out, energy from stdout power table."""
        cycles = 0
        instructions = 0
        energy = 0.0
        power = 0.0

        try:
            with open(sim_out_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Instructions") and "|" in line:
                        parts = [p.strip() for p in line.split("|")]
                        for p in parts[1:]:
                            try:
                                instructions += int(p)
                            except ValueError:
                                pass
                    elif line.startswith("Cycles") and "|" in line:
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) > 1:
                            try:
                                cycles = int(parts[1])
                            except ValueError:
                                pass
        except Exception as e:
            logger.warning(f"Failed to parse sim.out: {e}")

        # Parse energy from stdout power table (printed by Sniper --power)
        parsing_energy = False
        for line in stdout.split('\n'):
            line = line.strip()
            if "Power" in line and "Energy" in line and "Energy %" in line:
                parsing_energy = True
                continue
            if parsing_energy and "total" in line.lower():
                parts = line.split()
                try:
                    if len(parts) >= 4:
                        power = float(parts[1])
                        energy = float(parts[3])
                except (ValueError, IndexError):
                    pass
                break

        # Runtime from cycles at 1.5 GHz (epyc_9554p config frequency)
        runtime = cycles / 1.5e9 if cycles > 0 else 0.0
        ipc = instructions / cycles if cycles > 0 else 0.0

        return ExecutionMetrics(
            status='success',
            energy_joules=energy,
            runtime_seconds=runtime,
            cycles=cycles,
            instructions=instructions,
            power_watts=power,
            ipc=ipc
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics including energy reduction distributions"""
        total = max(self.stats['total_evaluations'], 1)

        energy_reductions = self.stats['energy_reductions'][-100:] if self.stats['energy_reductions'] else []
        rewards = self.stats['rewards'][-100:] if self.stats['rewards'] else []

        return {
            'total_evaluations': self.stats['total_evaluations'],
            'successes': self.stats['successes'],
            'compile_errors': self.stats['compile_errors'],
            'wrong_answers': self.stats['wrong_answers'],
            'timeouts': self.stats['timeouts'],
            'runtime_errors': self.stats['runtime_errors'],
            'partial_correctness': self.stats['partial_correctness'],
            'energy_improvements': self.stats['energy_improvements'],
            'success_rate': self.stats['successes'] / total,
            'compile_error_rate': self.stats['compile_errors'] / total,
            'wrong_answer_rate': self.stats['wrong_answers'] / total,
            'timeout_rate': self.stats['timeouts'] / total,
            'runtime_error_rate': self.stats['runtime_errors'] / total,
            'energy_improvement_rate': self.stats['energy_improvements'] / max(self.stats['successes'], 1),
            'mean_energy_reduction': float(np.mean(energy_reductions)) if energy_reductions else 0.0,
            'median_energy_reduction': float(np.median(energy_reductions)) if energy_reductions else 0.0,
            'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
            'median_reward': float(np.median(rewards)) if rewards else 0.0
        }


# ==============================================================================
# GRPO TRAINER WITH VLLM
# ==============================================================================

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, reward_function, *args, **kwargs):
        kwargs["reward_funcs"] = reward_function
        super().__init__(*args, **kwargs)
        self.reward_function = reward_function
        self._stats_snapshot = None
        fence_toks = self.processing_class.encode('```', add_special_tokens=False)
        if fence_toks:
            eos = self.generation_config.eos_token_id
            eos_list = [eos] if isinstance(eos, int) else list(eos or [])
            if fence_toks[0] not in eos_list:
                self.generation_config.eos_token_id = eos_list + [fence_toks[0]]
                logger.info(f"Added fence token {fence_toks[0]} as EOS for generation")

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        stats = self.reward_function.get_stats()
        prev = self._stats_snapshot or {}

        # Per-step deltas (since last log call)
        delta_evals = max(stats['total_evaluations'] - prev.get('total_evaluations', 0), 1)
        delta_success = stats['successes'] - prev.get('successes', 0)
        delta_compile = stats['compile_errors'] - prev.get('compile_errors', 0)
        delta_wrong = stats['wrong_answers'] - prev.get('wrong_answers', 0)
        delta_energy_impr = stats['energy_improvements'] - prev.get('energy_improvements', 0)

        logs['step/success_rate'] = delta_success / delta_evals
        logs['step/compile_error_rate'] = delta_compile / delta_evals
        logs['step/wrong_answer_rate'] = delta_wrong / delta_evals
        logs['step/energy_improvement_rate'] = delta_energy_impr / max(delta_success, 1)

        # Rolling reward stats (last 100 completions)
        recent = self.reward_function.stats['rewards'][-100:]
        if recent:
            logs['step/reward_mean'] = float(np.mean(recent))
            logs['step/reward_std'] = float(np.std(recent))
            logs['step/reward_min'] = float(np.min(recent))
            logs['step/reward_max'] = float(np.max(recent))

        # Cumulative totals
        logs['total/success_rate'] = stats.get('success_rate', 0.0)
        logs['total/compile_error_rate'] = stats.get('compile_error_rate', 0.0)
        logs['total/energy_improvement_rate'] = stats.get('energy_improvement_rate', 0.0)
        logs['sniper/total_evaluations'] = stats.get('total_evaluations', 0)
        logs['sniper/compile_fail_count'] = stats.get('compile_errors', 0)
        logs['sniper/timeout_count'] = stats.get('timeouts', 0)
        logs['energy/mean_reduction_pct'] = stats.get('mean_energy_reduction', 0.0)
        logs['energy/median_reduction_pct'] = stats.get('median_energy_reduction', 0.0)

        self._stats_snapshot = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
        super().log(logs, *args, **kwargs)


# ==============================================================================
# PROMPT TEMPLATE (PIE Paper Format + Problem Description)
# ==============================================================================

ENERGY_PROMPT_TEMPLATE = """This is an energy inefficient program we want to optimize to score 10/10.
### Program:
{baseline_code}

### Energy Optimized Version with score 10/10:
```cpp
"""


def create_prompt_template(example):
    """Format dataset example into GRPO prompt - matches SFT format exactly"""
    return ENERGY_PROMPT_TEMPLATE.format(
        baseline_code=example.get('baseline_code', '')
    )


def main():
    parser = argparse.ArgumentParser(
        description='GRPO training with vLLM and multi-objective rewards'
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, default=None)
    parser.add_argument('--sniper-root', type=str, required=True)
    parser.add_argument('--sniper-config', type=str, default='epyc_9554p')
    parser.add_argument('--output-dir', type=str, default='checkpoints/qwen_grpo_final')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-generations', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--beta', type=float, default=0.04)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='energy-code-generation', help='W&B project name')
    parser.add_argument('--eval-during-training', action='store_true', help='Enable periodic energy evaluation')
    parser.add_argument('--model-name', type=str, default='qwen-14b', help='Model identifier for organizing outputs and WandB tracking')
    parser.add_argument('--pie-dataset', type=str, default='../PIE_Dataset', help='Path to PIE dataset root (for test cases)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    parser.add_argument('--generation-log', type=str, default='data/grpo_generations/grpo_generations.jsonl')
    parser.add_argument('--compile-timeout', type=int, default=30)
    parser.add_argument('--simulation-timeout', type=int, default=300)
    parser.add_argument('--reward-type', type=str, default='edp', choices=['edp', 'energy', 'runtime'])

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("GRPO TRAINING WITH VLLM + MULTI-OBJECTIVE REWARDS")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Group size (K): {args.num_generations}")
    logger.info(f"Reward: EDP (energy x runtime)")

    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    # Load model for training (will use LoRA for efficiency)
    logger.info("Loading model for GRPO training...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )

    # Apply LoRA for memory efficiency during GRPO
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 128,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    model.print_trainable_parameters()

    # Initialize reward function
    logger.info("Initializing multi-objective reward function...")
    reward_fn = MultiObjectiveRewardFunction(
        sniper_root=args.sniper_root,
        pie_dataset_root=args.pie_dataset,
        sniper_config=args.sniper_config,
        compile_timeout=args.compile_timeout,
        simulation_timeout=args.simulation_timeout,
        log_file=args.generation_log,
        reward_type=args.reward_type
    )

    # Load dataset
    logger.info("\nLoading GRPO dataset...")
    dataset_files = {'train': args.train_data}
    if args.val_data:
        dataset_files['validation'] = args.val_data

    dataset = load_dataset('json', data_files=dataset_files)

    # Format prompts
    def format_example(example):
        prompt = create_prompt_template(example)
        return {'prompt': prompt, **example}

    dataset = dataset.map(format_example)

    logger.info(f"Loaded {len(dataset['train']):,} GRPO training examples")

    # GRPO Configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,

        # GRPO-specific
        num_generations=args.num_generations,  # K = group size
        temperature=1.0,
        beta=args.beta,  # KL penalty (like DPO beta)
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=768,  # 1024 causes OOM at step 26 (512 seqs x 1024 tokens KV cache)
        generation_batch_size=16,  # chunk generation: 1 prompt x K=16 per call vs 512 seqs at once
        mask_truncated_completions=True,  # clipped completions are always compile errors; mask their gradients

        # Optimization
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # Precision
        bf16=True,
        gradient_checkpointing=True,

        # Logging
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,

        # Evaluation
        eval_strategy="steps" if args.val_data else "no",
        eval_steps=25 if args.val_data else 25,
        load_best_model_at_end=True if args.val_data else False,
        metric_for_best_model="objective/rlhf_reward",
        greater_is_better=True,

        # Reporting
        report_to=["wandb"] if args.use_wandb else ["tensorboard"],
        logging_dir=f"{args.output_dir}/logs",

        # W&B Run Name
        run_name=f"{args.model_name}-grpo",
    )

    # Prepare callbacks
    callbacks = []
    if args.use_wandb:
        from utils.training_callbacks import get_wandb_callback
        wandb_callback = get_wandb_callback(
            stage="grpo",
            model_name=args.model_name,
            additional_config={
                'num_generations': args.num_generations,
                'reward': 'edp',
                'sniper_config': args.sniper_config
            }
        )
        callbacks.append(wandb_callback)
        logger.info("✓ W&B logging enabled")

    if args.eval_during_training:
        from utils.training_callbacks import EnergyEvaluationCallback
        energy_callback = EnergyEvaluationCallback(
            eval_model=model,
            eval_tokenizer=tokenizer,
            eval_dataset=dataset.get('validation', dataset['train']),
            sniper_root=args.sniper_root,
            eval_every_n_steps=50,
            max_eval_samples=10
        )
        callbacks.append(energy_callback)
        logger.info("✓ Energy evaluation during training enabled")

    # Create GRPO Trainer
    logger.info("\nInitializing CustomGRPOTrainer...")
    trainer = CustomGRPOTrainer(
        reward_function=reward_fn,
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        processing_class=tokenizer,
        callbacks=callbacks
    )

    logger.info("\n" + "=" * 70)
    logger.info("STARTING GRPO TRAINING")
    logger.info("=" * 70)
    logger.info("\nGRPO learns through online exploration:")
    logger.info("  1. Generate K=16 solutions per prompt (via vLLM)")
    logger.info("  2. Simulate each solution in Sniper (parallel)")
    logger.info("  3. Compute multi-objective rewards: R = R_status + R_algo + 0.5×R_arch")
    logger.info("  4. Normalize advantages within group (variance reduction)")
    logger.info("  5. Update policy to increase high-reward solution probabilities")
    logger.info("\nExpected speed:")
    logger.info("  - Generation: 10-20x faster with vLLM")
    logger.info("  - Simulation: Parallelized across 48 CPUs")
    logger.info("  - ~20-30s per training step")
    logger.info("=" * 70 + "\n")

    # Train (resume_from_checkpoint=None means fresh start)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info("\nSaving final model...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    # Print final statistics
    logger.info("\n" + "=" * 70)
    logger.info("GRPO TRAINING COMPLETE!")
    logger.info("=" * 70)

    final_stats = reward_fn.get_stats()
    logger.info("\nFinal Reward Function Statistics:")
    logger.info(f"  Total evaluations: {final_stats['total_evaluations']:,}")
    logger.info(f"  Success rate: {final_stats['success_rate']:.2%}")
    logger.info(f"  Energy improvement rate: {final_stats['energy_improvement_rate']:.2%}")
    logger.info(f"  IPC improvement rate: {final_stats.get('ipc_improvement_rate', 0):.2%}")

    logger.info(f"\nFinal model saved to: {args.output_dir}/final")
    logger.info("\nNext: Run comprehensive evaluation")
    logger.info(f"  python evaluation.py --model {args.output_dir}/final ...")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
