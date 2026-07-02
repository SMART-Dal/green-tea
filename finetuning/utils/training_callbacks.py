#!/usr/bin/env python3
"""
Training Callbacks for Monitoring and Evaluation

Provides:
- W&B logging integration
- Periodic energy evaluation during training
- Early stopping based on energy metrics
"""

import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

def timestamp():
    return datetime.now().strftime("%H:%M:%S")


class EnergyEvaluationCallback(TrainerCallback):
    """
    Periodic evaluation of energy metrics during training

    Monitors:
    - Compilation success rate
    - Energy reduction on held-out set
    - IPC improvement
    - EDP reduction

    Helps detect reward hacking early
    """

    def __init__(
        self,
        eval_model,
        eval_tokenizer,
        eval_dataset,
        sniper_root: str,
        pie_root: str = None,
        eval_every_n_steps: int = 100,
        max_eval_samples: int = 50,
        output_dir: str = None
    ):
        self.eval_model = eval_model
        self.eval_tokenizer = eval_tokenizer
        self.eval_dataset = eval_dataset
        self.sniper_root = Path(sniper_root)
        # sniper_root is {PROJECT_ROOT}/sniper/sniper; PIE_Dataset is at PROJECT_ROOT
        self.pie_root = Path(pie_root) if pie_root else self.sniper_root.parent.parent / 'PIE_Dataset'
        self.eval_every_n_steps = eval_every_n_steps
        self.max_eval_samples = max_eval_samples
        self.output_dir = Path(output_dir) if output_dir else None
        # Token 73594 in Qwen2.5-Coder vocab. Base model never generates <|im_end|> in
        # completion mode, so eos_token_id alone provides zero stopping signal.
        self.fence_token_id = eval_tokenizer.encode('```', add_special_tokens=False)[0]

        self.best_energy_reduction = -float('inf')
        self.steps_without_improvement = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs
    ):
        """Hook into TRL's standard evaluation to add energy metrics"""
        print(f"[{timestamp()}] DEBUG: on_evaluate called at step {state.global_step}, eval_every_n_steps={self.eval_every_n_steps}, check={state.global_step % self.eval_every_n_steps}")
        sys.stdout.flush()
        sys.stderr.flush()
        if state.global_step % self.eval_every_n_steps != 0:
            print(f"[{timestamp()}] DEBUG: Skipping energy eval (step check failed)")
            sys.stdout.flush()
            return control

        print(f"\n{'='*70}")
        print(f"[{timestamp()}] ENERGY EVALUATION @ Step {state.global_step}")
        print(f"{'='*70}")
        sys.stdout.flush()

        energy_metrics = self._evaluate_energy_metrics(state.global_step)
        print(f"[{timestamp()}] DEBUG: _evaluate_energy_metrics returned, updating metrics dict")

        # Add energy metrics to TRL's evaluation results (use eval_ prefix with underscores)
        metrics.update({
            "eval_energy_reduction": energy_metrics['mean_energy_reduction'],
            "eval_compilation_rate": energy_metrics['compilation_rate'],
            "eval_ipc_improvement": energy_metrics['mean_ipc_improvement'],
            "eval_edp_reduction": energy_metrics['mean_edp_reduction'],
            "eval_success_rate": energy_metrics['success_rate'],
            "eval_baseline_success_rate": energy_metrics['baseline_success_rate'],
            "eval_generated_success_rate": energy_metrics['generated_success_rate'],
            "eval_vs_gt_energy_reduction": energy_metrics['mean_vs_gt_reduction'],
            "eval_beats_gt_rate": energy_metrics['beats_gt_rate']
        })

        # Also log to W&B with eval/ prefix for consistency
        if state.is_world_process_zero:
            try:
                import wandb
                wandb.log({
                    "eval/compilation_rate": energy_metrics['compilation_rate'],
                    "eval/energy_reduction": energy_metrics['mean_energy_reduction'],
                    "eval/ipc_improvement": energy_metrics['mean_ipc_improvement'],
                    "eval/edp_reduction": energy_metrics['mean_edp_reduction'],
                    "eval/success_rate": energy_metrics['success_rate'],
                    "eval/baseline_success_rate": energy_metrics['baseline_success_rate'],
                    "eval/generated_success_rate": energy_metrics['generated_success_rate'],
                    "eval/vs_gt_energy_reduction": energy_metrics['mean_vs_gt_reduction'],
                    "eval/beats_gt_rate": energy_metrics['beats_gt_rate']
                }, step=state.global_step)
            except ImportError:
                pass

        logger.info(f"  Compilation rate: {energy_metrics['compilation_rate']:.2%}")
        logger.info(f"  Baseline success: {energy_metrics['baseline_success_rate']:.2%} ({int(energy_metrics['baseline_success_rate']*20)}/20)")
        logger.info(f"  Generated success: {energy_metrics['generated_success_rate']:.2%} ({int(energy_metrics['generated_success_rate']*20)}/20)")
        logger.info(f"  Both success (paired): {energy_metrics['success_rate']:.2%} ({int(energy_metrics['success_rate']*20)}/20)")
        logger.info(f"  Energy reduction: {energy_metrics['mean_energy_reduction']:.2%}")
        logger.info(f"  IPC improvement: {energy_metrics['mean_ipc_improvement']:.2%}")
        logger.info(f"  EDP reduction: {energy_metrics['mean_edp_reduction']:.2%}")
        logger.info(f"  Vs ground truth: {energy_metrics['mean_vs_gt_reduction']:.2%} (beats GT: {energy_metrics['beats_gt_rate']:.0%})")

        # Early warning for reward hacking
        if energy_metrics['success_rate'] < 0.6:
            logger.warning(f"\n⚠️  SUCCESS RATE DROPPED BELOW 60%!")
            logger.warning(f"  Model may be reward hacking (broken code with low energy)")
            logger.warning(f"  Consider stopping training or reducing learning rate")

        # Track improvement
        if energy_metrics['mean_energy_reduction'] > self.best_energy_reduction:
            self.best_energy_reduction = energy_metrics['mean_energy_reduction']
            self.steps_without_improvement = 0
            logger.info(f"  ✓ New best energy reduction!")
        else:
            self.steps_without_improvement += 1

        logger.info(f"{'='*70}\n")

        torch.cuda.empty_cache()
        print(f"[{timestamp()}] DEBUG: GPU cache cleared after energy evaluation")
        sys.stdout.flush()

        return control

    def _get_all_test_inputs(self, sample: Dict[str, Any]) -> list:
        """Load ALL test inputs for a sample.

        GRPO dataset: reads from test_input_path field (single input)
        SFT dataset: finds all test inputs via problem_id from PIE_Dataset
        """
        from typing import List

        if 'test_input_path' in sample:
            # GRPO: single test input path
            path = self.pie_root.parent / sample['test_input_path']
            if path.exists():
                return [path.read_text().strip()]
            logger.warning(f"test_input_path not found: {path}")
            return []

        if 'problem_id' in sample:
            # SFT: get ALL inputs from problem_id
            test_dir = self.pie_root / 'extracted_testcases' / 'merged_test_cases' / sample['problem_id']
            if test_dir.exists():
                inputs = sorted(test_dir.glob('input.*.txt'))
                return [f.read_text().strip() for f in inputs]

        return []

    def _evaluate_energy_metrics(self, global_step: int = 0) -> Dict[str, float]:
        """Run quick energy evaluation on held-out samples"""
        import json as _json
        logger.info(f"\n{'='*70}")
        logger.info(f"ENERGY EVALUATION @ STEP {global_step}")
        logger.info(f"{'='*70}")

        try:
            import evaluation
            SniperEvaluator = evaluation.SniperEvaluator
        except ImportError:
            from ..evaluation import SniperEvaluator

        evaluator = SniperEvaluator(str(self.sniper_root))
        # Deduplicate by baseline_code, then fixed-seed sample for cross-step comparability
        seen_codes = set()
        unique_indices = []
        for i in range(len(self.eval_dataset)):
            code = self.eval_dataset[i].get('baseline_code', self.eval_dataset[i].get('inefficient_code', ''))
            if code and code not in seen_codes:
                seen_codes.add(code)
                unique_indices.append(i)
        rng = np.random.default_rng(seed=42)
        n = min(self.max_eval_samples, len(unique_indices))
        selected = rng.choice(unique_indices, size=n, replace=False).tolist()
        eval_samples = self.eval_dataset.select(selected)
        logger.info(f"Evaluating {n} unique baseline samples (seed=42, {len(unique_indices)} unique in val)")

        compilation_successes = 0
        energy_reductions = []
        ipc_improvements = []
        edp_reductions = []
        total_successes = 0
        evaluated = 0
        generations = []
        baseline_successes = 0
        generated_successes = 0
        vs_gt_reductions = []
        beats_gt = 0

        for idx, sample in enumerate(eval_samples):
            print(f"[{timestamp()}] DEBUG: Evaluating sample {idx+1}/{len(eval_samples)}: {sample.get('problem_id', '?')}")
            sys.stdout.flush()

            # Support both SFT (inefficient_code) and GRPO (baseline_code) datasets
            baseline_code = sample.get('baseline_code', sample.get('inefficient_code', ''))
            test_inputs = self._get_all_test_inputs(sample)

            if not baseline_code:
                logger.warning(f"Skipping {sample.get('problem_id', '?')}: missing baseline_code")
                continue
            if not test_inputs:
                logger.warning(f"Skipping {sample.get('problem_id', '?')}: no test inputs")
                continue

            evaluated += 1
            print(f"[{timestamp()}] DEBUG: Generating code for {sample.get('problem_id', '?')} with {len(test_inputs)} test inputs")
            sys.stdout.flush()

            # Generate solution once
            prompt = self._create_prompt(sample)
            inputs = self.eval_tokenizer(prompt, return_tensors='pt').to(self.eval_model.device)

            with torch.no_grad():
                outputs = self.eval_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.2,
                    do_sample=True,
                    eos_token_id=[self.eval_tokenizer.eos_token_id, self.fence_token_id],
                    pad_token_id=self.eval_tokenizer.pad_token_id,
                    repetition_penalty=1.2
                )

            generated = self.eval_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            del outputs, inputs
            torch.cuda.empty_cache()

            code = self._extract_code(generated)
            print(f"[{timestamp()}] DEBUG: Generated code length: {len(code)} chars, measuring energy...")
            sys.stdout.flush()

            # Baseline: use pre-computed values from dataset if available (avoids redundant Sniper run)
            precomputed_baseline_energy = sample.get('baseline_energy', 0)
            print(f"[{timestamp()}]   DEBUG: Precomputed baseline_energy = {precomputed_baseline_energy}, will use: {precomputed_baseline_energy > 1e-9}")
            sys.stdout.flush()
            if precomputed_baseline_energy > 1e-9:
                baseline_result = {
                    'status': 'success',
                    'energy_joules': precomputed_baseline_energy,
                    'cycles': sample.get('baseline_cycles', 0),
                    'instructions': sample.get('baseline_instructions', 0)
                }
            else:
                # Fallback: run Sniper on baseline (for datasets without pre-computed values)
                print(f"[{timestamp()}]   DEBUG: No precomputed baseline, measuring {len(test_inputs)} inputs...")
                sys.stdout.flush()
                baseline_energies, baseline_cycles_list, baseline_instrs = [], [], []
                for input_idx, test_input in enumerate(test_inputs, 1):
                    print(f"[{timestamp()}]   DEBUG: Baseline Sniper {input_idx}/{len(test_inputs)}...")
                    sys.stdout.flush()
                    b_res = evaluator.measure_energy(baseline_code, test_input)
                    print(f"[{timestamp()}]   DEBUG: Baseline {input_idx} done: {b_res['status']}")
                    sys.stdout.flush()
                    if b_res['status'] == 'success' and b_res['energy_joules'] > 1e-9:
                        baseline_energies.append(b_res['energy_joules'])
                        baseline_cycles_list.append(b_res['cycles'])
                        baseline_instrs.append(b_res['instructions'])
                baseline_result = {
                    'status': 'success' if baseline_energies else 'failed',
                    'energy_joules': np.mean(baseline_energies) if baseline_energies else 0,
                    'cycles': int(np.mean(baseline_cycles_list)) if baseline_cycles_list else 0,
                    'instructions': int(np.mean(baseline_instrs)) if baseline_instrs else 0
                }

            # Generated: compile once, correctness check, Sniper only on correct inputs
            import subprocess as _sub, tempfile as _tmp, shutil as _shu
            generated_compile = False
            tests_passed = 0
            generated_energies, generated_cycles_list, generated_instrs = [], [], []
            compile_error_msg = ''

            _tdir = _tmp.mkdtemp(prefix='eval_gen_')
            try:
                _cpp = Path(_tdir) / 'sol.cpp'
                _cpp.write_text(code)
                _bin = Path(_tdir) / 'sol.bin'
                _cr = _sub.run(['g++', '-O3', '-std=c++17', '-static', str(_cpp), '-o', str(_bin)],
                               capture_output=True, text=True, timeout=10)
                generated_compile = _cr.returncode == 0
                if not generated_compile:
                    compile_error_msg = _cr.stderr[:200]
                else:
                    pid = sample.get('problem_id', '')
                    test_dir = self.pie_root / 'extracted_testcases' / 'merged_test_cases' / pid
                    input_files = sorted(test_dir.glob('input.*.txt')) if test_dir.exists() else []
                    for inf in input_files:
                        outf = test_dir / f"output.{inf.name.replace('input.', '')}"
                        if not outf.exists():
                            continue
                        try:
                            with open(inf) as inp:
                                r = _sub.run([str(_bin)], stdin=inp, capture_output=True, text=True, timeout=5)
                            if r.returncode != 0 or r.stdout.strip() != outf.read_text().strip():
                                continue
                        except Exception:
                            continue
                        tests_passed += 1
                        print(f"[{timestamp()}]   DEBUG: Running Sniper for test input {tests_passed}...")
                        sys.stdout.flush()
                        g_res = evaluator.measure_energy(code, inf.read_text().strip())
                        print(f"[{timestamp()}]   DEBUG: Sniper completed for input {tests_passed}: status={g_res['status']}")
                        sys.stdout.flush()
                        if g_res['status'] == 'success' and g_res['energy_joules'] > 1e-9:
                            generated_energies.append(g_res['energy_joules'])
                            generated_cycles_list.append(g_res['cycles'])
                            generated_instrs.append(g_res['instructions'])
            except Exception as e:
                logger.warning(f"Generated eval failed: {e}")
            finally:
                _shu.rmtree(_tdir, ignore_errors=True)

            generated_result = {
                'status': 'success' if generated_energies else ('compile_error' if not generated_compile else 'failed'),
                'energy_joules': np.mean(generated_energies) if generated_energies else 0,
                'cycles': int(np.mean(generated_cycles_list)) if generated_cycles_list else 0,
                'instructions': int(np.mean(generated_instrs)) if generated_instrs else 0
            }

            logger.debug(f"Sample {evaluated}: compiled={generated_compile}, passed={tests_passed}/{len(test_inputs)}, sniper_ok={len(generated_energies)}, baseline={'precomputed' if precomputed_baseline_energy > 1e-9 else 'simulated'}")

            gen_record = {
                'step': global_step,
                'problem_id': sample.get('problem_id', ''),
                'baseline_code': baseline_code,
                'generated_code': code,
                'optimized_code': sample.get('optimized_code', ''),
                'status': generated_result['status'],
                'compiled': generated_compile,
                'tests_passed': tests_passed,
                'num_inputs': len(test_inputs),
                'generated_success_count': len(generated_energies)
            }
            if compile_error_msg:
                gen_record['compile_error'] = compile_error_msg

            if generated_compile:
                compilation_successes += 1

            if baseline_result['status'] == 'success' and baseline_result['energy_joules'] > 1e-9:
                baseline_successes += 1

            if generated_result['status'] == 'success' and generated_result['energy_joules'] > 1e-9:
                generated_successes += 1
            else:
                logger.debug(f"Generated failed: {generated_result['status']}, E={generated_result.get('energy_joules', 0):.6f}")

            if (baseline_result['status'] == 'success' and
                generated_result['status'] == 'success' and
                baseline_result['energy_joules'] > 1e-9 and
                generated_result['energy_joules'] > 1e-9 and
                baseline_result['cycles'] > 0 and
                generated_result['cycles'] > 0):

                total_successes += 1

                e_reduction = (
                    (baseline_result['energy_joules'] - generated_result['energy_joules']) /
                    baseline_result['energy_joules']
                )
                energy_reductions.append(e_reduction)

                baseline_ipc = baseline_result['instructions'] / baseline_result['cycles']
                generated_ipc = generated_result['instructions'] / generated_result['cycles'] if generated_result['cycles'] > 0 else 0
                ipc_improvements.append((generated_ipc - baseline_ipc) / baseline_ipc if baseline_ipc > 0 else 0)

                baseline_edp = baseline_result['energy_joules'] * baseline_result['cycles']
                generated_edp = generated_result['energy_joules'] * generated_result['cycles']
                edp_reductions.append((baseline_edp - generated_edp) / baseline_edp)

                gen_record['energy_reduction'] = e_reduction
                gen_record['baseline_energy'] = baseline_result['energy_joules']
                gen_record['baseline_avg_cycles'] = baseline_result['cycles']
                gen_record['baseline_avg_instructions'] = baseline_result['instructions']
                gen_record['baseline_avg_ipc'] = baseline_ipc
                gen_record['generated_energy'] = generated_result['energy_joules']
                gen_record['generated_avg_cycles'] = generated_result['cycles']
                gen_record['generated_avg_instructions'] = generated_result['instructions']
                gen_record['generated_avg_ipc'] = generated_ipc
                gen_record['baseline_edp'] = baseline_edp
                gen_record['generated_edp'] = generated_edp
                gen_record['edp_reduction'] = (baseline_edp - generated_edp) / baseline_edp
                gen_record['speedup'] = baseline_result['cycles'] / generated_result['cycles'] if generated_result['cycles'] > 0 else 0
                gen_record['ipc_improvement_pct'] = (generated_ipc - baseline_ipc) / baseline_ipc * 100 if baseline_ipc > 0 else 0

                # Ground truth optimized metrics from dataset
                gt_energy = sample.get('optimized_energy', 0)
                if gt_energy > 1e-9:
                    vs_gt = (gt_energy - generated_result['energy_joules']) / gt_energy
                    vs_gt_reductions.append(vs_gt)
                    gen_record['optimized_energy'] = gt_energy
                    gen_record['optimized_cycles'] = sample.get('optimized_cycles', 0)
                    gen_record['optimized_ipc'] = sample.get('optimized_ipc', 0)
                    gen_record['optimized_edp'] = sample.get('optimized_edp', 0)
                    gen_record['vs_gt_reduction'] = vs_gt
                    if vs_gt > 0:
                        beats_gt += 1
            else:
                gen_record['error_msg'] = generated_result.get('error_msg', '')

            generations.append(gen_record)

        print(f"[{timestamp()}] DEBUG: Evaluation loop complete. Evaluated {evaluated} samples, compiled {compilation_successes}, total_successes {total_successes}")
        # Save generated code for manual inspection
        if self.output_dir and generations:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            gen_file = self.output_dir / f'eval_generations_step{global_step}.jsonl'
            with open(gen_file, 'w') as f:
                for rec in generations:
                    f.write(_json.dumps(rec) + '\n')
            logger.info(f"  Saved {len(generations)} generations to {gen_file}")

        if evaluated == 0:
            logger.warning("No samples could be evaluated - all missing baseline_code or test_input")

        logger.info(f"Evaluation complete: {evaluated} samples processed")
        logger.info(f"  Baseline successes: {baseline_successes}/{evaluated}")
        logger.info(f"  Generated successes: {generated_successes}/{evaluated}")
        logger.info(f"  Both succeeded (paired): {total_successes}/{evaluated}")

        print(f"[{timestamp()}] DEBUG: Computing final metrics and returning")
        return {
            'compilation_rate': compilation_successes / evaluated if evaluated > 0 else 0,
            'success_rate': total_successes / evaluated if evaluated > 0 else 0,
            'baseline_success_rate': baseline_successes / evaluated if evaluated > 0 else 0,
            'generated_success_rate': generated_successes / evaluated if evaluated > 0 else 0,
            'mean_energy_reduction': np.mean(energy_reductions) if energy_reductions else 0,
            'mean_ipc_improvement': np.mean(ipc_improvements) if ipc_improvements else 0,
            'mean_edp_reduction': np.mean(edp_reductions) if edp_reductions else 0,
            'mean_vs_gt_reduction': np.mean(vs_gt_reductions) if vs_gt_reductions else 0,
            'beats_gt_rate': beats_gt / len(vs_gt_reductions) if vs_gt_reductions else 0
        }

    def _create_prompt(self, sample: Dict[str, Any]) -> str:
        """Create prompt matching OPTIMIZATION_TEMPLATE from sft_train_trl.py.
        Must be identical to the training format so the model generates correctly.
        """
        baseline_code = sample.get('baseline_code', sample.get('inefficient_code', ''))
        return (
            f"This is an energy inefficient program we want to optimize to score 10/10.\n"
            f"### Program:\n{baseline_code}\n\n"
            f"### Energy Optimized Version with score 10/10:\n```cpp\n"
        )

    def _extract_code(self, generation: str) -> str:
        """Extract C++ code from generation.

        Prompt ends with ```cpp so the model outputs code directly.
        Expected generation: code...\n```
        split('```') produces ['code...', ''] -- code is at index 0.
        """
        if '```cpp' in generation:
            # Model repeated the opening tag (e.g. ```cpp\ncode\n```)
            code = generation.split('```cpp')[1].split('```')[0]
        elif '```' in generation:
            # Normal case: generation starts with code, ends with ```
            code = generation.split('```')[0]
        else:
            code = generation
        return code.strip()


class WandbInitCallback(TrainerCallback):
    """
    Initialize W&B with proper config tracking

    Automatically logs:
    - All training hyperparameters
    - Model architecture
    - Dataset statistics
    - System info
    """

    def __init__(
        self,
        project: str = "energy-code-generation",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict] = None
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.config = config or {}
        self.initialized = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Initialize W&B at training start"""
        if self.initialized or not state.is_world_process_zero:
            return

        try:
            import wandb

            # Merge training args with custom config
            full_config = {
                **vars(args),
                **self.config
            }

            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags,
                config=full_config,
                resume="allow"
            )

            logger.info(f"✓ W&B initialized: {wandb.run.url}")
            self.initialized = True

        except ImportError:
            logger.warning("W&B not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")


def get_wandb_callback(
    stage: str,
    model_name: str = "qwen-14b",
    additional_config: Optional[Dict] = None
) -> WandbInitCallback:
    """
    Factory for creating stage-specific W&B callbacks

    Args:
        stage: Training stage (syntax, optimization, dpo, grpo)
        model_name: Base model name
        additional_config: Extra config to log

    Returns:
        WandbInitCallback configured for the stage
    """
    return WandbInitCallback(
        project="energy-code-generation",
        name=f"{model_name}-{stage}",
        tags=[stage, model_name, "trl"],
        config=additional_config or {}
    )
