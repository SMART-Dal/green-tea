#!/usr/bin/env python3
"""
Dataset Preprocessing for Energy-Efficient Code Generation

Loads pre-aggregated solution metrics from analysis/ and creates training datasets:
- Stratified problem-level splitting (80/10/10) by energy difficulty
- Interpolation validation (10% solutions held out from training problems)
- Diverse pair generation (80% algorithmic jumps + 20% micro-optimizations)
- GRPO prompts for online RL with EDP-based rewards

Usage:
    python dataset_preprocessing.py \
        --output-dir data \
        --min-energy-reduction 0.20 \
        --min-solutions-per-group 2 \
        --create-grpo

Output:
    data/sft_pairs_train.jsonl       # SFT training pairs (extrapolation)
    data/sft_pairs_val.jsonl         # Validation pairs (extrapolation)
    data/interpolation_val.jsonl     # Interpolation validation pairs
    data/sft_pairs_test.jsonl        # Test pairs
    data/grpo_train.jsonl            # GRPO prompts (no solutions)
    data/grpo_val.jsonl
    data/dataset_stats.json          # Statistics (includes EDP metrics)
    data/problem_split.json          # Stratified problem split
"""

import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple
import numpy as np
from tqdm import tqdm
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnergyDatasetPreprocessor:
    """
    Preprocesses raw energy measurements into training pairs
    """

    def __init__(
        self,
        cache_dir: str,
        output_dir: str,
        min_energy_reduction: float = 0.20,
        min_solutions_per_group: int = 2,
        seed: int = 42
    ):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.min_energy_reduction = min_energy_reduction
        self.min_solutions_per_group = min_solutions_per_group
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized preprocessor:")
        logger.info(f"  Cache dir: {self.cache_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Min energy reduction: {min_energy_reduction * 100}%")

        # Load code lookup mapping
        self.code_lookup = self._load_code_lookup()

    def _load_code_lookup(self) -> Dict[str, str]:
        """
        Load code_hash -> code_string mapping from PIE dataset

        Returns:
            Dictionary mapping code_hash to actual code string
        """
        logger.info("Loading code lookup from PIE dataset...")

        pie_dataset_dir = Path(__file__).parent.parent / "PIE_Dataset"
        code_map = {}

        for dataset_file in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
            file_path = pie_dataset_dir / dataset_file
            if not file_path.exists():
                logger.warning(f"PIE dataset file not found: {file_path}")
                continue

            with open(file_path) as f:
                for line in tqdm(f, desc=f"Loading {dataset_file}"):
                    try:
                        data = json.loads(line)
                        for code_key in ['src_code', 'tgt_code']:
                            if code_key in data:
                                code = data[code_key]
                                code_hash = hashlib.md5(code.encode('utf-8')).hexdigest()[:12]
                                code_map[code_hash] = code
                    except:
                        continue

        logger.info(f"Loaded {len(code_map):,} unique code strings")
        return code_map

    def load_energy_measurements(self) -> List[Dict[str, Any]]:
        """
        Load pre-aggregated solution metrics from analysis/solution_metrics.jsonl

        This uses the already processed data which:
        - Aggregates across test inputs per (problem_id, code_hash)
        - Includes IPC, EDP, and all necessary metrics
        - Is much faster to load (17MB vs 731MB)

        Returns:
            List of solution dictionaries with aggregated energy measurements
        """
        logger.info("Loading pre-aggregated solution metrics from analysis...")

        # Check for pre-processed file in analysis directory
        analysis_dir = Path(__file__).parent.parent / "analysis"
        solution_metrics_file = analysis_dir / "solution_metrics.jsonl"

        if not solution_metrics_file.exists():
            raise FileNotFoundError(
                f"Pre-processed solution_metrics.jsonl not found at {solution_metrics_file}\n"
                f"Please run analysis/analysis.py first to generate aggregated metrics."
            )

        solutions = []
        failed_loads = 0

        logger.info(f"Reading from {solution_metrics_file}")

        with open(solution_metrics_file) as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading solution metrics"), 1):
                try:
                    data = json.loads(line)

                    # Extract fields
                    problem_id = data.get('problem_id')
                    code_hash = data.get('code_hash')
                    avg_energy = data.get('avg_energy')

                    # Filter invalid or zero-energy solutions (artifacts only)
                    if not problem_id or not code_hash or avg_energy is None or avg_energy == 0:
                        failed_loads += 1
                        continue

                    solutions.append({
                        'problem_id': problem_id,
                        'code_hash': code_hash,
                        'execution_id': f"{problem_id}_{code_hash}_avg",
                        'energy_joules': avg_energy,
                        'energy_std': data.get('std_energy', 0),
                        'power_watts': data.get('avg_power', 0),
                        'runtime_seconds': data.get('avg_runtime', 0),
                        'cycles': int(data.get('avg_cycles', 0)),
                        'instructions': int(data.get('avg_instructions', 0)),
                        'ipc': data.get('avg_ipc', 0),
                        'edp': data.get('avg_edp', 0),
                        'edp_std': data.get('std_edp', 0),
                        'num_test_inputs': data.get('num_test_inputs', 0),
                        'code': data.get('code', '')  # If available from analysis
                    })

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
                    failed_loads += 1
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    failed_loads += 1

        logger.info(f"Loaded {len(solutions):,} valid solution metrics")
        logger.info(f"Failed to load: {failed_loads:,} lines")
        logger.info(f"  Average test inputs per solution: {np.mean([s['num_test_inputs'] for s in solutions]):.1f}")

        return solutions

    def create_problem_split(
        self,
        solutions: List[Dict[str, Any]],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Split problems using STRATIFIED sampling based on energy difficulty

        CRITICAL:
        - Split by problem_id BEFORE creating pairs (prevents leakage)
        - Stratify by mean energy to ensure balanced difficulty distribution
        - Prevents all hard/easy problems ending up in one split

        Args:
            solutions: List of all solutions
            train_ratio: Fraction for training (default: 0.80)
            val_ratio: Fraction for validation (default: 0.10)
            test_ratio: Fraction for test (default: 0.10)

        Returns:
            (train_pids, val_pids, test_pids) - Sets of problem IDs
        """
        logger.info("Creating STRATIFIED problem-level data split...")

        # Calculate mean energy per problem (difficulty proxy)
        problem_energies = defaultdict(list)
        for sol in solutions:
            if sol['energy_joules'] > 0:  # Filter zeros
                problem_energies[sol['problem_id']].append(sol['energy_joules'])

        problem_stats = []
        for pid, energies in problem_energies.items():
            if energies:  # Skip if no valid energies
                problem_stats.append({
                    'problem_id': pid,
                    'mean_energy': np.mean(energies),
                    'num_solutions': len(energies)
                })

        logger.info(f"Found {len(problem_stats)} problems with valid energy measurements")

        # Sort by difficulty (mean energy)
        problem_stats.sort(key=lambda x: x['mean_energy'])

        # Stratified split: divide sorted list into small bins and sample from each
        train_pids, val_pids, test_pids = [], [], []
        chunk_size = 10  # Small bins ensure mixing of difficulties

        for i in range(0, len(problem_stats), chunk_size):
            chunk = problem_stats[i:i + chunk_size]
            if not chunk:
                continue

            # Shuffle within difficulty bin
            random.shuffle(chunk)

            # Allocate based on ratios
            n_chunk = len(chunk)
            n_train = int(n_chunk * train_ratio)
            n_val = int(n_chunk * val_ratio)

            c_train = chunk[:n_train]
            c_val = chunk[n_train:n_train + n_val]
            c_test = chunk[n_train + n_val:]

            train_pids.extend([p['problem_id'] for p in c_train])
            val_pids.extend([p['problem_id'] for p in c_val])
            test_pids.extend([p['problem_id'] for p in c_test])

        # Convert to sets
        train_pids, val_pids, test_pids = set(train_pids), set(val_pids), set(test_pids)

        # Verify no overlap
        assert len(train_pids & val_pids) == 0, "Train/Val overlap detected!"
        assert len(train_pids & test_pids) == 0, "Train/Test overlap detected!"
        assert len(val_pids & test_pids) == 0, "Val/Test overlap detected!"

        n_total = len(train_pids) + len(val_pids) + len(test_pids)
        logger.info(f"Stratified split results:")
        logger.info(f"  Train: {len(train_pids):,} problems ({len(train_pids)/n_total*100:.1f}%)")
        logger.info(f"  Val:   {len(val_pids):,} problems ({len(val_pids)/n_total*100:.1f}%)")
        logger.info(f"  Test:  {len(test_pids):,} problems ({len(test_pids)/n_total*100:.1f}%)")

        # Verify stratification worked
        train_energies = [p['mean_energy'] for p in problem_stats if p['problem_id'] in train_pids]
        val_energies = [p['mean_energy'] for p in problem_stats if p['problem_id'] in val_pids]
        test_energies = [p['mean_energy'] for p in problem_stats if p['problem_id'] in test_pids]

        logger.info(f"\nEnergy distribution check:")
        logger.info(f"  Train: mean={np.mean(train_energies):.4f}J, std={np.std(train_energies):.4f}J")
        logger.info(f"  Val:   mean={np.mean(val_energies):.4f}J, std={np.std(val_energies):.4f}J")
        logger.info(f"  Test:  mean={np.mean(test_energies):.4f}J, std={np.std(test_energies):.4f}J")

        # Save split for reproducibility
        split_file = self.output_dir / "problem_split.json"
        with open(split_file, 'w') as f:
            json.dump({
                'train': sorted(list(train_pids)),
                'val': sorted(list(val_pids)),
                'test': sorted(list(test_pids)),
                'seed': self.seed,
                'stratified': True,
                'train_mean_energy': float(np.mean(train_energies)),
                'val_mean_energy': float(np.mean(val_energies)),
                'test_mean_energy': float(np.mean(test_energies))
            }, f, indent=2)

        logger.info(f"Saved stratified problem split to {split_file}")

        return train_pids, val_pids, test_pids

    def calculate_energy_scores(
        self,
        solutions_group: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate 1-10 energy efficiency scores for a group

        Score 10 = most efficient (lowest energy)
        Score 1 = least efficient (highest energy)

        Args:
            solutions_group: Solutions for same (problem, test_input)

        Returns:
            Solutions with added 'energy_score' field
        """
        # Sort by energy (ascending)
        sorted_solutions = sorted(solutions_group, key=lambda x: x['energy_joules'])

        # Assign percentile-based scores: lowest energy = 10, highest = 1
        n = len(sorted_solutions)
        for i, sol in enumerate(sorted_solutions):
            percentile = i / (n - 1) if n > 1 else 0
            sol['energy_score'] = max(1, min(10, round((1 - percentile) * 9) + 1))

        return sorted_solutions

    def create_optimization_pairs(
        self,
        solutions: List[Dict[str, Any]],
        problem_ids: Set[str],
        is_training: bool = False,
        interpolation_ratio: float = 0.10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create optimization pairs with diverse strategies

        Strategy A (80%): Algorithmic jumps (best-vs-worst)
        Strategy B (20%): Micro-optimizations (mid-tier vs top-tier)

        Args:
            solutions: All solutions (already aggregated)
            problem_ids: Problem IDs to include
            is_training: If True, hold out solutions for interpolation validation
            interpolation_ratio: Fraction of solutions to hold out (default: 0.10)

        Returns:
            (main_pairs, interpolation_pairs) if is_training else (main_pairs, [])
        """
        logger.info(f"Creating optimization pairs for {len(problem_ids)} problems...")

        # STEP 1: Group by problem_id for comparison
        # (Solutions are already averaged in load_energy_measurements)
        problem_groups = defaultdict(list)
        for sol in solutions:
            if sol['problem_id'] in problem_ids:
                problem_groups[sol['problem_id']].append(sol)

        logger.info(f"Grouped into {len(problem_groups):,} problems")

        main_pairs = []
        interpolation_pairs = []
        skipped_groups = 0

        for problem_id, solutions_group in tqdm(
            problem_groups.items(),
            desc="Generating pairs"
        ):
            if len(solutions_group) < self.min_solutions_per_group:
                skipped_groups += 1
                continue

            scored_solutions = self.calculate_energy_scores(solutions_group)

            E_worst = scored_solutions[-1]['energy_joules']
            E_best = scored_solutions[0]['energy_joules']

            if E_worst / E_best < (1 + self.min_energy_reduction):
                skipped_groups += 1
                continue

            # Interpolation validation: hold out 10% of middle-tier solutions from training
            holdout_solutions = []
            if is_training and len(scored_solutions) >= 10:
                n_holdout = max(1, int(len(scored_solutions) * interpolation_ratio))
                mid_start = len(scored_solutions) // 3
                mid_end = 2 * len(scored_solutions) // 3
                candidates = scored_solutions[mid_start:mid_end]
                if len(candidates) >= n_holdout:
                    holdout_solutions = random.sample(candidates, n_holdout)
                    scored_solutions = [s for s in scored_solutions if s not in holdout_solutions]

            # Strategy A (80%): Algorithmic jumps (best-vs-worst)
            # Sample one per score level to avoid boundary bias from sorted slicing
            ineff_by_score = defaultdict(list)
            eff_by_score = defaultdict(list)
            for s in scored_solutions:
                if s['energy_score'] <= 4:
                    ineff_by_score[s['energy_score']].append(s)
                elif s['energy_score'] >= 7:
                    eff_by_score[s['energy_score']].append(s)

            baselines = [random.choice(v) for _, v in sorted(ineff_by_score.items())]  # score 1,2,3,4
            optimized = [random.choice(v) for _, v in sorted(eff_by_score.items(), reverse=True)]  # score 10,9,8,7

            strategy_a_pairs = []
            for baseline in baselines:
                for opt in optimized:
                    pair = self._make_pair(problem_id, baseline, opt)
                    if pair:
                        strategy_a_pairs.append(pair)

            # Strategy B (20%): Micro-optimizations (mid-tier vs top-tier)
            mid_by_score = defaultdict(list)
            for s in scored_solutions:
                if 5 <= s['energy_score'] <= 6:
                    mid_by_score[s['energy_score']].append(s)
            mid_tier = [random.choice(v) for _, v in sorted(mid_by_score.items())]

            strategy_b_pairs = []
            for baseline in mid_tier:
                for opt in optimized:
                    if baseline['energy_score'] < opt['energy_score']:
                        pair = self._make_pair(problem_id, baseline, opt)
                        if pair:
                            strategy_b_pairs.append(pair)

            # Mix strategies: 80% A, 20% B (at least 1 from A if non-empty)
            n_a = max(1, int(len(strategy_a_pairs) * 0.8)) if strategy_a_pairs else 0
            n_b = len(strategy_a_pairs) - n_a
            selected_pairs = strategy_a_pairs[:n_a] + strategy_b_pairs[:max(0, n_b)]
            main_pairs.extend(selected_pairs)

            # Create interpolation pairs from held-out solutions
            if holdout_solutions and optimized:
                for holdout in holdout_solutions:
                    for opt in optimized[:2]:
                        pair = self._make_pair(problem_id, holdout, opt)
                        if pair:
                            interpolation_pairs.append(pair)

        logger.info(f"Created {len(main_pairs):,} optimization pairs")
        logger.info(f"  Strategy A (algorithmic jumps): ~80%")
        logger.info(f"  Strategy B (micro-optimizations): ~20%")
        if interpolation_pairs:
            logger.info(f"Created {len(interpolation_pairs):,} interpolation validation pairs")
        logger.info(f"Skipped {skipped_groups:,} groups (insufficient diversity or improvement)")

        return main_pairs, interpolation_pairs

    def _make_pair(self, problem_id: str, baseline: Dict, optimized: Dict) -> Dict[str, Any]:
        """Helper to create a valid optimization pair"""
        energy_reduction = (baseline['energy_joules'] - optimized['energy_joules']) / baseline['energy_joules']

        if energy_reduction >= self.min_energy_reduction:
            baseline_code = self.code_lookup.get(baseline['code_hash'], '')
            optimized_code = self.code_lookup.get(optimized['code_hash'], '')

            baseline_cycles = baseline['cycles']
            optimized_cycles = optimized['cycles']
            baseline_runtime = baseline_cycles / 1.5e9 if baseline_cycles > 0 else 0
            optimized_runtime = optimized_cycles / 1.5e9 if optimized_cycles > 0 else 0
            baseline_ipc = baseline['instructions'] / baseline_cycles if baseline_cycles > 0 else 0
            optimized_ipc = optimized['instructions'] / optimized_cycles if optimized_cycles > 0 else 0
            baseline_edp = baseline['energy_joules'] * baseline_runtime
            optimized_edp = optimized['energy_joules'] * optimized_runtime

            return {
                'problem_id': problem_id,
                'baseline_execution_id': baseline['execution_id'],
                'baseline_code_hash': baseline['code_hash'],
                'inefficient_code': baseline_code,
                'baseline_energy': baseline['energy_joules'],
                'baseline_power': baseline.get('power_watts', 0),
                'baseline_runtime': baseline_runtime,
                'baseline_edp': baseline_edp,
                'baseline_ipc': baseline_ipc,
                'baseline_cycles': baseline_cycles,
                'baseline_instructions': baseline['instructions'],
                'baseline_score': baseline['energy_score'],
                'optimized_execution_id': optimized['execution_id'],
                'optimized_code_hash': optimized['code_hash'],
                'optimized_code': optimized_code,
                'optimized_energy': optimized['energy_joules'],
                'optimized_power': optimized.get('power_watts', 0),
                'optimized_runtime': optimized_runtime,
                'optimized_edp': optimized_edp,
                'optimized_ipc': optimized_ipc,
                'optimized_cycles': optimized_cycles,
                'optimized_instructions': optimized['instructions'],
                'optimized_score': optimized['energy_score'],
                'energy_reduction_pct': energy_reduction * 100,
                'speedup': baseline_cycles / optimized_cycles if optimized_cycles > 0 else 1.0,
                'edp_reduction_pct': ((baseline_edp - optimized_edp) / baseline_edp * 100) if baseline_edp > 0 else 0,
                'ipc_improvement': optimized_ipc - baseline_ipc
            }
        return None

    def calculate_statistics(
        self,
        pairs: List[Dict[str, Any]],
        split_name: str
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive dataset statistics

        Args:
            pairs: List of optimization pairs
            split_name: "train", "val", or "test"

        Returns:
            Statistics dictionary
        """
        if not pairs:
            return {}

        energy_reductions = [p['energy_reduction_pct'] for p in pairs]
        speedups = [p['speedup'] for p in pairs]
        baseline_energies = [p['baseline_energy'] for p in pairs]
        optimized_energies = [p['optimized_energy'] for p in pairs]

        stats = {
            'split': split_name,
            'num_pairs': len(pairs),
            'num_unique_problems': len(set(p['problem_id'] for p in pairs)),
            'energy_reduction': {
                'mean': np.mean(energy_reductions),
                'std': np.std(energy_reductions),
                'min': np.min(energy_reductions),
                'max': np.max(energy_reductions),
                'median': np.median(energy_reductions),
                'p25': np.percentile(energy_reductions, 25),
                'p75': np.percentile(energy_reductions, 75),
                'p95': np.percentile(energy_reductions, 95)
            },
            'speedup': {
                'mean': np.mean(speedups),
                'std': np.std(speedups),
                'min': np.min(speedups),
                'max': np.max(speedups),
                'median': np.median(speedups)
            },
            'baseline_energy': {
                'mean': np.mean(baseline_energies),
                'median': np.median(baseline_energies),
                'min': np.min(baseline_energies),
                'max': np.max(baseline_energies)
            },
            'optimized_energy': {
                'mean': np.mean(optimized_energies),
                'median': np.median(optimized_energies),
                'min': np.min(optimized_energies),
                'max': np.max(optimized_energies)
            }
        }

        return stats

    def _assign_scores_to_all(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign energy_score to all solutions based on per-problem percentiles"""
        problem_groups = defaultdict(list)
        for sol in solutions:
            problem_groups[sol['problem_id']].append(sol)

        scored = []
        for problem_id, group in problem_groups.items():
            scored.extend(self.calculate_energy_scores(group))

        return scored

    def _get_median_test_input_path(self, problem_id: str) -> str:
        """Find the median-sized test input for representative energy measurement."""
        pie_dataset_dir = Path(__file__).parent.parent / "PIE_Dataset"
        test_dir = pie_dataset_dir / "extracted_testcases" / "merged_test_cases" / problem_id
        
        if not test_dir.exists():
            return ""
            
        inputs = list(test_dir.glob("input.*.txt"))
        if not inputs:
            return ""
            
        # Sort by size to find median complexity
        inputs.sort(key=lambda p: p.stat().st_size)
        median_input = inputs[len(inputs) // 2]
        
        # Return path relative to project root for portability
        return str(median_input.relative_to(pie_dataset_dir.parent))

    def create_grpo_from_all_inefficient(
        self,
        solutions: List[Dict[str, Any]],
        problem_ids: Set[str],
        max_score: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Create GRPO dataset from ALL inefficient solutions in problem set
        """
        grpo_examples = []

        for solution in tqdm(solutions, desc=f"Creating GRPO examples (score ≤ {max_score})"):
            if solution['problem_id'] not in problem_ids:
                continue

            if solution.get('energy_score', 0) > max_score or solution.get('energy_score', 0) == 0:
                continue

            code = self.code_lookup.get(solution['code_hash'], '')
            if not code:
                continue

            problem_id = solution['problem_id']
            cycles = solution.get('cycles', 0)
            runtime = cycles / 1.5e9 if cycles > 0 else 0
            edp = solution['energy_joules'] * runtime
            ipc = solution.get('ipc', 0)
            
            # Find representative test input
            test_input_path = self._get_median_test_input_path(problem_id)

            grpo_examples.append({
                'problem_id': problem_id,
                'baseline_code': code,
                'baseline_code_hash': solution['code_hash'],
                'baseline_energy': solution['energy_joules'],
                'baseline_runtime': runtime,
                'baseline_edp': edp,
                'baseline_ipc': ipc,
                'baseline_power': solution.get('power_watts', 0),
                'baseline_cycles': cycles,
                'baseline_instructions': solution.get('instructions', 0),
                'energy_score': solution.get('energy_score', 5),
                'num_test_inputs': solution.get('num_test_inputs', 0),
                'test_input_path': test_input_path
            })

        return grpo_examples

    def save_datasets(
        self,
        solutions: List[Dict[str, Any]],
        train_pairs: List[Dict[str, Any]],
        val_pairs: List[Dict[str, Any]],
        test_pairs: List[Dict[str, Any]],
        train_pids: Set[str],
        val_pids: Set[str],
        test_pids: Set[str],
        interpolation_pairs: List[Dict[str, Any]] = None,
        create_grpo: bool = False
    ):
        """
        Save processed datasets to JSONL files

        Args:
            train_pairs: Training pairs
            val_pairs: Validation pairs
            test_pairs: Test pairs
            create_grpo: If True, also create GRPO-format datasets
        """
        logger.info("Saving processed datasets...")

        datasets = [
            ('sft_pairs_train.jsonl', train_pairs),
            ('sft_pairs_val.jsonl', val_pairs),
            ('sft_pairs_test.jsonl', test_pairs)
        ]

        if interpolation_pairs:
            datasets.append(('interpolation_val.jsonl', interpolation_pairs))

        for filename, data in datasets:
            if not data:
                logger.warning(f"Skipping empty dataset: {filename}")
                continue

            output_file = self.output_dir / filename
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')

            logger.info(f"Saved {len(data):,} samples to {output_file}")

        # Create GRPO-format datasets if requested
        if create_grpo:
            logger.info("\nCreating GRPO-format datasets from all inefficient solutions...")

            scored_solutions = self._assign_scores_to_all(solutions)

            grpo_train = self.create_grpo_from_all_inefficient(scored_solutions, train_pids, max_score=5)
            grpo_val = self.create_grpo_from_all_inefficient(scored_solutions, val_pids, max_score=7)
            grpo_test = self.create_grpo_from_all_inefficient(scored_solutions, test_pids, max_score=7)

            grpo_datasets = [
                ('grpo_train.jsonl', grpo_train),
                ('grpo_val.jsonl', grpo_val),
                ('grpo_test.jsonl', grpo_test)
            ]

            for filename, data in grpo_datasets:
                if not data:
                    continue

                output_file = self.output_dir / filename
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')

                logger.info(f"Saved {len(data):,} GRPO examples to {output_file}")

            logger.info(f"\n  GRPO strategy: All inefficient solutions from problems")
            logger.info(f"  Train (score ≤ 5): {len(grpo_train):,} examples ({len(grpo_train)/len(train_pairs):.1f}x SFT pairs)")
            logger.info(f"  Val   (score ≤ 7): {len(grpo_val):,} examples (tests broader difficulty range)")
            logger.info(f"  Test  (score ≤ 7): {len(grpo_test):,} examples")

        # Save comprehensive statistics
        stats = {
            'train': self.calculate_statistics(train_pairs, 'train'),
            'val': self.calculate_statistics(val_pairs, 'val'),
            'test': self.calculate_statistics(test_pairs, 'test'),
            'total_pairs': len(train_pairs) + len(val_pairs) + len(test_pairs)
        }

        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

    def process(self, create_grpo: bool = False):
        """
        Main processing pipeline

        1. Load energy measurements
        2. Create problem split (strict leakage prevention)
        3. Generate optimization pairs (SFT)
        4. Save datasets

        Args:
            create_grpo: If True, also create GRPO-format datasets
        """
        logger.info("=" * 70)
        logger.info("DATASET PREPROCESSING PIPELINE")
        logger.info("=" * 70)

        # Step 1: Load measurements (filter energy == 0)
        solutions = self.load_energy_measurements()
        solutions = [s for s in solutions if s['energy_joules'] > 0]
        logger.info(f"Filtered to {len(solutions):,} valid solutions (energy > 0)")

        # Step 2: Problem-level split (extrapolation: prevent leakage)
        train_pids, val_pids, test_pids = self.create_problem_split(solutions)

        # Step 3: Create optimization pairs for each split
        logger.info("\nGenerating training pairs...")
        train_pairs, interpolation_pairs = self.create_optimization_pairs(solutions, train_pids, is_training=True)

        logger.info("\nGenerating validation pairs...")
        val_pairs, _ = self.create_optimization_pairs(solutions, val_pids)

        logger.info("\nGenerating test pairs...")
        test_pairs, _ = self.create_optimization_pairs(solutions, test_pids)

        # Step 4: Save datasets (Replay buffer removed per user request)
        self.save_datasets(
            solutions,
            train_pairs, val_pairs, test_pairs,
            train_pids, val_pids, test_pids,
            interpolation_pairs=interpolation_pairs,
            create_grpo=create_grpo
        )

        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Training pairs:          {len(train_pairs):,}")
        logger.info(f"Validation pairs:        {len(val_pairs):,}")
        logger.info(f"Interpolation val pairs: {len(interpolation_pairs):,}")
        logger.info(f"Test pairs:              {len(test_pairs):,}")
        logger.info(f"Total samples:           {len(train_pairs) + len(val_pairs) + len(test_pairs):,}")
        if create_grpo:
            logger.info(f"\nGRPO datasets also created (grpo_train.jsonl, grpo_val.jsonl, grpo_test.jsonl)")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess energy measurements into training pairs'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='../pie_energy_cache',
        help='Path to energy cache directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for processed datasets'
    )
    parser.add_argument(
        '--min-energy-reduction',
        type=float,
        default=0.10,
        help='Minimum energy reduction for pairs (default: 0.10 = 10%%)'
    )
    parser.add_argument(
        '--min-solutions-per-group',
        type=int,
        default=2,
        help='Minimum solutions per (problem, input) group (default: 2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--create-grpo',
        action='store_true',
        help='Also create GRPO-format datasets (prompts only, no solutions)'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = EnergyDatasetPreprocessor(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        min_energy_reduction=args.min_energy_reduction,
        min_solutions_per_group=args.min_solutions_per_group,
        seed=args.seed
    )

    # Run processing pipeline
    preprocessor.process(create_grpo=args.create_grpo)


if __name__ == '__main__':
    main()
