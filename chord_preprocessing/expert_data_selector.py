#!/usr/bin/env python3
"""
Expert Data Selector for Energy-Efficient Code Generation

This module implements intelligent expert data selection strategies for Trinity-RFT CHORD,
focusing on high-quality energy-efficient code transformations that serve as expert
demonstrations for the CHORD algorithm.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import math
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExpertSelectionStrategy(Enum):
    """Strategies for selecting expert data."""
    TOP_ENERGY_REDUCTION = "top_energy_reduction"
    BALANCED_MULTI_OBJECTIVE = "balanced_multi_objective"
    PARETO_OPTIMAL = "pareto_optimal"
    DIVERSE_EXCELLENCE = "diverse_excellence"
    DIFFICULTY_AWARE = "difficulty_aware"


@dataclass
class ExpertSelectionConfig:
    """Configuration for expert data selection."""
    strategy: ExpertSelectionStrategy = ExpertSelectionStrategy.TOP_ENERGY_REDUCTION
    expert_ratio: float = 0.20  # 20% of samples as expert data
    min_energy_reduction: float = 1.10  # Minimum 10% energy reduction
    min_runtime_improvement: float = 0.95  # Allow up to 5% runtime regression
    energy_weight: float = 0.7
    runtime_weight: float = 0.3
    diversity_clusters: int = 5  # For diverse selection
    pareto_population_ratio: float = 0.5  # For Pareto optimal selection
    quality_threshold_percentile: float = 80  # For quality filtering


@dataclass
class SampleQualityMetrics:
    """Quality metrics for a code sample."""
    sample_id: str
    energy_reduction_ratio: float
    speedup_ratio: float
    power_reduction_ratio: float
    energy_efficiency_score: float
    code_length_reduction: float
    algorithmic_improvement_score: float
    multi_objective_score: float
    processing_quality: float

    @classmethod
    def from_pie_sample(cls, sample: Dict[str, Any]) -> 'SampleQualityMetrics':
        """Create quality metrics from PIE sample data."""
        src_energy = sample.get('src_energy_joules', 1.0)
        tgt_energy = sample.get('tgt_energy_joules', 1.0)
        src_runtime = sample.get('src_agg_runtime', 1.0)
        tgt_runtime = sample.get('tgt_agg_runtime', 1.0)
        src_power = sample.get('src_power_watts', 1.0)
        tgt_power = sample.get('tgt_power_watts', 1.0)

        # Calculate ratios
        energy_reduction_ratio = src_energy / max(tgt_energy, 1e-6)
        speedup_ratio = src_runtime / max(tgt_runtime, 1e-6)
        power_reduction_ratio = src_power / max(tgt_power, 1e-6)

        # Energy efficiency score (higher is better)
        energy_efficiency_score = 1.0 / (tgt_energy * tgt_runtime + 1e-6)

        # Code length analysis
        src_code = sample.get('src_code', '')
        tgt_code = sample.get('tgt_code', '')
        code_length_reduction = len(src_code) / max(len(tgt_code), 1) if tgt_code else 1.0

        # Algorithmic improvement heuristic (based on code complexity reduction)
        algorithmic_improvement_score = cls._estimate_algorithmic_improvement(src_code, tgt_code)

        # Multi-objective score (weighted combination)
        multi_objective_score = 0.7 * (energy_reduction_ratio - 1) + 0.3 * (speedup_ratio - 1)

        # Processing quality (based on success and error rate)
        processing_quality = 1.0 if sample.get('processing_success', False) else 0.0
        if sample.get('processing_error', ''):
            processing_quality *= 0.5

        return cls(
            sample_id=sample.get('src_id', 'unknown'),
            energy_reduction_ratio=energy_reduction_ratio,
            speedup_ratio=speedup_ratio,
            power_reduction_ratio=power_reduction_ratio,
            energy_efficiency_score=energy_efficiency_score,
            code_length_reduction=code_length_reduction,
            algorithmic_improvement_score=algorithmic_improvement_score,
            multi_objective_score=multi_objective_score,
            processing_quality=processing_quality
        )

    @staticmethod
    def _estimate_algorithmic_improvement(src_code: str, tgt_code: str) -> float:
        """Estimate algorithmic improvement based on code analysis."""
        if not src_code or not tgt_code:
            return 0.0

        # Simple heuristics for algorithmic improvement
        improvements = 0.0

        # Loop optimization detection
        src_loops = src_code.count('for') + src_code.count('while')
        tgt_loops = tgt_code.count('for') + tgt_code.count('while')
        if src_loops > tgt_loops:
            improvements += 0.2

        # Function call reduction
        src_calls = src_code.count('(')
        tgt_calls = tgt_code.count('(')
        if src_calls > tgt_calls:
            improvements += 0.1

        # Memory allocation patterns
        src_allocs = src_code.count('new') + src_code.count('malloc')
        tgt_allocs = tgt_code.count('new') + tgt_code.count('malloc')
        if src_allocs > tgt_allocs:
            improvements += 0.15

        # Vector/array operations efficiency
        if 'vector' in tgt_code and tgt_code.count('reserve') > src_code.count('reserve'):
            improvements += 0.1

        # Algorithm efficiency keywords
        efficiency_keywords = ['binary_search', 'sort', 'hash', 'unordered_map']
        for keyword in efficiency_keywords:
            if keyword in tgt_code and keyword not in src_code:
                improvements += 0.1

        return min(improvements, 1.0)  # Cap at 1.0


class ExpertDataSelector:
    """Selects expert data for Trinity-RFT CHORD training."""

    def __init__(self, config: ExpertSelectionConfig = None):
        self.config = config or ExpertSelectionConfig()
        self.sample_metrics: List[SampleQualityMetrics] = []
        self.selection_statistics: Dict[str, Any] = {}

        logger.info(f"Initialized expert data selector with strategy: {self.config.strategy.value}")

    def analyze_sample_quality(self, samples: List[Dict[str, Any]]) -> List[SampleQualityMetrics]:
        """Analyze quality metrics for all samples."""
        logger.info(f"Analyzing quality metrics for {len(samples)} samples")

        metrics_list = []
        for sample in samples:
            try:
                metrics = SampleQualityMetrics.from_pie_sample(sample)
                metrics_list.append(metrics)
            except Exception as e:
                logger.warning(f"Error analyzing sample {sample.get('src_id', 'unknown')}: {e}")

        self.sample_metrics = metrics_list
        logger.info(f"Analyzed {len(metrics_list)} samples successfully")

        return metrics_list

    def filter_quality_threshold(self, samples: List[Dict[str, Any]],
                                metrics_list: List[SampleQualityMetrics]) -> Tuple[List[Dict[str, Any]], List[SampleQualityMetrics]]:
        """Filter samples based on quality thresholds."""
        filtered_samples = []
        filtered_metrics = []

        # Calculate quality thresholds
        energy_reductions = [m.energy_reduction_ratio for m in metrics_list]
        energy_threshold = np.percentile(energy_reductions, self.config.quality_threshold_percentile)

        for sample, metrics in zip(samples, metrics_list):
            # Quality filters
            if (metrics.energy_reduction_ratio >= self.config.min_energy_reduction and
                metrics.speedup_ratio >= self.config.min_runtime_improvement and
                metrics.processing_quality > 0.5 and
                metrics.energy_reduction_ratio >= energy_threshold):

                filtered_samples.append(sample)
                filtered_metrics.append(metrics)

        logger.info(f"Quality filtering: {len(filtered_samples)}/{len(samples)} samples passed "
                   f"(threshold: {energy_threshold:.3f})")

        return filtered_samples, filtered_metrics

    def select_top_energy_reduction(self, samples: List[Dict[str, Any]],
                                   metrics_list: List[SampleQualityMetrics]) -> List[int]:
        """Select samples with highest energy reduction."""
        # Sort by energy reduction ratio
        indexed_metrics = [(i, m) for i, m in enumerate(metrics_list)]
        indexed_metrics.sort(key=lambda x: x[1].energy_reduction_ratio, reverse=True)

        # Select top percentage
        expert_count = int(len(samples) * self.config.expert_ratio)
        expert_indices = [idx for idx, _ in indexed_metrics[:expert_count]]

        return expert_indices

    def select_balanced_multi_objective(self, samples: List[Dict[str, Any]],
                                       metrics_list: List[SampleQualityMetrics]) -> List[int]:
        """Select samples based on balanced energy-runtime performance."""
        # Calculate balanced scores
        balanced_scores = []
        for metrics in metrics_list:
            score = (
                self.config.energy_weight * (metrics.energy_reduction_ratio - 1) +
                self.config.runtime_weight * (metrics.speedup_ratio - 1)
            )
            balanced_scores.append(score)

        # Sort by balanced score
        indexed_scores = [(i, score) for i, score in enumerate(balanced_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top percentage
        expert_count = int(len(samples) * self.config.expert_ratio)
        expert_indices = [idx for idx, _ in indexed_scores[:expert_count]]

        return expert_indices

    def select_pareto_optimal(self, samples: List[Dict[str, Any]],
                             metrics_list: List[SampleQualityMetrics]) -> List[int]:
        """Select Pareto optimal samples considering energy and runtime."""
        # Extract objectives (energy reduction and speedup)
        objectives = [(m.energy_reduction_ratio, m.speedup_ratio) for m in metrics_list]

        # Find Pareto front
        pareto_indices = self._find_pareto_front(objectives)

        # If we have more Pareto optimal samples than needed, rank them
        expert_count = int(len(samples) * self.config.expert_ratio)

        if len(pareto_indices) <= expert_count:
            # Use all Pareto optimal samples
            selected_indices = pareto_indices

            # Fill remaining slots with high-performing samples
            remaining_count = expert_count - len(selected_indices)
            if remaining_count > 0:
                non_pareto_indices = [i for i in range(len(samples)) if i not in pareto_indices]
                # Sort remaining by multi-objective score
                non_pareto_scored = [(i, metrics_list[i].multi_objective_score) for i in non_pareto_indices]
                non_pareto_scored.sort(key=lambda x: x[1], reverse=True)
                selected_indices.extend([idx for idx, _ in non_pareto_scored[:remaining_count]])
        else:
            # Rank Pareto optimal samples by distance from ideal point
            pareto_metrics = [metrics_list[i] for i in pareto_indices]
            max_energy = max(m.energy_reduction_ratio for m in pareto_metrics)
            max_runtime = max(m.speedup_ratio for m in pareto_metrics)

            distances = []
            for i, idx in enumerate(pareto_indices):
                m = metrics_list[idx]
                distance = math.sqrt(
                    (max_energy - m.energy_reduction_ratio) ** 2 +
                    (max_runtime - m.speedup_ratio) ** 2
                )
                distances.append((idx, distance))

            distances.sort(key=lambda x: x[1])
            selected_indices = [idx for idx, _ in distances[:expert_count]]

        return selected_indices

    def select_diverse_excellence(self, samples: List[Dict[str, Any]],
                                 metrics_list: List[SampleQualityMetrics]) -> List[int]:
        """Select diverse high-quality samples using clustering."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Extract features for clustering
        features = np.array([
            [m.energy_reduction_ratio, m.speedup_ratio, m.power_reduction_ratio,
             m.algorithmic_improvement_score, m.code_length_reduction]
            for m in metrics_list
        ])

        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        # Cluster samples
        n_clusters = min(self.config.diversity_clusters, len(samples) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_features)

        # Select best samples from each cluster
        expert_count = int(len(samples) * self.config.expert_ratio)
        samples_per_cluster = expert_count // n_clusters
        remainder = expert_count % n_clusters

        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

            if not cluster_indices:
                continue

            # Sort cluster samples by multi-objective score
            cluster_scored = [(i, metrics_list[i].multi_objective_score) for i in cluster_indices]
            cluster_scored.sort(key=lambda x: x[1], reverse=True)

            # Select top samples from this cluster
            cluster_select_count = samples_per_cluster + (1 if cluster_id < remainder else 0)
            cluster_selected = [idx for idx, _ in cluster_scored[:cluster_select_count]]
            selected_indices.extend(cluster_selected)

        return selected_indices[:expert_count]

    def select_difficulty_aware(self, samples: List[Dict[str, Any]],
                               metrics_list: List[SampleQualityMetrics]) -> List[int]:
        """Select samples considering problem difficulty and achievement."""
        # Estimate problem difficulty based on original code complexity
        difficulty_scores = []
        for sample in samples:
            src_code = sample.get('src_code', '')
            difficulty = self._estimate_problem_difficulty(src_code)
            difficulty_scores.append(difficulty)

        # Calculate achievement scores (improvement normalized by difficulty)
        achievement_scores = []
        for metrics, difficulty in zip(metrics_list, difficulty_scores):
            if difficulty > 0:
                achievement = metrics.multi_objective_score / math.sqrt(difficulty)
            else:
                achievement = metrics.multi_objective_score
            achievement_scores.append(achievement)

        # Sort by achievement score
        indexed_achievements = [(i, score) for i, score in enumerate(achievement_scores)]
        indexed_achievements.sort(key=lambda x: x[1], reverse=True)

        # Select top percentage
        expert_count = int(len(samples) * self.config.expert_ratio)
        expert_indices = [idx for idx, _ in indexed_achievements[:expert_count]]

        return expert_indices

    def _find_pareto_front(self, objectives: List[Tuple[float, float]]) -> List[int]:
        """Find Pareto optimal points (both objectives should be maximized)."""
        pareto_indices = []
        for i, (obj1_i, obj2_i) in enumerate(objectives):
            is_dominated = False
            for j, (obj1_j, obj2_j) in enumerate(objectives):
                if i != j and obj1_j >= obj1_i and obj2_j >= obj2_i and (obj1_j > obj1_i or obj2_j > obj2_i):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        return pareto_indices

    def _estimate_problem_difficulty(self, source_code: str) -> float:
        """Estimate problem difficulty based on code complexity."""
        if not source_code:
            return 1.0

        difficulty = 1.0

        # Code length factor
        difficulty += len(source_code) / 1000.0

        # Complexity indicators
        complexity_indicators = {
            'nested_loops': (source_code.count('for') + source_code.count('while')) * 0.3,
            'recursion': source_code.count('recursion') * 0.5,
            'data_structures': (source_code.count('vector') + source_code.count('map') +
                              source_code.count('set') + source_code.count('queue')) * 0.2,
            'algorithms': (source_code.count('sort') + source_code.count('search') +
                          source_code.count('binary') + source_code.count('tree')) * 0.4,
            'memory_management': (source_code.count('new') + source_code.count('delete') +
                                source_code.count('malloc') + source_code.count('free')) * 0.3
        }

        for indicator, score in complexity_indicators.items():
            difficulty += score

        return difficulty

    def select_expert_samples(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Main method to select expert samples based on configured strategy."""
        logger.info(f"Selecting expert samples using strategy: {self.config.strategy.value}")

        # Analyze sample quality
        metrics_list = self.analyze_sample_quality(samples)

        # Filter by quality threshold
        filtered_samples, filtered_metrics = self.filter_quality_threshold(samples, metrics_list)

        if not filtered_samples:
            logger.warning("No samples passed quality threshold")
            return [], []

        # Apply selection strategy
        if self.config.strategy == ExpertSelectionStrategy.TOP_ENERGY_REDUCTION:
            expert_indices = self.select_top_energy_reduction(filtered_samples, filtered_metrics)

        elif self.config.strategy == ExpertSelectionStrategy.BALANCED_MULTI_OBJECTIVE:
            expert_indices = self.select_balanced_multi_objective(filtered_samples, filtered_metrics)

        elif self.config.strategy == ExpertSelectionStrategy.PARETO_OPTIMAL:
            expert_indices = self.select_pareto_optimal(filtered_samples, filtered_metrics)

        elif self.config.strategy == ExpertSelectionStrategy.DIVERSE_EXCELLENCE:
            try:
                expert_indices = self.select_diverse_excellence(filtered_samples, filtered_metrics)
            except ImportError:
                logger.warning("sklearn not available, falling back to top energy reduction")
                expert_indices = self.select_top_energy_reduction(filtered_samples, filtered_metrics)

        elif self.config.strategy == ExpertSelectionStrategy.DIFFICULTY_AWARE:
            expert_indices = self.select_difficulty_aware(filtered_samples, filtered_metrics)

        else:
            raise ValueError(f"Unknown selection strategy: {self.config.strategy}")

        # Get expert samples
        expert_samples = [filtered_samples[i] for i in expert_indices]

        # Store selection statistics
        self._calculate_selection_statistics(filtered_samples, filtered_metrics, expert_indices)

        logger.info(f"Selected {len(expert_samples)} expert samples from {len(filtered_samples)} candidates")

        return expert_samples, expert_indices

    def _calculate_selection_statistics(self, samples: List[Dict[str, Any]],
                                       metrics_list: List[SampleQualityMetrics],
                                       expert_indices: List[int]):
        """Calculate and store selection statistics."""
        expert_metrics = [metrics_list[i] for i in expert_indices]
        all_metrics = metrics_list

        self.selection_statistics = {
            'total_candidates': len(samples),
            'expert_count': len(expert_indices),
            'expert_ratio': len(expert_indices) / len(samples),
            'expert_stats': {
                'energy_reduction_mean': np.mean([m.energy_reduction_ratio for m in expert_metrics]),
                'energy_reduction_std': np.std([m.energy_reduction_ratio for m in expert_metrics]),
                'speedup_mean': np.mean([m.speedup_ratio for m in expert_metrics]),
                'speedup_std': np.std([m.speedup_ratio for m in expert_metrics]),
                'multi_objective_mean': np.mean([m.multi_objective_score for m in expert_metrics]),
                'multi_objective_std': np.std([m.multi_objective_score for m in expert_metrics]),
            },
            'all_stats': {
                'energy_reduction_mean': np.mean([m.energy_reduction_ratio for m in all_metrics]),
                'energy_reduction_std': np.std([m.energy_reduction_ratio for m in all_metrics]),
                'speedup_mean': np.mean([m.speedup_ratio for m in all_metrics]),
                'speedup_std': np.std([m.speedup_ratio for m in all_metrics]),
                'multi_objective_mean': np.mean([m.multi_objective_score for m in all_metrics]),
                'multi_objective_std': np.std([m.multi_objective_score for m in all_metrics]),
            },
            'selection_improvement': {
                'energy_reduction_lift': (
                    np.mean([m.energy_reduction_ratio for m in expert_metrics]) /
                    np.mean([m.energy_reduction_ratio for m in all_metrics])
                ),
                'speedup_lift': (
                    np.mean([m.speedup_ratio for m in expert_metrics]) /
                    np.mean([m.speedup_ratio for m in all_metrics])
                ),
                'multi_objective_lift': (
                    np.mean([m.multi_objective_score for m in expert_metrics]) /
                    np.mean([m.multi_objective_score for m in all_metrics])
                )
            }
        }

    def get_selection_report(self) -> Dict[str, Any]:
        """Get detailed selection report."""
        return {
            'strategy': self.config.strategy.value,
            'config': asdict(self.config),
            'statistics': self.selection_statistics,
            'quality_distribution': self._analyze_quality_distribution()
        }

    def _analyze_quality_distribution(self) -> Dict[str, Any]:
        """Analyze quality distribution of samples."""
        if not self.sample_metrics:
            return {}

        energy_reductions = [m.energy_reduction_ratio for m in self.sample_metrics]
        speedups = [m.speedup_ratio for m in self.sample_metrics]
        multi_objectives = [m.multi_objective_score for m in self.sample_metrics]

        return {
            'energy_reduction': {
                'percentiles': {
                    '10': float(np.percentile(energy_reductions, 10)),
                    '25': float(np.percentile(energy_reductions, 25)),
                    '50': float(np.percentile(energy_reductions, 50)),
                    '75': float(np.percentile(energy_reductions, 75)),
                    '90': float(np.percentile(energy_reductions, 90)),
                    '95': float(np.percentile(energy_reductions, 95))
                }
            },
            'speedup': {
                'percentiles': {
                    '10': float(np.percentile(speedups, 10)),
                    '25': float(np.percentile(speedups, 25)),
                    '50': float(np.percentile(speedups, 50)),
                    '75': float(np.percentile(speedups, 75)),
                    '90': float(np.percentile(speedups, 90)),
                    '95': float(np.percentile(speedups, 95))
                }
            },
            'multi_objective': {
                'percentiles': {
                    '10': float(np.percentile(multi_objectives, 10)),
                    '25': float(np.percentile(multi_objectives, 25)),
                    '50': float(np.percentile(multi_objectives, 50)),
                    '75': float(np.percentile(multi_objectives, 75)),
                    '90': float(np.percentile(multi_objectives, 90)),
                    '95': float(np.percentile(multi_objectives, 95))
                }
            }
        }

    def save_selection_report(self, output_file: Path):
        """Save selection report to file."""
        report = self.get_selection_report()
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Selection report saved to {output_file}")


def main():
    """Main function for testing expert data selection."""
    import argparse

    parser = argparse.ArgumentParser(description="Select expert data for CHORD training")
    parser.add_argument("--input", required=True, help="Input energy dataset file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--strategy", choices=[s.value for s in ExpertSelectionStrategy],
                       default="top_energy_reduction", help="Selection strategy")
    parser.add_argument("--expert-ratio", type=float, default=0.20, help="Expert data ratio")
    parser.add_argument("--min-energy-reduction", type=float, default=1.10, help="Minimum energy reduction")

    args = parser.parse_args()

    # Create config
    config = ExpertSelectionConfig(
        strategy=ExpertSelectionStrategy(args.strategy),
        expert_ratio=args.expert_ratio,
        min_energy_reduction=args.min_energy_reduction
    )

    # Load samples
    samples = []
    with open(args.input, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Select expert data
    selector = ExpertDataSelector(config)
    expert_samples, expert_indices = selector.select_expert_samples(samples)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save expert samples
    expert_file = output_dir / "expert_samples.jsonl"
    with open(expert_file, 'w') as f:
        for sample in expert_samples:
            json.dump(sample, f)
            f.write('\n')

    # Save selection report
    report_file = output_dir / "expert_selection_report.json"
    selector.save_selection_report(report_file)

    print(f"Selected {len(expert_samples)} expert samples")
    print(f"Expert samples saved to {expert_file}")
    print(f"Selection report saved to {report_file}")


if __name__ == "__main__":
    main()