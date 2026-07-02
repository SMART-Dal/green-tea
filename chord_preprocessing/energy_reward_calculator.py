#!/usr/bin/env python3
"""
Energy Reward Calculator for Trinity-RFT CHORD

This module provides Trinity-RFT compatible reward calculation functions for energy-efficient
code generation, maintaining compatibility with the original CHORD algorithm while incorporating
energy-specific metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of rewards for energy-efficient code generation."""
    ENERGY_REDUCTION = "energy_reduction"
    POWER_EFFICIENCY = "power_efficiency"
    RUNTIME_ENERGY_BALANCE = "runtime_energy_balance"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class EnergyMetrics:
    """Container for energy measurement data."""
    src_energy_joules: float
    tgt_energy_joules: float
    src_power_watts: float
    tgt_power_watts: float
    src_runtime_seconds: float
    tgt_runtime_seconds: float
    energy_reduction_ratio: float
    speedup_ratio: float

    @property
    def energy_reduction_percentage(self) -> float:
        """Energy reduction as percentage."""
        return (self.energy_reduction_ratio - 1) * 100

    @property
    def power_reduction_ratio(self) -> float:
        """Power reduction ratio."""
        return self.src_power_watts / max(self.tgt_power_watts, 0.001)

    @property
    def energy_efficiency_score(self) -> float:
        """Energy efficiency score (higher is better)."""
        return 1.0 / (self.tgt_energy_joules * self.tgt_runtime_seconds + 1e-6)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    reward_type: RewardType = RewardType.ENERGY_REDUCTION
    energy_weight: float = 0.7
    runtime_weight: float = 0.3
    power_weight: float = 0.0
    baseline_energy_threshold: float = 1.05  # 5% improvement
    baseline_runtime_threshold: float = 1.0  # No regression
    use_log_scaling: bool = True
    clip_rewards: bool = True
    reward_range: Tuple[float, float] = (-1.0, 1.0)
    normalize_rewards: bool = True


class EnergyRewardCalculator:
    """Calculates rewards for energy-efficient code generation compatible with Trinity-RFT CHORD."""

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.reward_history = []  # For normalization

        logger.info(f"Initialized energy reward calculator with type: {self.config.reward_type.value}")

    def calculate_energy_reduction_reward(self, metrics: EnergyMetrics) -> float:
        """Calculate reward based purely on energy reduction."""
        if metrics.src_energy_joules <= 0 or metrics.tgt_energy_joules <= 0:
            return 0.0

        # Energy reduction ratio (higher is better)
        energy_ratio = metrics.energy_reduction_ratio

        # Convert to reward (positive for improvement, negative for regression)
        if energy_ratio >= self.config.baseline_energy_threshold:
            # Reward for energy reduction
            if self.config.use_log_scaling:
                reward = np.log(energy_ratio) / np.log(self.config.baseline_energy_threshold)
            else:
                reward = (energy_ratio - 1.0) / (self.config.baseline_energy_threshold - 1.0)
        else:
            # Penalty for energy increase or insufficient reduction
            if self.config.use_log_scaling:
                reward = -np.log(self.config.baseline_energy_threshold / energy_ratio)
            else:
                reward = (energy_ratio - self.config.baseline_energy_threshold) / (self.config.baseline_energy_threshold - 1.0)

        return reward

    def calculate_power_efficiency_reward(self, metrics: EnergyMetrics) -> float:
        """Calculate reward based on power efficiency."""
        if metrics.src_power_watts <= 0 or metrics.tgt_power_watts <= 0:
            return 0.0

        # Power reduction ratio
        power_ratio = metrics.power_reduction_ratio

        # Combine with runtime for efficiency
        efficiency_improvement = (power_ratio * metrics.speedup_ratio) - 1.0

        if self.config.use_log_scaling and efficiency_improvement > 0:
            reward = np.log(1 + efficiency_improvement)
        else:
            reward = efficiency_improvement

        return reward

    def calculate_runtime_energy_balance_reward(self, metrics: EnergyMetrics) -> float:
        """Calculate balanced reward considering both runtime and energy."""
        # Energy component
        energy_reward = self.calculate_energy_reduction_reward(metrics)

        # Runtime component
        if metrics.tgt_runtime_seconds > 0 and metrics.src_runtime_seconds > 0:
            runtime_ratio = metrics.speedup_ratio
            if runtime_ratio >= self.config.baseline_runtime_threshold:
                if self.config.use_log_scaling:
                    runtime_reward = np.log(runtime_ratio) if runtime_ratio > 1.0 else 0.0
                else:
                    runtime_reward = runtime_ratio - 1.0
            else:
                # Penalty for runtime regression
                runtime_reward = -(1.0 - runtime_ratio)
        else:
            runtime_reward = 0.0

        # Weighted combination
        combined_reward = (
            self.config.energy_weight * energy_reward +
            self.config.runtime_weight * runtime_reward
        )

        return combined_reward

    def calculate_multi_objective_reward(self, metrics: EnergyMetrics) -> float:
        """Calculate multi-objective reward including energy, runtime, and power."""
        # Individual components
        energy_reward = self.calculate_energy_reduction_reward(metrics)
        power_reward = self.calculate_power_efficiency_reward(metrics)

        # Runtime component
        runtime_reward = 0.0
        if metrics.speedup_ratio > 0:
            if metrics.speedup_ratio >= self.config.baseline_runtime_threshold:
                runtime_reward = np.log(metrics.speedup_ratio) if self.config.use_log_scaling else (metrics.speedup_ratio - 1.0)
            else:
                runtime_reward = -(1.0 - metrics.speedup_ratio)

        # Weighted combination
        total_weight = self.config.energy_weight + self.config.runtime_weight + self.config.power_weight
        if total_weight > 0:
            multi_objective_reward = (
                self.config.energy_weight * energy_reward +
                self.config.runtime_weight * runtime_reward +
                self.config.power_weight * power_reward
            ) / total_weight
        else:
            multi_objective_reward = energy_reward

        return multi_objective_reward

    def calculate_reward(self, metrics: EnergyMetrics) -> float:
        """Calculate reward based on configured reward type."""
        if self.config.reward_type == RewardType.ENERGY_REDUCTION:
            reward = self.calculate_energy_reduction_reward(metrics)
        elif self.config.reward_type == RewardType.POWER_EFFICIENCY:
            reward = self.calculate_power_efficiency_reward(metrics)
        elif self.config.reward_type == RewardType.RUNTIME_ENERGY_BALANCE:
            reward = self.calculate_runtime_energy_balance_reward(metrics)
        elif self.config.reward_type == RewardType.MULTI_OBJECTIVE:
            reward = self.calculate_multi_objective_reward(metrics)
        else:
            raise ValueError(f"Unknown reward type: {self.config.reward_type}")

        # Apply clipping if configured
        if self.config.clip_rewards:
            reward = np.clip(reward, self.config.reward_range[0], self.config.reward_range[1])

        # Store for normalization
        self.reward_history.append(reward)

        return reward

    def calculate_batch_rewards(self, batch_metrics: List[EnergyMetrics]) -> List[float]:
        """Calculate rewards for a batch of samples."""
        rewards = []
        for metrics in batch_metrics:
            reward = self.calculate_reward(metrics)
            rewards.append(reward)

        # Normalize batch if configured
        if self.config.normalize_rewards and len(rewards) > 1:
            rewards = self._normalize_batch_rewards(rewards)

        return rewards

    def _normalize_batch_rewards(self, rewards: List[float]) -> List[float]:
        """Normalize rewards within a batch."""
        rewards_array = np.array(rewards)

        # Z-score normalization
        if np.std(rewards_array) > 1e-6:
            normalized = (rewards_array - np.mean(rewards_array)) / np.std(rewards_array)
        else:
            normalized = rewards_array

        # Scale to reward range if needed
        if self.config.clip_rewards:
            # Scale to use most of the reward range
            min_val, max_val = np.min(normalized), np.max(normalized)
            if max_val > min_val:
                range_min, range_max = self.config.reward_range
                normalized = range_min + (normalized - min_val) * (range_max - range_min) / (max_val - min_val)

        return normalized.tolist()

    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about calculated rewards."""
        if not self.reward_history:
            return {}

        rewards = np.array(self.reward_history)
        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards)),
            'count': len(rewards),
            'positive_rate': float(np.mean(rewards > 0)),
            'percentile_25': float(np.percentile(rewards, 25)),
            'percentile_75': float(np.percentile(rewards, 75))
        }

    def reset_history(self):
        """Reset reward history."""
        self.reward_history = []

    @classmethod
    def from_pie_sample(cls, sample: Dict[str, Any], config: RewardConfig = None) -> Tuple['EnergyRewardCalculator', float]:
        """Create calculator and compute reward from PIE sample data."""
        calculator = cls(config)

        # Extract metrics from PIE sample
        metrics = EnergyMetrics(
            src_energy_joules=sample.get('src_energy_joules', 0.0),
            tgt_energy_joules=sample.get('tgt_energy_joules', 0.0),
            src_power_watts=sample.get('src_power_watts', 0.0),
            tgt_power_watts=sample.get('tgt_power_watts', 0.0),
            src_runtime_seconds=sample.get('src_agg_runtime', 0.0),
            tgt_runtime_seconds=sample.get('tgt_agg_runtime', 0.0),
            energy_reduction_ratio=sample.get('energy_reduction_ratio', 0.0),
            speedup_ratio=sample.get('speedup', 0.0)
        )

        reward = calculator.calculate_reward(metrics)
        return calculator, reward


class CHORDEnergyRewardWrapper:
    """Wrapper to make energy rewards compatible with Trinity-RFT CHORD."""

    def __init__(self, energy_calculator: EnergyRewardCalculator):
        self.energy_calculator = energy_calculator

    def compute_advantages(self, samples: List[Dict[str, Any]],
                          model_outputs: List[str] = None) -> List[float]:
        """
        Compute advantages/rewards for CHORD training.

        This method is designed to be compatible with Trinity-RFT CHORD's
        advantage calculation pipeline.
        """
        advantages = []

        for i, sample in enumerate(samples):
            # Extract energy metrics from sample
            try:
                if 'processing_metadata' in sample:
                    metadata = sample['processing_metadata']
                    metrics = EnergyMetrics(
                        src_energy_joules=metadata.get('src_energy_joules', 0.0),
                        tgt_energy_joules=metadata.get('tgt_energy_joules', 0.0),
                        src_power_watts=metadata.get('src_power_watts', 0.0),
                        tgt_power_watts=metadata.get('tgt_power_watts', 0.0),
                        src_runtime_seconds=metadata.get('src_runtime_seconds', 0.0),
                        tgt_runtime_seconds=metadata.get('tgt_runtime_seconds', 0.0),
                        energy_reduction_ratio=metadata.get('energy_reduction_ratio', 0.0),
                        speedup_ratio=metadata.get('speedup', 0.0)
                    )
                else:
                    # Fallback to sample-level fields
                    metrics = EnergyMetrics(
                        src_energy_joules=sample.get('src_energy_joules', 0.0),
                        tgt_energy_joules=sample.get('tgt_energy_joules', 0.0),
                        src_power_watts=sample.get('src_power_watts', 0.0),
                        tgt_power_watts=sample.get('tgt_power_watts', 0.0),
                        src_runtime_seconds=sample.get('src_agg_runtime', 0.0),
                        tgt_runtime_seconds=sample.get('tgt_agg_runtime', 0.0),
                        energy_reduction_ratio=sample.get('energy_reduction_ratio', 0.0),
                        speedup_ratio=sample.get('speedup', 0.0)
                    )

                advantage = self.energy_calculator.calculate_reward(metrics)
                advantages.append(advantage)

            except Exception as e:
                logger.warning(f"Error calculating advantage for sample {i}: {e}")
                advantages.append(0.0)

        return advantages

    def get_expert_advantages(self, expert_samples: List[Dict[str, Any]]) -> List[float]:
        """Get advantages for expert samples (should be consistently positive)."""
        expert_advantages = self.compute_advantages(expert_samples)

        # Ensure expert advantages are positive (they should be high-quality samples)
        expert_advantages = [max(adv, 0.1) for adv in expert_advantages]

        return expert_advantages


def create_trinity_compatible_reward_function(reward_config: RewardConfig = None):
    """
    Factory function to create Trinity-RFT compatible reward function.

    Returns a reward function that can be used directly with Trinity-RFT CHORD.
    """
    calculator = EnergyRewardCalculator(reward_config)
    wrapper = CHORDEnergyRewardWrapper(calculator)

    def compute_rewards(samples: List[Dict[str, Any]],
                       model_outputs: List[str] = None,
                       **kwargs) -> List[float]:
        """Trinity-RFT compatible reward function interface."""
        return wrapper.compute_advantages(samples, model_outputs)

    return compute_rewards


def main():
    """Main function for testing reward calculation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test energy reward calculation")
    parser.add_argument("--sample-file", help="JSON file with sample data")
    parser.add_argument("--reward-type", choices=[rt.value for rt in RewardType],
                       default="energy_reduction", help="Type of reward to calculate")
    parser.add_argument("--energy-weight", type=float, default=0.7, help="Energy weight")
    parser.add_argument("--runtime-weight", type=float, default=0.3, help="Runtime weight")

    args = parser.parse_args()

    # Create config
    config = RewardConfig(
        reward_type=RewardType(args.reward_type),
        energy_weight=args.energy_weight,
        runtime_weight=args.runtime_weight
    )

    # Create calculator
    calculator = EnergyRewardCalculator(config)

    if args.sample_file:
        # Load and test with actual samples
        with open(args.sample_file, 'r') as f:
            if args.sample_file.endswith('.jsonl'):
                samples = [json.loads(line) for line in f]
            else:
                samples = json.load(f)

        # Calculate rewards
        metrics_list = []
        for sample in samples[:10]:  # Test with first 10 samples
            metrics = EnergyMetrics(
                src_energy_joules=sample.get('src_energy_joules', 1.0),
                tgt_energy_joules=sample.get('tgt_energy_joules', 0.8),
                src_power_watts=sample.get('src_power_watts', 10.0),
                tgt_power_watts=sample.get('tgt_power_watts', 8.0),
                src_runtime_seconds=sample.get('src_agg_runtime', 1.0),
                tgt_runtime_seconds=sample.get('tgt_agg_runtime', 0.9),
                energy_reduction_ratio=sample.get('energy_reduction_ratio', 1.25),
                speedup_ratio=sample.get('speedup', 1.11)
            )
            metrics_list.append(metrics)

        rewards = calculator.calculate_batch_rewards(metrics_list)

        print(f"Calculated {len(rewards)} rewards:")
        for i, (metrics, reward) in enumerate(zip(metrics_list, rewards)):
            print(f"Sample {i+1}: Energy ratio={metrics.energy_reduction_ratio:.3f}, "
                  f"Speedup={metrics.speedup_ratio:.3f}, Reward={reward:.4f}")

        # Print statistics
        stats = calculator.get_reward_statistics()
        print(f"\nReward Statistics: {stats}")

    else:
        # Test with synthetic data
        test_metrics = [
            EnergyMetrics(1.0, 0.8, 10.0, 8.0, 1.0, 0.9, 1.25, 1.11),  # Good energy reduction
            EnergyMetrics(1.0, 0.95, 10.0, 9.5, 1.0, 0.98, 1.05, 1.02),  # Modest improvement
            EnergyMetrics(1.0, 1.1, 10.0, 11.0, 1.0, 1.05, 0.91, 0.95),  # Energy increase
        ]

        print("Testing with synthetic data:")
        for i, metrics in enumerate(test_metrics):
            reward = calculator.calculate_reward(metrics)
            print(f"Test {i+1}: Energy ratio={metrics.energy_reduction_ratio:.3f}, "
                  f"Speedup={metrics.speedup_ratio:.3f}, Reward={reward:.4f}")


if __name__ == "__main__":
    main()