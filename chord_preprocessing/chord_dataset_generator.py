#!/usr/bin/env python3
"""
CHORD Dataset Generator

This module orchestrates the complete pipeline for generating Trinity-RFT CHORD-compatible
datasets from PIE energy samples, including expert data selection, format conversion,
and dataset splitting.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from chord_preprocessing.pie_to_chord_converter import PIEToCHORDConverter, CHORDConversionConfig
from chord_preprocessing.expert_data_selector import ExpertDataSelector, ExpertSelectionConfig, ExpertSelectionStrategy
from chord_preprocessing.energy_reward_calculator import EnergyRewardCalculator, RewardConfig, RewardType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CHORDDatasetConfig:
    """Complete configuration for CHORD dataset generation."""
    # Input/Output
    input_pie_file: str
    output_dir: str

    # Data splits
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    expert_ratio: float = 0.20

    # Quality thresholds
    min_energy_reduction: float = 1.10  # 10% energy reduction
    min_success_rate: float = 0.8

    # Expert selection
    expert_strategy: ExpertSelectionStrategy = ExpertSelectionStrategy.BALANCED_MULTI_OBJECTIVE
    diversity_clusters: int = 5

    # Reward calculation
    reward_type: RewardType = RewardType.RUNTIME_ENERGY_BALANCE
    energy_weight: float = 0.7
    runtime_weight: float = 0.3

    # Prompt templates
    prompt_template: str = "energy_optimization"

    # Dataset size limits (for testing)
    max_samples: Optional[int] = None
    max_expert_samples: Optional[int] = None
    max_train_samples: Optional[int] = None

    # Trinity-RFT compatibility
    create_sft_dataset: bool = True
    create_openr1_format: bool = True
    include_metadata: bool = True


class CHORDDatasetGenerator:
    """Generates complete CHORD datasets from PIE energy samples."""

    def __init__(self, config: CHORDDatasetConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.expert_selector = None
        self.converter = None
        self.reward_calculator = None

        # Results storage
        self.generation_results = {}

        logger.info(f"Initialized CHORD dataset generator")
        logger.info(f"Output directory: {self.output_dir}")

    def load_pie_energy_samples(self) -> List[Dict[str, Any]]:
        """Load PIE energy samples from input file."""
        logger.info(f"Loading PIE energy samples from {self.config.input_pie_file}")

        samples = []
        input_file = Path(self.config.input_pie_file)

        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())

                    # Basic validation
                    if not sample.get('processing_success', False):
                        continue

                    # Energy validation
                    if (sample.get('energy_reduction_ratio', 0.0) < self.config.min_energy_reduction):
                        continue

                    samples.append(sample)

                    # Limit samples if specified
                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        break

                except json.JSONDecodeError as e:
                    logger.debug(f"Line {line_num}: JSON decode error - {e}")

                if line_num % 10000 == 0:
                    logger.info(f"Loaded {line_num} lines, {len(samples)} valid samples")

        logger.info(f"Loaded {len(samples)} valid PIE energy samples")
        return samples

    def setup_components(self):
        """Setup all processing components."""
        logger.info("Setting up processing components")

        # Expert selection config
        expert_config = ExpertSelectionConfig(
            strategy=self.config.expert_strategy,
            expert_ratio=self.config.expert_ratio,
            min_energy_reduction=self.config.min_energy_reduction,
            diversity_clusters=self.config.diversity_clusters,
            energy_weight=self.config.energy_weight,
            runtime_weight=self.config.runtime_weight
        )
        self.expert_selector = ExpertDataSelector(expert_config)

        # CHORD conversion config
        conversion_config = CHORDConversionConfig(
            input_file=self.config.input_pie_file,
            output_dir=str(self.output_dir),
            expert_data_ratio=self.config.expert_ratio,
            min_energy_reduction=self.config.min_energy_reduction,
            prompt_template=self.config.prompt_template,
            max_samples_per_split=self.config.max_train_samples,
            include_runtime_metrics=self.config.include_metadata,
            preserve_original_ids=True
        )
        self.converter = PIEToCHORDConverter(conversion_config)

        # Reward calculation config
        reward_config = RewardConfig(
            reward_type=self.config.reward_type,
            energy_weight=self.config.energy_weight,
            runtime_weight=self.config.runtime_weight,
            baseline_energy_threshold=self.config.min_energy_reduction
        )
        self.reward_calculator = EnergyRewardCalculator(reward_config)

        logger.info("All components initialized successfully")

    def select_and_split_data(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Select expert data and create train/test splits."""
        logger.info("Selecting expert data and creating splits")

        # Select expert samples
        expert_samples, expert_indices = self.expert_selector.select_expert_samples(samples)

        # Limit expert samples if specified
        if self.config.max_expert_samples and len(expert_samples) > self.config.max_expert_samples:
            expert_samples = expert_samples[:self.config.max_expert_samples]

        # Get remaining samples for regular training
        expert_ids = {s.get('src_id', '') for s in expert_samples}
        regular_samples = [s for s in samples if s.get('src_id', '') not in expert_ids]

        # Limit regular samples if specified
        if self.config.max_train_samples and len(regular_samples) > self.config.max_train_samples:
            regular_samples = regular_samples[:self.config.max_train_samples]

        # Create train/test split from regular samples
        train_size = int(len(regular_samples) * self.config.train_ratio)
        train_samples = regular_samples[:train_size]
        test_samples = regular_samples[train_size:]

        splits = {
            'expert': expert_samples,
            'train': train_samples,
            'test': test_samples,
            'all_regular': regular_samples
        }

        logger.info(f"Data splits created:")
        logger.info(f"  Expert: {len(expert_samples)} samples")
        logger.info(f"  Train: {len(train_samples)} samples")
        logger.info(f"  Test: {len(test_samples)} samples")

        return splits

    def convert_to_chord_format(self, data_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Convert all data splits to CHORD format."""
        logger.info("Converting data splits to CHORD format")

        chord_splits = {}

        for split_name, samples in data_splits.items():
            if not samples:
                chord_splits[split_name] = []
                continue

            logger.info(f"Converting {split_name} split ({len(samples)} samples)")

            chord_samples = []
            for sample in samples:
                is_expert = (split_name == 'expert')
                chord_sample = self.converter.convert_sample_to_chord_format(sample, is_expert)
                chord_samples.append(chord_sample)

            chord_splits[split_name] = chord_samples

        return chord_splits

    def calculate_reward_statistics(self, chord_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """Calculate reward statistics for all splits."""
        logger.info("Calculating reward statistics")

        statistics = {}

        for split_name, chord_samples in chord_splits.items():
            if not chord_samples:
                statistics[split_name] = {}
                continue

            rewards = []
            for sample in chord_samples:
                # Calculate reward from CHORD sample
                if 'processing_metadata' in sample:
                    metadata = sample['processing_metadata']
                    from chord_preprocessing.energy_reward_calculator import EnergyMetrics
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
                    reward = self.reward_calculator.calculate_reward(metrics)
                    rewards.append(reward)

            if rewards:
                statistics[split_name] = {
                    'mean_reward': float(np.mean(rewards)),
                    'std_reward': float(np.std(rewards)),
                    'min_reward': float(np.min(rewards)),
                    'max_reward': float(np.max(rewards)),
                    'positive_rate': float(np.mean(np.array(rewards) > 0))
                }
            else:
                statistics[split_name] = {}

        return statistics

    def save_chord_datasets(self, chord_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Save CHORD datasets in various formats."""
        logger.info("Saving CHORD datasets")

        saved_files = {}

        # Save individual splits
        for split_name, chord_samples in chord_splits.items():
            if not chord_samples:
                continue

            split_file = self.output_dir / f"{split_name}.jsonl"
            with open(split_file, 'w') as f:
                for sample in chord_samples:
                    json.dump(sample, f)
                    f.write('\n')

            saved_files[split_name] = str(split_file)
            logger.info(f"Saved {len(chord_samples)} {split_name} samples to {split_file}")

        # Create Trinity-RFT compatible datasets
        if self.config.create_sft_dataset and 'expert' in chord_splits:
            sft_data = []
            for sample in chord_splits['expert']:
                sft_sample = {
                    "messages": sample["messages"],
                    "energy_score": sample.get("energy_score", 0.0),
                    "runtime_score": sample.get("runtime_score", 0.0)
                }
                if self.config.include_metadata and "processing_metadata" in sample:
                    sft_sample["metadata"] = sample["processing_metadata"]
                sft_data.append(sft_sample)

            sft_file = self.output_dir / "openr1_sft_dataset.json"
            with open(sft_file, 'w') as f:
                json.dump(sft_data, f, indent=2)

            saved_files['sft'] = str(sft_file)
            logger.info(f"Saved SFT dataset with {len(sft_data)} samples to {sft_file}")

        # Create combined train dataset (expert + regular train)
        if 'expert' in chord_splits and 'train' in chord_splits:
            combined_train = chord_splits['expert'] + chord_splits['train']
            combined_file = self.output_dir / "train_combined.jsonl"

            with open(combined_file, 'w') as f:
                for sample in combined_train:
                    json.dump(sample, f)
                    f.write('\n')

            saved_files['train_combined'] = str(combined_file)
            logger.info(f"Saved combined training dataset with {len(combined_train)} samples")

        # Create OpenR1-style format if requested
        if self.config.create_openr1_format and 'train' in chord_splits:
            openr1_data = []
            for sample in chord_splits.get('train', []):
                openr1_sample = {
                    'problem': sample['problem'],
                    'answer': sample['answer'],
                    'energy_score': sample.get('energy_score', 0.0),
                    'runtime_score': sample.get('runtime_score', 0.0)
                }
                openr1_data.append(openr1_sample)

            openr1_file = self.output_dir / "openr1_dataset.json"
            with open(openr1_file, 'w') as f:
                json.dump(openr1_data, f, indent=2)

            saved_files['openr1'] = str(openr1_file)

        return saved_files

    def create_dataset_metadata(self, data_splits: Dict[str, List[Dict[str, Any]]],
                               chord_splits: Dict[str, List[Dict[str, Any]]],
                               reward_statistics: Dict[str, Dict[str, float]],
                               saved_files: Dict[str, str]) -> Dict[str, Any]:
        """Create comprehensive dataset metadata."""
        # Get expert selection report
        expert_report = self.expert_selector.get_selection_report() if self.expert_selector else {}

        # Get reward calculator statistics
        reward_calc_stats = self.reward_calculator.get_reward_statistics() if self.reward_calculator else {}

        metadata = {
            'dataset_info': {
                'name': 'PIE_Energy_CHORD_Dataset',
                'version': '1.0',
                'description': 'PIE dataset enhanced with energy measurements and converted to Trinity-RFT CHORD format',
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_file': self.config.input_pie_file
            },
            'configuration': {
                'expert_ratio': self.config.expert_ratio,
                'train_ratio': self.config.train_ratio,
                'test_ratio': self.config.test_ratio,
                'expert_strategy': self.config.expert_strategy.value,
                'reward_type': self.config.reward_type.value,
                'energy_weight': self.config.energy_weight,
                'runtime_weight': self.config.runtime_weight,
                'min_energy_reduction': self.config.min_energy_reduction,
                'prompt_template': self.config.prompt_template
            },
            'data_statistics': {
                'original_samples': sum(len(samples) for samples in data_splits.values()),
                'chord_samples': sum(len(samples) for samples in chord_splits.values()),
                'splits': {
                    split_name: len(samples)
                    for split_name, samples in chord_splits.items()
                }
            },
            'expert_selection': expert_report,
            'reward_statistics': reward_statistics,
            'reward_calculator_stats': reward_calc_stats,
            'files': saved_files,
            'quality_metrics': self._calculate_quality_metrics(chord_splits)
        }

        return metadata

    def _calculate_quality_metrics(self, chord_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate overall quality metrics for the dataset."""
        all_samples = []
        for samples in chord_splits.values():
            all_samples.extend(samples)

        if not all_samples:
            return {}

        energy_scores = []
        runtime_scores = []

        for sample in all_samples:
            energy_scores.append(sample.get('energy_score', 0.0))
            runtime_scores.append(sample.get('runtime_score', 0.0))

        import numpy as np

        quality_metrics = {
            'energy_scores': {
                'mean': float(np.mean(energy_scores)),
                'std': float(np.std(energy_scores)),
                'min': float(np.min(energy_scores)),
                'max': float(np.max(energy_scores)),
                'percentiles': {
                    '25': float(np.percentile(energy_scores, 25)),
                    '50': float(np.percentile(energy_scores, 50)),
                    '75': float(np.percentile(energy_scores, 75)),
                    '90': float(np.percentile(energy_scores, 90)),
                    '95': float(np.percentile(energy_scores, 95))
                }
            },
            'runtime_scores': {
                'mean': float(np.mean(runtime_scores)),
                'std': float(np.std(runtime_scores)),
                'min': float(np.min(runtime_scores)),
                'max': float(np.max(runtime_scores)),
                'percentiles': {
                    '25': float(np.percentile(runtime_scores, 25)),
                    '50': float(np.percentile(runtime_scores, 50)),
                    '75': float(np.percentile(runtime_scores, 75)),
                    '90': float(np.percentile(runtime_scores, 90)),
                    '95': float(np.percentile(runtime_scores, 95))
                }
            },
            'sample_quality': {
                'high_energy_efficiency_rate': float(np.mean(np.array(energy_scores) > 1.2)),  # >20% energy reduction
                'performance_improvement_rate': float(np.mean(np.array(runtime_scores) > 1.1)),  # >10% speedup
                'dual_improvement_rate': float(np.mean((np.array(energy_scores) > 1.1) & (np.array(runtime_scores) > 1.05)))
            }
        }

        return quality_metrics

    def generate_dataset(self) -> Dict[str, Any]:
        """Run the complete CHORD dataset generation pipeline."""
        logger.info("Starting CHORD dataset generation pipeline")
        start_time = time.time()

        try:
            # Step 1: Load PIE energy samples
            pie_samples = self.load_pie_energy_samples()

            # Step 2: Setup processing components
            self.setup_components()

            # Step 3: Select expert data and create splits
            data_splits = self.select_and_split_data(pie_samples)

            # Step 4: Convert to CHORD format
            chord_splits = self.convert_to_chord_format(data_splits)

            # Step 5: Calculate reward statistics
            reward_statistics = self.calculate_reward_statistics(chord_splits)

            # Step 6: Save datasets
            saved_files = self.save_chord_datasets(chord_splits)

            # Step 7: Create metadata
            metadata = self.create_dataset_metadata(data_splits, chord_splits, reward_statistics, saved_files)

            # Step 8: Save metadata
            metadata_file = self.output_dir / "dataset_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Step 9: Create generation summary
            end_time = time.time()
            generation_summary = {
                'success': True,
                'generation_time_seconds': end_time - start_time,
                'metadata_file': str(metadata_file),
                'output_directory': str(self.output_dir),
                **metadata['data_statistics']
            }

            logger.info("CHORD dataset generation completed successfully")
            logger.info(f"Generated {generation_summary['chord_samples']} CHORD samples")
            logger.info(f"Expert samples: {len(chord_splits.get('expert', []))}")
            logger.info(f"Training samples: {len(chord_splits.get('train', []))}")
            logger.info(f"Test samples: {len(chord_splits.get('test', []))}")
            logger.info(f"Generation time: {generation_summary['generation_time_seconds']:.1f} seconds")

            return generation_summary

        except Exception as e:
            logger.error(f"CHORD dataset generation failed: {e}")
            raise


def main():
    """Main entry point for CHORD dataset generation."""
    parser = argparse.ArgumentParser(description="Generate CHORD datasets from PIE energy data")

    # Required arguments
    parser.add_argument("--input", required=True, help="Input PIE energy dataset file")
    parser.add_argument("--output-dir", required=True, help="Output directory for CHORD datasets")

    # Data configuration
    parser.add_argument("--expert-ratio", type=float, default=0.20, help="Expert data ratio")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--min-energy-reduction", type=float, default=1.10, help="Minimum energy reduction")

    # Expert selection
    parser.add_argument("--expert-strategy", choices=[s.value for s in ExpertSelectionStrategy],
                       default="balanced_multi_objective", help="Expert selection strategy")

    # Reward configuration
    parser.add_argument("--reward-type", choices=[r.value for r in RewardType],
                       default="runtime_energy_balance", help="Reward calculation type")
    parser.add_argument("--energy-weight", type=float, default=0.7, help="Energy weight in rewards")
    parser.add_argument("--runtime-weight", type=float, default=0.3, help="Runtime weight in rewards")

    # Dataset limits (for testing)
    parser.add_argument("--max-samples", type=int, help="Maximum total samples to process")
    parser.add_argument("--max-expert", type=int, help="Maximum expert samples")
    parser.add_argument("--max-train", type=int, help="Maximum training samples")

    # Format options
    parser.add_argument("--prompt-template", choices=["energy_optimization", "performance_energy", "green_computing"],
                       default="energy_optimization", help="Prompt template to use")
    parser.add_argument("--no-sft", action="store_true", help="Don't create SFT dataset")
    parser.add_argument("--no-openr1", action="store_true", help="Don't create OpenR1 format")
    parser.add_argument("--no-metadata", action="store_true", help="Don't include processing metadata")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = CHORDDatasetConfig(
        input_pie_file=args.input,
        output_dir=args.output_dir,
        expert_ratio=args.expert_ratio,
        train_ratio=args.train_ratio,
        min_energy_reduction=args.min_energy_reduction,
        expert_strategy=ExpertSelectionStrategy(args.expert_strategy),
        reward_type=RewardType(args.reward_type),
        energy_weight=args.energy_weight,
        runtime_weight=args.runtime_weight,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples,
        max_expert_samples=args.max_expert,
        max_train_samples=args.max_train,
        create_sft_dataset=not args.no_sft,
        create_openr1_format=not args.no_openr1,
        include_metadata=not args.no_metadata
    )

    try:
        # Generate dataset
        generator = CHORDDatasetGenerator(config)
        summary = generator.generate_dataset()

        print("\nCHORD Dataset Generation Summary:")
        print("=" * 50)
        print(f"Success: {summary['success']}")
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Total CHORD Samples: {summary['chord_samples']:,}")
        if 'splits' in summary:
            for split_name, count in summary.get('splits', {}).items():
                print(f"  {split_name.title()}: {count:,}")
        print(f"Generation Time: {summary['generation_time_seconds']:.1f} seconds")
        print(f"Metadata: {summary['metadata_file']}")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()