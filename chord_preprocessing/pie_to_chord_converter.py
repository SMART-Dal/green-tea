#!/usr/bin/env python3
"""
PIE to CHORD Converter

This module converts PIE energy-enhanced dataset samples to Trinity-RFT CHORD format,
maintaining compatibility with the original CHORD loss function while adding energy-aware
expert data selection capabilities.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from energy_data_collection.energy_schema_manager import EnergySchemaManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CHORDConversionConfig:
    """Configuration for PIE to CHORD conversion."""
    input_file: str
    output_dir: str
    expert_data_ratio: float = 0.20  # Top 20% for expert data
    min_energy_reduction: float = 1.05  # Minimum 5% energy reduction
    min_success_rate: float = 0.8  # Minimum processing success rate
    max_samples_per_split: Optional[int] = None
    prompt_template: str = "energy_optimization"
    include_runtime_metrics: bool = True
    preserve_original_ids: bool = True


class PIEToCHORDConverter:
    """Converts PIE energy dataset to Trinity-RFT CHORD format."""

    def __init__(self, config: CHORDConversionConfig):
        self.config = config
        self.schema_manager = EnergySchemaManager()

        # Output paths
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prompt templates
        self.prompt_templates = self._load_prompt_templates()

        logger.info(f"Initialized PIE to CHORD converter")
        logger.info(f"Expert data ratio: {config.expert_data_ratio}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Load prompt templates for different optimization scenarios."""
        return {
            "energy_optimization": {
                "system_prompt": "You are an expert C++ programmer specializing in energy-efficient code optimization.",
                "user_prompt_template": "Optimize this C++ code for energy efficiency. Focus on reducing CPU cycles, memory access patterns, and algorithmic complexity while maintaining correctness:\n\n{src_code}",
                "context_template": "Original code energy: {src_energy:.4f}J, runtime: {src_runtime:.4f}s"
            },
            "performance_energy": {
                "system_prompt": "You are an expert in multi-objective code optimization, balancing performance and energy efficiency.",
                "user_prompt_template": "Optimize this C++ code for both performance and energy efficiency. Consider trade-offs between speed and power consumption:\n\n{src_code}",
                "context_template": "Original: {src_energy:.4f}J, {src_runtime:.4f}s, speedup potential: {speedup:.2f}x"
            },
            "green_computing": {
                "system_prompt": "You are a specialist in green computing and sustainable software development.",
                "user_prompt_template": "Refactor this C++ code to minimize its environmental impact through energy-efficient algorithms and data structures:\n\n{src_code}",
                "context_template": "Carbon footprint reduction target: {energy_reduction:.1f}%"
            }
        }

    def load_pie_energy_samples(self, input_file: Path) -> List[Dict[str, Any]]:
        """Load PIE energy-enhanced samples from file."""
        logger.info(f"Loading PIE energy samples from {input_file}")

        samples = []
        valid_samples = 0
        invalid_samples = 0

        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())

                    # Validate energy sample
                    is_valid, validation_errors = self.schema_manager.validate_energy_sample(sample)

                    if is_valid:
                        samples.append(sample)
                        valid_samples += 1
                    else:
                        logger.debug(f"Line {line_num}: Invalid sample - {validation_errors}")
                        invalid_samples += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error - {e}")
                    invalid_samples += 1

                if line_num % 10000 == 0:
                    logger.info(f"Loaded {line_num} lines, {valid_samples} valid samples")

        logger.info(f"Loaded {valid_samples} valid samples, {invalid_samples} invalid samples")
        return samples

    def filter_high_quality_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples for high-quality energy measurements."""
        logger.info("Filtering high-quality samples")

        filtered_samples = []

        for sample in samples:
            # Check processing success
            if not sample.get('processing_success', False):
                continue

            # Check energy reduction threshold
            energy_reduction_ratio = sample.get('energy_reduction_ratio', 0.0)
            if energy_reduction_ratio < self.config.min_energy_reduction:
                continue

            # Check for valid energy measurements
            src_energy = sample.get('src_energy_joules', 0.0)
            tgt_energy = sample.get('tgt_energy_joules', 0.0)

            if src_energy <= 0 or tgt_energy <= 0:
                continue

            # Check for reasonable runtime values
            src_runtime = sample.get('src_agg_runtime', 0.0)
            tgt_runtime = sample.get('tgt_agg_runtime', 0.0)

            if src_runtime <= 0 or tgt_runtime <= 0:
                continue

            # Check code quality (non-empty, reasonable length)
            src_code = sample.get('src_code', '').strip()
            tgt_code = sample.get('tgt_code', '').strip()

            if not src_code or not tgt_code:
                continue

            if len(src_code) < 50 or len(tgt_code) < 50:  # Too short to be meaningful
                continue

            if len(src_code) > 50000 or len(tgt_code) > 50000:  # Too long for model context
                continue

            filtered_samples.append(sample)

        logger.info(f"Filtered to {len(filtered_samples)} high-quality samples "
                   f"({100 * len(filtered_samples) / len(samples):.1f}% of original)")

        return filtered_samples

    def select_expert_samples(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Select expert samples based on energy efficiency scores."""
        logger.info(f"Selecting expert samples (top {self.config.expert_data_ratio:.1%})")

        # Sort by energy reduction ratio (higher is better)
        sorted_samples = sorted(samples, key=lambda x: x.get('energy_reduction_ratio', 0.0), reverse=True)

        # Calculate expert threshold
        expert_count = int(len(sorted_samples) * self.config.expert_data_ratio)
        expert_threshold = sorted_samples[expert_count - 1]['energy_reduction_ratio'] if expert_count > 0 else float('inf')

        # Split into expert and regular samples
        expert_samples = sorted_samples[:expert_count]
        regular_samples = sorted_samples[expert_count:]

        logger.info(f"Selected {len(expert_samples)} expert samples (threshold: {expert_threshold:.3f})")
        logger.info(f"Remaining {len(regular_samples)} regular samples")

        # Print statistics
        expert_energies = [s['energy_reduction_ratio'] for s in expert_samples]
        regular_energies = [s['energy_reduction_ratio'] for s in regular_samples]

        logger.info(f"Expert samples - Energy reduction: {np.mean(expert_energies):.3f} ± {np.std(expert_energies):.3f}")
        if regular_energies:
            logger.info(f"Regular samples - Energy reduction: {np.mean(regular_energies):.3f} ± {np.std(regular_energies):.3f}")

        return expert_samples, regular_samples

    def convert_sample_to_chord_format(self, sample: Dict[str, Any], is_expert: bool = False) -> Dict[str, Any]:
        """Convert a single PIE energy sample to CHORD format."""
        # Get prompt template
        template_name = self.config.prompt_template
        template = self.prompt_templates.get(template_name, self.prompt_templates["energy_optimization"])

        # Extract sample data
        src_code = sample.get('src_code', '').strip()
        tgt_code = sample.get('tgt_code', '').strip()
        src_energy = sample.get('src_energy_joules', 0.0)
        tgt_energy = sample.get('tgt_energy_joules', 0.0)
        src_runtime = sample.get('src_agg_runtime', 0.0)
        tgt_runtime = sample.get('tgt_agg_runtime', 0.0)
        energy_reduction_ratio = sample.get('energy_reduction_ratio', 0.0)
        speedup = sample.get('speedup', 0.0)

        # Create context information
        context_vars = {
            'src_energy': src_energy,
            'src_runtime': src_runtime,
            'energy_reduction': (energy_reduction_ratio - 1) * 100,  # Convert to percentage
            'speedup': speedup
        }

        # Generate user prompt
        user_prompt = template["user_prompt_template"].format(src_code=src_code)

        # Add context if template supports it
        if "context_template" in template:
            try:
                context_info = template["context_template"].format(**context_vars)
                user_prompt += f"\n\nContext: {context_info}"
            except KeyError:
                pass  # Skip context if variables not available

        # Create CHORD-compatible sample
        chord_sample = {
            'problem': self._extract_problem_description(sample),
            'answer': tgt_code,
            'messages': [
                {
                    "role": "system",
                    "content": template["system_prompt"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": tgt_code
                }
            ],
            'energy_score': energy_reduction_ratio,
            'runtime_score': speedup
        }

        # Add optional fields
        if self.config.preserve_original_ids:
            chord_sample['src_id'] = sample.get('src_id', '')
            chord_sample['tgt_id'] = sample.get('tgt_id', '')
            chord_sample['problem_id'] = sample.get('problem_id', '')

        if self.config.include_runtime_metrics:
            chord_sample['processing_metadata'] = {
                'src_energy_joules': src_energy,
                'tgt_energy_joules': tgt_energy,
                'src_runtime_seconds': src_runtime,
                'tgt_runtime_seconds': tgt_runtime,
                'energy_reduction_ratio': energy_reduction_ratio,
                'speedup': speedup,
                'power_reduction': (
                    (sample.get('src_power_watts', 0) - sample.get('tgt_power_watts', 0)) /
                    max(sample.get('src_power_watts', 1), 0.001)
                ),
                'is_expert_sample': is_expert,
                'sniper_architecture': sample.get('sniper_architecture', ''),
                'sniper_config': sample.get('sniper_config', ''),
                'processing_timestamp': sample.get('processing_timestamp', '')
            }

        return chord_sample

    def _extract_problem_description(self, sample: Dict[str, Any]) -> str:
        """Extract problem description from sample."""
        return self.schema_manager._extract_problem_description(sample)

    def convert_dataset_to_chord_format(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Convert entire dataset to CHORD format with expert/regular splits."""
        logger.info("Converting dataset to CHORD format")

        # Filter high-quality samples
        quality_samples = self.filter_high_quality_samples(samples)

        # Select expert samples
        expert_samples, regular_samples = self.select_expert_samples(quality_samples)

        # Convert expert samples
        logger.info("Converting expert samples")
        expert_chord_samples = []
        for sample in expert_samples:
            chord_sample = self.convert_sample_to_chord_format(sample, is_expert=True)
            expert_chord_samples.append(chord_sample)

        # Convert regular samples
        logger.info("Converting regular samples")
        regular_chord_samples = []
        for sample in regular_samples:
            chord_sample = self.convert_sample_to_chord_format(sample, is_expert=False)
            regular_chord_samples.append(chord_sample)

        # Limit samples if specified
        if self.config.max_samples_per_split:
            expert_chord_samples = expert_chord_samples[:self.config.max_samples_per_split]
            regular_chord_samples = regular_chord_samples[:self.config.max_samples_per_split]

        logger.info(f"Converted {len(expert_chord_samples)} expert samples")
        logger.info(f"Converted {len(regular_chord_samples)} regular samples")

        return {
            'expert': expert_chord_samples,
            'regular': regular_chord_samples,
            'all': expert_chord_samples + regular_chord_samples
        }

    def save_chord_datasets(self, chord_datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Save CHORD datasets in Trinity-RFT compatible format."""
        saved_files = {}

        # Save expert demonstrations (for SFT training)
        expert_file = self.output_dir / "expert_demonstrations.jsonl"
        with open(expert_file, 'w') as f:
            for sample in chord_datasets['expert']:
                json.dump(sample, f)
                f.write('\n')
        saved_files['expert'] = str(expert_file)
        logger.info(f"Saved {len(chord_datasets['expert'])} expert samples to {expert_file}")

        # Save regular samples (for RL training)
        train_file = self.output_dir / "train.jsonl"
        with open(train_file, 'w') as f:
            for sample in chord_datasets['regular']:
                json.dump(sample, f)
                f.write('\n')
        saved_files['train'] = str(train_file)
        logger.info(f"Saved {len(chord_datasets['regular'])} training samples to {train_file}")

        # Save combined dataset
        all_file = self.output_dir / "pie_chord_dataset.jsonl"
        with open(all_file, 'w') as f:
            for sample in chord_datasets['all']:
                json.dump(sample, f)
                f.write('\n')
        saved_files['all'] = str(all_file)
        logger.info(f"Saved {len(chord_datasets['all'])} total samples to {all_file}")

        # Create Trinity-RFT compatible SFT dataset
        sft_file = self.output_dir / "openr1_sft_dataset.json"
        sft_data = [
            {
                "messages": sample["messages"],
                "energy_score": sample.get("energy_score", 0.0),
                "runtime_score": sample.get("runtime_score", 0.0)
            }
            for sample in chord_datasets['expert']
        ]

        with open(sft_file, 'w') as f:
            json.dump(sft_data, f, indent=2)
        saved_files['sft'] = str(sft_file)
        logger.info(f"Saved SFT dataset with {len(sft_data)} samples to {sft_file}")

        # Create dataset metadata
        metadata = {
            'dataset_name': 'pie_energy_chord',
            'version': '1.0',
            'description': 'PIE dataset enhanced with energy measurements and converted to CHORD format',
            'total_samples': len(chord_datasets['all']),
            'expert_samples': len(chord_datasets['expert']),
            'regular_samples': len(chord_datasets['regular']),
            'expert_ratio': self.config.expert_data_ratio,
            'min_energy_reduction': self.config.min_energy_reduction,
            'prompt_template': self.config.prompt_template,
            'conversion_config': {
                'expert_data_ratio': self.config.expert_data_ratio,
                'min_energy_reduction': self.config.min_energy_reduction,
                'min_success_rate': self.config.min_success_rate,
                'include_runtime_metrics': self.config.include_runtime_metrics,
                'preserve_original_ids': self.config.preserve_original_ids
            },
            'files': saved_files,
            'sample_distribution': {
                'expert_energy_reduction_mean': np.mean([s.get('energy_score', 0) for s in chord_datasets['expert']]),
                'regular_energy_reduction_mean': np.mean([s.get('energy_score', 0) for s in chord_datasets['regular']]),
                'expert_runtime_score_mean': np.mean([s.get('runtime_score', 0) for s in chord_datasets['expert']]),
                'regular_runtime_score_mean': np.mean([s.get('runtime_score', 0) for s in chord_datasets['regular']])
            }
        }

        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_file)
        logger.info(f"Saved dataset metadata to {metadata_file}")

        return saved_files

    def run_conversion(self) -> Dict[str, Any]:
        """Run the complete PIE to CHORD conversion process."""
        logger.info("Starting PIE to CHORD conversion")

        # Load PIE energy samples
        input_file = Path(self.config.input_file)
        pie_samples = self.load_pie_energy_samples(input_file)

        if not pie_samples:
            raise ValueError("No valid PIE samples found in input file")

        # Convert to CHORD format
        chord_datasets = self.convert_dataset_to_chord_format(pie_samples)

        # Save datasets
        saved_files = self.save_chord_datasets(chord_datasets)

        # Create summary
        summary = {
            'input_file': str(input_file),
            'output_directory': str(self.output_dir),
            'original_samples': len(pie_samples),
            'converted_samples': len(chord_datasets['all']),
            'expert_samples': len(chord_datasets['expert']),
            'regular_samples': len(chord_datasets['regular']),
            'conversion_rate': len(chord_datasets['all']) / len(pie_samples),
            'expert_ratio': len(chord_datasets['expert']) / len(chord_datasets['all']),
            'saved_files': saved_files,
            'config': {
                'expert_data_ratio': self.config.expert_data_ratio,
                'min_energy_reduction': self.config.min_energy_reduction,
                'prompt_template': self.config.prompt_template
            }
        }

        # Save summary
        summary_file = self.output_dir / "conversion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("Conversion completed successfully")
        logger.info(f"Summary: {len(chord_datasets['all'])} samples converted "
                   f"({len(chord_datasets['expert'])} expert, {len(chord_datasets['regular'])} regular)")

        return summary


def main():
    """Main entry point for PIE to CHORD conversion."""
    parser = argparse.ArgumentParser(description="Convert PIE energy dataset to CHORD format")
    parser.add_argument("--input", required=True, help="Input PIE energy dataset file")
    parser.add_argument("--output-dir", required=True, help="Output directory for CHORD datasets")
    parser.add_argument("--expert-ratio", type=float, default=0.20, help="Expert data ratio")
    parser.add_argument("--min-energy-reduction", type=float, default=1.05, help="Minimum energy reduction ratio")
    parser.add_argument("--max-samples", type=int, help="Maximum samples per split")
    parser.add_argument("--prompt-template", choices=["energy_optimization", "performance_energy", "green_computing"],
                       default="energy_optimization", help="Prompt template to use")
    parser.add_argument("--no-runtime-metrics", action="store_true", help="Exclude runtime metrics")
    parser.add_argument("--no-preserve-ids", action="store_true", help="Don't preserve original IDs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create conversion config
    config = CHORDConversionConfig(
        input_file=args.input,
        output_dir=args.output_dir,
        expert_data_ratio=args.expert_ratio,
        min_energy_reduction=args.min_energy_reduction,
        max_samples_per_split=args.max_samples,
        prompt_template=args.prompt_template,
        include_runtime_metrics=not args.no_runtime_metrics,
        preserve_original_ids=not args.no_preserve_ids
    )

    try:
        # Run conversion
        converter = PIEToCHORDConverter(config)
        summary = converter.run_conversion()

        print("Conversion completed successfully!")
        print(f"Input samples: {summary['original_samples']:,}")
        print(f"Converted samples: {summary['converted_samples']:,}")
        print(f"Expert samples: {summary['expert_samples']:,}")
        print(f"Regular samples: {summary['regular_samples']:,}")
        print(f"Conversion rate: {summary['conversion_rate'] * 100:.1f}%")
        print(f"Output directory: {summary['output_directory']}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()