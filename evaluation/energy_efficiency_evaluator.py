#!/usr/bin/env python3
"""
Energy Efficiency Evaluator

This module provides comprehensive evaluation framework for assessing the performance
of energy-efficient code generation models, including CHORD and other RL fine-tuned models.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from finetuning.utils.chord_metrics import CHORDMetricsCalculator, EnergyCodeMetrics, CHORDEvaluationResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for energy efficiency evaluation."""
    model_path: str
    test_dataset_path: str
    output_dir: str
    sniper_root: Optional[str] = None
    enable_simulation: bool = False
    max_samples: Optional[int] = None
    batch_size: int = 10
    num_workers: int = mp.cpu_count() // 2
    timeout_per_sample: int = 300
    include_baseline_comparison: bool = True
    include_correctness_testing: bool = True
    generate_detailed_report: bool = True


@dataclass
class ModelEvaluationResults:
    """Results of model evaluation."""
    model_name: str
    evaluation_config: EvaluationConfig
    chord_results: CHORDEvaluationResult
    baseline_comparison: Optional[Dict[str, float]] = None
    inference_statistics: Dict[str, float] = None
    error_analysis: Dict[str, Any] = None
    recommendation_scores: Dict[str, float] = None


class EnergyEfficiencyEvaluator:
    """Comprehensive evaluator for energy-efficient code generation models."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics calculator
        self.metrics_calculator = CHORDMetricsCalculator(
            sniper_root=config.sniper_root,
            enable_simulation=config.enable_simulation
        )

        # Model and inference setup (placeholder for actual model loading)
        self.model = None
        self.tokenizer = None

        logger.info(f"Initialized Energy Efficiency Evaluator")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Simulation enabled: {config.enable_simulation}")

    def load_model(self):
        """Load the model for inference (placeholder - implement based on model type)."""
        logger.info(f"Loading model from: {self.config.model_path}")

        # This would be implemented based on the specific model framework
        # For now, we'll simulate model loading
        try:
            # Placeholder for actual model loading
            # self.model = load_model(self.config.model_path)
            # self.tokenizer = load_tokenizer(self.config.model_path)
            self.model = "placeholder_model"
            self.tokenizer = "placeholder_tokenizer"
            logger.info("Model loaded successfully (placeholder)")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load test dataset for evaluation."""
        logger.info(f"Loading test dataset from: {self.config.test_dataset_path}")

        test_samples = []
        dataset_path = Path(self.config.test_dataset_path)

        if dataset_path.is_file():
            # Single file - assume JSONL format
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        test_samples.append(sample)

                        if self.config.max_samples and len(test_samples) >= self.config.max_samples:
                            break

                    except json.JSONDecodeError as e:
                        logger.debug(f"Line {line_num}: JSON decode error - {e}")

        elif dataset_path.is_dir():
            # Directory with test files
            test_files = list(dataset_path.glob("test*.jsonl")) + list(dataset_path.glob("eval*.jsonl"))
            if not test_files:
                test_files = [dataset_path / "test.jsonl"]

            for test_file in test_files:
                if test_file.exists():
                    with open(test_file, 'r') as f:
                        for line in f:
                            try:
                                sample = json.loads(line.strip())
                                test_samples.append(sample)

                                if self.config.max_samples and len(test_samples) >= self.config.max_samples:
                                    break
                            except json.JSONDecodeError:
                                continue

                if self.config.max_samples and len(test_samples) >= self.config.max_samples:
                    break

        logger.info(f"Loaded {len(test_samples)} test samples")
        return test_samples

    def generate_model_outputs(self, test_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate model outputs for test samples."""
        logger.info(f"Generating model outputs for {len(test_samples)} samples")

        evaluation_samples = []

        for i, sample in enumerate(test_samples):
            try:
                # Extract source code
                src_code = sample.get('src_code', sample.get('original_code', ''))
                ground_truth_code = sample.get('tgt_code', sample.get('optimized_code', ''))

                # Generate optimized code using the model
                model_output = self._generate_optimized_code(src_code)

                # Create evaluation sample
                eval_sample = {
                    'sample_id': sample.get('src_id', f'eval_{i}'),
                    'original_code': src_code,
                    'ground_truth_code': ground_truth_code,
                    'model_generated_code': model_output,
                    'ground_truth_metrics': {
                        'src_energy_joules': sample.get('src_energy_joules', 0.0),
                        'tgt_energy_joules': sample.get('tgt_energy_joules', 0.0),
                        'src_runtime_seconds': sample.get('src_agg_runtime', 0.0),
                        'tgt_runtime_seconds': sample.get('tgt_agg_runtime', 0.0),
                        'src_power_watts': sample.get('src_power_watts', 0.0),
                        'tgt_power_watts': sample.get('tgt_power_watts', 0.0)
                    },
                    'problem_id': sample.get('problem_id', ''),
                    'test_cases': sample.get('test_cases', [])
                }

                evaluation_samples.append(eval_sample)

                if (i + 1) % 50 == 0:
                    logger.info(f"Generated outputs for {i + 1}/{len(test_samples)} samples")

            except Exception as e:
                logger.error(f"Error generating output for sample {i}: {e}")
                # Add placeholder for failed generation
                evaluation_samples.append({
                    'sample_id': f'failed_{i}',
                    'original_code': src_code,
                    'model_generated_code': '',
                    'generation_failed': True,
                    'error': str(e)
                })

        logger.info(f"Generated outputs for {len(evaluation_samples)} samples")
        return evaluation_samples

    def _generate_optimized_code(self, source_code: str) -> str:
        """Generate optimized code using the loaded model."""
        # Placeholder for actual model inference
        # This would implement the actual model inference logic

        # For now, simulate some basic optimizations
        if not source_code.strip():
            return ""

        # Simulate model-generated optimizations (placeholder)
        optimized_code = source_code

        # Add some common optimizations as examples
        simple_optimizations = [
            ('#include <vector>\n', '#include <vector>\n#include <algorithm>\n'),
            ('for (int i = 0; i < ', 'for (size_t i = 0; i < '),
            ('vector.push_back', 'vector.reserve(size); vector.push_back'),
            ('if (', 'if '),  # Simplification placeholder
        ]

        for old, new in simple_optimizations:
            if old in optimized_code:
                optimized_code = optimized_code.replace(old, new, 1)  # Apply only once

        return optimized_code

    def evaluate_model_performance(self, evaluation_samples: List[Dict[str, Any]]) -> CHORDEvaluationResult:
        """Evaluate model performance using CHORD metrics."""
        logger.info("Evaluating model performance with CHORD metrics")

        # Prepare samples for CHORD evaluation
        chord_samples = []
        for sample in evaluation_samples:
            if sample.get('generation_failed', False):
                continue

            chord_sample = {
                'original_code': sample['original_code'],
                'optimized_code': sample['model_generated_code'],
                'ground_truth': sample.get('ground_truth_metrics'),
                'test_cases': sample.get('test_cases', [])
            }
            chord_samples.append(chord_sample)

        # Run CHORD evaluation
        chord_results = self.metrics_calculator.evaluate_batch(
            chord_samples,
            model_name=f"energy_model_{int(time.time())}"
        )

        return chord_results

    def compare_with_baseline(self, evaluation_samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compare model performance with baseline (ground truth)."""
        if not self.config.include_baseline_comparison:
            return {}

        logger.info("Comparing with baseline (ground truth) performance")

        # Evaluate baseline performance using ground truth optimizations
        baseline_samples = []
        for sample in evaluation_samples:
            if 'ground_truth_code' in sample and sample['ground_truth_code']:
                baseline_sample = {
                    'original_code': sample['original_code'],
                    'optimized_code': sample['ground_truth_code'],  # Use ground truth as baseline
                    'ground_truth': sample.get('ground_truth_metrics'),
                    'test_cases': sample.get('test_cases', [])
                }
                baseline_samples.append(baseline_sample)

        if not baseline_samples:
            logger.warning("No baseline samples available for comparison")
            return {}

        baseline_results = self.metrics_calculator.evaluate_batch(
            baseline_samples,
            model_name="baseline_ground_truth"
        )

        # Calculate relative performance
        model_chord_results = self.evaluate_model_performance(evaluation_samples)

        comparison = {
            'energy_reduction_ratio': (
                model_chord_results.mean_metrics.energy_reduction_rate /
                max(baseline_results.mean_metrics.energy_reduction_rate, 0.001)
            ),
            'speedup_ratio': (
                model_chord_results.mean_metrics.speedup_ratio /
                max(baseline_results.mean_metrics.speedup_ratio, 0.001)
            ),
            'correctness_ratio': (
                model_chord_results.mean_metrics.correctness_score /
                max(baseline_results.mean_metrics.correctness_score, 0.001)
            ),
            'code_quality_ratio': (
                model_chord_results.mean_metrics.code_quality_score /
                max(baseline_results.mean_metrics.code_quality_score, 0.001)
            ),
            'overall_improvement_ratio': (
                model_chord_results.mean_metrics.overall_improvement /
                max(baseline_results.mean_metrics.overall_improvement, 0.001)
            )
        }

        return comparison

    def analyze_errors(self, evaluation_samples: List[Dict[str, Any]],
                      chord_results: CHORDEvaluationResult) -> Dict[str, Any]:
        """Analyze errors and failure modes."""
        logger.info("Analyzing errors and failure modes")

        error_analysis = {
            'generation_failures': 0,
            'compilation_failures': 0,
            'correctness_failures': 0,
            'low_energy_improvement': 0,
            'performance_regression': 0,
            'common_error_patterns': {},
            'failure_by_problem_type': {},
            'quality_distribution': {}
        }

        # Count different types of failures
        for i, (sample, metrics) in enumerate(zip(evaluation_samples, chord_results.sample_metrics)):
            if sample.get('generation_failed', False):
                error_analysis['generation_failures'] += 1
                continue

            if metrics.correctness_score < 0.5:
                error_analysis['compilation_failures'] += 1

            if metrics.correctness_score < 0.9:
                error_analysis['correctness_failures'] += 1

            if metrics.energy_reduction_rate < 0.05:  # Less than 5% improvement
                error_analysis['low_energy_improvement'] += 1

            if metrics.speedup_ratio < 0.95:  # More than 5% performance loss
                error_analysis['performance_regression'] += 1

        # Analyze quality distribution
        if chord_results.sample_metrics:
            quality_scores = [m.overall_improvement for m in chord_results.sample_metrics]
            error_analysis['quality_distribution'] = {
                'excellent': sum(1 for s in quality_scores if s > 0.5),
                'good': sum(1 for s in quality_scores if 0.2 <= s <= 0.5),
                'fair': sum(1 for s in quality_scores if 0.1 <= s < 0.2),
                'poor': sum(1 for s in quality_scores if s < 0.1)
            }

        return error_analysis

    def calculate_inference_statistics(self, evaluation_samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate inference-related statistics."""
        logger.info("Calculating inference statistics")

        # Placeholder for actual inference timing
        # In real implementation, this would measure actual inference time

        total_samples = len(evaluation_samples)
        successful_generations = sum(1 for s in evaluation_samples if not s.get('generation_failed', False))

        statistics = {
            'total_samples': total_samples,
            'successful_generations': successful_generations,
            'generation_success_rate': successful_generations / total_samples if total_samples > 0 else 0.0,
            'average_inference_time_seconds': 2.5,  # Placeholder
            'average_output_length': np.mean([
                len(s.get('model_generated_code', ''))
                for s in evaluation_samples
                if not s.get('generation_failed', False)
            ]) if successful_generations > 0 else 0.0,
            'throughput_samples_per_minute': successful_generations / max(total_samples * 2.5 / 60, 0.001),
        }

        return statistics

    def generate_recommendations(self, chord_results: CHORDEvaluationResult,
                               error_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Generate improvement recommendations with confidence scores."""
        logger.info("Generating improvement recommendations")

        recommendations = {}

        # Energy optimization recommendations
        if chord_results.mean_metrics.energy_reduction_rate < 0.15:  # Less than 15% improvement
            recommendations['improve_energy_optimization'] = 0.8
            recommendations['focus_on_algorithmic_improvements'] = 0.7

        # Performance recommendations
        if chord_results.mean_metrics.speedup_ratio < 1.05:  # Less than 5% speedup
            recommendations['balance_energy_performance_tradeoffs'] = 0.9

        # Correctness recommendations
        if chord_results.mean_metrics.correctness_score < 0.95:  # Less than 95% correct
            recommendations['improve_correctness_validation'] = 0.95
            recommendations['add_more_test_cases'] = 0.8

        # Code quality recommendations
        if chord_results.mean_metrics.code_quality_score < 0.7:
            recommendations['improve_code_quality_training'] = 0.6

        # Data-based recommendations
        if error_analysis['generation_failures'] > len(chord_results.sample_metrics) * 0.1:
            recommendations['improve_model_stability'] = 0.85

        if error_analysis['low_energy_improvement'] > len(chord_results.sample_metrics) * 0.5:
            recommendations['enhance_energy_awareness'] = 0.9

        return recommendations

    def run_comprehensive_evaluation(self) -> ModelEvaluationResults:
        """Run comprehensive evaluation of the energy efficiency model."""
        logger.info("Starting comprehensive energy efficiency evaluation")

        start_time = time.time()

        # Step 1: Load model and dataset
        self.load_model()
        test_samples = self.load_test_dataset()

        if not test_samples:
            raise ValueError("No test samples loaded")

        # Step 2: Generate model outputs
        evaluation_samples = self.generate_model_outputs(test_samples)

        # Step 3: Evaluate model performance
        chord_results = self.evaluate_model_performance(evaluation_samples)

        # Step 4: Compare with baseline
        baseline_comparison = None
        if self.config.include_baseline_comparison:
            baseline_comparison = self.compare_with_baseline(evaluation_samples)

        # Step 5: Error analysis
        error_analysis = self.analyze_errors(evaluation_samples, chord_results)

        # Step 6: Inference statistics
        inference_stats = self.calculate_inference_statistics(evaluation_samples)

        # Step 7: Generate recommendations
        recommendations = self.generate_recommendations(chord_results, error_analysis)

        end_time = time.time()
        evaluation_duration = end_time - start_time

        logger.info(f"Evaluation completed in {evaluation_duration:.1f} seconds")

        # Create comprehensive results
        results = ModelEvaluationResults(
            model_name=Path(self.config.model_path).name,
            evaluation_config=self.config,
            chord_results=chord_results,
            baseline_comparison=baseline_comparison,
            inference_statistics=inference_stats,
            error_analysis=error_analysis,
            recommendation_scores=recommendations
        )

        return results

    def save_evaluation_results(self, results: ModelEvaluationResults):
        """Save comprehensive evaluation results."""
        logger.info("Saving evaluation results")

        # Save main results
        results_file = self.output_dir / "evaluation_results.json"
        results_dict = {
            'model_name': results.model_name,
            'evaluation_timestamp': results.chord_results.evaluation_timestamp,
            'sample_count': results.chord_results.sample_count,
            'chord_metrics': {
                'mean_metrics': self.metrics_calculator._metrics_to_dict(results.chord_results.mean_metrics),
                'std_metrics': self.metrics_calculator._metrics_to_dict(results.chord_results.std_metrics),
                'success_rates': results.chord_results.success_rates
            },
            'baseline_comparison': results.baseline_comparison,
            'inference_statistics': results.inference_statistics,
            'error_analysis': results.error_analysis,
            'recommendations': results.recommendation_scores
        }

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save detailed CHORD results
        chord_results_file = self.output_dir / "chord_detailed_results.json"
        self.metrics_calculator.save_evaluation_results(results.chord_results, chord_results_file)

        logger.info(f"Results saved to {self.output_dir}")

    def generate_evaluation_report(self, results: ModelEvaluationResults):
        """Generate human-readable evaluation report."""
        if not self.config.generate_detailed_report:
            return

        logger.info("Generating detailed evaluation report")

        report_file = self.output_dir / "evaluation_report.md"

        with open(report_file, 'w') as f:
            f.write(f"# Energy Efficiency Evaluation Report\n\n")
            f.write(f"**Model**: {results.model_name}\n")
            f.write(f"**Evaluation Date**: {results.chord_results.evaluation_timestamp}\n")
            f.write(f"**Total Samples**: {results.chord_results.sample_count}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            mean_metrics = results.chord_results.mean_metrics
            f.write(f"- **Energy Reduction**: {mean_metrics.energy_reduction_rate:.1%}\n")
            f.write(f"- **Performance Impact**: {mean_metrics.speedup_ratio:.2f}x speedup\n")
            f.write(f"- **Correctness Rate**: {mean_metrics.correctness_score:.1%}\n")
            f.write(f"- **Overall Improvement**: {mean_metrics.overall_improvement:.3f}\n\n")

            # Success Rates
            f.write("## Success Rates\n\n")
            for metric, rate in results.chord_results.success_rates.items():
                f.write(f"- **{metric.replace('_', ' ').title()}**: {rate:.1%}\n")
            f.write("\n")

            # Baseline Comparison
            if results.baseline_comparison:
                f.write("## Baseline Comparison\n\n")
                f.write("Performance relative to ground truth optimizations:\n\n")
                for metric, ratio in results.baseline_comparison.items():
                    f.write(f"- **{metric.replace('_', ' ').title()}**: {ratio:.2f}x\n")
                f.write("\n")

            # Error Analysis
            if results.error_analysis:
                f.write("## Error Analysis\n\n")
                error = results.error_analysis
                total = results.chord_results.sample_count

                f.write(f"- **Generation Failures**: {error['generation_failures']} ({error['generation_failures']/total:.1%})\n")
                f.write(f"- **Compilation Failures**: {error['compilation_failures']} ({error['compilation_failures']/total:.1%})\n")
                f.write(f"- **Correctness Failures**: {error['correctness_failures']} ({error['correctness_failures']/total:.1%})\n")
                f.write(f"- **Low Energy Improvement**: {error['low_energy_improvement']} ({error['low_energy_improvement']/total:.1%})\n")
                f.write(f"- **Performance Regression**: {error['performance_regression']} ({error['performance_regression']/total:.1%})\n\n")

            # Recommendations
            if results.recommendation_scores:
                f.write("## Recommendations\n\n")
                for rec, score in sorted(results.recommendation_scores.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- **{rec.replace('_', ' ').title()}** (Confidence: {score:.1%})\n")
                f.write("\n")

            # Inference Statistics
            if results.inference_statistics:
                f.write("## Inference Performance\n\n")
                stats = results.inference_statistics
                f.write(f"- **Generation Success Rate**: {stats['generation_success_rate']:.1%}\n")
                f.write(f"- **Average Inference Time**: {stats['average_inference_time_seconds']:.2f} seconds\n")
                f.write(f"- **Throughput**: {stats['throughput_samples_per_minute']:.1f} samples/minute\n")
                f.write(f"- **Average Output Length**: {stats['average_output_length']:.0f} characters\n\n")

        logger.info(f"Evaluation report saved to {report_file}")

    def print_evaluation_summary(self, results: ModelEvaluationResults):
        """Print evaluation summary to console."""
        print("\n" + "="*80)
        print("ENERGY EFFICIENCY EVALUATION SUMMARY")
        print("="*80)
        print(f"Model: {results.model_name}")
        print(f"Samples Evaluated: {results.chord_results.sample_count}")

        mean_metrics = results.chord_results.mean_metrics
        print(f"\nKey Performance Metrics:")
        print(f"  Energy Reduction Rate: {mean_metrics.energy_reduction_rate:.1%}")
        print(f"  Speedup Ratio: {mean_metrics.speedup_ratio:.2f}x")
        print(f"  Correctness Score: {mean_metrics.correctness_score:.1%}")
        print(f"  Code Quality Score: {mean_metrics.code_quality_score:.3f}")
        print(f"  Overall Improvement: {mean_metrics.overall_improvement:.3f}")

        print(f"\nSuccess Rates:")
        for metric, rate in results.chord_results.success_rates.items():
            if rate > 0:
                print(f"  {metric.replace('_', ' ').title()}: {rate:.1%}")

        if results.baseline_comparison:
            print(f"\nBaseline Comparison (relative to ground truth):")
            for metric, ratio in results.baseline_comparison.items():
                print(f"  {metric.replace('_', ' ').title()}: {ratio:.2f}x")

        print("="*80)


def main():
    """Main entry point for energy efficiency evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate energy-efficient code generation models")

    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test-data", required=True, help="Path to test dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--sniper-root", help="Sniper root directory for energy simulation")
    parser.add_argument("--enable-simulation", action="store_true", help="Enable energy simulation")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=10, help="Evaluation batch size")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--no-report", action="store_true", help="Skip detailed report generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create evaluation configuration
    config = EvaluationConfig(
        model_path=args.model,
        test_dataset_path=args.test_data,
        output_dir=args.output_dir,
        sniper_root=args.sniper_root,
        enable_simulation=args.enable_simulation,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        include_baseline_comparison=not args.no_baseline,
        generate_detailed_report=not args.no_report
    )

    try:
        # Run evaluation
        evaluator = EnergyEfficiencyEvaluator(config)
        results = evaluator.run_comprehensive_evaluation()

        # Save and display results
        evaluator.save_evaluation_results(results)
        evaluator.print_evaluation_summary(results)

        if config.generate_detailed_report:
            evaluator.generate_evaluation_report(results)

        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()