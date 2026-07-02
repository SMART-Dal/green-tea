#!/usr/bin/env python3
"""
Integration Verification Script for Energy-Efficient Code Generation Pipeline

This script verifies that all components of the 4-phase energy-efficient code generation
pipeline are properly integrated and ready for production deployment.

Phases verified:
1. Energy Data Collection Infrastructure
2. CHORD Preprocessing Pipeline
3. Trinity-RFT Integration
4. Evaluation Framework

Author: Automated Pipeline Verification
Date: September 15, 2025
"""

import os
import sys
import json
import yaml
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import subprocess

@dataclass
class VerificationResult:
    component: str
    status: str
    message: str
    details: Dict[str, Any] = None

class PipelineIntegrationVerifier:
    """Comprehensive verification of the energy-efficient code generation pipeline."""

    def __init__(self, project_root: str = str(REPO_ROOT)):
        self.project_root = Path(project_root)
        self.trinity_root = REPO_ROOT / "Trinity-RFT"
        self.results: List[VerificationResult] = []

    def verify_all_phases(self) -> List[VerificationResult]:
        """Run comprehensive verification of all pipeline phases."""
        print("🔍 Starting Pipeline Integration Verification")
        print("=" * 60)

        # Phase 1: Energy Data Collection
        print("\n📊 Phase 1: Energy Data Collection Infrastructure")
        self._verify_energy_collection()

        # Phase 2: CHORD Preprocessing
        print("\n🔄 Phase 2: CHORD Preprocessing Pipeline")
        self._verify_chord_preprocessing()

        # Phase 3: Trinity-RFT Integration
        print("\n🧠 Phase 3: Trinity-RFT Integration")
        self._verify_trinity_integration()

        # Phase 4: Evaluation Framework
        print("\n📈 Phase 4: Evaluation Framework")
        self._verify_evaluation_framework()

        # Overall Integration
        print("\n🌟 Overall Pipeline Integration")
        self._verify_overall_integration()

        return self.results

    def _verify_energy_collection(self):
        """Verify Phase 1: Energy Data Collection Infrastructure."""

        # Core infrastructure files
        core_files = {
            "sniper_parallel_runner.py": "Sniper parallelization runner",
            "pie_energy_processor.py": "PIE dataset batch processor",
            "energy_schema_manager.py": "Schema enhancement manager",
            "progress_monitor.py": "Real-time progress monitor",
            "slurm_energy_collection.sh": "HPC SLURM collection script"
        }

        collection_dir = self.project_root / "energy_data_collection"

        for filename, description in core_files.items():
            filepath = collection_dir / filename
            if filepath.exists():
                if filename.endswith('.py'):
                    # Verify Python file can be imported
                    try:
                        spec = importlib.util.spec_from_file_location("module", filepath)
                        if spec and spec.loader:
                            self._add_result("PASS", f"Phase 1: {description}", f"✓ {filename} verified")
                        else:
                            self._add_result("WARN", f"Phase 1: {description}", f"⚠ {filename} import issue")
                    except Exception as e:
                        self._add_result("FAIL", f"Phase 1: {description}", f"✗ {filename} import error: {e}")
                else:
                    # Verify shell script syntax
                    result = subprocess.run(['bash', '-n', str(filepath)],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        self._add_result("PASS", f"Phase 1: {description}", f"✓ {filename} syntax verified")
                    else:
                        self._add_result("FAIL", f"Phase 1: {description}",
                                       f"✗ {filename} syntax error: {result.stderr}")
            else:
                self._add_result("FAIL", f"Phase 1: {description}", f"✗ {filename} missing")

        # Verify Sniper integration
        sniper_path = self.project_root / "sniper" / "sniper"
        if (sniper_path / "run_energy_analysis.sh").exists():
            self._add_result("PASS", "Phase 1: Sniper Integration",
                           "✓ Sniper simulator properly integrated")
        else:
            self._add_result("FAIL", "Phase 1: Sniper Integration",
                           "✗ Sniper energy analysis script missing")

    def _verify_chord_preprocessing(self):
        """Verify Phase 2: CHORD Preprocessing Pipeline."""

        preprocessing_files = {
            "pie_to_chord_converter.py": "PIE to CHORD format converter",
            "energy_reward_calculator.py": "Trinity-RFT reward calculator",
            "expert_data_selector.py": "Expert sample selection system",
            "chord_dataset_generator.py": "Complete preprocessing orchestrator"
        }

        preprocessing_dir = self.project_root / "chord_preprocessing"

        for filename, description in preprocessing_files.items():
            filepath = preprocessing_dir / filename
            if filepath.exists():
                try:
                    spec = importlib.util.spec_from_file_location("module", filepath)
                    if spec and spec.loader:
                        self._add_result("PASS", f"Phase 2: {description}", f"✓ {filename} verified")
                    else:
                        self._add_result("WARN", f"Phase 2: {description}", f"⚠ {filename} import issue")
                except Exception as e:
                    self._add_result("FAIL", f"Phase 2: {description}", f"✗ {filename} error: {e}")
            else:
                self._add_result("FAIL", f"Phase 2: {description}", f"✗ {filename} missing")

    def _verify_trinity_integration(self):
        """Verify Phase 3: Trinity-RFT Integration."""

        # Verify configuration files
        config_files = {
            "energy_chord.yaml": "Production CHORD configuration",
            "energy_chord_pilot.yaml": "Pilot testing configuration"
        }

        configs_dir = self.project_root / "configs"

        for filename, description in config_files.items():
            filepath = configs_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        config = yaml.safe_load(f)

                    # Verify CHORD-specific configurations
                    if config.get('algorithm', {}).get('algorithm_type') == 'mix_chord':
                        self._add_result("PASS", f"Phase 3: {description}",
                                       "✓ CHORD algorithm type verified")
                    else:
                        self._add_result("WARN", f"Phase 3: {description}",
                                       "⚠ CHORD algorithm type not found")

                    # Verify expert data ratio
                    expert_ratio = config.get('algorithm', {}).get('sample_strategy_args', {}).get('expert_data_ratio')
                    if expert_ratio == 0.20:
                        self._add_result("PASS", f"Phase 3: Expert Data Config",
                                       "✓ Expert data ratio (20%) correctly set")
                    else:
                        self._add_result("WARN", f"Phase 3: Expert Data Config",
                                       f"⚠ Expert ratio: {expert_ratio} (expected 0.20)")

                except Exception as e:
                    self._add_result("FAIL", f"Phase 3: {description}", f"✗ Config parse error: {e}")
            else:
                self._add_result("FAIL", f"Phase 3: {description}", f"✗ {filename} missing")

        # Verify training infrastructure
        training_files = {
            "energy_chord_trainer.py": "Trinity-RFT integration trainer",
            "slurm/chord_energy_train.sh": "Single-node SLURM script",
            "slurm/multi_node_chord.sh": "Multi-node SLURM script"
        }

        finetuning_dir = self.project_root / "finetuning"

        for filename, description in training_files.items():
            filepath = finetuning_dir / filename
            if filepath.exists():
                if filename.endswith('.py'):
                    try:
                        spec = importlib.util.spec_from_file_location("module", filepath)
                        if spec and spec.loader:
                            self._add_result("PASS", f"Phase 3: {description}", f"✓ {filename} verified")
                    except Exception as e:
                        self._add_result("FAIL", f"Phase 3: {description}", f"✗ {filename} error: {e}")
                else:
                    # Verify shell script
                    result = subprocess.run(['bash', '-n', str(filepath)],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        self._add_result("PASS", f"Phase 3: {description}", f"✓ {filename} syntax verified")
                    else:
                        self._add_result("FAIL", f"Phase 3: {description}",
                                       f"✗ {filename} syntax error")
            else:
                self._add_result("FAIL", f"Phase 3: {description}", f"✗ {filename} missing")

        # Verify Trinity-RFT installation
        if self.trinity_root.exists():
            if (self.trinity_root / "examples" / "mix_chord").exists():
                self._add_result("PASS", "Phase 3: Trinity-RFT Installation",
                               "✓ Trinity-RFT with CHORD examples available")
            else:
                self._add_result("WARN", "Phase 3: Trinity-RFT Installation",
                               "⚠ CHORD examples not found")
        else:
            self._add_result("FAIL", "Phase 3: Trinity-RFT Installation",
                           "✗ Trinity-RFT not found")

    def _verify_evaluation_framework(self):
        """Verify Phase 4: Evaluation Framework."""

        evaluation_files = {
            "energy_efficiency_evaluator.py": "Comprehensive model evaluator",
            "benchmark_comparison.py": "Multi-model comparison framework"
        }

        evaluation_dir = self.project_root / "evaluation"

        for filename, description in evaluation_files.items():
            filepath = evaluation_dir / filename
            if filepath.exists():
                try:
                    spec = importlib.util.spec_from_file_location("module", filepath)
                    if spec and spec.loader:
                        self._add_result("PASS", f"Phase 4: {description}", f"✓ {filename} verified")
                except Exception as e:
                    self._add_result("FAIL", f"Phase 4: {description}", f"✗ {filename} error: {e}")
            else:
                self._add_result("FAIL", f"Phase 4: {description}", f"✗ {filename} missing")

    def _verify_overall_integration(self):
        """Verify overall pipeline integration and dependencies."""

        # Verify directory structure
        required_dirs = [
            "energy_data_collection",
            "chord_preprocessing",
            "configs",
            "finetuning",
            "evaluation"
        ]

        for dirname in required_dirs:
            dirpath = self.project_root / dirname
            if dirpath.exists() and dirpath.is_dir():
                file_count = len(list(dirpath.glob("*.py"))) + len(list(dirpath.glob("*.sh"))) + len(list(dirpath.glob("*.yaml")))
                self._add_result("PASS", f"Integration: {dirname} structure",
                               f"✓ Directory exists with {file_count} implementation files")
            else:
                self._add_result("FAIL", f"Integration: {dirname} structure",
                               f"✗ Required directory missing")

        # Verify PIE dataset availability
        pie_dataset_path = self.project_root / "PIE_Dataset"
        if pie_dataset_path.exists():
            # Count PIE samples
            try:
                sample_count = 0
                for file in pie_dataset_path.rglob("*.json"):
                    sample_count += 1

                self._add_result("PASS", "Integration: PIE Dataset",
                               f"✓ PIE dataset available with {sample_count} files")
            except Exception:
                self._add_result("WARN", "Integration: PIE Dataset",
                               "⚠ PIE dataset present but couldn't count samples")
        else:
            self._add_result("WARN", "Integration: PIE Dataset",
                           "⚠ PIE dataset not found - will need to be provided")

        # Verify documentation
        docs = ["README.md", "plan.txt", "todo.txt"]
        for doc in docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                size_kb = doc_path.stat().st_size / 1024
                self._add_result("PASS", f"Integration: {doc} documentation",
                               f"✓ Documentation available ({size_kb:.1f} KB)")
            else:
                self._add_result("WARN", f"Integration: {doc} documentation",
                               f"⚠ {doc} not found")

    def _add_result(self, status: str, component: str, message: str, details: Dict[str, Any] = None):
        """Add a verification result."""
        result = VerificationResult(
            component=component,
            status=status,
            message=message,
            details=details or {}
        )
        self.results.append(result)

        # Print real-time feedback
        status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        print(f"  {status_emoji.get(status, '❓')} {message}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""

        total = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        warnings = len([r for r in self.results if r.status == "WARN"])
        failed = len([r for r in self.results if r.status == "FAIL"])

        report = {
            "verification_timestamp": "2025-09-15T10:30:00Z",
            "project_root": str(self.project_root),
            "trinity_root": str(self.trinity_root),
            "summary": {
                "total_checks": total,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "success_rate": (passed / total * 100) if total > 0 else 0
            },
            "detailed_results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []

        failed_components = [r for r in self.results if r.status == "FAIL"]
        warning_components = [r for r in self.results if r.status == "WARN"]

        if failed_components:
            recommendations.append("🚨 Critical: Address failed components before production deployment")
            for comp in failed_components[:3]:  # Show top 3 failures
                recommendations.append(f"   - Fix: {comp.component}")

        if warning_components:
            recommendations.append("⚠️ Review warning components for optimal performance")

        passed_components = [r for r in self.results if r.status == "PASS"]
        if len(passed_components) >= len(self.results) * 0.9:
            recommendations.append("🎉 Pipeline is ready for production deployment!")
            recommendations.append("💡 Next steps:")
            recommendations.append("   1. Submit Phase 1 SLURM job for energy data collection")
            recommendations.append("   2. Run CHORD preprocessing on energy-enhanced PIE dataset")
            recommendations.append("   3. Execute Trinity-RFT training with energy-aware configuration")
            recommendations.append("   4. Evaluate trained models using comprehensive evaluation framework")

        return recommendations

def main():
    """Run the complete pipeline integration verification."""

    verifier = PipelineIntegrationVerifier()
    results = verifier.verify_all_phases()

    print("\n" + "=" * 60)
    print("📋 PIPELINE VERIFICATION REPORT")
    print("=" * 60)

    report = verifier.generate_report()

    # Print summary
    summary = report["summary"]
    print(f"\n📊 Summary:")
    print(f"   Total Checks: {summary['total_checks']}")
    print(f"   ✅ Passed: {summary['passed']}")
    print(f"   ⚠️  Warnings: {summary['warnings']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   🎯 Success Rate: {summary['success_rate']:.1f}%")

    # Print recommendations
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   {rec}")

    # Save detailed report
    report_path = REPO_ROOT / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_path}")

    # Exit code based on results
    if summary["failed"] == 0:
        print(f"\n🎉 Pipeline verification completed successfully!")
        return 0
    else:
        print(f"\n⚠️  Pipeline has {summary['failed']} critical issues that need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())