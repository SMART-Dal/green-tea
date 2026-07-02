#!/usr/bin/env python3
"""
Quick Test Script for PIE Validation Suite

This script runs a quick test of the validation framework on a small subset
of samples to verify that everything is working correctly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Add the validation directory to path
sys.path.append(str(Path(__file__).parent))

from run_validation_suite import PIEValidationSuite

def main():
    print("🧪 Quick Validation Test")
    print("=" * 30)

    validation_dir = str(REPO_ROOT / "validation")
    sniper_root = str(REPO_ROOT / "sniper")

    try:
        # Test with just 3 samples and 1 test case each
        suite = PIEValidationSuite(validation_dir, sniper_root)

        print("Testing with 3 samples...")
        report = suite.run_validation_suite(max_samples=3, num_test_cases=1)

        # Print quick summary
        if 'validation_summary' in report:
            summary = report['validation_summary']
            accuracy = report['validation_accuracy']

            print(f"\n✅ Quick Test Results:")
            print(f"   Success Rate: {summary['success_rate']:.1f}%")
            print(f"   Speedup Correlation: {accuracy['speedup_correlation_mean']:.2f}")
            print(f"   Correct Predictions: {accuracy['samples_with_correct_direction']}")

            if summary['success_rate'] > 50:
                print(f"\n🎉 Validation framework is working correctly!")
                return 0
            else:
                print(f"\n⚠️ Low success rate - check configuration")
                return 1
        else:
            print(f"\n❌ No validation results generated")
            return 1

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())