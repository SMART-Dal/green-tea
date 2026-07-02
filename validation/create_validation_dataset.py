#!/usr/bin/env python3
"""
Create Validation Dataset for Sniper+McPAT Energy Analysis

This script:
1. Randomly samples 100 entries from PIE train.jsonl
2. Creates a validation dataset file
3. Copies corresponding test input/output files
4. Generates C++ source files for validation

Author: Energy-Efficient Code Generation Pipeline
"""

import json
import random
import shutil
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
from typing import List, Dict, Any
import argparse

class ValidationDatasetCreator:
    """Creates validation dataset with random PIE samples and their test cases."""

    def __init__(self, pie_dataset_root: str, validation_dir: str):
        self.pie_root = Path(pie_dataset_root)
        self.validation_dir = Path(validation_dir)
        self.testcases_dir = self.pie_root / "extracted_testcases" / "merged_test_cases"

        # Create validation subdirectories
        self.samples_dir = self.validation_dir / "samples"
        self.testcases_validation_dir = self.validation_dir / "testcases"
        self.cpp_files_dir = self.validation_dir / "cpp_files"

        # Create directories
        for dir_path in [self.samples_dir, self.testcases_validation_dir, self.cpp_files_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_train_data(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load PIE training data."""
        train_file = self.pie_root / "train.jsonl"

        if not train_file.exists():
            raise FileNotFoundError(f"PIE train.jsonl not found: {train_file}")

        samples = []
        with open(train_file, 'r') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {idx}: {e}")
                    continue

        print(f"Loaded {len(samples)} samples from train.jsonl")
        return samples

    def sample_random_entries(self, samples: List[Dict[str, Any]], n_samples: int = 100) -> List[Dict[str, Any]]:
        """Randomly sample n_samples from the dataset."""
        if len(samples) < n_samples:
            print(f"Warning: Only {len(samples)} samples available, using all")
            return samples

        # Set deterministic seed for reproducible validation dataset
        random.seed(42)  # Fixed seed ensures same 100 samples every time
        sampled = random.sample(samples, n_samples)

        print(f"Randomly sampled {len(sampled)} entries")
        return sampled

    def copy_test_cases(self, problem_id: str) -> Dict[str, int]:
        """Copy test cases for a given problem_id."""
        source_testcase_dir = self.testcases_dir / problem_id
        target_testcase_dir = self.testcases_validation_dir / problem_id

        if not source_testcase_dir.exists():
            print(f"Warning: Test cases not found for problem {problem_id}")
            return {"input_files": 0, "output_files": 0}

        # Create target directory
        target_testcase_dir.mkdir(parents=True, exist_ok=True)

        # Copy all input and output files
        input_files = 0
        output_files = 0

        for file_path in source_testcase_dir.glob("*"):
            if file_path.is_file():
                target_file = target_testcase_dir / file_path.name
                shutil.copy2(file_path, target_file)

                if file_path.name.startswith("input."):
                    input_files += 1
                elif file_path.name.startswith("output."):
                    output_files += 1

        return {"input_files": input_files, "output_files": output_files}

    def create_cpp_files(self, sample: Dict[str, Any], index: int) -> Dict[str, str]:
        """Create C++ source and target files for a sample."""
        sample_id = sample.get('src_id', f'sample_{index}')
        problem_id = sample.get('problem_id', 'unknown')

        # Source file
        src_filename = f"{index:03d}_{problem_id}_{sample_id}_src.cpp"
        src_path = self.cpp_files_dir / src_filename

        with open(src_path, 'w') as f:
            f.write(f"// PIE Validation Sample {index}\n")
            f.write(f"// Problem ID: {problem_id}\n")
            f.write(f"// Source ID: {sample_id}\n")
            f.write(f"// Source Version (Less Efficient)\n\n")
            f.write(sample.get('src_code', ''))

        # Target file
        tgt_id = sample.get('tgt_id', f'tgt_{index}')
        tgt_filename = f"{index:03d}_{problem_id}_{tgt_id}_tgt.cpp"
        tgt_path = self.cpp_files_dir / tgt_filename

        with open(tgt_path, 'w') as f:
            f.write(f"// PIE Validation Sample {index}\n")
            f.write(f"// Problem ID: {problem_id}\n")
            f.write(f"// Target ID: {tgt_id}\n")
            f.write(f"// Target Version (More Efficient)\n\n")
            f.write(sample.get('tgt_code', ''))

        return {
            "src_file": str(src_path),
            "tgt_file": str(tgt_path),
            "src_filename": src_filename,
            "tgt_filename": tgt_filename
        }

    def create_validation_dataset(self, n_samples: int = 100) -> str:
        """Create complete validation dataset."""
        print(f"Creating validation dataset with {n_samples} samples...")

        # Load and sample data
        all_samples = self.load_train_data()
        sampled_data = self.sample_random_entries(all_samples, n_samples)

        # Process each sample
        validation_samples = []
        testcase_stats = {"total_problems": 0, "total_input_files": 0, "total_output_files": 0}

        for idx, sample in enumerate(sampled_data):
            print(f"Processing sample {idx + 1}/{len(sampled_data)}: {sample.get('problem_id', 'unknown')}")

            # Copy test cases
            problem_id = sample.get('problem_id', 'unknown')
            if problem_id != 'unknown':
                test_stats = self.copy_test_cases(problem_id)
                testcase_stats["total_problems"] += 1
                testcase_stats["total_input_files"] += test_stats["input_files"]
                testcase_stats["total_output_files"] += test_stats["output_files"]

            # Create C++ files
            cpp_files = self.create_cpp_files(sample, idx)

            # Add metadata for validation
            validation_sample = sample.copy()
            validation_sample.update({
                "validation_index": idx,
                "cpp_files": cpp_files,
                "testcase_dir": str(self.testcases_validation_dir / problem_id) if problem_id != 'unknown' else None,
                "has_testcases": (self.testcases_validation_dir / problem_id).exists() if problem_id != 'unknown' else False
            })

            validation_samples.append(validation_sample)

        # Save validation dataset
        validation_file = self.validation_dir / "pie_validation_100_samples.jsonl"
        with open(validation_file, 'w') as f:
            for sample in validation_samples:
                f.write(json.dumps(sample) + '\n')

        # Create summary report
        self.create_summary_report(validation_samples, testcase_stats, validation_file)

        print(f"\n✅ Validation dataset created successfully!")
        print(f"📄 Dataset file: {validation_file}")
        print(f"📁 C++ files: {self.cpp_files_dir}")
        print(f"🧪 Test cases: {self.testcases_validation_dir}")

        return str(validation_file)

    def create_summary_report(self, samples: List[Dict[str, Any]], testcase_stats: Dict[str, int], dataset_file: Path):
        """Create a summary report of the validation dataset."""

        report_file = self.validation_dir / "validation_dataset_summary.md"

        # Analyze samples
        problems_with_testcases = sum(1 for s in samples if s.get('has_testcases', False))
        unique_problems = len(set(s.get('problem_id', 'unknown') for s in samples))
        avg_speedup = sum(s.get('speedup', 1.0) for s in samples) / len(samples)

        # Find speedup range
        speedups = [s.get('speedup', 1.0) for s in samples]
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        with open(report_file, 'w') as f:
            f.write("# PIE Validation Dataset Summary\n\n")
            f.write(f"**Generated**: {Path(__file__).name}\n")
            f.write(f"**Dataset File**: `{dataset_file.name}`\n")
            f.write(f"**Total Samples**: {len(samples)}\n\n")

            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Unique Problems**: {unique_problems}\n")
            f.write(f"- **Problems with Test Cases**: {problems_with_testcases}/{len(samples)} ({problems_with_testcases/len(samples)*100:.1f}%)\n")
            f.write(f"- **Average Speedup**: {avg_speedup:.2f}x\n")
            f.write(f"- **Speedup Range**: {min_speedup:.2f}x - {max_speedup:.2f}x\n\n")

            f.write("## Test Cases\n\n")
            f.write(f"- **Total Problems with Test Cases**: {testcase_stats['total_problems']}\n")
            f.write(f"- **Total Input Files**: {testcase_stats['total_input_files']}\n")
            f.write(f"- **Total Output Files**: {testcase_stats['total_output_files']}\n\n")

            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write("validation/\n")
            f.write("├── pie_validation_100_samples.jsonl\n")
            f.write("├── validation_dataset_summary.md\n")
            f.write("├── cpp_files/\n")
            f.write("│   ├── 000_pXXXXX_sXXXXXXXXX_src.cpp\n")
            f.write("│   ├── 000_pXXXXX_sXXXXXXXXX_tgt.cpp\n")
            f.write("│   └── ...\n")
            f.write("└── testcases/\n")
            f.write("    ├── p00000/\n")
            f.write("    │   ├── input.0.txt\n")
            f.write("    │   ├── output.0.txt\n")
            f.write("    │   └── ...\n")
            f.write("    └── ...\n")
            f.write("```\n\n")

            f.write("## Sample Problems (First 10)\n\n")
            f.write("| Index | Problem ID | Speedup | Has Tests | Src ID | Tgt ID |\n")
            f.write("|-------|------------|---------|-----------|---------|--------|\n")

            for i, sample in enumerate(samples[:10]):
                f.write(f"| {i:03d} | {sample.get('problem_id', 'unknown')} | {sample.get('speedup', 1.0):.2f}x | {'✅' if sample.get('has_testcases', False) else '❌'} | {sample.get('src_id', 'unknown')} | {sample.get('tgt_id', 'unknown')} |\n")

            if len(samples) > 10:
                f.write(f"| ... | ... | ... | ... | ... | ... |\n")
                f.write(f"| Total: {len(samples)} samples | | | | | |\n")

            f.write("\n## Usage\n\n")
            f.write("This validation dataset can be used to:\n")
            f.write("1. Test Sniper+McPAT energy analysis on real PIE samples\n")
            f.write("2. Validate energy calculation accuracy\n")
            f.write("3. Compare energy measurements between src and tgt versions\n")
            f.write("4. Benchmark the energy data collection pipeline\n\n")

            f.write("### Example Usage\n\n")
            f.write("```bash\n")
            f.write("# Compile and test a sample\n")
            f.write("g++ -O3 cpp_files/000_p03352_s743140059_src.cpp -o test_src\n")
            f.write("g++ -O3 cpp_files/000_p03352_s147468699_tgt.cpp -o test_tgt\n")
            f.write("\n")
            f.write("# Run with test input\n")
            f.write("./test_src < testcases/p03352/input.0.txt\n")
            f.write("./test_tgt < testcases/p03352/input.0.txt\n")
            f.write("\n")
            f.write("# Energy analysis with Sniper\n")
            f.write("cd ../../sniper/sniper\n")
            f.write("./run-sniper -n 1 -c config/epyc_9554p.cfg -d validation_output --power -- ../../validation/test_tgt < ../../validation/testcases/p03352/input.0.txt\n")
            f.write("```\n")

        print(f"📋 Summary report created: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Create PIE validation dataset")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to extract")
    parser.add_argument("--pie-root", default=str(REPO_ROOT / "PIE_Dataset"),
                       help="PIE dataset root directory")
    parser.add_argument("--validation-dir", default=str(REPO_ROOT / "validation"),
                       help="Validation directory")

    args = parser.parse_args()

    print("🔍 Creating PIE Validation Dataset")
    print("=" * 50)
    print(f"PIE Dataset: {args.pie_root}")
    print(f"Validation Directory: {args.validation_dir}")
    print(f"Number of Samples: {args.samples}")
    print("=" * 50)

    try:
        creator = ValidationDatasetCreator(args.pie_root, args.validation_dir)
        dataset_file = creator.create_validation_dataset(args.samples)

        print("\n🎉 Validation dataset creation completed successfully!")
        print(f"Use the dataset for Sniper+McPAT validation testing.")

        return 0

    except Exception as e:
        print(f"❌ Error creating validation dataset: {e}")
        return 1

if __name__ == "__main__":
    exit(main())