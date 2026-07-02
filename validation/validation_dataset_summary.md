# PIE Validation Dataset Summary

**Generated**: create_validation_dataset.py
**Dataset File**: `pie_validation_100_samples.jsonl`
**Total Samples**: 100

## Dataset Statistics

- **Unique Problems**: 68
- **Problems with Test Cases**: 100/100 (100.0%)
- **Average Speedup**: 4.55x
- **Speedup Range**: 1.11x - 27.44x

## Test Cases

- **Total Problems with Test Cases**: 100
- **Total Input Files**: 8978
- **Total Output Files**: 8982

## Directory Structure

```
validation/
├── pie_validation_100_samples.jsonl
├── validation_dataset_summary.md
├── cpp_files/
│   ├── 000_pXXXXX_sXXXXXXXXX_src.cpp
│   ├── 000_pXXXXX_sXXXXXXXXX_tgt.cpp
│   └── ...
└── testcases/
    ├── p00000/
    │   ├── input.0.txt
    │   ├── output.0.txt
    │   └── ...
    └── ...
```

## Sample Problems (First 10)

| Index | Problem ID | Speedup | Has Tests | Src ID | Tgt ID |
|-------|------------|---------|-----------|---------|--------|
| 000 | p02960 | 1.11x | ✅ | s165197801 | s284290848 |
| 001 | p03769 | 1.23x | ✅ | s430540074 | s434545279 |
| 002 | p02723 | 5.33x | ✅ | s462811989 | s672730515 |
| 003 | p02629 | 5.37x | ✅ | s499974225 | s868002642 |
| 004 | p03164 | 2.11x | ✅ | s351122842 | s046681716 |
| 005 | p02948 | 3.51x | ✅ | s184429370 | s976970780 |
| 006 | p03965 | 6.66x | ✅ | s603179693 | s903661032 |
| 007 | p02658 | 5.28x | ✅ | s483056776 | s010757892 |
| 008 | p03277 | 2.51x | ✅ | s260954079 | s934061417 |
| 009 | p03838 | 5.23x | ✅ | s071133047 | s227660755 |
| ... | ... | ... | ... | ... | ... |
| Total: 100 samples | | | | | |

## Usage

This validation dataset can be used to:
1. Test Sniper+McPAT energy analysis on real PIE samples
2. Validate energy calculation accuracy
3. Compare energy measurements between src and tgt versions
4. Benchmark the energy data collection pipeline

### Example Usage

```bash
# Compile and test a sample
g++ -O3 cpp_files/000_p03352_s743140059_src.cpp -o test_src
g++ -O3 cpp_files/000_p03352_s147468699_tgt.cpp -o test_tgt

# Run with test input
./test_src < testcases/p03352/input.0.txt
./test_tgt < testcases/p03352/input.0.txt

# Energy analysis with Sniper
cd ../../sniper/sniper
./run-sniper -n 1 -c config/epyc_9554p.cfg -d validation_output --power -- ../../validation/test_tgt < ../../validation/testcases/p03352/input.0.txt
```
