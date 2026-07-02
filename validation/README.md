# PIE Validation Suite

This directory contains a comprehensive validation framework for testing the accuracy of Sniper+McPAT energy simulation against real execution metrics.

## 🎯 Purpose

The validation suite verifies that Sniper+McPAT energy simulations provide reliable **relative** performance and energy measurements needed for training energy-efficient code generation models. It compares:

- ⚡ **Real execution time** vs **Simulated execution time**
- 🏃 **Real speedup ratios** vs **Simulated speedup ratios**
- 🔋 **Energy consumption patterns** between optimized/unoptimized code pairs
- 📊 **Correlation accuracy** for ranking code variants by efficiency

## 📁 Directory Structure

```
validation/
├── README.md                          # This documentation
├── create_validation_dataset.py       # Dataset creation script
├── run_validation_suite.py           # Main validation framework
├── test_validation.py                # Quick test script
├── detailed_energy_analyzer.py       # Energy analysis utilities
├── pie_validation_100_samples.jsonl  # 100 PIE samples (deterministic)
├── validation_dataset_summary.md     # Dataset documentation
├── cpp_files/                        # 200 C++ files (100 src + 100 tgt)
│   ├── 000_p02960_s123456789_src.cpp
│   ├── 000_p02960_s987654321_tgt.cpp
│   └── ...
├── testcases/                        # Test input/output files
│   ├── p02960/
│   │   ├── input.0.txt
│   │   ├── output.0.txt
│   │   └── ...
│   └── ...
└── validation_results/               # Generated validation reports
    └── validation_results.json
```

## 🚀 Quick Start

### 1. Run Quick Test (3 samples)
```bash
cd validation
python test_validation.py
```

### 2. Run Full Validation Suite (100 samples)
```bash
python run_validation_suite.py
```

### 3. Run Custom Validation
```bash
# Validate 10 samples with 5 test cases each
python run_validation_suite.py --max-samples 10 --test-cases 5
```

## 📊 Validation Metrics

The suite measures several key validation metrics:

### **Performance Correlation**
- **Real Speedup**: `real_src_time / real_tgt_time`
- **Simulated Speedup**: `sim_src_time / sim_tgt_time`
- **Speedup Correlation**: Whether simulation predicts correct optimization direction

### **Energy Analysis**
- **Energy Reduction Ratio**: `src_energy / tgt_energy`
- **Component Breakdown**: Core, Cache, DRAM energy distribution
- **Power Patterns**: Static vs Dynamic power consumption

### **Accuracy Scores**
- **Success Rate**: Percentage of samples successfully validated
- **Direction Accuracy**: Correct prediction of which version is faster/more efficient
- **Timing Correlation**: How well simulated times correlate with real execution

## 🔬 How Validation Works

### **Real Execution Phase**
1. **Compile** source and target C++ files with `-O3`
2. **Execute** both versions with real test inputs
3. **Monitor** runtime, CPU usage, memory consumption
4. **Verify** output correctness against expected results

### **Simulation Phase**
1. **Run Sniper** simulation with same inputs
2. **Extract** performance metrics (instructions, cycles, IPC)
3. **Calculate** energy consumption via McPAT integration
4. **Parse** component-wise energy breakdown

### **Comparison Phase**
1. **Compare** speedup ratios between real and simulated
2. **Evaluate** energy reduction patterns
3. **Score** correlation accuracy and prediction reliability
4. **Generate** comprehensive validation report

## 📈 Expected Results

### **Good Validation Indicators**
- ✅ **Success Rate > 80%**: Most samples validate successfully
- ✅ **Speedup Correlation > 0.7**: Simulation predicts optimization direction correctly
- ✅ **Direction Accuracy > 90%**: Correctly identifies which version is more efficient
- ✅ **Energy Ratios 1.1-5.0x**: Realistic energy improvements for optimized code

### **Validation Report Example**
```json
{
  "validation_summary": {
    "success_rate": 87.5,
    "successful_validations": 87,
    "total_samples": 100
  },
  "validation_accuracy": {
    "speedup_correlation_mean": 0.73,
    "speedup_correlation_rate": 84.2,
    "samples_with_correct_direction": 89
  },
  "performance_analysis": {
    "real_speedup_stats": {
      "mean": 2.34,
      "median": 1.87,
      "min": 1.02,
      "max": 12.45
    }
  }
}
```

## ⚙️ Configuration

### **Key Parameters**
- **`max_samples`**: Number of validation samples (default: all 100)
- **`test_cases`**: Test cases per sample (default: 3)
- **`timeout`**: Execution timeout in seconds (default: 120 for Sniper)

### **Architecture Settings**
- **Sniper Config**: `config/epyc_9554p.cfg` (EPYC 9554P architecture)
- **Compilation**: `-O3 -std=c++17` (optimized compilation)
- **Cores**: Single core simulation (`-n 1`)

## 🎯 Use Cases

### **1. Validate Sniper Setup**
Ensure Sniper+McPAT is correctly configured and producing reliable energy measurements.

### **2. Benchmark Simulation Accuracy**
Quantify how well simulated metrics correlate with real execution for relative comparisons.

### **3. Energy Model Validation**
Verify that energy patterns between optimized/unoptimized code pairs are realistic.

### **4. Pipeline Integration Testing**
Test the complete energy data collection pipeline before running on the full 170k PIE dataset.

## 🚨 Troubleshooting

### **Common Issues**

**Low Success Rate (<50%)**
- Check Sniper installation and configuration
- Verify test case files are present and readable
- Ensure adequate system resources

**Compilation Failures**
- Check g++ compiler installation
- Verify C++17 standard support
- Review C++ file syntax in `cpp_files/`

**Sniper Timeouts**
- Increase timeout values for complex samples
- Check Sniper configuration validity
- Monitor system memory usage

**Missing Test Cases**
- Verify `testcases/` directory structure
- Check PIE dataset extraction completeness
- Ensure input/output file pairing

## 🔧 Advanced Usage

### **Custom Validation Sets**
```bash
# Create new validation dataset with different seed
python create_validation_dataset.py --samples 50

# Run validation on specific problems
python run_validation_suite.py --max-samples 20 --test-cases 1
```

### **Energy Analysis Only**
```bash
# Use detailed energy analyzer for specific files
python detailed_energy_analyzer.py cpp_files/000_p02960_s123456789_tgt.cpp
```

### **Batch Processing**
```bash
# Run validation in parallel (if supported)
for i in {1..10}; do
    python run_validation_suite.py --max-samples 10 &
done
wait
```

## 📝 Next Steps

After successful validation:

1. **Deploy Energy Collection**: Run full 170k PIE dataset energy enhancement
2. **CHORD Preprocessing**: Convert energy-enhanced data to Trinity-RFT format
3. **Model Training**: Train energy-efficient code generation models
4. **Evaluation**: Assess trained model performance

---

**🎉 The validation framework ensures reliable energy simulation for training energy-efficient code generation models!**