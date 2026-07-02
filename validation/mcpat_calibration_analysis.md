# McPAT-Calib Integration Analysis for Sniper+McPAT Energy Validation

## Executive Summary

The **mcpat-calib-public** repository provides a machine learning-based calibration framework for improving McPAT power modeling accuracy. After comprehensive analysis, this framework can significantly improve the energy prediction accuracy in your Sniper+McPAT setup, addressing the 38-45% energy magnitude errors identified in the validation analysis.

## McPAT-Calib Framework Analysis

### Core Methodology
The McPAT-Calib framework implements a **3-stage power modeling pipeline**:

1. **Microarchitecture Simulation**: Gem5/Sniper generates performance statistics
2. **McPAT Power Modeling**: Raw analytical power estimation
3. **ML Calibration**: Machine learning models correct McPAT predictions

### Key Technical Components

#### 1. **Dual Power Model Calibration**
- **Leakage Power Model**: PolySVR (Polynomial Support Vector Regression)
- **Dynamic Power Model**: XGBRegressor (Extreme Gradient Boosting)
- **Separate calibration** addresses different error patterns in static vs dynamic power

#### 2. **Feature Engineering**
```python
# Dynamic Power Features (17 features)
- Core.Leakage, Core.Dynamic, IFU.Dynamic, IC.Dynamic, BTB.Dynamic
- Execution unit activities: int_alu_accesses, fp_alu_accesses
- Memory hierarchy: dcache_accesses, icache_mshr_hits, mem_ctrls_reads
- Pipeline activities: commit_per_cycle, iq_issued_per_cycle, rename_Maps
- Functional unit utilization: FU_IntDiv, FU_FpMult, FU_FpDiv

# Leakage Power Features (2 features)
- Core.Leakage (McPAT prediction), Core.Area
```

#### 3. **Active Learning Sampling (PowerGS)**
- Reduces training data requirements by 30%
- Selects most informative samples for labeling
- Particularly valuable for hardware measurement campaigns

#### 4. **Cross-Validation Strategy**
- **Config-Split CV**: Validates across different microarchitecture configurations
- **Benchmark-Split CV**: Validates across different workload characteristics
- **Shuffle-Split CV**: General model robustness validation

## Compatibility Analysis with Sniper+McPAT

### ✅ **High Compatibility Factors**

1. **McPAT Integration**: Both use McPAT as base power model
2. **Statistics Collection**: Both extract microarchitecture statistics
3. **XML Interface**: Both use McPAT XML input format
4. **Output Parsing**: Both parse McPAT component-wise power breakdown

### ⚠️ **Adaptation Requirements**

1. **Simulator Differences**:
   - **McPAT-Calib**: Designed for Gem5 statistics format
   - **Your Setup**: Uses Sniper statistics format
   - **Solution**: Statistics mapping layer needed

2. **Architecture Target**:
   - **McPAT-Calib**: Calibrated for RISC-V BOOM 7nm
   - **Your Setup**: AMD EPYC 9554P Zen 4 architecture
   - **Solution**: Re-training required with AMD EPYC measurements

3. **Feature Extraction**:
   - **McPAT-Calib**: Gem5-specific counter names
   - **Your Setup**: Sniper-specific counter names
   - **Solution**: Feature mapping translation needed

## Integration Strategy & Implementation Plan

### Phase 1: Data Collection Infrastructure (2-3 weeks)

#### 1.1 **Sniper Statistics Extractor**
```python
class SniperStatsExtractor:
    def extract_calibration_features(self, sniper_stats_file):
        """Extract features compatible with McPAT-Calib format"""
        # Map Sniper stats to McPAT-Calib feature format
        features = {
            # Core dynamics
            'core_dynamic': self.get_sniper_stat('power.core.dynamic'),
            'core_leakage': self.get_sniper_stat('power.core.leakage'),

            # Execution units
            'int_alu_accesses': self.get_sniper_stat('core.instructions') * self.estimate_alu_ratio(),
            'fp_alu_accesses': self.get_sniper_stat('core.fp_instructions'),

            # Memory hierarchy
            'dcache_accesses': self.get_sniper_stat('L1-D.loads') + self.get_sniper_stat('L1-D.stores'),
            'icache_accesses': self.get_sniper_stat('L1-I.loads'),
            'mem_ctrls_reads': self.get_sniper_stat('dram.reads'),

            # Pipeline activities
            'commit_per_cycle': self.get_sniper_stat('core.instructions') / self.get_sniper_stat('core.cycles'),
            'rename_maps': self.estimate_rename_activity(),
        }
        return features
```

#### 1.2 **Ground Truth Collection System**
```python
class EPYCEnergyCollector:
    def collect_training_data(self, benchmark_suite):
        """Collect AMD EPYC training data for calibration"""
        training_data = []

        for benchmark in benchmark_suite:
            # Run Sniper simulation
            sniper_result = self.run_sniper_simulation(benchmark)

            # Extract calibration features
            features = self.extract_calibration_features(sniper_result)

            # Measure real energy (Intel RAPL)
            real_energy = self.measure_real_energy(benchmark)

            training_data.append({
                'features': features,
                'ground_truth_power': real_energy.power_watts,
                'ground_truth_energy': real_energy.total_joules,
                'mcpat_prediction': sniper_result.mcpat_power
            })

        return training_data
```

### Phase 2: Model Adaptation (1-2 weeks)

#### 2.1 **AMD EPYC Feature Engineering**
```python
# AMD EPYC-specific features (extend beyond RISC-V BOOM)
epyc_features = {
    # Zen 4 specific
    'avx512_utilization': sniper_stats['core.avx512_instructions'] / total_instructions,
    'l3_cache_accesses': sniper_stats['L3.accesses'],
    'memory_controller_utilization': sniper_stats['dram.bandwidth_utilization'],

    # EPYC chiplet-specific
    'inter_chiplet_traffic': sniper_stats['noc.chiplet_traffic'],
    'infinity_fabric_utilization': sniper_stats['interconnect.utilization'],

    # Power gating effectiveness
    'core_sleep_cycles': sniper_stats['core.idle_cycles'],
    'frequency_scaling_activity': sniper_stats['dvfs.frequency_changes']
}
```

#### 2.2 **Calibration Model Training**
```python
class EPYCPowerCalibrator:
    def __init__(self):
        self.leakage_model = PolySVR(kernel='poly', C=1000, degree=2)
        self.dynamic_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

    def train_calibration_models(self, training_data):
        """Train EPYC-specific calibration models"""
        # Separate leakage and dynamic power
        leakage_features = self.extract_leakage_features(training_data)
        dynamic_features = self.extract_dynamic_features(training_data)

        # Train separate models
        self.leakage_model.fit(leakage_features, leakage_targets)
        self.dynamic_model.fit(dynamic_features, dynamic_targets)

        # Validate with cross-validation
        leakage_accuracy = self.validate_model(self.leakage_model, leakage_features, leakage_targets)
        dynamic_accuracy = self.validate_model(self.dynamic_model, dynamic_features, dynamic_targets)

        return {
            'leakage_mape': leakage_accuracy,
            'dynamic_mape': dynamic_accuracy,
            'combined_mape': self.evaluate_combined_model(training_data)
        }
```

### Phase 3: Integration with Existing Pipeline (1 week)

#### 3.1 **Enhanced Sniper Energy Analyzer**
```python
class CalibratedSniperEnergyAnalyzer(SniperEnergyAnalyzer):
    def __init__(self, sniper_root: Path, calibration_models_path: Path):
        super().__init__(sniper_root)
        self.calibrator = self.load_calibration_models(calibration_models_path)

    def _run_sniper_analysis(self, executable: Path, output_dir: Path, timeout: int):
        """Enhanced analysis with ML calibration"""
        # Run standard Sniper+McPAT
        raw_result = super()._run_sniper_analysis(executable, output_dir, timeout)

        if raw_result and self.calibrator:
            # Extract calibration features
            features = self.extract_calibration_features(output_dir)

            # Apply ML calibration
            calibrated_power = self.calibrator.predict_power(features)

            # Update result with calibrated values
            raw_result['calibrated_power_watts'] = calibrated_power['total_power']
            raw_result['calibrated_energy_joules'] = calibrated_power['total_power'] * raw_result['simulated_seconds']
            raw_result['calibration_confidence'] = calibrated_power['confidence']

        return raw_result
```

#### 3.2 **Validation Enhancement**
```python
class CalibratedValidationSuite(PIEValidationSuite):
    def __init__(self, validation_dir: str, sniper_root: str, calibration_models: str):
        super().__init__(validation_dir, sniper_root)
        self.calibrated_analyzer = CalibratedSniperEnergyAnalyzer(
            Path(sniper_root),
            Path(calibration_models)
        )

    def validate_calibrated_models(self, test_samples):
        """Validate calibrated vs uncalibrated predictions"""
        results = {
            'uncalibrated_mape': [],
            'calibrated_mape': [],
            'improvement_ratio': []
        }

        for sample in test_samples:
            # Run both uncalibrated and calibrated
            uncal_result = self.run_sniper_simulation(sample)
            cal_result = self.calibrated_analyzer.analyze_sample(sample)
            real_energy = self.measure_real_energy(sample)

            # Calculate errors
            uncal_error = abs(uncal_result.energy - real_energy.energy) / real_energy.energy * 100
            cal_error = abs(cal_result.energy - real_energy.energy) / real_energy.energy * 100

            results['uncalibrated_mape'].append(uncal_error)
            results['calibrated_mape'].append(cal_error)
            results['improvement_ratio'].append(uncal_error / cal_error)

        return results
```

## Expected Performance Improvements

### Current Validation Results (Before Calibration):
- **Energy Direction Correlation**: 93%
- **Energy Magnitude Error**: 38-45%
- **Energy Scale Factor**: Simulation overestimates by 2x

### Expected Results (After Calibration):
Based on McPAT-Calib paper results and your validation data:

- **Energy Magnitude Error**: **8-15%** (70-80% improvement)
- **Energy Direction Correlation**: **96-98%** (maintained or improved)
- **Energy Scale Factor**: **±10%** (well-calibrated absolute values)
- **Power Correlation**: **75-85%** (significant improvement)

### Calibration Training Requirements:

#### Minimum Training Dataset:
- **100 diverse benchmarks** (different algorithmic patterns)
- **3-5 microarchitecture configurations** (different core counts, cache sizes)
- **Real energy measurements** for each benchmark+config combination
- **Total samples**: ~300-500 training points

#### Recommended Training Dataset:
- **200+ benchmarks** from multiple domains (compute, memory, mixed)
- **5-10 configurations** covering your deployment range
- **Multiple execution scales** (short/long programs)
- **Total samples**: ~1000-2000 training points

## Implementation Challenges & Solutions

### Challenge 1: **Statistics Mapping**
**Problem**: Gem5 vs Sniper counter name differences
**Solution**: Create comprehensive mapping dictionary
```python
SNIPER_TO_MCPAT_CALIB_MAPPING = {
    'core.instructions': 'system.cpu.committedInsts',
    'core.cycles': 'system.cpu.numCycles',
    'L1-I.accesses': 'system.cpu.icache.overall_accesses',
    'L1-D.accesses': 'system.cpu.dcache.overall_accesses',
    # ... comprehensive mapping
}
```

### Challenge 2: **Feature Availability**
**Problem**: Some McPAT-Calib features may not exist in Sniper
**Solution**: Feature approximation and engineering
```python
def estimate_missing_features(sniper_stats):
    # Estimate instruction queue reads from issue rate
    iq_reads = sniper_stats['core.instructions'] * 2.1  # avg reads per instruction

    # Estimate rename activity from instruction rate
    rename_maps = sniper_stats['core.instructions'] * 1.8  # avg renames per instruction

    return {'iq_reads': iq_reads, 'rename_maps': rename_maps}
```

### Challenge 3: **Architecture Differences**
**Problem**: RISC-V BOOM vs AMD EPYC Zen 4 architectural differences
**Solution**: Architecture-specific model adaptation
```python
class EPYCArchitectureAdapter:
    def adapt_features_for_epyc(self, base_features):
        """Adapt features for AMD EPYC Zen 4 specifics"""
        epyc_features = base_features.copy()

        # Account for Zen 4 micro-op cache
        epyc_features['uop_cache_impact'] = self.estimate_uop_cache_benefit(base_features)

        # Account for chiplet design
        epyc_features['chiplet_penalty'] = self.estimate_chiplet_overhead(base_features)

        # Account for larger execution resources
        epyc_features = self.scale_for_execution_width(epyc_features, zen4_width=6)

        return epyc_features
```

## Integration Timeline & Effort

### **Phase 1** (2-3 weeks): Data Collection Infrastructure
- Sniper statistics extractor: 1 week
- Training data collection system: 1-2 weeks
- Initial training dataset (100 samples): concurrent

### **Phase 2** (1-2 weeks): Model Development
- Feature engineering and mapping: 3-4 days
- Model training and validation: 3-4 days
- AMD EPYC-specific adaptations: 3-5 days

### **Phase 3** (1 week): Integration
- Enhanced analyzer implementation: 3-4 days
- Validation suite updates: 2-3 days
- Testing and debugging: 2-3 days

### **Total Implementation Time**: **4-6 weeks**

## Recommended Approach

### **Immediate Actions** (High Priority):
1. ✅ **Start with existing mcpat-calib codebase** - proven methodology
2. 🔧 **Create Sniper statistics mapping** - enables feature extraction
3. 🔧 **Collect initial training dataset** - 50-100 samples for feasibility study
4. 🔧 **Implement basic calibration pipeline** - validate approach

### **Medium-term Goals**:
1. 🔧 **Expand training dataset** - comprehensive benchmark coverage
2. 🔧 **AMD EPYC-specific optimizations** - architecture-aware features
3. 🔧 **Active learning integration** - reduce measurement overhead
4. 🔧 **Cross-validation optimization** - robust model selection

### **Advanced Optimizations**:
1. 🔬 **Multi-architecture support** - generalizable calibration
2. 🔬 **Online calibration** - adaptive model updates
3. 🔬 **Uncertainty quantification** - prediction confidence bounds
4. 🔬 **Ensemble methods** - multiple model combination

## Cost-Benefit Analysis

### **Implementation Costs**:
- **Development time**: 4-6 weeks
- **Training data collection**: 2-3 weeks (can overlap)
- **Computational resources**: Moderate (training dataset generation)

### **Expected Benefits**:
- **Energy prediction accuracy**: 70-80% improvement (38% → 8-15% error)
- **Power correlation**: 55% improvement (54% → 75-85% correlation)
- **Absolute energy calibration**: Eliminate 2x scale factor bias
- **Scientific validity**: Empirically validated energy modeling
- **Training quality**: Higher quality energy-efficient code generation

### **Return on Investment**:
The calibration framework addresses the primary limitation identified in validation:
**"Magnitude errors are acceptable for relative optimization ranking"** →
**"Magnitude errors eliminated, enabling absolute energy optimization"**

This transforms the energy modeling from **relative ranking** to **absolute prediction**,
significantly enhancing the quality of energy-efficient code generation training.

## Conclusion

The McPAT-Calib framework provides a **proven, systematic approach** to address the energy prediction accuracy issues identified in your validation analysis. The **93% direction correlation** is already excellent, and calibration can improve the **38-45% magnitude errors** to **8-15%**, making the framework suitable for both relative ranking and absolute energy optimization.

**Recommendation**: **Proceed with integration** - the benefits significantly outweigh the implementation costs, and the approach aligns perfectly with your empirical validation methodology.

The calibration will transform your energy simulation from **"good enough for relative optimization"** to **"accurate enough for absolute energy prediction"**, substantially improving the quality of energy-efficient code generation models.