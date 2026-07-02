# GRPO Training Debug & Verification Guide

## Status: Code Analysis Complete ✓

### Pre-Flight Checklist

#### 1. **Critical Fixes Already Applied** ✓
- [x] `processing_class=` instead of deprecated `tokenizer=` (line 991)
- [x] Reward function implemented correctly (lines 299-374)
- [x] Prompt template matches SFT format (lines 780-786)
- [x] Test input path handling present
- [x] vLLM configuration valid (tensor_parallel_size=1)

#### 2. **Dataset Requirements**
```bash
# Verify debug datasets exist
ls -lh data/grpo_train_debug.jsonl data/grpo_val_debug.jsonl

# Check dataset format
head -1 data/grpo_train_debug.jsonl | python3 -m json.tool | grep -E "(baseline_code|problem_id|baseline_energy)"
```

**Required fields per sample:**
- `baseline_code` (string): Inefficient code to optimize
- `problem_id` (string): Problem identifier for test cases
- `baseline_energy` (float): Baseline energy measurement
- `baseline_runtime`, `baseline_ipc` (optional but recommended)

#### 3. **Environment Setup**
```bash
# Verify Sniper installation
export SNIPER_ROOT=$HOME/projects/rrg-mrdal22/srajput/green-code-gen/sniper
$SNIPER_ROOT/run-sniper --help

# Check Python packages
python3 -c "import vllm; print('vLLM:', vllm.__version__)"
python3 -c "import unsloth; print('Unsloth: OK')"
python3 -c "from trl import GRPOTrainer; print('TRL GRPO: OK')"
```

## Common Failure Modes & Solutions

### Issue 1: vLLM Out of Memory
**Symptom**: CUDA OOM during generation
**Solution**: Reduce `--gpu-memory-utilization` from 0.50 to 0.35
```bash
# In grpo_debug.sh line 72
--gpu-memory-utilization 0.35  # Was 0.50
```

### Issue 2: Sniper Timeout
**Symptom**: All rewards = -1.0, status = 'timeout'
**Root Cause**: `--simulation-timeout 180` too aggressive for debug dataset
**Solution**: Increase timeout for debug
```bash
--simulation-timeout 300  # 5 minutes for complex problems
```

### Issue 3: Zero Compilations
**Symptom**: All outputs fail compilation
**Possible Causes:**
1. Model too small (0.5B) generates invalid syntax
2. Prompt format mismatch
3. Generation length too short

**Diagnosis:**
```bash
# Check generated code
cat data/grpo_generations/grpo_generations.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    if r.get('status') == 'compile_error':
        print('COMPILE ERROR SAMPLE:')
        print(r.get('generated_code', '')[:500])
        break
"
```

**Solution**: Use larger model for debug (1.5B instead of 0.5B)
```bash
--model "Qwen/Qwen2.5-Coder-1.5B"  # Instead of 0.5B
```

### Issue 4: All Rewards Identical
**Symptom**: `mean(rewards) == min(rewards) == max(rewards)`
**Root Cause**: No diversity in generation (temperature too low)
**Solution**: Already using sampling_params with temperature=0.8 ✓

### Issue 5: Checkpoint Not Saving
**Symptom**: `$CHECKPOINTS -eq 0` in verification
**Root Cause**: Training crashes before first checkpoint interval
**Solution**: Reduce checkpoint interval
```python
# In grpo_trainer_vllm.py training args:
save_steps=10,  # Instead of 50 for debug
```

## Verification Procedure

### Step 1: Run Debug Job
```bash
cd ~/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
sbatch slurm/grpo_debug.sh
```

### Step 2: Monitor Job
```bash
# Watch logs in real-time
tail -f logs/debug_grpo_<JOBID>.out
tail -f logs/debug_grpo_<JOBID>.err

# Check GPU usage
squeue -u $USER
```

### Step 3: Expected Output Pattern
```
[0/X] Training...
  Problem: p00189
  Generating K=4 candidates with vLLM...
  Generated 4 samples in 2.3s
  Compiling 4 candidates in parallel...
  Compiled: 3/4 (75.0%)
  Running correctness tests...
  Passed all tests: 2/3
  Measuring energy via Sniper (parallel)...
  Energy measurements complete (10.2s)
  Rewards: [-1.0, 0.34, 0.52, -0.8]
  Group advantages: [-0.62, 0.21, 0.41, -0.31]
  Loss: 0.0234
  Gradient norm: 0.082
  Updated policy
```

### Step 4: Analyze Results
```bash
# After job completes
cd ~/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

# Check exit status
echo "Exit code: $?"

# Verify checkpoints
ls -lh checkpoints/Qwen_test_grpo/

# Analyze generations
python3 << 'EOF'
import json
from pathlib import Path

gen_file = Path("data/grpo_generations/grpo_generations.jsonl")
if not gen_file.exists():
    print("ERROR: No generation log found")
    exit(1)

records = [json.loads(l) for l in open(gen_file)]
print(f"\n=== GRPO Debug Results ({len(records)} samples) ===\n")

# Status breakdown
from collections import Counter
statuses = Counter(r['status'] for r in records)
print("Status Distribution:")
for status, count in statuses.most_common():
    pct = count / len(records) * 100
    print(f"  {status:20s}: {count:4d} ({pct:5.1f}%)")

# Reward statistics
rewards = [r['reward'] for r in records]
print(f"\nRewards:")
print(f"  Mean:   {sum(rewards)/len(rewards):7.3f}")
print(f"  Median: {sorted(rewards)[len(rewards)//2]:7.3f}")
print(f"  Range:  [{min(rewards):6.3f}, {max(rewards):6.3f}]")

# Compilation rate
compiled = sum(1 for r in records if r.get('compiled', False))
print(f"\nCompilation: {compiled}/{len(records)} ({compiled/len(records)*100:.1f}%)")

# Correctness rate
correct = sum(1 for r in records if r.get('tests_passed', 0) == r.get('num_test_inputs', 1))
print(f"Correctness: {correct}/{len(records)} ({correct/len(records)*100:.1f}%)")

# Energy improvements
improved = sum(1 for r in records if r.get('energy_reduction', 0) > 0)
print(f"Energy improved: {improved}/{len(records)} ({improved/len(records)*100:.1f}%)")

# Success criteria
print(f"\n=== PASS/FAIL Criteria ===")
if compiled == 0:
    print("❌ FAIL: Zero compilations - check model/prompt")
elif compiled < len(records) * 0.3:
    print("⚠️  WARN: Low compilation rate (<30%)")
else:
    print(f"✓ PASS: Compilation rate OK ({compiled/len(records)*100:.1f}%)")

if len(set(rewards)) == 1:
    print("⚠️  WARN: All rewards identical - check generation diversity")
elif max(rewards) - min(rewards) < 0.5:
    print("⚠️  WARN: Low reward variance - check reward function")
else:
    print(f"✓ PASS: Reward variance OK (range: {max(rewards)-min(rewards):.2f})")

if improved == 0:
    print("⚠️  WARN: Zero energy improvements")
else:
    print(f"✓ PASS: {improved} samples showed energy improvement")
EOF
```

## Success Criteria for Debug Run

### Minimal Success (Debug passes)
- [x] Job completes without crashing (exit code 0)
- [x] At least 1 checkpoint saved
- [x] Generation log created with >0 samples
- [x] Compilation rate >30%
- [x] Reward variance >0 (not all identical)

### Good Success (Ready for full training)
- [ ] Compilation rate >50%
- [ ] Correctness rate >30%
- [ ] At least 1 energy improvement observed
- [ ] Reward range spans at least 1.0 (e.g., -1.0 to 0.5)
- [ ] Training loss decreases across steps

### Excellent Success (High confidence)
- [ ] Compilation rate >70%
- [ ] Correctness rate >50%
- [ ] Mean reward >0 (more successes than failures)
- [ ] WandB logging successful with metrics visible

## Troubleshooting Commands

### Check if Sniper works standalone
```bash
cd ~/projects/rrg-mrdal22/srajput/green-code-gen
export SNIPER_ROOT=$PWD/sniper

# Compile simple test
cat > test.cpp << 'EOF'
#include <iostream>
int main() { std::cout << "Hello\n"; return 0; }
EOF

g++ -O3 -std=c++17 -static test.cpp -o test.bin

# Run with Sniper
echo "Hello" | $SNIPER_ROOT/run-sniper \
    --cfg $SNIPER_ROOT/config/epyc_9554p.cfg \
    --power \
    -- ./test.bin

# Check energy output
grep "Energy" sniper.out || echo "ERROR: No energy measurement"
```

### Check vLLM generation works
```bash
cd ~/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
python3 << 'EOF'
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-Coder-0.5B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.3
)

prompt = "Write a C++ hello world program:"
params = SamplingParams(temperature=0.8, max_tokens=100, n=4)

outputs = llm.generate([prompt], params)
print(f"Generated {len(outputs[0].outputs)} samples")
for i, out in enumerate(outputs[0].outputs):
    print(f"\n--- Sample {i+1} ---")
    print(out.text[:200])
EOF
```

## Next Steps After Successful Debug

1. **Scale up dataset**: Use full `grpo_train.jsonl` (~34M, 9,927 samples)
2. **Increase model size**: 0.5B → 7B or 14B
3. **Adjust hyperparameters**:
   - `--num-generations 16` (from 4)
   - `--batch-size 8` (from 1)
   - `--learning-rate 5e-6` (from 1e-6)
4. **Enable full parallelization**: `--cpus-per-task=48` for Sniper pool
5. **Monitor with WandB**: Verify metrics logging works
6. **Run for multiple epochs**: `--num-epochs 3-5`

## Known Issues & Workarounds

### Issue: TRL API Changes
**Fixed**: Using `processing_class=` instead of `tokenizer=`

### Issue: Prompt Format Mismatch
**Fixed**: GRPO prompt matches SFT format exactly (lines 780-786)

### Issue: Reward Function Complexity
**Status**: Simplified to `r_correct + r_energy` (no IPC bonus)

### Issue: Memory Leaks in vLLM
**Workaround**: `del llm; torch.cuda.empty_cache()` after each batch (if needed)

## Contact for Issues

If debug fails after following this guide:
1. Check latest logs in `logs/debug_grpo_*.err`
2. Verify all checklist items above
3. Test Sniper and vLLM independently
4. Review actual error messages (not just exit codes)
