# Speed Up Guide: Redistributing Remaining Work

## Problem
When running 440 SLURM jobs, most finish quickly because their batches are mostly completed (cached). Only ~130 jobs have significant remaining work, leading to poor parallelization.

## Solution
Redistribute the remaining uncompleted work evenly across all 440 jobs to maximize parallelization.

---

## Quick Start

### Step 1: Check Current Status
```bash
cd /project/6090549/srajput/green-code-gen

# Check completed work
find pie_energy_cache/completed -name "*.done" | wc -l

# Check failed work
find pie_energy_cache/failed -name "*.failed" | wc -l

# Check total executions across all batches
cat PIE_Dataset/batches/execution_master_batch_*.jsonl | wc -l
```

### Step 2: Dry Run (Preview)
```bash
python3 energy_data_collection/redistribute_remaining_work.py \
    --batch-dir PIE_Dataset/batches \
    --cache-dir pie_energy_cache \
    --output-dir PIE_Dataset/batches_remaining \
    --num-jobs 440 \
    --dry-run
```

This will show you:
- How many executions are remaining
- How they'll be distributed
- No files will be created

### Step 3: Create Redistributed Batches
```bash
python3 energy_data_collection/redistribute_remaining_work.py \
    --batch-dir PIE_Dataset/batches \
    --cache-dir pie_energy_cache \
    --output-dir PIE_Dataset/batches_remaining \
    --num-jobs 440
```

Expected output:
```
Redistributing X executions across 440 batch files
Output directory: PIE_Dataset/batches_remaining
Total batch files: 440
```

### Step 4: Verify Redistributed Batches
```bash
# Check that 440 batch files were created
ls PIE_Dataset/batches_remaining/execution_master_batch_*.jsonl | wc -l

# Check distribution
for f in PIE_Dataset/batches_remaining/execution_master_batch_{000..009}.jsonl; do
    echo "$f: $(wc -l < $f) executions"
done
```

### Step 5: Submit SLURM Jobs
The SLURM script is already configured to use redistributed batches!

```bash
sbatch energy_data_collection/slurm_execution_master.sh
```

Monitor progress:
```bash
squeue -u $USER
```

---

## Configuration Details

### SLURM Script Configuration
The file `slurm_execution_master.sh` (lines 46-73) has two options:

**Option 1 - Original Batches (commented out):**
```bash
#BATCH_SOURCE="${BATCH_DIR}"  # PIE_Dataset/batches
```

**Option 2 - Redistributed Batches (ACTIVE):**
```bash
BATCH_SOURCE="${PIE_DATASET}/batches_remaining"
```

### Switching Between Modes

**To use redistributed batches (recommended for speedup):**
- Keep line 71 uncommented: `BATCH_SOURCE="${PIE_DATASET}/batches_remaining"`
- Comment line 68: `#BATCH_SOURCE="${BATCH_DIR}"`

**To use original batches (run everything from scratch):**
- Uncomment line 68: `BATCH_SOURCE="${BATCH_DIR}"`
- Comment line 71: `#BATCH_SOURCE="${PIE_DATASET}/batches_remaining"`

---

## Safety Features

### What's Protected
1. **Original batch files** - Never modified
2. **Cache directory** - Never deleted or modified
3. **Completed work** - Always skipped (via SharedExecutionCache)
4. **Dataset** - Never touched

### What's Created
1. New directory: `PIE_Dataset/batches_remaining/`
2. 440 new batch files with remaining work only
3. Log files in `logs/` as usual

### Easy Rollback
To go back to original batches:
1. Edit `slurm_execution_master.sh`
2. Change `BATCH_SOURCE` to `"${BATCH_DIR}"`
3. Re-submit jobs

---

## Expected Performance Improvement

### Before (Current Situation)
```
440 jobs submitted
~310 jobs finish in < 5 minutes (mostly cached)
~130 jobs run for 7 days
→ Underutilized parallelization
```

### After (With Redistribution)
```
440 jobs submitted
All 440 jobs process remaining work evenly
All jobs finish around the same time
→ Maximum parallelization efficiency
```

### Speedup Calculation
If you have ~130 jobs worth of remaining work:
- **Before:** 130 jobs × 7 days = 910 job-days
- **After:** Redistributed across 440 jobs = ~2.9 days × 440 jobs = ~910 job-days
- **Wall-clock time:** ~7 days → ~2.9 days = **2.4× faster!**

Actual speedup depends on remaining work amount.

---

## Troubleshooting

### Issue: "Batch file not found"
**Solution:** Make sure you created redistributed batches:
```bash
python3 energy_data_collection/redistribute_remaining_work.py \
    --batch-dir PIE_Dataset/batches \
    --cache-dir pie_energy_cache \
    --output-dir PIE_Dataset/batches_remaining \
    --num-jobs 440
```

### Issue: "No remaining executions found"
**Solution:** All work is complete! Check:
```bash
find pie_energy_cache/completed -name "*.done" | wc -l
```

### Issue: Jobs finishing too quickly
**Solution:** The work might already be done. Check cache:
```bash
# Compare completed vs total
COMPLETED=$(find pie_energy_cache/completed -name "*.done" | wc -l)
TOTAL=$(cat PIE_Dataset/batches/execution_master_batch_*.jsonl | wc -l)
echo "Progress: $COMPLETED / $TOTAL"
```

### Issue: Want to regenerate redistributed batches
**Solution:** Simply delete and recreate:
```bash
rm -rf PIE_Dataset/batches_remaining
python3 energy_data_collection/redistribute_remaining_work.py \
    --batch-dir PIE_Dataset/batches \
    --cache-dir pie_energy_cache \
    --output-dir PIE_Dataset/batches_remaining \
    --num-jobs 440
```

---

## Monitoring Progress

### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# Count running jobs
squeue -u $USER -h | wc -l

# View detailed job info
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R"
```

### Check Completion Rate
```bash
#!/bin/bash
COMPLETED=$(find pie_energy_cache/completed -name "*.done" | wc -l)
FAILED=$(find pie_energy_cache/failed -name "*.failed" | wc -l)
TOTAL=$(cat PIE_Dataset/batches/execution_master_batch_*.jsonl | wc -l)
REMAINING=$((TOTAL - COMPLETED - FAILED))

echo "==================================="
echo "Simulation Progress"
echo "==================================="
echo "Total executions:     $TOTAL"
echo "Completed:            $COMPLETED"
echo "Failed:               $FAILED"
echo "Remaining:            $REMAINING"
echo "Progress:             $(echo "scale=2; $COMPLETED * 100 / $TOTAL" | bc)%"
echo "==================================="
```

Save this as `check_progress.sh`, make executable, and run:
```bash
chmod +x check_progress.sh
./check_progress.sh
```

---

## Additional Notes

### Re-running After Completion
If you need to run more batches later:
1. New work will be skipped automatically (cache check)
2. You can regenerate redistributed batches anytime
3. The redistribution script always checks current cache state

### Partial Runs
You can test with fewer jobs:
```bash
# Test with 10 jobs instead of 440
python3 energy_data_collection/redistribute_remaining_work.py \
    --batch-dir PIE_Dataset/batches \
    --cache-dir pie_energy_cache \
    --output-dir PIE_Dataset/batches_test \
    --num-jobs 10
```

Then update SLURM script:
```bash
#SBATCH --array=0-9  # 10 jobs instead of 440
```

### Cleanup Old Logs (Optional)
```bash
# Archive old logs before new run
mkdir -p logs_archive/$(date +%Y%m%d)
mv logs/*.out logs/*.err logs_archive/$(date +%Y%m%d)/ 2>/dev/null
```

---

## Summary Checklist

- [ ] Check current completion status
- [ ] Run dry-run to preview redistribution
- [ ] Create redistributed batch files
- [ ] Verify 440 batch files created
- [ ] Confirm SLURM script points to redistributed batches
- [ ] Submit SLURM jobs
- [ ] Monitor progress
- [ ] Celebrate faster completion! 🎉

---

**Questions?** Check logs in `logs/` directory or review SLURM output files.
