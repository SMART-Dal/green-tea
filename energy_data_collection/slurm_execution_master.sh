#!/bin/bash
#SBATCH --job-name=pie-energy-exec
#SBATCH --account=rrg-mrdal22
#SBATCH --time=168:00:00         # 7 days (conservative with 3× safety margin)
#SBATCH --signal=B:TERM@300      # Send SIGTERM 300 seconds before timeout
#SBATCH --signal=B:USR1@750      # Send SIGUSR1 750 seconds before timeout (early warning)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1        # Sequential execution
#SBATCH --mem=16G                # Reduced memory need
#SBATCH --array=0-439            # 440 jobs for 440 batch files
#SBATCH --output=logs/pie_energy_exec_%A_%a.out
#SBATCH --error=logs/pie_energy_exec_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=saurabh@dal.ca

# PIE Dataset Energy Collection - Execution Master Approach
# - Processes execution_master_batch_*.jsonl files
# - Uses SharedExecutionCache for automatic resumability
# - One execution at a time (simple, robust)
# - 3× safety margin (75s/simulation estimate)

echo "========================================="
echo "PIE Energy Collection (Execution Master)"
echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "========================================="

# Load central configuration
# Try multiple possible locations for config.env
if [ -f "../config.env" ]; then
    CONFIG_FILE="../config.env"
elif [ -f "/home/srajput/energy_simulations/green-code-gen/config.env" ]; then
    CONFIG_FILE="/home/srajput/energy_simulations/green-code-gen/config.env"
elif [ -f "/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen/config.env" ]; then
    CONFIG_FILE="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen/config.env"
else
    echo "ERROR: config.env not found. Please create it from config.env.template"
    exit 1
fi

echo "Loading configuration from: ${CONFIG_FILE}"
source "${CONFIG_FILE}"

# ============================================================================
# BATCH DIRECTORY SELECTION
# ============================================================================
# You can toggle between original batches and redistributed remaining work:
#
# Option 1 - ORIGINAL BATCHES (run everything from scratch):
#   Use this for first-time runs or to restart from beginning
#   BATCH_SOURCE="${BATCH_DIR}"
#
# Option 2 - REDISTRIBUTED REMAINING WORK (speedup):
#   Use this to redistribute remaining work across all 440 jobs for better parallelization
#   First run: python3 energy_data_collection/redistribute_remaining_work.py \
#              --batch-dir PIE_Dataset/batches \
#              --cache-dir pie_energy_cache \
#              --output-dir PIE_Dataset/batches_remaining \
#              --num-jobs 440
#   BATCH_SOURCE="${PIE_DATASET}/batches_remaining"
# ============================================================================

# ACTIVE CONFIGURATION: Uncomment ONE of the following lines

# Option 1: Original batches (all work)
#BATCH_SOURCE="${BATCH_DIR}"

# Option 2: Redistributed remaining work (recommended for speedup)
BATCH_SOURCE="${PIE_DATASET}/batches_remaining"

# ============================================================================

# Batch file for this job
BATCH_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
BATCH_FILE="${BATCH_SOURCE}/execution_master_batch_${BATCH_ID}.jsonl"

echo "Batch source: ${BATCH_SOURCE}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Sniper root: ${SNIPER_ROOT}"
echo "PIE dataset: ${PIE_DATASET}"
echo "Batch file: ${BATCH_FILE}"
echo "Cache directory: ${CACHE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Verify batch file exists
if [ ! -f "${BATCH_FILE}" ]; then
    echo "ERROR: Batch file not found: ${BATCH_FILE}"
    exit 1
fi

# Count executions in this batch
BATCH_SIZE=$(wc -l < "${BATCH_FILE}")
echo "Batch size: ${BATCH_SIZE} executions"

# Load modules
echo "Loading modules..."
module load python/3.11
module load StdEnv/2023
module load boost

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load required modules"
    exit 1
fi

echo "Modules loaded successfully"

# Set up Sniper environment
export PYTHONPATH="${SNIPER_ROOT}/mbuild:${PYTHONPATH}"
export PIN_HOME="${SNIPER_ROOT}/pin_kit"
export LD_LIBRARY_PATH="${PIN_HOME}/intel64/lib-ext:${PIN_HOME}/intel64/lib:${LD_LIBRARY_PATH}"

# Verify Sniper
if [ ! -d "${SNIPER_ROOT}" ]; then
    echo "ERROR: Sniper not found at ${SNIPER_ROOT}"
    exit 1
fi

if [ ! -f "${SNIPER_ROOT}/run-sniper" ]; then
    echo "ERROR: run-sniper not found"
    exit 1
fi

echo "Sniper verified"

# Python virtual environment
if [ ! -d "${PROJECT_ROOT}/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${PROJECT_ROOT}/venv"
fi

echo "Activating virtual environment..."
source "${PROJECT_ROOT}/venv/bin/activate"

echo "Python: $(python --version)"

# Change to project directory
cd "${PROJECT_ROOT}"

# Sniper configuration (using epyc_9554p.cfg - matches actual 64-core processor)
SNIPER_CONFIG="${SNIPER_ROOT}/config/epyc_9554p.cfg"

if [ ! -f "${SNIPER_CONFIG}" ]; then
    echo "ERROR: Sniper config not found: ${SNIPER_CONFIG}"
    exit 1
fi

echo "Sniper config: ${SNIPER_CONFIG}"

# Performance estimation
ESTIMATED_TIME_PER_EXEC=75  # Conservative: 75s per execution (3× safety margin)
ESTIMATED_TOTAL_SECONDS=$((BATCH_SIZE * ESTIMATED_TIME_PER_EXEC))
ESTIMATED_HOURS=$((ESTIMATED_TOTAL_SECONDS / 3600))
ESTIMATED_DAYS=$(echo "scale=1; ${ESTIMATED_HOURS} / 24" | bc)

echo "Performance estimates:"
echo "  Time per execution: ${ESTIMATED_TIME_PER_EXEC}s (conservative 3× margin)"
echo "  Total executions: ${BATCH_SIZE}"
echo "  Estimated time: ${ESTIMATED_HOURS} hours (~${ESTIMATED_DAYS} days)"
echo "  Time limit: 168 hours (7 days)"

# Define cleanup function with signal-based timeout handling
cleanup_and_save() {
    local reason="$1"
    local signal_name="$2"

    echo "========================================="
    echo "CLEANUP TRIGGERED: ${reason}"
    if [ -n "${signal_name}" ]; then
        echo "Signal: ${signal_name}"
    fi
    echo "========================================="

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    REMAINING_TIME=$((168 * 3600 - DURATION))  # 7 days (168 hours) - elapsed time

    echo "Job duration: ${DURATION} seconds ($((DURATION / 3600)) hours)"
    echo "Time remaining: ${REMAINING_TIME} seconds ($((REMAINING_TIME / 3600)) hours)"

    # Count completed executions from cache
    local completed_count=0
    if [ -d "${CACHE_DIR}/completed" ]; then
        completed_count=$(find "${CACHE_DIR}/completed" -name "*.done" 2>/dev/null | wc -l)
    fi

    local failed_count=0
    if [ -d "${CACHE_DIR}/failed" ]; then
        failed_count=$(find "${CACHE_DIR}/failed" -name "*.failed" 2>/dev/null | wc -l)
    fi

    echo "Completed executions: ${completed_count}"
    echo "Failed executions: ${failed_count}"

    # Create comprehensive metadata
    cat > "${LOG_DIR}/job_metadata_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json" << EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "array_task_id": "${SLURM_ARRAY_TASK_ID}",
  "batch_file": "${BATCH_FILE}",
  "batch_size": ${BATCH_SIZE},
  "node": "${SLURMD_NODENAME}",
  "start_time": ${START_TIME},
  "end_time": ${END_TIME},
  "duration_seconds": ${DURATION},
  "duration_hours": $((DURATION / 3600)),
  "remaining_seconds": ${REMAINING_TIME},
  "status": "${reason}",
  "signal": "${signal_name:-none}",
  "completed_executions": ${completed_count},
  "failed_executions": ${failed_count},
  "cache_dir": "${CACHE_DIR}",
  "sniper_config": "${SNIPER_CONFIG}"
}
EOF

    # Create summary file
    cat > "${LOG_DIR}/summary_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" << EOF
Job Summary - ${reason}
=========================================
Job ID: ${SLURM_JOB_ID}
Array Task ID: ${SLURM_ARRAY_TASK_ID}
Batch File: ${BATCH_FILE}
Batch Size: ${BATCH_SIZE} executions
Node: ${SLURMD_NODENAME}
Duration: ${DURATION} seconds ($((DURATION / 3600)) hours)
Remaining: ${REMAINING_TIME} seconds ($((REMAINING_TIME / 3600)) hours)
Completed: ${completed_count} executions
Failed: ${failed_count} executions
Status: ${reason}
Signal: ${signal_name:-none}
Time: $(date)
=========================================
EOF

    echo "Metadata saved to ${LOG_DIR}"
    echo "========================================="
}

# Enhanced signal handlers that work with background processes
handle_signal() {
    local signal_name="$1"
    echo "========================================="
    echo "SIGNAL RECEIVED: ${signal_name}"
    echo "Timestamp: $(date)"
    echo "Python PID: ${PYTHON_PID:-not_set}"
    echo "========================================="

    # Save current state
    cleanup_and_save "TIMEOUT" "${signal_name}"

    # Wait for Python process to finish gracefully
    if [ -n "${PYTHON_PID}" ] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
        echo "Waiting for Python process ${PYTHON_PID} to finish gracefully..."
        # Give it 60 seconds to finish current execution
        for i in {1..60}; do
            if ! kill -0 "${PYTHON_PID}" 2>/dev/null; then
                echo "Python process finished"
                break
            fi
            sleep 1
        done

        # If still running, force kill
        if kill -0 "${PYTHON_PID}" 2>/dev/null; then
            echo "Force killing Python process..."
            kill -9 "${PYTHON_PID}" 2>/dev/null || true
        fi
    fi

    echo "Signal handling completed"
    exit 0
}

# Trap signals to preserve partial results
# SIGTERM is sent 120 seconds before job timeout (as configured in #SBATCH --signal)
trap 'handle_signal "SIGTERM"' SIGTERM
trap 'handle_signal "SIGINT"' SIGINT
trap 'handle_signal "SIGUSR1"' SIGUSR1

# Check cache stats before starting
if [ -d "${CACHE_DIR}/completed" ]; then
    ALREADY_COMPLETED=$(find "${CACHE_DIR}/completed" -name "*.done" 2>/dev/null | wc -l)
    echo "Cache: ${ALREADY_COMPLETED} executions already completed"
else
    echo "Cache: empty (first run)"
fi

# Start timestamp
START_TIME=$(date +%s)
echo "Starting execution processing at $(date)"

# Run the execution-based runner in background so we can trap signals
echo "========================================="
python3 energy_data_collection/sniper_execution_runner.py \
    --batch-file "${BATCH_FILE}" \
    --cache-dir "${CACHE_DIR}" \
    --sniper-path "${SNIPER_ROOT}" \
    --sniper-config "${SNIPER_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" &

# Save Python PID for signal handling
PYTHON_PID=$!
echo "Python process started with PID: ${PYTHON_PID}"

# Wait for the Python process to complete or for signals
wait $PYTHON_PID
PROCESSING_EXIT_CODE=$?

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_HOURS=$(echo "scale=2; ${DURATION} / 3600" | bc)

echo "========================================="
echo "Processing completed at $(date)"
echo "Duration: ${DURATION} seconds (${DURATION_HOURS} hours)"

# Check processing results and save state
if [ ${PROCESSING_EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Execution master processing completed successfully"
    cleanup_and_save "SUCCESS" "NONE"
else
    echo "ERROR: Execution master processing failed with exit code ${PROCESSING_EXIT_CODE}"
    cleanup_and_save "ERROR_${PROCESSING_EXIT_CODE}" "NONE"
fi

# Check cache stats after completion
if [ -d "${CACHE_DIR}/completed" ]; then
    TOTAL_COMPLETED=$(find "${CACHE_DIR}/completed" -name "*.done" 2>/dev/null | wc -l)
    echo "Cache: ${TOTAL_COMPLETED} total executions completed"
fi

if [ -d "${CACHE_DIR}/failed" ]; then
    TOTAL_FAILED=$(find "${CACHE_DIR}/failed" -name "*.failed" 2>/dev/null | wc -l)
    echo "Cache: ${TOTAL_FAILED} total executions failed"
fi

# Final status
echo "========================================="
echo "Job ${SLURM_ARRAY_TASK_ID} Final Status"
echo "Batch: ${BATCH_FILE}"
echo "Batch size: ${BATCH_SIZE} executions"
echo "Exit code: ${PROCESSING_EXIT_CODE}"
echo "Duration: ${DURATION_HOURS} hours"
if [ ${PROCESSING_EXIT_CODE} -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED"
fi
echo "========================================="

# Exit with processing code
exit ${PROCESSING_EXIT_CODE}
