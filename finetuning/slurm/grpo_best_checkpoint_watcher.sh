#!/bin/bash
# Monitors GRPO training log and keeps checkpoint-best updated with the best checkpoint seen so far.
# Run from login node ONLY. Read-only access to log + checkpoint dir. Does NOT touch running job.
#
# Usage: bash slurm/grpo_best_checkpoint_watcher.sh [check_interval_seconds]
# Default interval: 600s (10 min). Runs until job ends or killed.
#
# Logic: every interval, compute rolling avg reward over last 100 log steps.
# If better than previous best, replace checkpoint-best with the latest checkpoint.
# Step-to-checkpoint offset (~1304) is auto-detected from existing checkpoint dirs.

CKPT_DIR="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen/finetuning/checkpoints/qwen-coder-base-14b_grpo_20260227_033015"
LOG="/home/srajput/projects/rrg-mrdal22/srajput/green-code-gen/finetuning/logs/qwen-coder-base-14b_grpo_20260227_033015_grpo_vllm_9400877.log"
WATCHER_LOG="$CKPT_DIR/checkpoint-best-watcher.log"
INTERVAL=${1:-600}

echo "$(date): Watcher started. Interval=${INTERVAL}s. Log=$WATCHER_LOG" | tee -a "$WATCHER_LOG"

# Recover from interrupted copy on startup
if [ -d "$CKPT_DIR/checkpoint-best-tmp" ] && [ ! -d "$CKPT_DIR/checkpoint-best" ]; then
    mv "$CKPT_DIR/checkpoint-best-tmp" "$CKPT_DIR/checkpoint-best"
    echo "$(date): Recovered checkpoint-best from tmp on startup." | tee -a "$WATCHER_LOG"
fi

# Restore BEST_REWARD from disk to survive restarts without resetting to -9999
BEST_REWARD=-9999
if [ -f "$CKPT_DIR/checkpoint-best/best_reward.txt" ]; then
    BEST_REWARD=$(head -1 "$CKPT_DIR/checkpoint-best/best_reward.txt" | awk '{print $1}')
    echo "$(date): Restored BEST_REWARD=$BEST_REWARD from checkpoint-best/best_reward.txt" | tee -a "$WATCHER_LOG"
fi

while true; do
    # Compute avg reward over last 100 log steps
    CURRENT_REWARD=$(python3 -c "
import re, sys
steps = []
with open('$LOG') as f:
    for line in f:
        line = line.strip()
        if line.startswith('{') and \"'reward':\" in line:
            try:
                d = eval(line)
                steps.append(d.get('reward', -9999))
            except:
                pass
if not steps:
    print(-9999)
else:
    window = steps[-100:]
    print(sum(window)/len(window))
" 2>/dev/null)

    # Find latest checkpoint (highest number)
    LATEST_CKPT=$(ls -d "$CKPT_DIR"/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n | tail -1)

    if [ -z "$LATEST_CKPT" ] || [ -z "$CURRENT_REWARD" ]; then
        echo "$(date): No checkpoints or reward found, skipping." >> "$WATCHER_LOG"
        sleep "$INTERVAL"
        continue
    fi

    CKPT_NUM=$(basename "$LATEST_CKPT" | cut -d- -f2)

    # Compare with best (python float comparison)
    IS_BETTER=$(python3 -c "print('yes' if float('$CURRENT_REWARD') > float('$BEST_REWARD') else 'no')" 2>/dev/null)

    # Recreate checkpoint-best if it was deleted externally
    if [ ! -d "$CKPT_DIR/checkpoint-best" ] && [ -n "$LATEST_CKPT" ]; then
        BEST_CKPT_NUM=$(python3 -c "
import os, glob
bests = sorted(glob.glob('$CKPT_DIR/checkpoint-best-[0-9]*'), key=lambda x: int(x.split('-')[-1]))
print(bests[-1].split('-')[-1] if bests else '')
" 2>/dev/null)
        if [ -n "$BEST_CKPT_NUM" ] && [ -d "$CKPT_DIR/checkpoint-best-$BEST_CKPT_NUM" ]; then
            cp -rl "$CKPT_DIR/checkpoint-best-$BEST_CKPT_NUM" "$CKPT_DIR/checkpoint-best"
            echo "$(date): Recreated checkpoint-best from checkpoint-best-$BEST_CKPT_NUM (was deleted externally)" | tee -a "$WATCHER_LOG"
        elif [ -n "$LATEST_CKPT" ]; then
            cp -r "$LATEST_CKPT" "$CKPT_DIR/checkpoint-best"
            echo "$BEST_REWARD" > "$CKPT_DIR/checkpoint-best/best_reward.txt"
            echo "$(date): Recreated checkpoint-best from $LATEST_CKPT (no numbered copy found)" | tee -a "$WATCHER_LOG"
        fi
    fi

    if [ "$IS_BETTER" = "yes" ]; then
        echo "$(date): New best reward=$CURRENT_REWARD (was $BEST_REWARD) at ckpt-$CKPT_NUM. Updating checkpoint-best." | tee -a "$WATCHER_LOG"
        # Atomic copy to tmp, then rename to checkpoint-best
        TMPDIR="$CKPT_DIR/checkpoint-best-tmp"
        rm -rf "$TMPDIR"
        cp -r "$LATEST_CKPT" "$TMPDIR"
        echo "$CURRENT_REWARD" > "$TMPDIR/best_reward.txt"
        echo "checkpoint-$CKPT_NUM" > "$TMPDIR/source_checkpoint.txt"
        echo "$CKPT_NUM" > "$TMPDIR/step_number.txt"
        rm -rf "$CKPT_DIR/checkpoint-best"
        mv "$TMPDIR" "$CKPT_DIR/checkpoint-best"
        # Keep numbered copy of each best (hard-linked, near-zero extra space)
        NUMBERED="$CKPT_DIR/checkpoint-best-$CKPT_NUM"
        rm -rf "$NUMBERED"
        cp -rl "$CKPT_DIR/checkpoint-best" "$NUMBERED"
        echo "$(date): Saved numbered copy: checkpoint-best-$CKPT_NUM" >> "$WATCHER_LOG"
        BEST_REWARD=$CURRENT_REWARD
    else
        echo "$(date): reward=$CURRENT_REWARD <= best=$BEST_REWARD (ckpt-$CKPT_NUM). No update." >> "$WATCHER_LOG"
    fi

    sleep "$INTERVAL"
done
