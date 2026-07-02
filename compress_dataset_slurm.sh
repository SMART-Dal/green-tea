#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --account=rrg-mrdal22
#SBATCH --job-name=analysis_results_backup
#SBATCH --output=logs/analysis_results_backup_%j.out
#SBATCH --error=logs/analysis_results_backup_%j.err
#SBATCH --mem=128G
#SBATCH --time=1:00:00

set -euo pipefail

echo "Analysis Results Backup - Job $SLURM_JOB_ID - $(date)"
echo "Node: $SLURM_NODELIST | CPUs: $SLURM_CPUS_PER_TASK"

module load StdEnv/2023

CACHE_DIR="analysis"
OUTPUT="analysis_results_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "Input: $CACHE_DIR ($(du -sh $CACHE_DIR | cut -f1))"
echo "Output: $OUTPUT"
echo "Compressing with pigz -p $SLURM_CPUS_PER_TASK -6..."

tar -I "pigz -p $SLURM_CPUS_PER_TASK -6" -cf "$OUTPUT" "$CACHE_DIR"

echo "Done: $(du -sh $OUTPUT | cut -f1) ($(date))"
md5sum "$OUTPUT" > "${OUTPUT}.md5"
echo "Checksum: $(cat ${OUTPUT}.md5)"