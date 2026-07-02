#!/bin/bash
# Submit all new-model experiment jobs with proper dependency chains.
# Run from: $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning/
# Usage: bash slurm/submit_all_new_models.sh

set -e
cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning

echo "=============================================="
echo "SUBMITTING NEW MODEL EXPERIMENTS"
echo "$(date)"
echo "=============================================="

# ---- W2 FRONTIER ZERO-SHOT BASELINES (run immediately, no deps) ----
echo ""
echo "--- W2 Frontier Baselines ---"

JOB_FQ_GEN=$(sbatch --parsable slurm/eval_frontier_qwen32b_gen.sh)
echo "Qwen32B gen:    $JOB_FQ_GEN (def-tusharma_gpu)"

JOB_FD_GEN=$(sbatch --parsable slurm/eval_frontier_dscoder_gen.sh)
echo "DSCoder-V2 gen: $JOB_FD_GEN (rrg-mrdal22)"

# Sim jobs depend on gen jobs
JOB_FQ_SIM=$(sbatch --parsable --dependency=afterok:$JOB_FQ_GEN slurm/eval_frontier_qwen32b_sim.sh)
echo "Qwen32B sim:    $JOB_FQ_SIM (rrg-mrdal22, afterok:$JOB_FQ_GEN)"

JOB_FD_SIM=$(sbatch --parsable --dependency=afterok:$JOB_FD_GEN slurm/eval_frontier_dscoder_sim.sh)
echo "DSCoder-V2 sim: $JOB_FD_SIM (def-tusharma_gpu, afterok:$JOB_FD_GEN)"

# ---- W1 SECOND MODEL: QWEN-7B (SFT -> GRPO -> gen -> sim) ----
echo ""
echo "--- W1 Second Model: Qwen2.5-Coder-7B ---"

JOB_Q7_SFT=$(sbatch --parsable slurm/sft_qwen7b.sh)
echo "Qwen7B SFT:     $JOB_Q7_SFT (def-tusharma_gpu)"

JOB_Q7_GRPO=$(sbatch --parsable --dependency=afterok:$JOB_Q7_SFT slurm/grpo_qwen7b.sh)
echo "Qwen7B GRPO:    $JOB_Q7_GRPO (rrg-mrdal22, afterok:$JOB_Q7_SFT)"

JOB_Q7_GEN=$(sbatch --parsable --dependency=afterok:$JOB_Q7_GRPO slurm/eval_qwen7b_gen.sh)
echo "Qwen7B gen:     $JOB_Q7_GEN (def-tusharma_gpu, afterok:$JOB_Q7_GRPO)"

JOB_Q7_SIM=$(sbatch --parsable --dependency=afterok:$JOB_Q7_GEN slurm/eval_qwen7b_sim.sh)
echo "Qwen7B sim:     $JOB_Q7_SIM (rrg-mrdal22, afterok:$JOB_Q7_GEN)"

# ---- W1 SECOND MODEL: DEEPSEEK-CODER-7B (SFT -> GRPO -> gen -> sim) ----
echo ""
echo "--- W1 Second Model: DeepSeek-Coder-7B-v1.5 ---"

JOB_DS7_SFT=$(sbatch --parsable slurm/sft_dscoder7b.sh)
echo "DSCoder7B SFT:  $JOB_DS7_SFT (rrg-mrdal22)"

JOB_DS7_GRPO=$(sbatch --parsable --dependency=afterok:$JOB_DS7_SFT slurm/grpo_dscoder7b.sh)
echo "DSCoder7B GRPO: $JOB_DS7_GRPO (def-tusharma_gpu, afterok:$JOB_DS7_SFT)"

JOB_DS7_GEN=$(sbatch --parsable --dependency=afterok:$JOB_DS7_GRPO slurm/eval_dscoder7b_gen.sh)
echo "DSCoder7B gen:  $JOB_DS7_GEN (rrg-mrdal22, afterok:$JOB_DS7_GRPO)"

JOB_DS7_SIM=$(sbatch --parsable --dependency=afterok:$JOB_DS7_GEN slurm/eval_dscoder7b_sim.sh)
echo "DSCoder7B sim:  $JOB_DS7_SIM (def-tusharma_gpu, afterok:$JOB_DS7_GEN)"

# ---- SCORE SENSITIVITY (uses existing GRPO-14B, run immediately) ----
echo ""
echo "--- Score Sensitivity (8/10, 9/10, 10/10) ---"
JOB_SCORE=$(sbatch --parsable slurm/eval_score_sensitivity.sh)
echo "Score sensitivity gen: $JOB_SCORE (def-tusharma_gpu)"

JOB_SCORE_SIM=$(sbatch --parsable --dependency=afterok:$JOB_SCORE slurm/eval_score_sensitivity_sim.sh)
echo "Score sensitivity sim: $JOB_SCORE_SIM (def-tusharma_gpu, afterok:$JOB_SCORE)"

# ---- SUMMARY ----
echo ""
echo "=============================================="
echo "ALL JOBS SUBMITTED"
echo "=============================================="
echo ""
echo "Frontier baselines (W2, immediate):"
echo "  Qwen32B:     gen=$JOB_FQ_GEN -> sim=$JOB_FQ_SIM"
echo "  DSCoder-V2:  gen=$JOB_FD_GEN -> sim=$JOB_FD_SIM"
echo ""
echo "Second models (W1, ~2-9 day chains):"
echo "  Qwen7B:   SFT=$JOB_Q7_SFT -> GRPO=$JOB_Q7_GRPO -> gen=$JOB_Q7_GEN -> sim=$JOB_Q7_SIM"
echo "  DSCoder7B: SFT=$JOB_DS7_SFT -> GRPO=$JOB_DS7_GRPO -> gen=$JOB_DS7_GEN -> sim=$JOB_DS7_SIM"
echo ""
echo "Misc: score_sensitivity gen=$JOB_SCORE sim=$JOB_SCORE_SIM"
echo ""
echo "Monitor: squeue -u srajput"
echo "=============================================="
