# Energy-Efficient Code Generation: Training Pipeline

This repository contains the training infrastructure for fine-tuning Large Language Models (LLMs) to generate energy-efficient C++ code. The pipeline uses a hybrid alignment strategy combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) with hardware simulation feedback.

## 🚀 Training Pipeline

We use a **3-Stage Pipeline** to transform a general-purpose coding model into an energy optimization specialist:

1.  **Base Model**: `Qwen/Qwen2.5-Coder-14B-Instruct`
2.  **Stage 1: SFT Optimization (Refinement)**
    *   **Goal:** Teach the model to transform "Slow" code into "Fast" code using contrastive pairs.
    *   **Method:** Performance-Conditioned Generation ("Optimize to score 10/10").
    *   **Data:** 100% Refinement (Optimization).
3.  **Stage 2: GRPO (Online RL)**
    *   **Goal:** Discover novel optimizations beyond the training data.
    *   **Method:** Generate 16 solutions, simulate them in Sniper (cycle-accurate), and reward Energy reduction + IPC improvement.
    *   **Feedback:** Real-time ground truth from the simulator.

---

## 🛠️ Usage

### 1. Data Preprocessing
Generate the training datasets from the raw PIE analysis data.
```bash
python dataset_preprocessing.py --create-grpo
```
*Outputs: `data/sft_pairs_train.jsonl`, `data/grpo_train.jsonl`*

### 2. SFT Training (Optimization)
Train the model to perform energy optimization edits.
```bash
sbatch slurm/sft_optimization.sh
```
*   **Input:** Base Model
*   **Output:** `checkpoints/qwen-14b_sft_optimization/final`
*   **Key Script:** `sft_train_trl.py`

### 3. GRPO Training (RL)
Align the model using online simulation feedback.
```bash
sbatch slurm/grpo_vllm_train.sh
```
*   **Input:** SFT Optimization Checkpoint
*   **Output:** `checkpoints/qwen-14b_grpo/final`
*   **Key Script:** `grpo_trainer_vllm.py`
*   **Features:**
    *   vLLM for fast generation (16 samples/prompt).
    *   Parallel Sniper simulation (48 CPUs).
    *   Multi-objective Reward: `Energy_Improvement + 0.1 * IPC_Bonus`.
    *   Ghost Execution Filter (penalizes trivial/empty code).

---

---

## Attention Hotspot Analysis

`analyze_grpo_checkpoint.py --mode attention` runs attention extraction on the trained GRPO model to test whether it has learned to internally attend to energy hotspots (code regions it ends up modifying).

### What it does

For each example (highest energy-reduction cases from `data/grpo_sim_results`):
1. Runs a forward pass on `[prompt + generated_code]` with `output_attentions=True`
2. Extracts the last N layers of attention from output tokens back to input code tokens
3. Identifies hotspot tokens via diff of baseline vs generated code (changed lines)
4. Computes hotspot attention ratio: mean attention on changed tokens / mean attention on unchanged tokens
5. Saves all intermediate arrays and 5 plots per example

**Key metric:** ratio > 1.0 means the model attends disproportionately to the regions it modifies.

### Run

```bash
sbatch --account=rrg-mrdal22_gpu --gpus-per-node=h100:1 \
  --cpus-per-task=14 --mem=80G --time=2:00:00 \
  --job-name=attn_analysis --output=logs/attn_analysis_%j.out \
  --wrap="cd $HOME/projects/rrg-mrdal22/srajput/green-code-gen/finetuning && \
    source ../config.env && source ../venv/bin/activate && \
    python3 -u analyze_grpo_checkpoint.py --mode attention \
      --model checkpoints/qwen-coder-base-14b_grpo_latest/checkpoint-best \
      --sim-dir data/grpo_sim_results \
      --output-dir ../analysis/attention_plots \
      --n-samples 10 --n-layers 8"
```

Add `--save-raw` to also save the full per-layer-per-head tensor `(layers, heads, out_tokens, code_tokens)`.

### Output structure

```
analysis/attention_plots/
  ex00_<problem_id>/
    01_token_code_heatmap.png      # code tokens colored by attention (explainability style)
    02_attention_matrix.png        # output x input attention matrix with token labels
    03_layer_profile.png           # per-layer attention to hotspot vs non-hotspot
    04_head_diversity.png          # per-head hotspot ratio grid (layers x heads)
    05_token_importance_bar.png    # token importance bar chart
    attn_mean.npy                  # (n_out, n_code)
    attn_per_layer.npy             # (n_layers, n_out, n_code)
    tok_importance.npy             # (n_code,)
    layer_tok_importance.npy       # (n_layers, n_code)
    head_tok_importance.npy        # (n_layers, n_heads, n_code)
    hotspot_mask.npy               # (n_code,) bool
    layer_hot_profile.npy          # (n_layers,)
    layer_nothot_profile.npy       # (n_layers,)
    head_ratios.npy                # (n_layers, n_heads)
    metadata.json                  # all token strings, codes, indices, array shapes
  summary_hotspot_attention.png    # 3-panel: bar + scatter + per-example ratios
  hotspot_attention_stats.json     # aggregate stats
```

### Reload for offline analysis (no re-run needed)

```python
from analyze_grpo_checkpoint import load_example, load_all_examples

ex = load_example("analysis/attention_plots/ex00_p02948")
attn_matrix  = ex['arrays']['attn_mean']           # (n_out, n_code)
tok_imp      = ex['arrays']['tok_importance']       # (n_code,)
head_imp     = ex['arrays']['head_tok_importance']  # (n_layers, n_heads, n_code)
hotspot_mask = ex['arrays']['hotspot_mask']         # (n_code,) bool
code_tokens  = ex['code_tokens']                    # list[str]

all_examples = load_all_examples("analysis/attention_plots")
```

---

## 📂 Directory Structure

*   `dataset_preprocessing.py`: Prepares data for all stages.
*   `sft_train_trl.py`: TRL-based SFT trainer (using PIE templates).
*   `grpo_trainer_vllm.py`: Advanced GRPO trainer with vLLM and Sniper integration.
*   `slurm/`: Job submission scripts.
*   `archive/`: Deprecated scripts (DPO, old SFT stages).
*   `utils/`: Helper functions and callbacks.

## ⚡ Reward Function (GRPO)

The RL stage optimizes for **Energy Efficiency**, avoiding the "runtime trap" (where models just write shorter, power-hungry code).

$$ Reward = R_{Energy} + \beta \cdot R_{IPC} $$

*   **Energy ($R_{Energy}$):** Relative improvement over baseline ($ \frac{E_{base} - E_{new}}{E_{base}} $).
*   **IPC ($R_{IPC}$):** Bonus for architectural efficiency (Instructions Per Cycle), applied only if energy improves.
*   **Correctness:** Solutions must pass **all** unit tests for the problem before energy is measured.
*   **Safety:** "Ghost" executions (0 energy artifacts) result in severe penalties (-1.0).

WandB (Weights & Biases)

1. Login (one-time setup):
# SSH to compute node or login node
cd ~/projects/rrg-mrdal22/srajput/green-code-gen
source venv/bin/activate
wandb login
# Paste your API key from https://wandb.ai/authorize

2. Enable in training:
# SFT
sbatch finetuning/slurm/sft_debug.sh --use-wandb

# GRPO
sbatch finetuning/slurm/grpo_debug.sh --use-wandb

3. View dashboard:
- Go to https://wandb.ai
- Navigate to project: energy-code-generation
- See runs: debug-sft, debug-grpo, etc.

Current status:
- SFT: report_to=["tensorboard", "wandb"] if --use-wandb (line 248-250)
- GRPO: report_to=["wandb"] if --use-wandb, else ["tensorboard"] (line 685)

TensorBoard (Always enabled)

1. Training logs saved to:
# SFT
./checkpoints/debug_sft/logs/

# GRPO
./checkpoints/debug_grpo/logs/

2. View locally (after job completes):
# Copy logs from cluster to local machine
scp -r username@cluster:/path/to/checkpoints/debug_sft/logs ./

# Run TensorBoard locally
tensorboard --logdir ./logs
# Open http://localhost:6006

3. View on cluster (while job running):
# SSH tunnel from local machine
ssh -L 6006:localhost:6006 username@cluster

# On cluster
cd ~/projects/rrg-mrdal22/srajput/green-code-gen/finetuning
source venv/bin/activate
tensorboard --logdir checkpoints/debug_sft/logs --host 0.0.0.0 --port 6006

# Open http://localhost:6006 on your local browser

Add --use-wandb to scripts:


# sft_debug.sh line 69 (add after --model-name):
    --model-name "debug-sft" \
    --use-wandb \
    --sniper-root "$SNIPER_ROOT"

# grpo_debug.sh line 58 (add after --model-name):
    --model-name "debug-grpo" \
    --use-wandb \
    --eval-during-training \