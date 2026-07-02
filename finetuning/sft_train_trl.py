#!/usr/bin/env python3
"""
TRL-based SFT Training with Two-Stage Curriculum

Stage 1 (Syntax): Problem Description → Working Solution
  - Goal: Maximize compilation success rate
  - Focus: Correctness, syntax, basic algorithmic thinking

Stage 2 (Optimization): Problem + Slow Code → Fast/Efficient Code
  - Goal: Energy reduction via contrastive learning
  - Focus: Optimization patterns, energy-aware transformations

Usage:
    python sft_train_trl.py \
        --stage syntax \
        --model Qwen/Qwen2.5-Coder-14B-Instruct \
        --data-path data \
        --output-dir checkpoints/qwen_sft_syntax

    python sft_train_trl.py \
        --stage optimization \
        --model checkpoints/qwen_sft_syntax/final \
        --data-path data \
        --output-dir checkpoints/qwen_sft_optimization

Based on: https://huggingface.co/docs/trl/main/en/sft_trainer
"""

from unsloth import FastLanguageModel
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

OPTIMIZATION_TEMPLATE = """This is an energy inefficient program we want to optimize to score {target_score}/10.
### Program:
{baseline_code}

### Energy Optimized Version with score {target_score}/10:
```cpp
{optimized_solution}
```"""

RUNTIME_TEMPLATE = """This is a slow program we want to optimize for speed to score {target_score}/10.
### Program:
{baseline_code}

### Speed Optimized Version with score {target_score}/10:
```cpp
{optimized_solution}
```"""


# ==============================================================================
# DATASET FORMATTING
# ==============================================================================

def format_optimization_example(examples: Dict[str, Any]) -> List[str]:
    """
    Format examples for Energy Optimization (Refinement).
    Handles both single examples (during trainer init) and batches (during training).
    """
    # Check if input is a batch (values are lists) by checking one key
    # Use 'inefficient_code' or 'baseline_code' as probe
    probe_key = next(iter(examples.keys())) if examples else None
    if not probe_key:
        return []
    
    is_batch = isinstance(examples[probe_key], list)
    
    if not is_batch:
        text = OPTIMIZATION_TEMPLATE.format(
            target_score=examples.get('optimized_score', examples.get('energy_score', 10)),
            baseline_code=examples.get('inefficient_code', examples.get('baseline_code', '')),
            optimized_solution=examples.get('optimized_code', '')
        )
        return [text]

    else:
        output_texts = []
        for i in range(len(examples[probe_key])):
            opt_score = examples['optimized_score'][i] if 'optimized_score' in examples else \
                        (examples['energy_score'][i] if 'energy_score' in examples else 10)
            baseline = examples['inefficient_code'][i] if 'inefficient_code' in examples else \
                       (examples['baseline_code'][i] if 'baseline_code' in examples else '')
            optimized = examples['optimized_code'][i] if 'optimized_code' in examples else ''

            output_texts.append(OPTIMIZATION_TEMPLATE.format(
                target_score=opt_score,
                baseline_code=baseline,
                optimized_solution=optimized
            ))
        return output_texts


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRL-based SFT training for Energy Optimization'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-Coder-14B',
        help='Base model or checkpoint path'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='energy',
        choices=['energy', 'runtime'],
        help='Prompt template: energy (default) or runtime (W1 ablation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Per-device batch size'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=16,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=16384,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=64,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=128,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='energy-code-generation',
        help='W&B project name'
    )
    parser.add_argument(
        '--eval-during-training',
        action='store_true',
        help='Enable periodic energy evaluation during training'
    )
    parser.add_argument(
        '--sniper-root',
        type=str,
        default=None,
        help='Sniper root directory (required if --eval-during-training)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='qwen-14b',
        help='Model identifier for organizing outputs and WandB tracking (e.g., qwen-14b, codellama-13b)'
    )
    parser.add_argument(
        '--eval-output-dir',
        type=str,
        default='data/sft_generations_base',
        help='Output dir for energy eval generation files'
    )
    parser.add_argument(
        '--eval-every-n-steps',
        type=int,
        default=100,
        help='Run EnergyEvaluationCallback every N steps'
    )
    parser.add_argument(
        '--max-eval-samples',
        type=int,
        default=20,
        help='Max val samples per energy eval trigger'
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=100,
        help='TRL eval_steps (eval_loss checkpoint selection)'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=100,
        help='TRL save_steps (checkpoint interval)'
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"TRL SFT TRAINING - ENERGY OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Max seq length: {args.max_seq_length}")

    # Load model and tokenizer
    logger.info("\nLoading model and tokenizer...")
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model,
    #     dtype=torch.bfloat16,
    #     trust_remote_code=True
    # )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model,
    #     trust_remote_code=True
    # )

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        load_in_4bit = False
        )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # # Ensure eos_token is in vocabulary, if not, set it to a known token or add it
    # if tokenizer.eos_token is None:
    #     logger.warning("Tokenizer has no EOS token! Setting to <|endoftext|>")
    #     tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

    # Apply LoRA
    logger.info(f"Applying LoRA (rank={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()

    # Load and format dataset
    logger.info("\nLoading dataset...")
    data_path = Path(args.data_path)

    # Load optimization pairs
    train_file = data_path / 'sft_pairs_train.jsonl'
    val_file = data_path / 'sft_pairs_val.jsonl'
    _tmpl = RUNTIME_TEMPLATE if getattr(args, 'template', 'energy') == 'runtime' else OPTIMIZATION_TEMPLATE
    def _fmt(examples, _t=_tmpl):
        probe = next(iter(examples.keys())) if examples else None
        if not probe:
            return []
        is_batch = isinstance(examples[probe], list)
        if not is_batch:
            return [_t.format(
                target_score=examples.get('optimized_score', examples.get('energy_score', 10)),
                baseline_code=examples.get('inefficient_code', examples.get('baseline_code', '')),
                optimized_solution=examples.get('optimized_code', ''))]
        out = []
        for i in range(len(examples[probe])):
            s = examples['optimized_score'][i] if 'optimized_score' in examples else 10
            b = examples['inefficient_code'][i] if 'inefficient_code' in examples else examples.get('baseline_code', [''])[i]
            o = examples['optimized_code'][i] if 'optimized_code' in examples else ''
            out.append(_t.format(target_score=s, baseline_code=b, optimized_solution=o))
        return out
    formatting_func = _fmt

    dataset = load_dataset(
        'json',
        data_files={
            'train': str(train_file),
            'validation': str(val_file)
        }
    )

    logger.info(f"Loaded {len(dataset['train']):,} training examples")
    logger.info(f"Loaded {len(dataset['validation']):,} validation examples")

    # Prepare reporting backends
    report_to_backends = ["tensorboard"]
    if args.use_wandb:
        report_to_backends.append("wandb")
        logger.info("✓ W&B logging enabled")

    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,

        # WE are using dynamic fine tuning loss type
        loss_type="dft",

        # Sequence length
        max_length=args.max_seq_length,

        # Optimization
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,

        # Precision
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging and saving
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_energy_reduction" if args.eval_during_training else "eval_loss",
        greater_is_better=True if args.eval_during_training else False,

        # DeepSpeed
        # deepspeed="config/ds_zero2.json",

        # Reporting
        report_to=report_to_backends,
        logging_dir=f"{args.output_dir}/logs",

        # Dataset handling
        packing=False,

        # W&B Run Name
        run_name=f"{args.model_name}-sft",
    )

    # Prepare callbacks
    callbacks = []

    # W&B initialization callback
    if args.use_wandb:
        from utils.training_callbacks import get_wandb_callback
        wandb_callback = get_wandb_callback(
            stage="optimization",
            model_name=args.model_name,
            additional_config={
                'max_seq_length': args.max_seq_length,
                'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha
            }
        )
        callbacks.append(wandb_callback)

    # Energy evaluation callback (only for optimization stage)
    if args.eval_during_training:
        if not args.sniper_root:
            logger.warning("--eval-during-training requires --sniper-root, skipping energy evaluation")
        else:
            from utils.training_callbacks import EnergyEvaluationCallback
            energy_callback = EnergyEvaluationCallback(
                eval_model=model,
                eval_tokenizer=tokenizer,
                eval_dataset=dataset["validation"],
                sniper_root=args.sniper_root,
                eval_every_n_steps=args.eval_every_n_steps,
                max_eval_samples=args.max_eval_samples,
                output_dir=args.eval_output_dir
            )
            callbacks.append(energy_callback)
            logger.info("✓ Energy evaluation during training enabled")
            print(f"✓ Energy callback initialized: sniper_root={args.sniper_root}, max_eval_samples={args.max_eval_samples}, eval_every_n_steps={args.eval_every_n_steps}")

    # Create SFT Trainer
    logger.info("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=callbacks
    )

    # Train
    logger.info("\n" + "=" * 70)
    logger.info(f"STARTING ENERGY OPTIMIZATION TRAINING")
    logger.info("=" * 70)
    if args.use_wandb:
        logger.info(f"📊 W&B Project: {args.wandb_project}")
    if args.eval_during_training:
        logger.info(f"⚡ Energy evaluation enabled (every 500 steps)")
    logger.info("=" * 70)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info("\nSaving final model...")
    # trainer.save_model(f"{args.output_dir}/final") # This only saves adapters
    
    # Save merged model for vLLM usage
    logger.info("Saving merged 16bit model for vLLM compatibility...")
    model.save_pretrained_merged(
        f"{args.output_dir}/final", 
        tokenizer, 
        save_method="merged_16bit",
    )
    # Also save adapters separately just in case
    model.save_pretrained_merged(
        f"{args.output_dir}/adapters", 
        tokenizer, 
        save_method="lora",
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"ENERGY OPTIMIZATION TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    logger.info(f"Final model saved to: {args.output_dir}/final")

    logger.info("\nNext step: Run GRPO alignment training")
    logger.info(f"  python grpo_trainer_vllm.py --model-path {args.output_dir}/final ...")

    logger.info("=" * 70)


if __name__ == '__main__':
    main()
