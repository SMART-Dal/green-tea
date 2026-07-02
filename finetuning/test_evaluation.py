
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import os
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finetuning.utils.training_callbacks import EnergyEvaluationCallback
from finetuning.dataset_preprocessing import apply_chat_template

# --- CONFIGURATION ---
MODEL_PATH = "finetuning/checkpoints/qwen-coder-14b_sft_20260114_133855/final"
TOKENIZER_PATH = "Qwen/Qwen2.5-Coder-14B-Instruct" # Use base tokenizer
DATASET_PATH = "finetuning/data/sft_pairs_val.jsonl"
EVAL_DATASET_SIZE = 50 # Number of samples to evaluate on, keep it small for a quick test
MAX_NEW_TOKENS = 768 # Increased from 512 to be safer

def main():
    """
    Runs a standalone evaluation of the EnergyEvaluationCallback to quickly test
    if the model generates compilable and successful code after fixes.
    """
    print("--- STANDALONE EVALUATION TEST ---")

    # 1. Load Model and Tokenizer
    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # 2. Load and Prepare Dataset
    print(f"Loading validation dataset from: {DATASET_PATH}")
    eval_dataset = load_dataset("json", data_files=DATASET_PATH, split=f'train[:{EVAL_DATASET_SIZE}]')
    
    # The callback expects the dataset formatted in a specific way
    original_columns = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda x: {"text": apply_chat_template(x, tokenizer, "validation")},
        remove_columns=original_columns,
    )
    print(f"Prepared {len(eval_dataset)} samples for evaluation.")

    # 3. Instantiate the Callback
    print("Instantiating EnergyEvaluationCallback...")
    evaluation_callback = EnergyEvaluationCallback(
        eval_model=model,
        eval_tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        eval_batch_size=2, # Small batch size for evaluation
        log_predictions=True,
    )

    # 4. Run the Evaluation
    print("\n--- Starting Evaluation ---")
    trainer_state_mock = type('TrainerState', (), {'global_step': 999})
    metrics = evaluation_callback._evaluate_energy_metrics(trainer_state_mock)
    
    # 5. Print Results
    print("\n--- EVALUATION COMPLETE ---")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n--- DETAILED PREDICTIONS ---")
    # Access the logged predictions if available
    if hasattr(evaluation_callback, 'prediction_log_df'):
        df = evaluation_callback.prediction_log_df
        # Display relevant columns for analysis
        display_cols = ['step', 'problem_name', 'compiles', 'success', 'energy_reduction', 'edp_reduction', 'error_type']
        
        # Add generated code for failed cases
        pd.set_option('display.max_colwidth', 200) # Show more of the code
        for index, row in df.iterrows():
            if not row['success']:
                print(f"\n--- FAILURE: {row['problem_name']} ---")
                print(f"Compilation: {'OK' if row['compiles'] else 'FAILED'}")
                if not row['compiles']:
                    print(f"Error Type: {row['error_type']}")
                print(f"Generated Code:\n---\n{row['generated_code']}\n---")


if __name__ == "__main__":
    main()
