import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import sys
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Import model definitions
# Assuming models are importable from the current environment/path
try:
    from models.recursive_reasoning.trm_dyt import TinyRecursiveReasoningModel_ACTV1_DyT as TRM_DyT
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as TRM
    from models.recursive_reasoning.trm_dyt_e1 import TinyRecursiveReasoningModel_ACTV1_DyT_EBT1 as TRM_DyT_EB1
    from models.recursive_reasoning.trm_dyt_e2 import TinyRecursiveReasoningModel_ACTV1_DyT_EBT2 as TRM_DyT_EB2
    from models.recursive_reasoning.trm_ebt2 import TinyRecursiveReasoningModel_ACTV1_EBT2 as TRM_EB2
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please ensure model files are in the correct directory or PYTHONPATH.")
    sys.exit(1)

# --- Constants ---
SORTING_VOCAB_SIZE = 11 # Digits 0-9 (10 tokens) + 1 PAD token
PAD_TOKEN_ID = 10      # Use 10 as the PAD token ID
IGNORE_LABEL_ID = -100 # Standard ignore index for cross-entropy
ALL_DIGITS = np.arange(10) # Digits 0-9


# --- Data Generation with Gradual Duplicates ---
def generate_sequence_with_duplicates(seq_len):
    """Generates a single sequence with controlled duplicates based on length."""
    if seq_len <= 3: # Unique
        return np.random.choice(ALL_DIGITS, size=seq_len, replace=False)
    elif 4 <= seq_len <= 6: # Exactly 1 pair
        unique_count = seq_len - 2
        uniques = np.random.choice(ALL_DIGITS, size=unique_count, replace=False)
        available_digits = np.setdiff1d(ALL_DIGITS, uniques)
        duplicate_digit = np.random.choice(available_digits if len(available_digits) > 0 else ALL_DIGITS) # Fallback
        seq = np.concatenate((uniques, [duplicate_digit, duplicate_digit]))
        np.random.shuffle(seq)
        return seq
    elif 7 <= seq_len <= 8: # Exactly 2 pairs
        unique_count = seq_len - 4
        uniques = np.random.choice(ALL_DIGITS, size=unique_count, replace=False)
        available_digits = np.setdiff1d(ALL_DIGITS, uniques)
        duplicate_digits = np.random.choice(available_digits if len(available_digits) >= 2 else ALL_DIGITS, size=2, replace=False)
        seq = np.concatenate((uniques, [duplicate_digits[0]]*2, [duplicate_digits[1]]*2))
        np.random.shuffle(seq)
        return seq
    elif seq_len == 9: # Exactly 3 pairs
        unique_count = seq_len - 6
        uniques = np.random.choice(ALL_DIGITS, size=unique_count, replace=False)
        available_digits = np.setdiff1d(ALL_DIGITS, uniques)
        duplicate_digits = np.random.choice(available_digits if len(available_digits) >= 3 else ALL_DIGITS, size=3, replace=False)
        seq = np.concatenate((uniques, [duplicate_digits[0]]*2, [duplicate_digits[1]]*2, [duplicate_digits[2]]*2))
        np.random.shuffle(seq)
        return seq
    else: # seq_len >= 10, allow free replacement
        return np.random.choice(ALL_DIGITS, size=seq_len, replace=True)

# --- MODIFIED Data Generation with Positional Jitter ---
def generate_sorting_task_data(num_samples, batch_size, seq_len, max_model_len, device, task_name="Sorting"):
    """
    Generates data batches for the sorting task with gradual duplicates
    AND positional jitter.
    
    Args:
        seq_len (int): The length of the *active* sequence.
        max_model_len (int): The *total* tensor length to pad to.
    """
    data = []
    num_batches = num_samples // batch_size
    if num_batches == 0:
        return []
    elif num_samples % batch_size != 0:
        actual_samples = num_batches * batch_size

    # Ensure active sequence fits within the max model length
    if seq_len > max_model_len:
        print(f"Warning: Skipping {task_name} for seq_len {seq_len} as it exceeds max_model_len {max_model_len}.")
        return []

    # Calculate the maximum possible start index for jittering
    max_start_idx = max_model_len - seq_len

    for _ in range(num_batches):
        # Create full-size arrays initialized with pad/ignore tokens
        # Use int64 for numpy defaults, will be cast to torch.long
        unsorted_batch_full = np.full((batch_size, max_model_len), PAD_TOKEN_ID, dtype=np.int64)
        sorted_batch_full = np.full((batch_size, max_model_len), IGNORE_LABEL_ID, dtype=np.int64)

        for i in range(batch_size):
            # 1. Generate the active sequence
            unsorted_active = generate_sequence_with_duplicates(seq_len)
            sorted_active = np.sort(unsorted_active)
            
            # 2. Choose a random start index for jittering
            # np.random.randint is exclusive of the high value, so +1
            start_idx = np.random.randint(0, max_start_idx + 1)
            end_idx = start_idx + seq_len
            
            # 3. Place the active sequences into the full-size arrays
            unsorted_batch_full[i, start_idx:end_idx] = unsorted_active
            sorted_batch_full[i, start_idx:end_idx] = sorted_active

        # Convert the *full-size* arrays to tensors
        unsorted_batch = torch.tensor(unsorted_batch_full, device=device, dtype=torch.long)
        sorted_batch = torch.tensor(sorted_batch_full, device=device, dtype=torch.long)
        data.append((unsorted_batch, sorted_batch))
        
    return data
# --- End Data Generation ---


# --- NEW COMBINED LOSS WRAPPER ---
class TRMCombinedLoss(nn.Module):
    """
    A loss wrapper that correctly computes and combines both the
    Language Model (task) loss and the ACT (halting) loss.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, carry, batch):
        """Calculates combined LM and ACT loss."""
        # --- 1. Model Forward Pass ---
        current_carry, outputs = self.model(carry, batch)
        
        # Extract all relevant tensors
        logits = outputs["logits"] # [B, MaxSeqLen, Vocab]
        labels = batch["labels"]   # [B, MaxSeqLen]
        q_halt_logits = outputs["q_halt_logits"] # [B]
        halted = current_carry.halted # [B]

        logit_seq_len = logits.shape[1]
        label_seq_len = labels.shape[1]

        # Safeguard: Align lengths if they somehow mismatch
        if logit_seq_len != label_seq_len:
            min_len = min(logit_seq_len, label_seq_len)
            print(f"Warning: Jittered Logit length ({logit_seq_len}) != Label length ({label_seq_len}). Using min length {min_len}.")
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]

        # --- 2. Calculate Task Loss (Language Model Loss) ---
        # We must use reduction='none' to get per-token loss
        lm_loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), # [B*MaxSeqLen, Vocab]
            labels.reshape(-1).long(),            # [B*MaxSeqLen]
            ignore_index=IGNORE_LABEL_ID,
            reduction='none'
        )
        
        lm_loss_per_seq = lm_loss_per_token.view(labels.shape).sum(dim=-1) # [B]
        
        # Normalize by the number of valid tokens in each sequence
        with torch.no_grad():
            valid_mask = (labels != IGNORE_LABEL_ID)
            loss_counts = valid_mask.sum(dim=-1).clamp_min(1) # [B], avoid div by zero
            
        # Final normalized LM loss per sequence
        lm_loss = (lm_loss_per_seq / loss_counts).mean() # Mean over the batch

        # --- 3. Calculate ACT Loss (Halting Loss) ---
        q_halt_loss = torch.tensor(0.0, device=logits.device)
        q_continue_loss = torch.tensor(0.0, device=logits.device)
        
        # This logic is only active if halt_steps > 1
        # We need the "ground truth" of whether the sequence is correct *now*
        with torch.no_grad():
            # valid_mask is from above
            # loss_counts is from above
            preds = torch.argmax(logits, dim=-1)
            is_correct = valid_mask & (preds == labels)
            # A sequence is "correct" if all its valid tokens are correct
            seq_is_correct = (is_correct.sum(dim=-1) == valid_mask.sum(dim=-1))
            
        # Q-Halt Loss: Binary Cross-Entropy
        # Target is 1.0 if the sequence is correct, 0.0 otherwise
        q_halt_target = seq_is_correct.to(q_halt_logits.dtype)
        # Use mean() to keep it scaled similarly to lm_loss
        q_halt_loss = F.binary_cross_entropy_with_logits(q_halt_logits, q_halt_target, reduction='mean')

        # Q-Continue Loss (if model provides it, e.g., non-"no_ACT_continue" models)
        if "target_q_continue" in outputs:
            q_continue_logits = outputs["q_continue_logits"]
            target_q_continue = outputs["target_q_continue"]
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits, target_q_continue, reduction='mean')

        # --- 4. Combine Losses ---
        # Weighting from original models/losses.py (using mean reduction, so 0.5 is a relative weight)
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

        # Metrics for logging (optional, but good practice)
        metrics = {
            "logits": outputs["logits"], 
            "original_labels": batch["labels"],
            "loss_lm": lm_loss.detach(),
            "loss_q_halt": q_halt_loss.detach(),
            "loss_q_cont": q_continue_loss.detach(),
            "acc_q_halt": ((q_halt_logits > 0) == seq_is_correct).float().mean()
        }
        
        return current_carry, total_loss, metrics, {}, halted
# --- END LOSS WRAPPER ---


# --- Training and Evaluation Logic ---
def train_one_epoch(model, optimizer, train_data_batches, device, epoch_num, total_epochs, model_name):
    """
    Trains the model for one epoch.
    train_data_batches is now a pre-computed list of batches (x_train, y_train).
    """
    model.train() # Set model to training mode
    epoch_loss = 0.0
    num_batches = len(train_data_batches) # Use the length of the provided batch list
    if num_batches == 0: return float('nan') # Handle case with no data

    # Progress bar for the epoch
    batch_iterator = tqdm(train_data_batches, desc=f"Epoch {epoch_num+1}/{total_epochs} [{model_name}]", leave=False)

    for (x_train, y_train) in batch_iterator: # Iterate directly over batches
        optimizer.zero_grad(set_to_none=True) # Reset gradients efficiently
        
        # x_train and y_train are now full-length, jittered tensors
        # The model's forward pass will handle this full-length tensor

        # Prepare batch dictionary expected by the model
        batch_data = {
            "inputs": x_train,
            "labels": y_train,
            "puzzle_identifiers": torch.zeros(x_train.shape[0], device=device, dtype=torch.long) # Dummy identifiers
        }

        loss_fn = TRMCombinedLoss(model) # Instantiate loss wrapper
        current_carry = model.initial_carry(batch_data) # Get initial state if needed

        # Forward pass and loss calculation
        _, loss, _, _, _ = loss_fn(carry=current_carry, batch=batch_data)

        # Handle potential NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"!!! NaN/Inf loss detected in {model_name} at Epoch {epoch_num+1}. Skipping backward/step. !!!")
            continue # Skip this batch

        # Backward pass and optimizer step
        loss.backward()

        # --- Gradient Integrity Check ---
        # Clip gradients and get the total norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # If the norm is NaN or Inf, gradients are bad
        if not torch.isfinite(total_norm):
            print(f"!!! NaN/Inf gradients detected in {model_name} at Epoch {epoch_num+1}. Skipping optimizer step. !!!")
            optimizer.zero_grad(set_to_none=True) # Clear the bad gradients
            continue # Skip this optimizer step
        # --- End Check ---

        optimizer.step()

        batch_loss_item = loss.item()
        epoch_loss += batch_loss_item
        batch_iterator.set_postfix(loss=f"{batch_loss_item:.6f}") # Update progress bar

    # Return average loss for the epoch
    return epoch_loss / num_batches if num_batches > 0 else float('nan')

def evaluate_model(model, test_data, device, current_test_len):
    """Evaluates the model on the test dataset."""
    model.eval() # Set model to evaluation mode
    total_test_loss = 0
    total_correct_tokens = 0
    total_non_ignored_tokens = 0

    if not test_data: return float('nan'), float('nan') # Handle case with no data

    with torch.no_grad(): # Disable gradient calculations for evaluation
        test_batch_iterator = tqdm(test_data, desc=f"Evaluating Active L={current_test_len}", leave=False)
        for x_test, y_test in test_batch_iterator:
            # x_test and y_test are full-length, jittered tensors
            
            # Prepare batch dictionary
            test_batch_data = {
                "inputs": x_test,
                "labels": y_test,
                "puzzle_identifiers": torch.zeros(x_test.shape[0], device=device, dtype=torch.long) # Dummy identifiers
            }

            loss_fn = TRMCombinedLoss(model) # Instantiate loss wrapper
            test_carry = model.initial_carry(test_batch_data) # Get initial state

            # Forward pass
            _, loss, metrics_output, _, _ = loss_fn(carry=test_carry, batch=test_batch_data)
            logits = metrics_output.get("logits") # Get full-length logits
            original_labels = metrics_output.get("original_labels") # Get full-length labels

            # Accumulate loss if valid
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_test_loss += loss.item()

                # Calculate accuracy if logits and labels are available
                if logits is not None and original_labels is not None:
                    # No slicing needed here, logits and labels are aligned
                    preds = torch.argmax(logits, dim=-1) # Get predicted tokens
                    valid_mask = (original_labels != IGNORE_LABEL_ID) # Mask for non-ignored (active) tokens

                    batch_correct = ((preds == original_labels) & valid_mask).sum().item() # Count correct predictions
                    batch_total_valid = valid_mask.sum().item() # Count total valid tokens

                    total_correct_tokens += batch_correct
                    total_non_ignored_tokens += batch_total_valid
            else:
                print(f"Warning: NaN/Inf loss during evaluation for {model.__class__.__name__}. Skipping batch metrics.")

    # Calculate average loss and accuracy
    avg_test_loss = total_test_loss / len(test_data) if len(test_data) > 0 else float('nan')
    
    # --- FIX: Prevent ZeroDivisionError ---
    avg_test_accuracy = total_correct_tokens / total_non_ignored_tokens if total_non_ignored_tokens > 0 else float('nan')
    # --- END FIX ---

    return avg_test_loss, avg_test_accuracy

# --- Plotting ---
def plot_curriculum_results(results, curriculum_lengths, metric='loss', filename_prefix="curriculum_results"):
    """Plots the results (loss or accuracy) across curriculum stages."""
    plt.figure(figsize=(12, 8))
    metric_label = "Test Loss" if metric == 'loss' else "Test Accuracy"
    metric_key = 'test_loss' if metric == 'loss' else 'test_accuracy'
    
    # Ensure curriculum_lengths is sorted for plotting
    plotted_lengths = sorted(list(curriculum_lengths))

    for model_name, model_results in results.items():
        # Extract metric values for each stage length
        metric_values = [model_results.get(length, {}).get(metric_key, float('nan')) for length in plotted_lengths]
        # Filter out NaN values for plotting
        valid_lengths = [l for l, v in zip(plotted_lengths, metric_values) if not math.isnan(v)]
        valid_metrics = [v for v in metric_values if not math.isnan(v)]

        if valid_lengths: # Plot if there's valid data
            plt.plot(valid_lengths, valid_metrics, marker='o', linestyle='-', label=model_name)

    # Configure and save the plot
    plt.title(f"Model {metric_label} vs. Training Stage (Gradual Duplicates, Jittered)")
    plt.xlabel("Training Stage (Max Active Sequence Length)")
    plt.ylabel(f"Final {metric_label} on Test Length (Train Length + Gap)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(plotted_lengths) # Ensure all curriculum lengths are marked
    plot_filename = f"{filename_prefix}_{metric}.png"
    plt.savefig(plot_filename)
    print(f"\nCurriculum {metric} plot saved to {plot_filename}")
    plt.close() # Close the figure to free memory


# --- Main Execution Block ---
if __name__ == '__main__':

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Train TRM models on a sorting task using curriculum learning with positional jitter.")
    
    parser.add_argument("--curriculum-lengths", type=int, nargs='+', 
                        default=[0,1,2, 3, 4, 5, 10, 25, 50, 100], 
                        help="Sparse list of *active* training sequence lengths for the curriculum.")
    parser.add_argument("--final-test-lengths", type=int, nargs='+', 
                        default=[7, 15, 35, 75, 150], 
                        help="List of unseen *active* sequence lengths for the final generalization test.")

    # --- NEW ARG ---
    parser.add_argument("--max-model-len", type=int, default=200, 
                        help="Maximum sequence length the model is initialized with and tensors are padded to. MUST be >= max(curriculum_lengths) + test_len_gap and max(final_test_lengths).")

    parser.add_argument("--test-len-gap", type=int, default=1, help="Difference between test and train length (test_len = train_len + gap) during the curriculum phase.")
    parser.add_argument("--epochs-per-stage", type=int, default=50, help="Maximum epochs to train for each sequence length.")
    parser.add_argument("--early-stop-threshold", type=float, default=1e-6, help="Loss threshold for early stopping within a stage.")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Number of consecutive epochs loss must be below threshold to stop.")

    # General training args
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per device.")
    parser.add_argument("--pos-encodings", type=str, default="rope", choices=["rope", "learned", "none"], help="Type of positional encoding to use.")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers in the reasoning module.")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--d-ffn", type=int, default=512, help="Feedforward network dimension.")
    parser.add_argument("--h-cycles", type=int, default=3, help="Number of T cycles (outer loop).")
    parser.add_argument("--l-cycles", type=int, default=6, help="Number of n cycles (inner loop).")
    parser.add_argument("--halt-steps", type=int, default=16, help="Maximum halt steps for ACT. (Default: 16)")
    parser.add_argument("--dyt-alpha", type=float, default=0.80, help="Initial alpha for DyT.")
    parser.add_argument("--dyt-slope", type=float, default=0.072, help="Slope for DyT alpha initialization.")
    parser.add_argument("--ebt-beta", type=float, default=0.5, help="Beta value (strength) for EBT guidance.")
    parser.add_argument("--ebt-hessian-eps", type=float, default=1e-8, help="Epsilon for EBT2 Hessian approximation.")
    parser.add_argument("--train-batches-per-stage", type=int, default=200, help="Total number of training *batches* (steps) per stage.")
    parser.add_argument("--test-batches-per-stage", type=int, default=10, help="Total number of test *batches* (steps) per stage.")
    parser.add_argument("--results-filename", type=str, default="curriculum_results_jitter.json", help="Filename to save results JSON.")
    parser.add_argument("--plot-filename-prefix", type=str, default="curriculum_results_jitter", help="Prefix for saving plot filenames.")
    parser.add_argument("--final-results-filename", type=str, default="final_test_results_jitter.json", help="Filename to save final generalization test results.")


    args = parser.parse_args()

    # --- Validate Curriculum Args ---
    if not args.curriculum_lengths or min(args.curriculum_lengths) < 0:
        raise ValueError("--curriculum-lengths must contain values >= 0")
    args.curriculum_lengths = sorted(list(set(args.curriculum_lengths))) # Ensure sorted and unique
    if not args.final_test_lengths:
        print("Warning: --final-test-lengths is empty. No final generalization test will be run.")
    
    # --- MODIFIED: Use max-model-len for validation ---
    max_curriculum_len = max(args.curriculum_lengths) if args.curriculum_lengths else 0
    max_curriculum_test_len = max_curriculum_len + args.test_len_gap
    max_final_test_len = max(args.final_test_lengths) if args.final_test_lengths else 0
    
    # This is the single, most important length for model init
    MAX_LEN_MODEL_INIT = args.max_model_len
    
    required_len = max(max_curriculum_test_len, max_final_test_len)
    
    if MAX_LEN_MODEL_INIT <= 0:
         raise ValueError("--max-model-len must be > 0.")
    if MAX_LEN_MODEL_INIT < required_len:
        raise ValueError(f"--max-model-len ({MAX_LEN_MODEL_INIT}) must be >= max curriculum test length ({max_curriculum_test_len}) and max final test length ({max_final_test_len}). Required: {required_len}")
    # --- END MODIFICATION ---

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using positional encoding: {args.pos_encodings}")
    print(f"Using curriculum *active* lengths: {args.curriculum_lengths}")
    print(f"Using final test *active* lengths: {args.final_test_lengths}")
    print(f"Initializing models with *total* padded length = {MAX_LEN_MODEL_INIT}")

    # We ignore the 'base' (e.g., 10000) and dynamically calculate the
    # "biologically optimal" base from the GridPE paper for 1D space (p=1).
    # The optimal wavelength ratio r = p_root(e) = e.
    # RoPE's wavelength ratio is base**(2/dim).
    # Setting base**(2/dim) = e  =>  log(base) = dim / 2  =>  base = e**(dim / 2)
    # Here 'dim' is the head_dim, which is correct.

    # --- Define Model Configurations (using MAX_LEN_MODEL_INIT) ---
    base_config = {
        "vocab_size": SORTING_VOCAB_SIZE, "hidden_size": args.d_model, "L_layers": args.n_layers,
        "num_heads": args.n_head, "expansion": args.d_ffn / args.d_model, "H_cycles": args.h_cycles,
        "L_cycles": args.l_cycles, "halt_max_steps": args.halt_steps,
        "pos_encodings": args.pos_encodings,
        "batch_size": args.batch_size, 
        "seq_len": MAX_LEN_MODEL_INIT, # CRITICAL: Use max length for model init
        "num_puzzle_identifiers": 1, "H_layers": 1, "halt_exploration_prob": 0.0,
        "puzzle_emb_ndim": 0, "puzzle_emb_len": 0, # Assuming no puzzle embedding for sorting task
        "rope_theta": math.exp( (args.d_model // args.n_head) / 2),     
        "forward_dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32" # Auto-select dtype


    }

    # Derive specific model configurations
    trm_config = base_config.copy()
    trm_dyt_config = base_config.copy()
    trm_dyt_config.update({"dyt_init_a": args.dyt_alpha, "dyt_init_a_slope": args.dyt_slope})
    trm_dyt_ebt1_config = trm_dyt_config.copy()
    trm_dyt_ebt1_config.update({"ebt_beta": args.ebt_beta})
    trm_dyt_ebt2_config = trm_dyt_config.copy()
    trm_dyt_ebt2_config.update({"ebt_beta": args.ebt_beta, "ebt_hessian_eps": args.ebt_hessian_eps})
    trm_ebt2_config = base_config.copy()
    trm_ebt2_config.update({"ebt_beta": args.ebt_beta, "ebt_hessian_eps": args.ebt_hessian_eps})

    # --- List of Models to Train ---
    model_defs = [
        (f"TRM DyT EBT2 (beta={args.ebt_beta:.3f}, eps={args.ebt_hessian_eps:.1E})", TRM_DyT_EB2, trm_dyt_ebt2_config),
        (f"TRM EBT2 (beta={args.ebt_beta:.3f}, eps={args.ebt_hessian_eps:.1E})", TRM_EB2, trm_ebt2_config),
        (f"TRM DyT EBT1 (beta={args.ebt_beta:.3f})", TRM_DyT_EB1, trm_dyt_ebt1_config),
        (f"TRM DyT (a={args.dyt_alpha:.3f}, slope={args.dyt_slope:.3f})", TRM_DyT, trm_dyt_config),
        ("Baseline TRM", TRM, trm_config),
    ]

    # --- Initialize Models and Optimizers ---
    models = {}
    optimizers = {}
    for name, model_class, config in model_defs:
        print(f"Initializing {name}...")
        try:
            # Models are initialized with seq_len=MAX_LEN_MODEL_INIT
            models[name] = model_class(config).to(device)
            optimizers[name] = optim.AdamW(models[name].parameters(), lr=args.lr)
        except Exception as e:
            print(f"Error initializing model {name}: {e}")
            # Optionally skip this model or exit
            # sys.exit(1)

    # --- Curriculum Loop ---
    all_results = defaultdict(dict) # Stores results per model per stage length
    curriculum_lengths = args.curriculum_lengths # Use the sparse list from args
    
    cumulative_lengths_so_far = []

    print(f"\n--- Starting Mixed-Batch Jittered Curriculum Learning ---")

    for current_train_len in curriculum_lengths:
        cumulative_lengths_so_far.append(current_train_len)
        current_test_len = current_train_len + args.test_len_gap
        print(f"\n--- Stage: Training on active lengths {cumulative_lengths_so_far}, Testing Active Length {current_test_len} ---")

        # --- MODIFIED BATCH ALLOCATION LOGIC ---
        # Allocate 50% of batches to the new longest length,
        # and 50% split evenly among all previous lengths.
        total_batches_per_stage = args.train_batches_per_stage # Use the new argument directly
        stage_train_batches = []

        if total_batches_per_stage == 0:
            print(f"Warning: --train-batches-per-stage is 0. Not generating training data for this stage.")
        
        else:
            # Identify current (longest) length and previous lengths
            current_longest_len = current_train_len # This is the newest length for this stage
            previous_lengths = [l for l in cumulative_lengths_so_far if l != current_longest_len]
            num_previous_lengths = len(previous_lengths)

            # Allocate batches
            if num_previous_lengths == 0:
                # This is the first stage (e.g., L=2), 100% of batches go to this length
                batches_for_longest_len = total_batches_per_stage
                samples_for_longest_len = batches_for_longest_len * args.batch_size
                samples_per_previous_len = 0
                print(f"First stage: Allocating all {batches_for_longest_len} batches to Active L={current_longest_len}.")
            
            else:
                # Subsequent stages: ~50% to longest, ~50% to previous
                batches_for_longest_len = int(math.ceil(total_batches_per_stage * 0.5))
                samples_for_longest_len = batches_for_longest_len * args.batch_size
                
                batches_for_previous_total = total_batches_per_stage - batches_for_longest_len
                
                if batches_for_previous_total > 0:
                    # Give at least 1 batch to each previous length if possible
                    batches_per_previous_len = max(1, batches_for_previous_total // num_previous_lengths)
                    samples_per_previous_len = batches_per_previous_len * args.batch_size
                    
                    print(f"Allocating {batches_for_longest_len} batches to Active L={current_longest_len} (New).")
                    print(f"Allocating {batches_per_previous_len} batches to each of {num_previous_lengths} previous lengths.")
                
                else:
                    # Not enough batches to split, give all to the longest length
                    batches_for_longest_len = total_batches_per_stage
                    samples_for_longest_len = batches_for_longest_len * args.batch_size
                    samples_per_previous_len = 0
                    print(f"Total batches ({total_batches_per_stage}) not enough to split. Allocating all to Active L={current_longest_len}.")

            # Generate data for the longest length
            if samples_for_longest_len > 0:
                data_for_len = generate_sorting_task_data(
                    samples_for_longest_len,
                    args.batch_size,
                    current_longest_len,
                    MAX_LEN_MODEL_INIT,
                    device,
                    task_name=f"Train Active L={current_longest_len} (New)"
                )
                stage_train_batches.extend(data_for_len)

            # Generate data for all previous lengths
            if samples_per_previous_len > 0 and num_previous_lengths > 0:
                for length in previous_lengths:
                    data_for_len = generate_sorting_task_data(
                        samples_per_previous_len,
                        args.batch_size,
                        length,
                        MAX_LEN_MODEL_INIT,
                        device,
                        task_name=f"Train Active L={length} (Previous)"
                    )
                    stage_train_batches.extend(data_for_len)
        # --- END MODIFIED BATCH ALLOCATION ---


        # Generate test data (only for the current stage's test length)
        test_data = generate_sorting_task_data(
            args.test_batches_per_stage * args.batch_size, # Use arg.batch_size 
            args.batch_size, 
            current_test_len, # active seq_len
            MAX_LEN_MODEL_INIT, # max_model_len
            device, 
            task_name=f"Test Active L={current_test_len}"
        )

        if not stage_train_batches:
            print(f"Skipping stage L={current_train_len} due to insufficient training samples.")
            # Store NaN results for plotting consistency
            for model_name in models.keys():
                 all_results[model_name][current_train_len] = {
                     'test_loss': float('nan'), 'test_accuracy': float('nan'),
                     'epochs_trained': 0, 'final_train_loss': float('nan')
                 }
            continue

        for model_name, model in models.items():
            if model_name not in optimizers: # Skip if initialization failed
                print(f"Skipping training for {model_name} due to initialization error.")
                all_results[model_name][current_train_len] = {
                    'test_loss': float('nan'), 'test_accuracy': float('nan'),
                    'epochs_trained': 0, 'final_train_loss': float('nan')
                }
                continue

            print(f"\nTraining {model_name} on active lengths {cumulative_lengths_so_far}...")
            optimizer = optimizers[model_name]
            consecutive_below_threshold = 0
            stage_train_losses = []
            final_epoch_num = 0
            avg_loss = float('nan') 

            for epoch in range(args.epochs_per_stage):
                final_epoch_num = epoch + 1
                
                # Shuffle combined batches *every* epoch
                np.random.shuffle(stage_train_batches)

                # --- Add try/except for stability ---
                try:
                    # Pass the shuffled mixed-batch list to the epoch trainer
                    avg_loss = train_one_epoch(
                        model, optimizer, stage_train_batches, device, epoch, args.epochs_per_stage, model_name
                    )
                except Exception as e:
                    print(f"!!! CRITICAL ERROR in {model_name} during training epoch {epoch+1}: {e} !!!")
                    print("Skipping rest of stage for this model.")
                    avg_loss = float('nan') # Mark as NaN
                    break # Exit the epoch loop for this model
                # --- End try/except ---
                
                stage_train_losses.append(avg_loss)

                # Check for NaN loss
                if math.isnan(avg_loss):
                     print(f"Stopping stage early for {model_name} at epoch {epoch+1} due to NaN loss.")
                     break # Stop training this model for this stage

                # Early stopping logic based on training loss threshold
                if avg_loss < args.early_stop_threshold:
                    consecutive_below_threshold += 1
                else:
                    consecutive_below_threshold = 0

                # Stop if loss is below threshold for enough consecutive epochs
                if consecutive_below_threshold >= args.early_stop_patience:
                    print(f"Early stopping for {model_name} at epoch {epoch+1} (loss {avg_loss:.6f} < {args.early_stop_threshold} for {args.early_stop_patience} epochs).")
                    break # Stop training this model for this stage

            # Evaluate after finishing the stage (or early stopping)
            # --- Add try/except for stability ---
            try:
                test_loss, test_accuracy = evaluate_model(model, test_data, device, current_test_len)
            except Exception as e:
                print(f"!!! CRITICAL ERROR in {model_name} during evaluation: {e} !!!")
                test_loss, test_accuracy = float('nan'), float('nan')
            # --- End try/except ---
            
            print(f"{model_name} Stage Active L={current_train_len}: Test Loss={test_loss:.6f}, Test Acc={test_accuracy:.4f}")

            # Store results for this stage
            all_results[model_name][current_train_len] = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'epochs_trained': final_epoch_num,
                'final_train_loss': avg_loss if not math.isnan(avg_loss) else float('nan') # Store last valid avg_loss
            }
            torch.cuda.empty_cache() # Clear cache between models/stages

    # --- Save Curriculum Results ---
    print("\n--- Curriculum Learning Complete ---")
    try:
        # Prepare results for JSON serialization (handle NaN)
        serializable_results = defaultdict(dict)
        for model, stage_data in all_results.items():
             for length, metrics in stage_data.items():
                 serializable_results[model][length] = {
                     k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                     for k, v in metrics.items()
                 }

        # Use pos_encodings in the filename
        results_filename_final = args.results_filename.replace(".json", f"_pe-{args.pos_encodings}.json")
        with open(results_filename_final, "w") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Saved final curriculum results to {results_filename_final}")
    except Exception as e:
        print(f"Error saving curriculum results to JSON: {e}")

    # --- Plot Curriculum Results ---
    plot_filename_prefix_final = f"{args.plot_filename_prefix}_pe-{args.pos_encodings}"
    plot_curriculum_results(all_results, curriculum_lengths, metric='loss', filename_prefix=plot_filename_prefix_final)
    plot_curriculum_results(all_results, curriculum_lengths, metric='accuracy', filename_prefix=plot_filename_prefix_final)


    # --- Final Generalization Test ---
    print(f"\n--- Final Generalization Test (Active Lengths: {args.final_test_lengths}) ---")
    final_results = defaultdict(dict) # Store results per model AND per length

    if not args.final_test_lengths:
        print("Skipping final generalization test as no lengths were specified.")
    else:
        for final_len in args.final_test_lengths:
            print(f"\n--- Testing Active Length {final_len} ---")
            final_test_data = generate_sorting_task_data(
                args.test_batches_per_stage * args.batch_size, # Use arg.batch_size
                args.batch_size, 
                final_len, # active seq_len
                MAX_LEN_MODEL_INIT, # max_model_len
                device, 
                task_name=f"Final Test Active L={final_len}"
            )
            
            if not final_test_data:
                print(f"Skipping final test for L={final_len} due to insufficient samples or length > max_model_len.")
                for model_name in models.keys():
                    final_results[model_name][final_len] = {'test_loss': float('nan'), 'test_accuracy': float('nan')}
                continue

            for model_name, model in models.items():
                 if model_name not in optimizers: # Skip if initialization failed
                     print(f"Skipping final test for {model_name} due to initialization error.")
                     final_results[model_name][final_len] = {'test_loss': float('nan'), 'test_accuracy': float('nan')}
                     continue

                 print(f"Evaluating {model_name} on final test active length {final_len}...")
                 # --- Add try/except for stability ---
                 try:
                    test_loss, test_accuracy = evaluate_model(model, final_test_data, device, final_len)
                 except Exception as e:
                    print(f"!!! CRITICAL ERROR in {model_name} during final test (L={final_len}): {e} !!!")
                    test_loss, test_accuracy = float('nan'), float('nan')
                 # --- End try/except ---
                 
                 print(f"{model_name} Final Test Active L={final_len}: Test Loss={test_loss:.6f}, Test Acc={test_accuracy:.4f}")
                 final_results[model_name][final_len] = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
                 torch.cuda.empty_cache() # Clear cache between models

        # --- Save Final Test Results ---
        try:
            serializable_final_results = defaultdict(dict)
            for model, length_data in final_results.items():
                 for length, metrics in length_data.items():
                     serializable_final_results[model][length] = {
                         k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                         for k, v in metrics.items()
                     }
            
            final_results_filename_final = args.final_results_filename.replace(".json", f"_pe-{args.pos_encodings}.json")
            with open(final_results_filename_final, "w") as f:
                json.dump(serializable_final_results, f, indent=4)
            print(f"Saved final test results to {final_results_filename_final}")
        except Exception as e:
            print(f"Exception during saving results: {e}")





