# CLAUDE.md - TinyRecursiveModels Development Guide

## Project Overview

**TinyRecursiveModels** implements the Tiny Recursion Model (TRM), a recursive reasoning approach that achieves impressive scores on ARC-AGI benchmarks (45% on ARC-AGI-1, 8% on ARC-AGI-2) using only a 7M parameter neural network.

**Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (https://arxiv.org/abs/2510.04871)

### Core Philosophy
- Small models can solve hard reasoning problems through recursive processing
- Efficiency over scale: recursive reasoning allows tiny networks to compete with large language models
- Simplified approach compared to Hierarchical Reasoning Model (HRM)

### Key Achievements
- 45% accuracy on ARC-AGI-1 with 7M parameters
- 8% accuracy on ARC-AGI-2 with 7M parameters
- Competitive performance on Sudoku-Extreme and Maze-Hard tasks

## Repository Structure

```
TinyRecursiveModels/
├── models/                      # Neural network architectures
│   ├── recursive_reasoning/     # Core TRM implementations
│   │   ├── trm.py              # Main TRM model (ACT version)
│   │   ├── trm_singlez.py      # Single z variant
│   │   ├── trm_hier6.py        # Hierarchical variant
│   │   ├── hrm.py              # Hierarchical Reasoning Model
│   │   └── transformers_baseline.py
│   ├── layers.py               # Attention, MLP, RoPE, embeddings
│   ├── losses.py               # Loss functions (ACT, cross-entropy)
│   ├── ema.py                  # Exponential Moving Average
│   ├── sparse_embedding.py     # Puzzle-specific embeddings
│   └── common.py               # Model utilities
├── dataset/                     # Dataset building scripts
│   ├── build_arc_dataset.py    # ARC-AGI dataset prep
│   ├── build_sudoku_dataset.py # Sudoku dataset prep
│   ├── build_maze_dataset.py   # Maze dataset prep
│   └── common.py               # Dataset utilities, transforms
├── evaluators/                  # Evaluation logic
│   └── arc.py                  # ARC-specific evaluator with voting
├── config/                      # Hydra configuration files
│   ├── cfg_pretrain.yaml       # Main training config
│   └── arch/                   # Model architecture configs
│       ├── trm.yaml
│       ├── hrm.yaml
│       └── ...
├── utils/                       # Utility functions
│   └── functions.py            # Model loading helpers
├── pretrain.py                  # Main training script
├── puzzle_dataset.py           # PyTorch dataset implementation
├── requirements.txt            # Python dependencies
└── README.md                   # User-facing documentation
```

## Core Components

### 1. Training Pipeline (`pretrain.py`)

**Entry Point**: Hydra-based configuration management
- Distributed training support via `torchrun` (NCCL backend)
- Multi-GPU training with gradient synchronization
- Exponential Moving Average (EMA) for model weights
- Checkpoint management and model saving

**Key Classes**:
- `PretrainConfig`: Pydantic model for type-safe configuration
- `TrainState`: Training state container (model, optimizers, carry, step)
- `TrainState.carry`: Model-specific state that persists across iterations

**Training Loop Flow**:
1. Initialize distributed setup (if using multi-GPU)
2. Load dataset with rank-specific sharding
3. Create model and optimizers (AdamATan2 for weights, SignSGD for embeddings)
4. Training iterations with carry state management
5. Periodic evaluation and checkpointing
6. EMA model evaluation

### 2. Models (`models/recursive_reasoning/`)

#### TRM Architecture (`trm.py`)

**Core Concept**: Adaptive Computation Time (ACT) with recursive reasoning

**Key Components**:
- `TinyRecursiveReasoningModel_ACTV1_Inner`: Core reasoning module
  - Embedding layers (token + puzzle + positional)
  - L-level reasoning layers (transformer or MLP)
  - LM head for predictions
  - Q-head for halting decisions

- `TinyRecursiveReasoningModel_ACTV1`: ACT wrapper
  - Manages halting logic
  - Tracks computation steps
  - Implements Q-learning for adaptive computation

**Recursive Process**:
1. Start with embedded input `x` and initial latent states `z_H`, `z_L`
2. For H_cycles high-level iterations:
   - For L_cycles low-level iterations:
     - Update `z_L` given `z_H + x` and current `z_L`
   - Update `z_H` given `z_L`
3. Generate output predictions from final `z_H`
4. Q-head determines if more computation needed (during training)

**Configuration Parameters**:
- `H_cycles`: High-level recursion depth
- `L_cycles`: Low-level recursion depth
- `L_layers`: Number of transformer/MLP layers in L-level
- `halt_max_steps`: Maximum ACT iterations
- `puzzle_emb_ndim`: Puzzle embedding dimension
- `mlp_t`: Use MLP instead of transformer for L-level

#### Model Variants:
- `trm.py`: Full ACT version (default)
- `trm_singlez.py`: Single latent state variant
- `trm_hier6.py`: Hierarchical 6-level variant
- `hrm.py`: Original Hierarchical Reasoning Model
- `transformers_baseline.py`: Standard transformer baseline

### 3. Dataset System

#### Dataset Building (`dataset/`)

**ARC Dataset** (`build_arc_dataset.py`):
- Loads ARC-AGI tasks from JSON
- Applies data augmentation (dihedral transformations)
- Creates train/eval/concept splits
- Generates puzzle identifiers for embedding lookup
- Outputs: `.npy` files with inputs, labels, puzzle_indices, group_indices

**Data Augmentation**:
- 8 dihedral transformations (rotations + flips)
- Configurable augmentation count
- Inverse transform tracking for evaluation

**Output Format**:
```
data/{dataset_name}/
├── train/
│   ├── dataset.json              # Metadata
│   ├── {set_name}__inputs.npy    # Tokenized inputs
│   ├── {set_name}__labels.npy    # Target outputs
│   ├── {set_name}__puzzle_identifiers.npy
│   ├── {set_name}__puzzle_indices.npy
│   └── {set_name}__group_indices.npy
└── test/
    └── (same structure)
```

#### Dataset Loading (`puzzle_dataset.py`)

**PuzzleDataset**: Iterable dataset with two modes
- **Train mode**: Randomly samples from groups, balances puzzles
- **Test mode**: Sequential iteration over all examples

**Key Features**:
- Memory-mapped file loading for large datasets
- Automatic padding to batch size
- Multi-dataset merging support
- Distributed training support (rank-based sharding)

**Batch Format**:
```python
{
    "inputs": Tensor[B, SeqLen],      # Tokenized input grids
    "labels": Tensor[B, SeqLen],      # Target output grids
    "puzzle_identifiers": Tensor[B]   # Puzzle ID for embeddings
}
```

### 4. Loss Functions (`models/losses.py`)

**ACTLossHead**: Main loss head for ACT models
- **LM Loss**: Cross-entropy on output predictions
  - Uses `softmax_cross_entropy` or `stablemax_cross_entropy`
  - Ignores padding tokens (IGNORE_LABEL_ID = -100)
- **Q-Halt Loss**: BCE for halting prediction
  - Target: sequence correctness (all tokens correct)
- **Q-Continue Loss**: (Optional) Bootstrapping loss for Q-learning

**Total Loss**: `lm_loss + 0.5 * (q_halt_loss + q_continue_loss)`

**Metrics Tracked**:
- Token-level accuracy
- Exact sequence accuracy
- Q-halt prediction accuracy
- Average computation steps

### 5. Evaluation (`evaluators/arc.py`)

**ARC Evaluator**: Sophisticated voting-based evaluation
- Collects predictions across all augmented versions
- Inverse transforms predictions back to original orientation
- Aggregates predictions using Q-value weighted voting
- Computes pass@K metrics (K=1,2,5,10,100,1000)
- Generates submission JSON for Kaggle

**Evaluation Flow**:
1. Process all test examples through model
2. Inverse transform predictions to canonical form
3. Group predictions by (puzzle, input) using hash
4. Rank predictions by Q-value
5. Vote across augmented versions
6. Generate top-K submission

## Development Workflows

### Setting Up Environment

```bash
# Python 3.10+ required
pip install --upgrade pip wheel setuptools

# Install PyTorch (adjust for your CUDA version)
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Install dependencies
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2

# Configure W&B (optional)
wandb login YOUR-LOGIN
```

### Preparing Datasets

**ARC-AGI-1**:
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

**ARC-AGI-2**:
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

**Note**: Cannot train on both ARC-AGI-1 and ARC-AGI-2 simultaneously (data leakage).

**Sudoku-Extreme**:
```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
```

**Maze-Hard**:
```bash
python dataset/build_maze_dataset.py  # 1000 examples, 8 augments
```

### Training Models

#### Single GPU Training
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name="my_experiment" ema=True
```

#### Multi-GPU Training (torchrun)
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name="my_experiment" ema=True
```

**Key Training Parameters**:
- `arch`: Model architecture (trm, hrm, transformers_baseline, etc.)
- `data_paths`: List of dataset directories
- `global_batch_size`: Total batch size across all GPUs (default: 768)
- `lr`: Learning rate for model weights (default: 1e-4)
- `puzzle_emb_lr`: Learning rate for puzzle embeddings (default: 1e-2)
- `epochs`: Total training epochs
- `eval_interval`: Evaluate every N epochs
- `ema`: Use Exponential Moving Average (recommended: True)
- `ema_rate`: EMA decay rate (default: 0.999)

### Configuration System (Hydra)

**Base Config**: `config/cfg_pretrain.yaml`
**Architecture Configs**: `config/arch/*.yaml`

**Override Syntax**:
```bash
# Override single value
python pretrain.py lr=5e-4

# Add new parameter
python pretrain.py +run_name="experiment1"

# Override nested value
python pretrain.py arch.H_cycles=5

# Use different architecture config
python pretrain.py arch=hrm
```

**Common Architecture Parameters** (`config/arch/trm.yaml`):
```yaml
name: trm@TinyRecursiveReasoningModel_ACTV1
loss:
  name: losses@ACTLossHead
  loss_type: softmax_cross_entropy

# Model dimensions
hidden_size: 384
num_heads: 6
expansion: 1.5

# Recursive structure
H_cycles: 3          # High-level cycles
L_cycles: 4          # Low-level cycles
L_layers: 2          # Layers per level

# Puzzle embeddings
puzzle_emb_ndim: 1536
puzzle_emb_len: 16

# ACT parameters
halt_max_steps: 1
halt_exploration_prob: 0.0

# Position encodings
pos_encodings: rope  # Options: rope, learned, none

# Other
mlp_t: false         # Use MLP instead of attention
```

### Checkpointing and Evaluation

**Checkpoint Location**: `checkpoints/{project_name}/{run_name}/`

**Saved Files**:
- `step_{N}`: Model checkpoint at step N
- `all_config.yaml`: Full configuration dump
- Model source code files (for reproducibility)
- `evaluator_{name}_step_{N}/submission.json`: Evaluation results

**Loading Checkpoint**:
```bash
python pretrain.py \
  load_checkpoint=checkpoints/my_project/my_run/step_10000 \
  ...
```

**Note**: Puzzle embeddings are automatically resized if the number of puzzles changes.

## Conventions and Best Practices

### Code Style

1. **Type Hints**: Use Pydantic models for configuration, type hints for functions
2. **Torch Conventions**:
   - Use `torch.device("cuda")` context for initialization
   - Cast to float16/bfloat16 for forward pass
   - Use float32 for loss computation
3. **Distributed Training**:
   - Always check `rank == 0` before logging/saving
   - Use `dist.all_reduce()` for gradient synchronization
   - Use `dist.broadcast()` for parameter synchronization

### Model Development

**Adding a New Model**:
1. Create file in `models/recursive_reasoning/`
2. Implement `nn.Module` with:
   - `__init__(config_dict: dict)`: Take dict, create config
   - `initial_carry(batch)`: Return initial state
   - `forward(carry, batch)`: Return (new_carry, outputs_dict)
3. Create config in `config/arch/`
4. Register in `utils/functions.py` if needed

**Model Interface Requirements**:
```python
class MyModel(nn.Module):
    def __init__(self, config_dict: dict):
        self.config = MyModelConfig(**config_dict)
        # ... initialization

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # Return initial carry state
        return MyCarry(...)

    def forward(self, carry: MyCarry, batch: Dict[str, torch.Tensor]):
        # Return (new_carry, outputs_dict)
        outputs = {"logits": ..., "q_halt_logits": ..., ...}
        return new_carry, outputs
```

### Dataset Development

**Adding a New Dataset**:
1. Create `dataset/build_{name}_dataset.py`
2. Generate `.npy` files with required fields:
   - `inputs`: [N, SeqLen] int32 array
   - `labels`: [N, SeqLen] int32 array (use ignore_label_id for padding)
   - `puzzle_identifiers`: [NumPuzzles] int32 array
   - `puzzle_indices`: [NumPuzzles+1] int64 cumulative indices
   - `group_indices`: [NumGroups+1] int64 cumulative indices
3. Create `dataset.json` metadata file
4. Split into `train/` and `test/` directories

**Metadata JSON Structure**:
```json
{
  "pad_id": 0,
  "ignore_label_id": 1,
  "blank_identifier_id": 0,
  "vocab_size": 20,
  "seq_len": 900,
  "num_puzzle_identifiers": 500,
  "total_groups": 1000,
  "mean_puzzle_examples": 5.5,
  "total_puzzles": 500,
  "sets": ["training", "evaluation"]
}
```

### Evaluation Development

**Adding a New Evaluator**:
1. Create class in `evaluators/{name}.py`
2. Implement required methods:
   - `required_outputs`: Set of output keys needed
   - `begin_eval()`: Reset state
   - `update_batch(batch, preds)`: Process batch
   - `result(save_path, rank, world_size, group)`: Return metrics dict
3. Add to `config/cfg_pretrain.yaml` evaluators list

**Evaluator Template**:
```python
class MyEvaluator:
    required_outputs = {"preds", "inputs"}

    def begin_eval(self):
        self._results = []

    def update_batch(self, batch, preds):
        # Process predictions
        self._results.append(...)

    def result(self, save_path, rank, world_size, group):
        # Aggregate across ranks using dist.gather_object
        # Return metrics dict (only on rank 0)
        return {"my_metric/accuracy": 0.95}
```

## Architecture Deep Dive

### TRM Recursive Reasoning

**Two-Level Hierarchy**:
- **H-level** (High): Coarse reasoning over answer space
- **L-level** (Low): Fine-grained refinement

**Recursive Flow**:
```
Input x → Embeddings
Initialize z_H, z_L

For h in H_cycles:
    For l in L_cycles:
        z_L ← L_level(z_L, z_H + x)
    z_H ← L_level(z_H, z_L)

Output ← LM_head(z_H)
```

**Key Insight**: Only last H-cycle has gradients, earlier cycles run inference-only for efficiency.

### Adaptive Computation Time (ACT)

**Q-Learning for Halting**:
- Q-head predicts halt/continue value
- Training signal: sequence correctness
- Exploration: Random minimum steps during training
- Evaluation: Always run max steps for batch consistency

**Benefits**:
- Variable computation based on difficulty
- Learned halting criterion
- No manual step tuning needed

### Sparse Puzzle Embeddings

**Design**:
- Each puzzle gets unique embedding vector
- Stored sparsely (only updated for seen puzzles)
- Optimized with SignSGD (different optimizer than main model)
- Higher learning rate (1e-2 vs 1e-4)

**Purpose**:
- Capture puzzle-specific patterns
- Enable quick adaptation to new puzzle types
- Low-dimensional inductive bias (1536-dim → 384-dim model)

### Position Encodings

**Options**:
1. **RoPE** (Rotary Position Embedding): Default, best for variable length
2. **Learned**: Absolute position embeddings
3. **None**: For tasks with no positional structure (e.g., Sudoku)

**Implementation**: See `models/layers.py:RotaryEmbedding`

## Common Issues and Solutions

### Memory Issues

**Problem**: OOM during training
**Solutions**:
1. Reduce `global_batch_size`
2. Enable gradient checkpointing (modify model)
3. Use smaller `hidden_size` or fewer `L_layers`
4. Reduce `halt_max_steps`

### Slow Training

**Problem**: Training too slow
**Solutions**:
1. Disable compilation: `export DISABLE_COMPILE=1`
2. Reduce `L_cycles` or `H_cycles`
3. Use `mlp_t=True` for faster L-level
4. Increase `eval_interval` to evaluate less frequently

### Poor Convergence

**Problem**: Model not learning
**Solutions**:
1. Check dataset quality (correct labels, proper padding)
2. Verify puzzle embeddings are updating (check `puzzle_emb_lr`)
3. Try different loss type: `softmax_cross_entropy` vs `stablemax_cross_entropy`
4. Enable EMA: `ema=True`
5. Adjust learning rate warmup: `lr_warmup_steps`

### Distributed Training Issues

**Problem**: Hangs or crashes with multi-GPU
**Solutions**:
1. Ensure `global_batch_size % world_size == 0`
2. Check NCCL environment variables
3. Verify all ranks have same data
4. Use `NCCL_DEBUG=INFO` for debugging

## Performance Benchmarks

### Expected Training Times (4x H100)

| Task | Runtime | Epochs | Steps |
|------|---------|--------|-------|
| ARC-AGI-1 | ~3 days | 100,000 | ~1M |
| ARC-AGI-2 | ~3 days | 100,000 | ~1M |
| Sudoku-Extreme | <36 hours | 50,000 | ~500K |
| Maze-Hard | <24 hours | 50,000 | ~500K |

### Model Sizes

| Config | Parameters | Hidden Size | Layers |
|--------|-----------|-------------|--------|
| TRM-7M | ~7M | 384 | 2 L-layers |
| TRM-Small | ~3M | 256 | 2 L-layers |
| TRM-Large | ~15M | 512 | 3 L-layers |

## Testing and Validation

### Quick Smoke Test
```bash
# Test data loading
python pretrain.py \
  data_paths="[data/arc1concept-aug-1000]" \
  epochs=1 eval_interval=1 \
  global_batch_size=32

# Test model forward pass
python -c "
from pretrain import *
config = PretrainConfig(**{...})
model, _, _ = create_model(config, metadata, rank=0, world_size=1)
batch = {...}
carry = model.initial_carry(batch)
new_carry, outputs = model(carry, batch)
print('Success!')
"
```

### Validation Checklist
- [ ] Dataset builds without errors
- [ ] Training runs for 100 steps
- [ ] Evaluation completes without OOM
- [ ] Checkpoints save and load correctly
- [ ] Metrics logged to W&B
- [ ] Multi-GPU training works (if applicable)

## References and Related Work

**Original Paper**: https://arxiv.org/abs/2510.04871

**Based On**:
- Hierarchical Reasoning Model (HRM): https://arxiv.org/abs/2506.21734
- HRM Code: https://github.com/sapientinc/HRM
- HRM Analysis: https://github.com/arcprize/hierarchical-reasoning-model-analysis

**Key Differences from HRM**:
1. Simplified architecture (no hierarchy requirement)
2. Removed biological brain analogies
3. No fixed-point theorem needed
4. Pure recursive refinement approach
5. More flexible configuration

## Citation

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Alexia Jolicoeur-Martineau},
  year={2025},
  eprint={2510.04871},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.04871},
}
```

## Quick Reference

### File Locations

| Purpose | Path |
|---------|------|
| Main training | `pretrain.py` |
| Model configs | `config/arch/*.yaml` |
| TRM model | `models/recursive_reasoning/trm.py` |
| Dataset loader | `puzzle_dataset.py` |
| Loss functions | `models/losses.py` |
| ARC evaluator | `evaluators/arc.py` |
| Build ARC data | `dataset/build_arc_dataset.py` |

### Key Constants

```python
IGNORE_LABEL_ID = -100          # Padding label for loss masking
DEFAULT_BATCH_SIZE = 768        # Global batch size
DEFAULT_LR = 1e-4              # Model learning rate
DEFAULT_PUZZLE_EMB_LR = 1e-2   # Embedding learning rate
```

### Common Commands

```bash
# Train on ARC with default settings
torchrun --nproc-per-node 4 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]"

# Resume from checkpoint
python pretrain.py load_checkpoint=checkpoints/my_run/step_10000

# Override hyperparameters
python pretrain.py lr=5e-4 global_batch_size=512 arch.H_cycles=5

# Disable W&B logging
WANDB_MODE=disabled python pretrain.py ...
```

---

*Last Updated: 2025-11-15*
*For questions or issues, refer to the GitHub repository.*
