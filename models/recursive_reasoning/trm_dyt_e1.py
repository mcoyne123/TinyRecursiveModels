from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
# Assuming these imports are in the correct relative path in your project structure
from models.common import trunc_normal_init_
from models.layers import DyT, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry_DyT:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config_DyT(BaseModel):
    # Core dimensions and sizes
    batch_size: int
    seq_len: int # Represents MAX sequence length the model is initialized for
    vocab_size: int
    hidden_size: int
    num_heads: int
    L_layers: int # Number of transformer blocks in the reasoning module

    # Recursion cycles
    H_cycles: int # Outer loop (state update)
    L_cycles: int # Inner loop (reasoning)

    # Embeddings
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    puzzle_emb_len: int = 16 # Default puzzle embedding length if used

    # Transformer specifics
    expansion: float # Expansion factor for SwiGLU MLP
    pos_encodings: str # Type of positional encoding ('rope', 'learned', 'none')
    rms_norm_eps: float = 1e-5 # Ignored when using DyT
    rope_theta: float = 10000.0

    # Adaptive Computation (ACT)
    halt_max_steps: int # Maximum number of ACT steps
    halt_exploration_prob: float # Probability for random exploration steps in ACT
    no_ACT_continue: bool = True # Simplified ACT loss (only halt signal)

    # Data type
    forward_dtype: str = "bfloat16"

    # Dynamic Tanh (DyT) specific config
    dyt_init_a: float = 1.0 # Initial alpha value for DyT layers
    dyt_init_a_slope: float = 0.0 # Slope for increasing alpha across layers

    # Energy-Based Transformer (EBT) specific config
    ebt_beta: float = 0.1 # Strength factor for the EBT gradient guidance

    # Other flags (e.g., for ablations)
    mlp_t: bool = False # Use MLP instead of Transformer block (ablation)
    H_layers: int = 1 # Ignored in TRM, kept for config compatibility if needed

    # Ensure Pydantic handles extra fields if any come from config files
    class Config:
        extra = 'allow'


class TinyRecursiveReasoningModel_ACTV1Block_DyT(nn.Module):
    """A single Transformer block using DyT normalization."""
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config_DyT, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        # Calculate initial alpha for DyT based on layer index and slope
        base_init_a = self.config.dyt_init_a
        slope = self.config.dyt_init_a_slope
        # Each block uses two normalization layers
        init_a_attn = base_init_a + (layer_idx * 2) * slope
        init_a_mlp = base_init_a + (layer_idx * 2 + 1) * slope

        if self.config.mlp_t: # Ablation: Use MLP-based block instead of attention
            # Calculate puzzle embedding length contribution if applicable
            # Use config.seq_len which holds MAX_SEQ_LEN_MODEL
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            mlp_t_hidden_size = self.config.seq_len + self.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=mlp_t_hidden_size,
                expansion=config.expansion,
            )
            # Use the MLP-associated alpha for normalization
            self.norm_mlp_t = DyT(hidden_size=mlp_t_hidden_size, init_a=init_a_mlp)
        else: # Standard Transformer block with Attention
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads, # No Grouped Query Attention
                causal=False # Not causal for this architecture
            )
            # Normalization before the MLP layer (after attention residual)
            self.norm_attn = DyT(hidden_size=config.hidden_size, init_a=init_a_attn)

        # SwiGLU Feedforward Network
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        # Normalization after the MLP residual connection
        self.norm_mlp = DyT(hidden_size=config.hidden_size, init_a=init_a_mlp)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer block."""
        residual = hidden_states

        if self.config.mlp_t: # MLP block ablation path
            hidden_states_t = hidden_states.transpose(1,2) # Reshape for MLP if needed
            out = self.mlp_t(hidden_states_t)
            # Apply residual connection *before* normalization (Post-Norm style)
            hidden_states = self.norm_mlp_t(residual.transpose(1,2) + out)
            hidden_states = hidden_states.transpose(1,2) # Reshape back
        else: # Standard Attention path
            attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            # Apply residual connection *before* normalization
            hidden_states = self.norm_attn(residual + attn_output)

        # Apply MLP sub-block
        residual = hidden_states # Update residual for the MLP block
        out = self.mlp(hidden_states)
        # Apply residual connection *before* normalization
        hidden_states = self.norm_mlp(residual + out)

        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule_DyT(nn.Module):
    """Module representing one pass through the L_layers."""
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block_DyT]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        """Applies input injection and passes through the layers."""
        hidden_states = hidden_states + input_injection # Additive input injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner_DyT_EBT1(nn.Module):
    """The core inner loop model combining TRM-DyT with 1st Order EBT guidance."""
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config_DyT) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Input/Output Embeddings and Heads
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False) # Output projection
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True) # ACT Q-value head

        # EBT Energy Head (New) - Predicts scalar energy from hidden state
        self.energy_head = CastedLinear(self.config.hidden_size, 1, bias=False)
        # Initialize energy head weights near zero to avoid large initial gradients
        with torch.no_grad():
             trunc_normal_init_(self.energy_head.weight, std=1e-4)

        # Puzzle ID Embedding (Optional)
        # Use config.seq_len here which holds MAX_SEQ_LEN_MODEL
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Positional Embeddings
        if self.config.pos_encodings == "rope":
            # Use config.seq_len which holds MAX_SEQ_LEN_MODEL for RoPE init
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
             # Use config.seq_len which holds MAX_SEQ_LEN_MODEL for learned pos emb
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
             pass

        # Core Reasoning Module (using DyT blocks)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule_DyT(
            layers=[TinyRecursiveReasoningModel_ACTV1Block_DyT(self.config, layer_idx=_i)
                    for _i in range(self.config.L_layers)]
        )

        # Initial hidden states (learned buffers)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Initialize Q-head
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input_tokens: torch.Tensor, puzzle_identifiers: torch.Tensor, target_seq_len: Optional[int] = None):
        """Creates input embeddings *without* padding."""
        # Token embeddings
        embedding = self.embed_tokens(input_tokens.to(torch.int32))
        current_seq_len_no_puzzle = embedding.shape[1]

        # Puzzle ID embeddings (if used)
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
            current_total_seq_len = embedding.shape[1] # Includes puzzle emb len
        else:
            current_total_seq_len = current_seq_len_no_puzzle

        # Add learned positional embeddings (if used)
        if self.config.pos_encodings == "learned":
            pos_ids = torch.arange(current_total_seq_len, device=embedding.device).unsqueeze(0)
            pos_embeddings = self.embed_pos(pos_ids)
            embedding = 0.707106781 * (embedding + pos_embeddings.to(self.forward_dtype))

        # --- MODIFICATION: Removed padding logic ---
        
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device):
        """Creates an initial empty carry state."""
        # Use config.seq_len which holds MAX_SEQ_LEN_MODEL for shape
        state_shape = (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT(
            z_H=torch.empty(state_shape, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(state_shape, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT):
        """Resets the carry state for halted sequences."""
        reset_view = reset_flag.view(-1, 1, 1)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT(
            z_H=torch.where(reset_view, self.H_init, carry.z_H),
            z_L=torch.where(reset_view, self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Performs one step of the inner reasoning loop with EBT1 guidance."""
        
        # --- MODIFICATION: Get actual sequence length and slice RoPE ---
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            target_seq_len=None # Explicitly tell it NOT to pad
        )
        current_total_seq_len = input_embeddings.shape[1]

        cos_sin_full = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        cos_sin_sliced = None
        if cos_sin_full is not None:
            cos, sin = cos_sin_full
            cos_sin_sliced = (cos[:current_total_seq_len], sin[:current_total_seq_len])
        
        seq_info = dict(cos_sin=cos_sin_sliced)
        # --- END MODIFICATION ---

        # --- MODIFICATION: Slice carry states ---
        z_H, z_L = carry.z_H[:, :current_total_seq_len, :], carry.z_L[:, :current_total_seq_len, :]
        # --- END MODIFICATION ---

        # --- EBT Integration within TRM Cycles ---
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles - 1):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

        beta = self.config.ebt_beta

        # --- Refined EBT Integration Loop ---
        for _L_step in range(self.config.L_cycles):
            z_L_curr = z_L.detach().requires_grad_(True)
            trm_output = self.L_level(z_L_curr, z_H + input_embeddings, **seq_info)

            # EBT Guidance Calculation (First Order)
            with torch.enable_grad():
                energy_scalar = self.energy_head(z_L_curr.float()).sum()
                energy_grad = torch.autograd.grad(energy_scalar, z_L_curr, create_graph=True)[0]
            
            energy_grad = energy_grad.to(trm_output.dtype)

            # Combine Updates
            z_L = trm_output - (beta * energy_grad)

        # Final H update
        z_H = self.L_level(z_H, z_L, **seq_info)


        # --- MODIFICATION: Pad carry states back to max length ---
        pad_needed = (self.config.seq_len + self.puzzle_emb_len) - current_total_seq_len
        if pad_needed > 0:
            z_H_padded = F.pad(z_H, (0, 0, 0, pad_needed), "constant", 0)
            z_L_padded = F.pad(z_L, (0, 0, 0, pad_needed), "constant", 0)
        else:
            z_H_padded = z_H
            z_L_padded = z_L
            
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry_DyT(z_H=z_H_padded.detach(), z_L=z_L_padded.detach())
        # --- END MODIFICATION ---

        output_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output_logits, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1_DyT_EBT1(nn.Module):
    """ACT wrapper managing the outer loop (steps, halting) for the EBT-enhanced TRM."""
    def __init__(self, config_dict: dict):
        super().__init__()
        # Load configuration and initialize the inner model
        self.config = TinyRecursiveReasoningModel_ACTV1Config_DyT(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner_DyT_EBT1(self.config)

    @property
    def puzzle_emb(self):
        """Allows access to the puzzle embedding module if it exists."""
        if hasattr(self.inner, 'puzzle_emb'):
             return self.inner.puzzle_emb
        return None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Creates the initial carry state for the ACT loop."""
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device # Ensure states are on the correct device
        return TinyRecursiveReasoningModel_ACTV1Carry_DyT(
            inner_carry=self.inner.empty_carry(batch_size, device=device), # Initialize inner model states
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device), # Step counter starts at 0
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device), # Start all sequences in halted state
            current_data={k: torch.empty_like(v) for k, v in batch.items()} # Placeholder for current batch data
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry_DyT, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry_DyT, Dict[str, torch.Tensor]]:
        """Performs one step of the ACT outer loop."""
        # Reset inner carry and update current data for sequences that halted previously
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps) # Reset steps counter for halted sequences
        # Update data for sequences starting a new computation
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # === Execute one step of the inner model ===
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        # Store outputs for loss calculation and potential return
        outputs = {
            "logits": logits, # Shape: [B, ActualSeqLen, Vocab]
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # === ACT Halting Logic (executed without gradient tracking) ===
        with torch.no_grad():
            new_steps = new_steps + 1 # Increment step counter
            is_last_step = new_steps >= self.config.halt_max_steps # Check if max steps reached
            halted = is_last_step # Always halt at max steps

            # Apply adaptive halting logic only during training (if enabled)
            if self.training and (self.config.halt_max_steps > 1):
                # Determine halt signal based on Q-values
                if self.config.no_ACT_continue:
                    # Simplified: Halt if Q(halt) is positive (sigmoid > 0.5)
                    halt_signal = (q_halt_logits > 0)
                else:
                    # Standard: Halt if Q(halt) > Q(continue)
                    halt_signal = (q_halt_logits > q_continue_logits)
                halted = halted | halt_signal # Halt if max steps reached OR halt signal is true

                # Apply exploration policy (force continuation sometimes)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q-value for Q-learning loss (if not using simplified ACT loss)
                if not self.config.no_ACT_continue:
                    # --- MODIFICATION: Pass current_data to inner for no_grad call ---
                    with torch.no_grad():
                        _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    # --- END MODIFICATION ---
                    target_q = torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    outputs["target_q_continue"] = torch.sigmoid(target_q) # Apply sigmoid for BCE loss

        # Return the updated carry state and the computed outputs
        return TinyRecursiveReasoningModel_ACTV1Carry_DyT(new_inner_carry, new_steps, halted, new_current_data), outputs
