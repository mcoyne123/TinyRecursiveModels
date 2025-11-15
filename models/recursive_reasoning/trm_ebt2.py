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
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100
DEFAULT_EPSILON = 1e-8 # Epsilon for RMSProp

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry_EBT2:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config_EBT2(BaseModel):
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
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Adaptive Computation (ACT)
    halt_max_steps: int # Maximum number of ACT steps
    halt_exploration_prob: float # Probability for random exploration steps in ACT
    no_ACT_continue: bool = True # Simplified ACT loss (only halt signal)

    # Data type
    forward_dtype: str = "bfloat16"

    # Energy-Based Transformer (EBT) specific config
    ebt_beta: float = 0.1 # Strength factor for the EBT gradient guidance
    ebt_hessian_eps: float = DEFAULT_EPSILON # Epsilon for stabilizing RMSProp division
    ebt_beta2: float = 0.99 # Beta2 for RMSProp running average

    # Other flags (e.g., for ablations)
    mlp_t: bool = False # Use MLP instead of Transformer block (ablation)
    H_layers: int = 1 # Ignored in TRM, kept for config compatibility if needed

    # Ensure Pydantic handles extra fields if any come from config files
    class Config:
        extra = 'allow'


class TinyRecursiveReasoningModel_ACTV1Block_EBT2(nn.Module):
    """A single Transformer block using RMSNorm normalization."""
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config_EBT2) -> None:
        super().__init__()
        self.config = config

        if self.config.mlp_t: # Ablation: Use MLP-based block instead of attention
            # Calculate puzzle embedding length contribution if applicable
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            mlp_t_hidden_size = self.config.seq_len + self.puzzle_emb_len # Use config.seq_len which is MAX length
            self.mlp_t = SwiGLU(
                hidden_size=mlp_t_hidden_size,
                expansion=config.expansion,
            )
            # Use RMSNorm for the MLP-transpose block
            self.norm_mlp_t = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)
        else: # Standard Transformer block with Attention
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads, # No Grouped Query Attention
                causal=False # Not causal for this architecture
            )
            # Use RMSNorm before the MLP layer (after attention residual)
            self.norm_attn = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)

        # SwiGLU Feedforward Network
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        # Use RMSNorm after the MLP residual connection
        self.norm_mlp = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)
        self.norm_eps = config.rms_norm_eps # Keep eps accessible if needed elsewhere

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer block using RMSNorm."""
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

class TinyRecursiveReasoningModel_ACTV1ReasoningModule_EBT2(nn.Module):
    """Module representing one pass through the L_layers using RMSNorm blocks."""
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block_EBT2]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        """Applies input injection and passes through the layers."""
        hidden_states = hidden_states + input_injection # Additive input injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner_EBT2(nn.Module):
    """The core inner loop model combining Baseline TRM (RMSNorm) with 2nd Order EBT guidance (RMSProp Preconditioner)."""
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config_EBT2) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Input/Output Embeddings and Heads
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False) # Output projection
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True) # ACT Q-value head

        # EBT Energy Head
        self.energy_head = CastedLinear(self.config.hidden_size, 1, bias=False)
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

        # Core Reasoning Module (using RMSNorm blocks)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule_EBT2(
            layers=[TinyRecursiveReasoningModel_ACTV1Block_EBT2(self.config)
                    for _i in range(self.config.L_layers)]
        )

        # Initial hidden states (learned buffers)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # --- RMSProp State Buffer ---
        self.register_buffer('exp_avg_sq', None, persistent=False)
        self.beta2 = self.config.ebt_beta2
        # --- End RMSProp State ---

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
        # Initialize exp_avg_sq buffer here
        self.exp_avg_sq = torch.zeros(state_shape, dtype=self.forward_dtype, device=device)

        return TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2(
            z_H=torch.empty(state_shape, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(state_shape, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2):
        """Resets the carry state for halted sequences."""
        reset_view = reset_flag.view(-1, 1, 1)
        if self.exp_avg_sq is None:
             raise RuntimeError("exp_avg_sq buffer not initialized.")
        if self.exp_avg_sq.device != reset_flag.device:
            self.exp_avg_sq = self.exp_avg_sq.to(reset_flag.device)
            
        # --- MODIFICATION: Ensure buffer shape matches max model shape ---
        state_shape = (carry.z_L.shape[0], self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        if self.exp_avg_sq.shape[1] != state_shape[1]:
             self.exp_avg_sq = torch.zeros(state_shape, dtype=carry.z_L.dtype, device=carry.z_L.device)
        # --- END MODIFICATION ---

        # Reset exp_avg_sq where halted
        self.exp_avg_sq = torch.where(reset_view, torch.zeros_like(self.exp_avg_sq), self.exp_avg_sq)

        return TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2(
            z_H=torch.where(reset_view, self.H_init, carry.z_H),
            z_L=torch.where(reset_view, self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Performs one step of the inner reasoning loop with EBT2 guidance."""
        
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

        # --- MODIFICATION: Slice carry states and RMSProp buffer ---
        z_H, z_L = carry.z_H[:, :current_total_seq_len, :], carry.z_L[:, :current_total_seq_len, :]
        
        if self.exp_avg_sq is None or self.exp_avg_sq.shape[1] != z_L.shape[1]:
             self.exp_avg_sq = torch.zeros_like(z_L)
        if self.exp_avg_sq.device != z_L.device or self.exp_avg_sq.dtype != z_L.dtype:
            self.exp_avg_sq = self.exp_avg_sq.to(device=z_L.device, dtype=z_L.dtype)
            
        exp_avg_sq_sliced = self.exp_avg_sq[:, :current_total_seq_len, :]
        # --- END MODIFICATION ---

        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles - 1):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

        beta = self.config.ebt_beta
        hessian_eps = self.config.ebt_hessian_eps

        # --- Refined EBT Integration Loop ---
        for _L_step in range(self.config.L_cycles):
            z_L_curr = z_L.detach().requires_grad_(True)
            trm_output = self.L_level(z_L_curr, z_H + input_embeddings, **seq_info)

            # EBT Guidance Calculation (Second Order - RMSProp Preconditioner)
            with torch.enable_grad():
                energy_scalar = self.energy_head(z_L_curr.float()).sum()
                energy_grad = torch.autograd.grad(energy_scalar, z_L_curr, create_graph=True)[0]

            # Update RMSProp buffer
            current_grad_sq = energy_grad.detach().square()
            exp_avg_sq_sliced.mul_(self.beta2).add_(current_grad_sq, alpha=1 - self.beta2)

            # Precondition the gradient: ∇ E / (sqrt(avg_sq) + ε)
            preconditioned_grad = energy_grad / (torch.sqrt(exp_avg_sq_sliced) + hessian_eps)
            preconditioned_grad = preconditioned_grad.to(trm_output.dtype)


            # Combine Updates: z_{L, next} = TRM_update(z_L_curr) - β * Preconditioned_∇ E(z_L_curr)
            z_L = trm_output - (beta * preconditioned_grad)

        # Final H update
        z_H = self.L_level(z_H, z_L, **seq_info)


        # --- MODIFICATION: Pad carry states and RMSProp buffer back ---
        pad_needed = (self.config.seq_len + self.puzzle_emb_len) - current_total_seq_len
        if pad_needed > 0:
            z_H_padded = F.pad(z_H, (0, 0, 0, pad_needed), "constant", 0)
            z_L_padded = F.pad(z_L, (0, 0, 0, pad_needed), "constant", 0)
            self.exp_avg_sq[:, :current_total_seq_len, :] = exp_avg_sq_sliced
        else:
            z_H_padded = z_H
            z_L_padded = z_L
            self.exp_avg_sq = exp_avg_sq_sliced
            
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2(z_H=z_H_padded.detach(), z_L=z_L_padded.detach())
        # --- END MODIFICATION ---
        
        output_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output_logits, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1_EBT2(nn.Module):
    """ACT wrapper for the baseline TRM with EBT2 guidance."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config_EBT2(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner_EBT2(self.config)

    @property
    def puzzle_emb(self):
        if hasattr(self.inner, 'puzzle_emb'):
             return self.inner.puzzle_emb
        return None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry_EBT2(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry_EBT2, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry_EBT2, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halt_signal = (q_halt_logits > 0)
                else:
                    halt_signal = (q_halt_logits > q_continue_logits)
                halted = halted | halt_signal

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # --- MODIFICATION: Pass current_data to inner for no_grad call ---
                    with torch.no_grad():
                        _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    # --- END MODIFICATION ---
                    target_q = torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    outputs["target_q_continue"] = torch.sigmoid(target_q)

        return TinyRecursiveReasoningModel_ACTV1Carry_EBT2(new_inner_carry, new_steps, halted, new_current_data), outputs
