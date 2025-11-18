"""
Enhanced Energy-Based TRM (TRM-EBT2-Enhanced)

This model implements advanced Energy-Based Model (EBM) techniques on top of the baseline TRM:

1. Spectral Normalization: Smooths the energy landscape by constraining Lipschitz constant
2. Hard Negative Mining: Generates near-miss negatives for sharper energy manifolds
3. Momentum + Nesterov Lookahead: Better inner-loop optimization with curvature awareness
4. Multi-Scale Energy Heads: Coarse + Fine energy guidance at different abstraction levels
5. Dynamic Halting: Adaptive stopping based on energy convergence (ΔE monitoring)
6. L2 Regularization: Information bottleneck on latent z to prevent degenerate solutions

Author: Enhanced version based on original TRM-EBT2
"""

from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from pydantic import BaseModel
import random

from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100
DEFAULT_EPSILON = 1e-8


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced:
    z_H: torch.Tensor
    z_L: torch.Tensor
    # NEW: Momentum buffers for energy gradient descent
    momentum_z_L: Optional[torch.Tensor] = None


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry_EBT2Enhanced:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    # NEW: Energy tracking for dynamic halting
    prev_energy: Optional[torch.Tensor] = None


class TinyRecursiveReasoningModel_ACTV1Config_EBT2Enhanced(BaseModel):
    # Core dimensions and sizes
    batch_size: int
    seq_len: int
    vocab_size: int
    hidden_size: int
    num_heads: int
    L_layers: int

    # Recursion cycles
    H_cycles: int
    L_cycles: int

    # Embeddings
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    puzzle_emb_len: int = 16

    # Transformer specifics
    expansion: float
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Adaptive Computation (ACT)
    halt_max_steps: int
    halt_exploration_prob: float
    no_ACT_continue: bool = True

    # Data type
    forward_dtype: str = "bfloat16"

    # Energy-Based Transformer (EBT) config - ENHANCED
    ebt_beta: float = 0.1  # Strength factor for energy gradient guidance
    ebt_hessian_eps: float = DEFAULT_EPSILON  # Epsilon for RMSProp
    ebt_beta2: float = 0.99  # Beta2 for RMSProp running average

    # NEW: Advanced EBM parameters
    ebt_momentum: float = 0.9  # Momentum coefficient for energy gradient descent
    ebt_use_nesterov: bool = True  # Use Nesterov lookahead
    ebt_hard_negative_prob: float = 0.2  # Probability of generating hard negatives
    ebt_hard_negative_noise: float = 0.1  # Noise scale for hard negative generation
    ebt_z_l2_penalty: float = 1e-4  # L2 regularization on z_L latent
    ebt_use_spectral_norm: bool = True  # Apply spectral normalization to reasoning layers
    ebt_multiscale_energy: bool = True  # Use multi-scale (coarse+fine) energy heads
    ebt_dynamic_halt_threshold: float = 1e-3  # ΔE threshold for dynamic halting
    ebt_use_dynamic_halt: bool = False  # Enable dynamic halting based on energy variance

    # Other flags
    mlp_t: bool = False
    H_layers: int = 1

    class Config:
        extra = 'allow'


class TinyRecursiveReasoningModel_ACTV1Block_EBT2Enhanced(nn.Module):
    """Transformer block with optional Spectral Normalization."""

    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config_EBT2Enhanced) -> None:
        super().__init__()
        self.config = config

        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            mlp_t_hidden_size = self.config.seq_len + self.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=mlp_t_hidden_size,
                expansion=config.expansion,
            )
            self.norm_mlp_t = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
            self.norm_attn = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        # ENHANCEMENT 1: Apply Spectral Normalization to MLP weights
        if self.config.ebt_use_spectral_norm:
            # Spectral norm constrains the Lipschitz constant, smoothing the energy landscape
            spectral_norm(self.mlp.down_proj.weight, name='weight', n_power_iterations=1)
            spectral_norm(self.mlp.gate_up_proj.weight, name='weight', n_power_iterations=1)

        self.norm_mlp = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        if self.config.mlp_t:
            hidden_states_t = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states_t)
            hidden_states = self.norm_mlp_t(residual.transpose(1,2) + out)
            hidden_states = hidden_states.transpose(1,2)
        else:
            attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = self.norm_attn(residual + attn_output)

        residual = hidden_states
        out = self.mlp(hidden_states)
        hidden_states = self.norm_mlp(residual + out)

        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule_EBT2Enhanced(nn.Module):
    """Reasoning module with optional spectral normalization."""

    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block_EBT2Enhanced]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class MultiScaleEnergyHead(nn.Module):
    """
    ENHANCEMENT 4: Multi-Scale Energy Head

    Operates at two levels of abstraction:
    - Coarse: Global energy from mean-pooled representation
    - Fine: Token-level energy summed across sequence

    This provides both global structure guidance (early reasoning)
    and local refinement (late reasoning).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Coarse energy head: looks at global state (mean-pooled)
        self.coarse_energy_head = CastedLinear(hidden_size, 1, bias=False)
        # Fine energy head: looks at each token
        self.fine_energy_head = CastedLinear(hidden_size, 1, bias=False)

        # Initialize near zero to avoid large initial gradients
        with torch.no_grad():
            trunc_normal_init_(self.coarse_energy_head.weight, std=1e-4)
            trunc_normal_init_(self.fine_energy_head.weight, std=1e-4)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, SeqLen, Hidden]
        Returns:
            total_energy: [B] - Combined energy
            coarse_energy: [B] - Global energy
            fine_energy: [B] - Token-level energy
        """
        # Coarse: Mean-pool across sequence, then predict energy
        z_global = z.mean(dim=1)  # [B, Hidden]
        coarse_energy = self.coarse_energy_head(z_global.float()).squeeze(-1)  # [B]

        # Fine: Per-token energy, summed across sequence
        fine_energy_per_token = self.fine_energy_head(z.float()).squeeze(-1)  # [B, SeqLen]
        fine_energy = fine_energy_per_token.sum(dim=1)  # [B]

        # Total energy is the sum of both scales
        total_energy = coarse_energy + fine_energy

        return total_energy, coarse_energy, fine_energy


class TinyRecursiveReasoningModel_ACTV1_Inner_EBT2Enhanced(nn.Module):
    """
    Enhanced TRM-EBT2 with advanced Energy-Based Model techniques.
    """

    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config_EBT2Enhanced) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Input/Output Embeddings and Heads
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # ENHANCEMENT 4: Multi-Scale Energy Head
        if self.config.ebt_multiscale_energy:
            self.energy_head = MultiScaleEnergyHead(self.config.hidden_size)
        else:
            # Fallback to single-scale
            self.energy_head = CastedLinear(self.config.hidden_size, 1, bias=False)
            with torch.no_grad():
                trunc_normal_init_(self.energy_head.weight, std=1e-4)

        # Puzzle embedding
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Positional Embeddings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Core Reasoning Module with Spectral Normalization
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule_EBT2Enhanced(
            layers=[TinyRecursiveReasoningModel_ACTV1Block_EBT2Enhanced(self.config)
                    for _i in range(self.config.L_layers)]
        )

        # Initial hidden states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # RMSProp State Buffer
        self.register_buffer('exp_avg_sq', None, persistent=False)
        self.beta2 = self.config.ebt_beta2

        # Initialize Q-head
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input_tokens: torch.Tensor, puzzle_identifiers: torch.Tensor, target_seq_len: Optional[int] = None):
        """Creates input embeddings without padding."""
        embedding = self.embed_tokens(input_tokens.to(torch.int32))
        current_seq_len_no_puzzle = embedding.shape[1]

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
            current_total_seq_len = embedding.shape[1]
        else:
            current_total_seq_len = current_seq_len_no_puzzle

        if self.config.pos_encodings == "learned":
            pos_ids = torch.arange(current_total_seq_len, device=embedding.device).unsqueeze(0)
            pos_embeddings = self.embed_pos(pos_ids)
            embedding = 0.707106781 * (embedding + pos_embeddings.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device):
        """Creates initial empty carry state with momentum buffers."""
        state_shape = (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)

        # Initialize RMSProp buffer
        self.exp_avg_sq = torch.zeros(state_shape, dtype=self.forward_dtype, device=device)

        return TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced(
            z_H=torch.empty(state_shape, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(state_shape, dtype=self.forward_dtype, device=device),
            # ENHANCEMENT 3: Initialize momentum buffer
            momentum_z_L=torch.zeros(state_shape, dtype=self.forward_dtype, device=device) if self.config.ebt_momentum > 0 else None,
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced):
        """Resets carry state for halted sequences, including momentum."""
        reset_view = reset_flag.view(-1, 1, 1)

        if self.exp_avg_sq is None:
            raise RuntimeError("exp_avg_sq buffer not initialized.")
        if self.exp_avg_sq.device != reset_flag.device:
            self.exp_avg_sq = self.exp_avg_sq.to(reset_flag.device)

        state_shape = (carry.z_L.shape[0], self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        if self.exp_avg_sq.shape[1] != state_shape[1]:
            self.exp_avg_sq = torch.zeros(state_shape, dtype=carry.z_L.dtype, device=carry.z_L.device)

        # Reset RMSProp buffer and momentum where halted
        self.exp_avg_sq = torch.where(reset_view, torch.zeros_like(self.exp_avg_sq), self.exp_avg_sq)

        new_momentum = None
        if carry.momentum_z_L is not None:
            new_momentum = torch.where(reset_view, torch.zeros_like(carry.momentum_z_L), carry.momentum_z_L)

        return TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced(
            z_H=torch.where(reset_view, self.H_init, carry.z_H),
            z_L=torch.where(reset_view, self.L_init, carry.z_L),
            momentum_z_L=new_momentum,
        )

    def _generate_hard_negatives(self, z_L: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        ENHANCEMENT 2: Hard Negative Mining

        Generates "near-miss" negatives by adding controlled noise to the current latent state.
        These negatives should have HIGH energy, sharpening the energy manifold around correct solutions.

        Args:
            z_L: Current latent state [B, SeqLen, Hidden]
            batch_size: Number of examples
        Returns:
            z_L_negative: Perturbed latent state
        """
        noise_scale = self.config.ebt_hard_negative_noise
        # Add Gaussian noise to create "almost correct but wrong" states
        noise = torch.randn_like(z_L) * noise_scale
        z_L_negative = z_L + noise
        return z_L_negative

    def _compute_energy(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute energy using multi-scale or single-scale head.

        Returns:
            energy_scalar: Total energy for backprop [1]
            energy_dict: Dictionary with detailed energy components
        """
        if self.config.ebt_multiscale_energy and isinstance(self.energy_head, MultiScaleEnergyHead):
            total_energy, coarse_energy, fine_energy = self.energy_head(z)
            energy_scalar = total_energy.sum()  # Sum over batch for scalar loss
            energy_dict = {
                "total_energy": total_energy.detach(),
                "coarse_energy": coarse_energy.detach(),
                "fine_energy": fine_energy.detach(),
            }
        else:
            energy = self.energy_head(z.float()).squeeze(-1)  # [B]
            energy_scalar = energy.sum()
            energy_dict = {"total_energy": energy.detach()}

        return energy_scalar, energy_dict

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs one step of the inner reasoning loop with ENHANCED EBT guidance.
        """

        # Get input embeddings and sequence info
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            target_seq_len=None
        )
        current_total_seq_len = input_embeddings.shape[1]

        cos_sin_full = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        cos_sin_sliced = None
        if cos_sin_full is not None:
            cos, sin = cos_sin_full
            cos_sin_sliced = (cos[:current_total_seq_len], sin[:current_total_seq_len])

        seq_info = dict(cos_sin=cos_sin_sliced)

        # Slice carry states and buffers
        z_H, z_L = carry.z_H[:, :current_total_seq_len, :], carry.z_L[:, :current_total_seq_len, :]

        if self.exp_avg_sq is None or self.exp_avg_sq.shape[1] != z_L.shape[1]:
            self.exp_avg_sq = torch.zeros_like(z_L)
        if self.exp_avg_sq.device != z_L.device or self.exp_avg_sq.dtype != z_L.dtype:
            self.exp_avg_sq = self.exp_avg_sq.to(device=z_L.device, dtype=z_L.dtype)

        exp_avg_sq_sliced = self.exp_avg_sq[:, :current_total_seq_len, :]

        # Slice momentum buffer if it exists
        momentum_z_L = None
        if carry.momentum_z_L is not None:
            momentum_z_L = carry.momentum_z_L[:, :current_total_seq_len, :]

        # No-grad H cycles (if H_cycles > 1)
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles - 1):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

        beta = self.config.ebt_beta
        hessian_eps = self.config.ebt_hessian_eps
        momentum_coeff = self.config.ebt_momentum
        use_nesterov = self.config.ebt_use_nesterov
        hard_negative_prob = self.config.ebt_hard_negative_prob
        z_l2_penalty = self.config.ebt_z_l2_penalty

        # Track energy for diagnostics
        energy_diagnostics = {}

        # --- ENHANCED EBT Integration Loop ---
        for _L_step in range(self.config.L_cycles):
            # ENHANCEMENT 3: Nesterov Lookahead
            # Instead of computing gradient at z_L_curr, compute it at the "lookahead" position
            if use_nesterov and momentum_z_L is not None:
                # Nesterov momentum: look ahead by current momentum
                z_L_lookahead = (z_L + momentum_coeff * momentum_z_L).detach().requires_grad_(True)
                trm_input = z_L_lookahead
            else:
                z_L_lookahead = z_L.detach().requires_grad_(True)
                trm_input = z_L_lookahead

            # TRM forward pass
            trm_output = self.L_level(trm_input, z_H + input_embeddings, **seq_info)

            # --- Energy Guidance Calculation ---
            with torch.enable_grad():
                # Compute energy at lookahead position
                energy_scalar, energy_dict = self._compute_energy(z_L_lookahead)

                # ENHANCEMENT 2: Hard Negative Mining (probabilistic)
                if self.training and random.random() < hard_negative_prob:
                    # Generate hard negatives and penalize them with high energy
                    z_L_negative = self._generate_hard_negatives(z_L_lookahead, batch_size=z_L.shape[0])
                    energy_negative_scalar, _ = self._compute_energy(z_L_negative)

                    # Contrastive loss: minimize energy of correct state, maximize energy of negatives
                    # We want: E(correct) < E(negative), so loss = E(correct) - E(negative)
                    # Or equivalently, maximize: E(negative) - E(correct)
                    # We'll add a margin-based loss
                    margin = 1.0
                    contrastive_term = F.relu(energy_scalar - energy_negative_scalar + margin)
                    total_energy_scalar = energy_scalar + 0.5 * contrastive_term
                else:
                    total_energy_scalar = energy_scalar

                # ENHANCEMENT 6: L2 Regularization on z_L (information bottleneck)
                if z_l2_penalty > 0:
                    l2_reg = (z_L_lookahead ** 2).sum()
                    total_energy_scalar = total_energy_scalar + z_l2_penalty * l2_reg

                # Compute gradient
                energy_grad = torch.autograd.grad(total_energy_scalar, z_L_lookahead, create_graph=True)[0]

            # Update RMSProp buffer (Hessian approximation)
            current_grad_sq = energy_grad.detach().square()
            exp_avg_sq_sliced.mul_(self.beta2).add_(current_grad_sq, alpha=1 - self.beta2)

            # Precondition gradient
            preconditioned_grad = energy_grad / (torch.sqrt(exp_avg_sq_sliced) + hessian_eps)
            preconditioned_grad = preconditioned_grad.to(trm_output.dtype)

            # ENHANCEMENT 3: Momentum Update
            if momentum_coeff > 0 and momentum_z_L is not None:
                # Update momentum: v_{t+1} = μ * v_t - α * ∇E
                momentum_z_L = momentum_coeff * momentum_z_L - beta * preconditioned_grad
                # Update z_L: z_{t+1} = TRM_output + v_{t+1}
                z_L = trm_output + momentum_z_L
            else:
                # Standard update without momentum
                z_L = trm_output - beta * preconditioned_grad

            # Store diagnostics from final L cycle
            if _L_step == self.config.L_cycles - 1:
                energy_diagnostics = energy_dict

        # Final H update
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Pad carry states and buffers back to max length
        pad_needed = (self.config.seq_len + self.puzzle_emb_len) - current_total_seq_len
        if pad_needed > 0:
            z_H_padded = F.pad(z_H, (0, 0, 0, pad_needed), "constant", 0)
            z_L_padded = F.pad(z_L, (0, 0, 0, pad_needed), "constant", 0)
            self.exp_avg_sq[:, :current_total_seq_len, :] = exp_avg_sq_sliced
            if momentum_z_L is not None:
                momentum_z_L_padded = F.pad(momentum_z_L, (0, 0, 0, pad_needed), "constant", 0)
            else:
                momentum_z_L_padded = None
        else:
            z_H_padded = z_H
            z_L_padded = z_L
            self.exp_avg_sq = exp_avg_sq_sliced
            momentum_z_L_padded = momentum_z_L

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry_EBT2Enhanced(
            z_H=z_H_padded.detach(),
            z_L=z_L_padded.detach(),
            momentum_z_L=momentum_z_L_padded.detach() if momentum_z_L_padded is not None else None,
        )

        output_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output_logits, (q_logits[..., 0], q_logits[..., 1]), energy_diagnostics


class TinyRecursiveReasoningModel_ACTV1_EBT2Enhanced(nn.Module):
    """
    ACT wrapper for enhanced TRM-EBT2 with dynamic halting and energy tracking.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config_EBT2Enhanced(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner_EBT2Enhanced(self.config)

    @property
    def puzzle_emb(self):
        if hasattr(self.inner, 'puzzle_emb'):
            return self.inner.puzzle_emb
        return None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry_EBT2Enhanced(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            # ENHANCEMENT 5: Track previous energy for dynamic halting
            prev_energy=None,
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry_EBT2Enhanced, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry_EBT2Enhanced, Dict[str, torch.Tensor]]:
        """Performs one step of ACT outer loop with optional dynamic halting."""

        # Reset inner carry
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Execute one step of inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), energy_diagnostics = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # Add energy diagnostics to outputs (for logging)
        for key, value in energy_diagnostics.items():
            outputs[f"energy/{key}"] = value

        # ACT Halting Logic
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            # ENHANCEMENT 5: Dynamic Halting based on Energy Variance
            current_energy = energy_diagnostics.get("total_energy")
            if (self.config.ebt_use_dynamic_halt and
                current_energy is not None and
                carry.prev_energy is not None and
                self.training):

                # Compute energy change (ΔE)
                delta_energy = torch.abs(current_energy - carry.prev_energy)
                # Halt if energy has plateaued (ΔE < threshold)
                energy_plateau = delta_energy < self.config.ebt_dynamic_halt_threshold
                halted = halted | energy_plateau

                outputs["energy/delta_energy"] = delta_energy

            # Store current energy for next step
            new_prev_energy = current_energy if current_energy is not None else carry.prev_energy

            # Standard ACT halting logic
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halt_signal = (q_halt_logits > 0)
                else:
                    halt_signal = (q_halt_logits > q_continue_logits)
                halted = halted | halt_signal

                # Exploration policy
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                # Q-learning target
                if not self.config.no_ACT_continue:
                    with torch.no_grad():
                        _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    target_q = torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    outputs["target_q_continue"] = torch.sigmoid(target_q)

        return TinyRecursiveReasoningModel_ACTV1Carry_EBT2Enhanced(
            new_inner_carry, new_steps, halted, new_current_data, new_prev_energy
        ), outputs
