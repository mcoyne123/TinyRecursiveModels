from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

class ETRMConfig:
    def __init__(self, **kwargs):
        self.batch_size = 32
        self.seq_len = 20
        self.puzzle_emb_ndim = 0
        self.num_puzzle_identifiers = 1
        self.vocab_size = 100
        self.L_layers = 2
        self.hidden_size = 128
        self.expansion = 2
        self.num_heads = 4
        self.pos_encodings = "rope"
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.max_supervision_steps = 8
        self.forward_dtype = "bfloat16"
        self.puzzle_emb_len = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

class ETRMNetwork(nn.Module):
    """The core Transformer network for the Simplified E-TRM."""
    def __init__(self, d_model, n_layers, n_head, d_ffn):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, d_ffn, batch_first=True)
            for _ in range(n_layers)
        ])
        self.update_head = nn.Linear(d_model, d_model)
        self.z_head = nn.Linear(d_model, d_model)

    def forward(self, x_embed, y_embed, z_embed):
        combined_input = torch.cat([x_embed, y_embed, z_embed], dim=1).float()

        output = combined_input
        for layer in self.layers:
            output = layer(output)

        _, y_processed, z_processed = torch.split(output, [x_embed.size(1), y_embed.size(1), z_embed.size(1)], dim=1)

        update_vector = self.update_head(y_processed)
        z_new = self.z_head(z_processed)

        return update_vector, z_new


class ETRM_Inner(nn.Module):
    def __init__(self, config: 'ETRMConfig') -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.update_head = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.z_head = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.puzzle_emb_len = self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.network = ETRMNetwork(
            d_model=self.config.hidden_size,
            n_layers=self.config.L_layers,
            n_head=self.config.num_heads,
            d_ffn=int(self.config.hidden_size * self.config.expansion),
        )

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def forward(self, x_embed: torch.Tensor, y_embed: torch.Tensor, z_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.network(x_embed, y_embed, z_embed)


class ETRM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ETRMConfig(**config_dict)
        self.inner = ETRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def forward(self, carry, batch: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, torch.Tensor]]:
        x_tokens = batch["inputs"]

        x_embed = self.inner._input_embeddings(x_tokens, batch["puzzle_identifiers"])

        y_embed = torch.randn_like(x_embed)
        z_embed = torch.randn_like(x_embed)

        for _ in range(self.config.max_supervision_steps):
            update_vector, z_new = self.inner(x_embed, y_embed, z_embed)

            y_embed = y_embed + update_vector

            y_embed = y_embed.detach()
            z_embed = z_new.detach()

        logits = self.inner.lm_head(y_embed)[:, self.inner.puzzle_emb_len:]
        return None, {"logits": logits}

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        return None

    def predict(self, batch: Dict[str, torch.Tensor]):
        """Generates a final prediction using the iterative refinement process."""
        self.eval()
        with torch.no_grad():
            x_tokens = batch["inputs"]
            x_embed = self.inner._input_embeddings(x_tokens, batch["puzzle_identifiers"])
            y_embed = torch.randn_like(x_embed)
            z_embed = torch.randn_like(x_embed)

            for _ in range(self.config.max_supervision_steps):
                update_vector, z_new = self.inner(x_embed, y_embed, z_embed)
                y_embed = y_embed + update_vector

                y_embed = y_embed.detach()
                z_embed = z_new.detach()

            logits = self.inner.lm_head(y_embed)[:, self.inner.puzzle_emb_len:]
        return logits
