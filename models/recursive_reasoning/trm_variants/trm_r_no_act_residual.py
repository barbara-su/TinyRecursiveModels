from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


import os
import logging
import tqdm
import torch
import torch.distributed as dist


IGNORE_LABEL_ID = -100

# small util for printing
def log0(msg: str):
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        tqdm.tqdm.write(msg)
        
def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


@dataclass
class TinyRecursiveReasoningModelInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModelCarry:
    inner_carry: TinyRecursiveReasoningModelInnerCarry


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            # TODO: why?
            if self.config.puzzle_emb_ndim == 0:
                self.puzzle_emb_len = 0
            else:
                computed = ceil_div(self.config.puzzle_emb_ndim, self.config.hidden_size)
                self.puzzle_emb_len = computed if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
                        
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # TODO: why?
        if self.config.puzzle_emb_ndim == 0:
            self.puzzle_emb_len = 0
        else:
            computed = ceil_div(self.config.puzzle_emb_ndim, self.config.hidden_size)
            self.puzzle_emb_len = computed if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div

        
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
        else:
            self.puzzle_emb = None
            
        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        # TODO: why?
        device = self.H_init.device
        L_total = self.config.seq_len + self.puzzle_emb_len
        return TinyRecursiveReasoningModelInnerCarry(
            z_H=torch.empty(batch_size, L_total, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, L_total, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModelInnerCarry):
        return TinyRecursiveReasoningModelInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )
    
    def forward(
        self,
        carry: TinyRecursiveReasoningModelInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModelInnerCarry, torch.Tensor, Dict[str, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        
        # Hyperparameters
        eps = 1e-6
        tau = 1e-3 # residual threshold
        patience_req = 2 # require residual < tau for this many consecutive cycles
        max_no_grad_H = 3 # avoid infinite recursions
        
        # H_cycles-1 without grad
        with torch.no_grad():
            B = z_H.shape[0] # obtain batch size
            device = z_H.device

            active = torch.ones((B,), dtype=torch.bool, device=device)
            patience = torch.zeros((B,), dtype=torch.int32, device=device)
            iters_used = torch.zeros((B,), dtype=torch.int32, device=device)
            last_r = torch.full((B,), float("nan"), dtype=torch.float32, device=device)
            r_at_stop = torch.full((B,), float("nan"), dtype=torch.float32, device=device)

            for h_iter in range(max_no_grad_H):
                if not active.any():
                    break

                active_prev = active
                iters_used += active_prev.to(torch.int32)

                z_H_prev = z_H
                z_L_prev = z_L

                # Candidate update for everyone
                z_L_new = z_L
                for _L_step in range(self.config.L_cycles):
                    z_L_new = self.L_level(z_L_new, z_H + input_embeddings, **seq_info)
                z_H_new = self.L_level(z_H, z_L_new, **seq_info)

                # Residual in fp32
                dH = (z_H_new.float() - z_H_prev.float())
                dL = (z_L_new.float() - z_L_prev.float())

                rms_dH = torch.sqrt(torch.mean(dH * dH, dim=(1, 2)) + eps)
                rms_H  = torch.sqrt(torch.mean(z_H_prev.float() * z_H_prev.float(), dim=(1, 2)) + eps)
                r_H = rms_dH / (rms_H + eps)

                rms_dL = torch.sqrt(torch.mean(dL * dL, dim=(1, 2)) + eps)
                rms_L  = torch.sqrt(torch.mean(z_L_prev.float() * z_L_prev.float(), dim=(1, 2)) + eps)
                r_L = rms_dL / (rms_L + eps)

                r = torch.maximum(r_H, r_L)  # (B,)
                last_r = r

                # Apply updates only to active samples
                m = active_prev.view(B, 1, 1)
                z_H = torch.where(m, z_H_new, z_H)
                z_L = torch.where(m, z_L_new, z_L)

                # Consecutive patience update, reset to 0 on failure
                good = (r < tau)
                
                patience = torch.where(
                    active_prev & good,
                    patience + 1,
                    torch.where(active_prev, torch.zeros_like(patience), patience),
                )

                newly_done = active_prev & (patience >= patience_req)
                r_at_stop = torch.where(newly_done & torch.isnan(r_at_stop), r, r_at_stop)

                active = active_prev & ~newly_done

            converged = (patience >= patience_req)
            r_at_stop = torch.where(torch.isnan(r_at_stop), last_r, r_at_stop) 
        
        # debug message
        if converged.any():
            log0(f"stop early: {int(converged.sum().item())}/{converged.numel()} converged, "
                f"mean iters {iters_used.float().mean().item():.2f}/{max_no_grad_H}")
                            
        # Save no-grad fixed-point candidate
        z_H_fp = z_H
        z_L_fp = z_L

        # Detach the candidate so gradients flow only through the mapping F_theta
        z_H0 = z_H_fp.detach()
        z_L0 = z_L_fp.detach()

        # One probe application of the update map WITH grad
        z_L_probe = z_L0
        for _L_step in range(self.config.L_cycles):
            z_L_probe = self.L_level(z_L_probe, z_H0 + input_embeddings, **seq_info)
        z_H_probe = self.L_level(z_H0, z_L_probe, **seq_info)

        # Differentiable fixed-point residual (fp32 for stability)
        dH = (z_H_probe.float() - z_H0.float())
        dL = (z_L_probe.float() - z_L0.float())

        rms_dH = torch.sqrt(torch.mean(dH * dH, dim=(1, 2)) + eps)
        rms_H  = torch.sqrt(torch.mean(z_H0.float() * z_H0.float(), dim=(1, 2)) + eps)
        r_H = rms_dH / (rms_H + eps)

        rms_dL = torch.sqrt(torch.mean(dL * dL, dim=(1, 2)) + eps)
        rms_L  = torch.sqrt(torch.mean(z_L0.float() * z_L0.float(), dim=(1, 2)) + eps)
        r_L = rms_dL / (rms_L + eps)

        fp_residual_probe = torch.maximum(r_H, r_L)  # (B,)

        # Use probe state for logits (this is your "one-step grad" style)
        output = self.lm_head(z_H_probe)[:, self.puzzle_emb_len:]

        # New carry stays detached as before
        new_inner_carry = TinyRecursiveReasoningModelInnerCarry(
            z_H=z_H_probe.detach(),
            z_L=z_L_probe.detach(),
        )

        # Package stats here (this is the ONLY place where these tensors exist)
        stats: Dict[str, torch.Tensor] = {
            # keep WITH grad, used by fp_residual_weight
            "fp_residual_probe": fp_residual_probe,
            
            # no_grad loop diagnostics (logging only)
            "iters_used": iters_used.to(torch.float32),
            "r_at_stop": r_at_stop.to(torch.float32),
            "converged": converged.to(torch.int32),
        }

        return new_inner_carry, output, stats


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModelCarry(
            inner_carry=self.inner.empty_carry(batch_size)
        )
        

    def forward(
        self,
        carry: TinyRecursiveReasoningModelCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModelCarry, Dict[str, torch.Tensor]]:
        
        B = batch["inputs"].shape[0]
        device = batch["inputs"].device

        # Always start fresh each call (no cross-call ACT)
        reset_all = torch.ones((B,), dtype=torch.bool, device=device)
        inner_carry = self.inner.reset_carry(reset_all, carry.inner_carry)
        
        # Single inner call
        new_inner_carry, logits, stats = self.inner(inner_carry, batch)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            **stats,
        }

        return TinyRecursiveReasoningModelCarry(inner_carry=new_inner_carry), outputs