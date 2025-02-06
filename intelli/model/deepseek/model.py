import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .kernel import act_quant, weight_dequant, fp8_gemm


@dataclass
class ModelArgs:
    """Model configuration arguments."""
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.qk_rope_head_dim
        base = args.rope_theta
        
        # Compute position embeddings but don't register as buffers
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Apply scaling for extended context
        if args.max_seq_len > args.original_seq_len:
            scale = math.log(args.rope_factor) / 2.0
            self.inv_freq = self.inv_freq * args.rope_factor ** (scale / dim)
        
        # Store dimensions for use in forward pass
        self.dim = dim

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # Move inv_freq to correct device
        self.inv_freq = self.inv_freq.to(x.device)
        
        # Get sequence length and compute position embeddings
        seq_len = x.shape[1]
        t = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        
        # Compute cos and sin
        cos = torch.cos(freqs)  # [seq_len, dim/2]
        sin = torch.sin(freqs)  # [seq_len, dim/2]
        
        # Reshape x to match expected dimensions
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, 2)  # [..., dim/2, 2]
        
        # Reshape cos and sin for broadcasting
        cos = cos.view(1, seq_len, 1, cos.shape[-1])  # [1, seq_len, 1, dim/2]
        sin = sin.view(1, seq_len, 1, sin.shape[-1])  # [1, seq_len, 1, dim/2]
        
        # Ensure cos and sin match x's dimension
        cos = cos.expand(x_shape[0], -1, x_shape[2], -1)  # [batch, seq_len, heads, dim/2]
        sin = sin.expand(x_shape[0], -1, x_shape[2], -1)  # [batch, seq_len, heads, dim/2]
        
        # Split input into half for rotation
        x1, x2 = x.unbind(-1)  # [..., dim/2], [..., dim/2]
        
        # Apply rotation using the RoPE formulation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)
        
        return rotated


class Attention(nn.Module):
    """Multi-head attention with support for rotary embeddings."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Query uses full dimension
        self.q_proj = nn.Linear(args.dim, args.dim, bias=True)
        
        # Key and Value use reduced dimension (256 total, not per head)
        self.k_proj = nn.Linear(args.dim, 256, bias=True)
        self.v_proj = nn.Linear(args.dim, 256, bias=True)
        self.o_proj = nn.Linear(args.dim, args.dim, bias=False)
        
        self.rope = RotaryEmbedding(args)
        self.scale = self.head_dim ** -0.5
        
        # Store rope dimensions
        self.rope_dim = args.qk_rope_head_dim
        
        # Apply extended context scaling if needed
        if args.max_seq_len > args.original_seq_len:
            self.scale *= args.mscale

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.n_heads
        
        # Linear projections
        q = self.q_proj(x).view(B, T, H, -1)  # [B, T, H, head_dim]
        k = self.k_proj(x).view(B, T, 1, 256).expand(B, T, H, 256)  # [B, T, H, 256]
        v = self.v_proj(x).view(B, T, 1, 256).expand(B, T, H, 256)  # [B, T, H, 256]
        
        # Apply rotary embeddings only to the query projection
        # Reshape query to match RoPE dimensions
        q_rope_dim = min(q.shape[-1], self.rope_dim * 2)  # Ensure we don't exceed tensor dimensions
        q_rope = q[..., :q_rope_dim]  # [B, T, H, rope_dim*2]
        q_rope = self.rope(q_rope, start_pos)  # Apply RoPE
        
        # Concatenate with remaining dimensions if any
        if q.shape[-1] > q_rope_dim:
            q = torch.cat([q_rope, q[..., q_rope_dim:]], dim=-1)
        else:
            q = q_rope
        
        # Compute attention
        attn = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        
        # Compute output
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.dim, args.inter_dim, bias=False)
        self.up_proj = nn.Linear(args.dim, args.inter_dim, bias=False)
        self.down_proj = nn.Linear(args.inter_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), start_pos, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    """DeepSeek transformer model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        B, T = tokens.size()
        
        # Get embeddings
        h = self.embed_tokens(tokens)
        
        # Create attention mask
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, mask)
        
        # Output projection
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits 

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    theta = args.rope_theta
    
    # Compute frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    
    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (..., head_dim).
        freqs_cis (torch.Tensor): Precomputed complex exponential values.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    # Ensure x is contiguous and reshape for complex view
    x = x.contiguous()
    
    # Reshape input to complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Expand freqs_cis for broadcasting
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, x_complex.shape[-1])
    
    # Apply rotation
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real and restore original dtype
    x_out = torch.view_as_real(x_rotated).flatten(start_dim=-2)
    return x_out.type_as(x)


class MLA(nn.Module):
    """Multi-Head Linear Attention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x 