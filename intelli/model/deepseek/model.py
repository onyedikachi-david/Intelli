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
        
        # Compute position embeddings
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Apply scaling for extended context
        if args.max_seq_len > args.original_seq_len:
            scale = math.log(args.rope_factor) / 2.0
            self.inv_freq = self.inv_freq * args.rope_factor ** (scale / dim)
        
        # Store dimensions for use in forward pass
        self.dim = dim
        self.max_seq_len = args.max_seq_len

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, heads, head_dim]
            start_pos: Starting position for computing position embeddings
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Move inv_freq to correct device
        self.inv_freq = self.inv_freq.to(x.device)
        
        # Get sequence length and compute position embeddings
        seq_len = x.shape[1]
        t = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        
        # Compute cos and sin
        cos = torch.cos(freqs)  # [seq_len, dim/2]
        sin = torch.sin(freqs)  # [seq_len, dim/2]
        
        # Reshape cos and sin for broadcasting
        cos = cos.view(1, seq_len, 1, cos.shape[-1])  # [1, seq_len, 1, dim/2]
        sin = sin.view(1, seq_len, 1, sin.shape[-1])  # [1, seq_len, 1, dim/2]
        
        # Ensure input tensor has correct shape by splitting last dimension
        x_shape = x.shape
        x_reshaped = x.view(x_shape[0], x_shape[1], x_shape[2], -1, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        
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
        # Ensure q_rope has correct dimensions for RoPE
        q_rope_dim = self.rope_dim * 2  # Multiply by 2 since we need pairs for rotation
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