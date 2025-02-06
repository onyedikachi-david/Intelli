import torch
import math
import warnings
from typing import Tuple

# Only import triton if CUDA is available
USE_TRITON = False
if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        USE_TRITON = True
    except:
        warnings.warn("Triton import failed, falling back to CPU implementation")

# CPU implementations
def act_quant_cpu(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU implementation of activation quantization"""
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    x_reshaped = x.view(-1, block_size)
    s = torch.max(torch.abs(x_reshaped), dim=1)[0] / 448.
    y = (x_reshaped / s.unsqueeze(1))
    if hasattr(torch, 'float8_e4m3fn'):
        y = y.to(torch.float8_e4m3fn)
    else:
        y = y.to(torch.float16)  # Fallback to float16 if float8 not available
    return y.view_as(x), s.view(*x.size()[:-1], -1)

def weight_dequant_cpu(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """CPU implementation of weight dequantization"""
    M, N = x.size()
    n_blocks = (N + block_size - 1) // block_size
    s_expanded = s.unsqueeze(-1).expand(-1, -1, block_size)
    s_expanded = s_expanded[:, :n_blocks].reshape(M, -1)[:, :N]
    return x.float() * s_expanded

def fp8_gemm_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CPU implementation of matrix multiplication"""
    return torch.matmul(a, b)

# Main interface functions that choose between CPU and GPU implementations
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes activations using block-wise scaling"""
    return act_quant_cpu(x, block_size)

def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantizes weights using scaling factors"""
    return weight_dequant_cpu(x, s, block_size)

def fp8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with optional GPU acceleration"""
    return fp8_gemm_cpu(a, b)

# Only define Triton kernels if CUDA is available
if USE_TRITON:
    # Optimized GEMM configurations
    fp8_gemm_configs = [
        triton.Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, 
               num_stages=num_stages, num_warps=8)
        for block_m in [16, 32, 64] 
        for block_n in [32, 64, 128] 
        for num_stages in [3, 4, 5, 6]
    ]

    @triton.jit
    def act_quant_kernel(x_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = tl.round(x / scale) * scale
        tl.store(x_ptr + offsets, output, mask=mask)

    @triton.jit
    def weight_dequant_kernel(x_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = x * scale
        tl.store(x_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def fp8_gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask) 