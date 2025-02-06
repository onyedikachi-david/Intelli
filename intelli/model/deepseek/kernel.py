import torch
from typing import Tuple
import platform
import math
import warnings

# Only import triton if CUDA is available
USE_TRITON = False
if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl
        USE_TRITON = True
    except:
        warnings.warn("Triton import failed, falling back to CPU implementation")


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors.
    
    Args:
        x_ptr: Pointer to quantized weights
        s_ptr: Pointer to scaling factors
        y_ptr: Pointer to output buffer
        M: Number of rows
        N: Number of columns
        BLOCK_SIZE: Block size for tiling
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.
    Falls back to PyTorch implementation on non-Linux platforms.
    
    Args:
        x: Quantized weight tensor of shape (M, N)
        s: Scale tensor
        block_size: Block size for dequantization (used only with Triton)
        
    Returns:
        Dequantized weight tensor
    """
    # PyTorch implementation
    M, N = x.size()
    n_blocks = (N + block_size - 1) // block_size
    s_expanded = s.unsqueeze(-1).expand(-1, -1, block_size)
    s_expanded = s_expanded[:, :n_blocks].reshape(M, -1)[:, :N]
    return x.float() * s_expanded


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes activations using block-wise scaling.
    
    Args:
        x_ptr: Pointer to input tensor
        y_ptr: Pointer to output tensor
        s_ptr: Pointer to scaling factors
        BLOCK_SIZE: Block size for quantization
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor using block-wise quantization.
    Falls back to PyTorch implementation on non-Linux platforms.
    
    Args:
        x: Input tensor to quantize
        block_size: Block size for quantization
        
    Returns:
        Tuple of (quantized tensor, scaling factors)
    """
    # PyTorch implementation
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


# CPU implementations
def act_quant_cpu(x, scale):
    return torch.round(x / scale) * scale

def weight_dequant_cpu(x, scale):
    return x * scale

def fp8_gemm_cpu(a, b):
    return torch.matmul(a, b)

# Main interface functions that choose between CPU and GPU implementations
def act_quant(x, scale):
    if USE_TRITON and x.is_cuda:
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (math.ceil(n_elements / BLOCK_SIZE),)
        act_quant_kernel[grid](x, scale, n_elements, BLOCK_SIZE)
        return x
    else:
        return act_quant_cpu(x, scale)

def weight_dequant(x, scale):
    if USE_TRITON and x.is_cuda:
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (math.ceil(n_elements / BLOCK_SIZE),)
        weight_dequant_kernel[grid](x, scale, n_elements, BLOCK_SIZE)
        return x
    else:
        return weight_dequant_cpu(x, scale)

def fp8_gemm(a, b):
    if USE_TRITON and a.is_cuda and b.is_cuda:
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        fp8_gemm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c
    else:
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