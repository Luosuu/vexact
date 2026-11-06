# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch-config tuning for VeOmni's Triton grouped GEMM (the fused-MoE expert GEMMs).

VeOmni ships no pre-tuned configs, so every call uses the fallback tile (BLOCK_M=128, BLOCK_N=128,
BLOCK_K=32, 4 warps, 3 stages). On H100 that leaves the decode-time expert GEMMs 1.1-1.5x slower
than streaming the active experts' weights.

The wrapper below launches the *same* JIT kernel with a tile chosen from the number of routed rows.
Results are bitwise identical to the fallback: the kernel accumulates K sequentially from zero
for every tile shape, and on H100 neither the tile nor the emitted MMA instruction changes the bits
for a fixed K order (tests/batch_invariant_ops/test_moe_group_gemm_tuning.py). Training and
rollout therefore stay aligned whichever side applies it. Only the forward call pattern
(``transpose_b=True``, no activation, no accumulation, bf16, contiguous) is retuned; every other
call goes to VeOmni's original function unchanged.
"""

import torch
import triton


def _choose_config(max_m: int) -> tuple[int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) by routed rows (tokens x top-k)."""
    if max_m <= 128:
        return 64, 256, 64, 8, 4
    if max_m <= 512:
        return 64, 128, 128, 4, 4
    if max_m <= 4096:
        return 64, 128, 64, 8, 3
    return 128, 128, 32, 4, 4


def apply_moe_group_gemm_tuning() -> None:
    """Patch VeOmni's fused-MoE module to use tuned launches. Idempotent."""
    from veomni.ops.kernels.moe import group_gemm as fused_moe_module
    from veomni.ops.kernels.moe._kernels.kernel import group_gemm as kernel_module

    if getattr(fused_moe_module.group_gemm_same_nk, "_vexact_tuned", False):
        return
    original = kernel_module.group_gemm_same_nk
    jit_kernel = kernel_module.group_gemm_same_nk_kernel.fn  # below VeOmni's pretuned-config layer

    def group_gemm_same_nk(
        a,
        b,
        cumsum_M,
        max_M,
        transpose_a=False,
        transpose_b=False,
        activation=None,
        save_activation=False,
        c=None,
    ):
        if (
            transpose_a
            or not transpose_b
            or activation is not None
            or save_activation
            or c is not None
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
            or not (a.is_contiguous() and b.is_contiguous())
            or len(cumsum_M) != b.shape[0]
        ):
            return original(
                a,
                b,
                cumsum_M,
                max_M,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                activation=activation,
                save_activation=save_activation,
                c=c,
            )
        G, N, K = b.shape
        block_m, block_n, block_k, warps, stages = _choose_config(max_M)
        out = torch.empty((a.shape[0], N), dtype=a.dtype, device=a.device)
        with torch.cuda.device(a.device):
            jit_kernel[(triton.cdiv(max_M, block_m) * triton.cdiv(N, block_n), G)](
                a_ptr=a,
                b_ptr=b,
                c_ptr=out,
                act_ptr=None,
                cumsum_M=cumsum_M,
                max_M=max_M,
                total_M=a.shape[0],
                G=G,
                K=K,
                N=N,
                TRANSPOSE_A=False,
                TRANSPOSE_B=True,
                ACCUMULATE_TO_C=False,
                ACTIVATION=None,
                SAVE_ACTIVATION=False,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP=8,
                num_warps=warps,
                num_stages=stages,
            )
        return out

    group_gemm_same_nk._vexact_tuned = True
    fused_moe_module.group_gemm_same_nk = group_gemm_same_nk
