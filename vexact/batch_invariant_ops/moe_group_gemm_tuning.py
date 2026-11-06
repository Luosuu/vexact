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

Above ``_OGS_MIN_ROWS`` routed rows (prefill and mixed steps) the GEMM instead runs triton_kernels'
ragged ``matmul_ogs`` (persistent, TMA, warp-specialized), which reaches ~400-530 TFLOP/s where
VeOmni's kernel stays near 200. It is used with no gather/scatter/activation fusion and with split-K
pinned to 1 (its default split-K depends on the batch and would change the reduction tree), so it too
accumulates K sequentially from zero and returns the same bits (same test).
"""

import torch
import triton


# Routed rows (tokens x top-k) above which matmul_ogs is used. Below it both kernels stream expert
# weights at the same rate; above it matmul_ogs is 1.2-2.6x faster on H100.
_OGS_MIN_ROWS = 2048


def _load_matmul_ogs():
    """Return a callable ``(a, w_ekn, splits, topk=1, gather=None) -> out`` using triton_kernels' ragged
    matmul_ogs. With ``gather=(src, dst)`` the rows of ``a`` are tokens, gathered into expert order by the
    GEMM's loads."""
    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, matmul_ogs
    from triton_kernels.matmul_ogs_details.opt_flags import update_opt_flags_constraints
    from triton_kernels.tensor import make_ragged_tensor_metadata

    # Batch-dependent split-K would change the reduction tree; keep one CTA per output tile.
    update_opt_flags_constraints({"split_k": 1})

    def run(a, w_ekn, splits, topk=1, gather=None):
        rows = a.shape[0] if gather is None else gather[0].shape[0]
        routing = RoutingData(None, splits, splits.shape[0], topk, make_ragged_tensor_metadata(splits, rows))
        gather_indx = None if gather is None else GatherIndx(src_indx=gather[0], dst_indx=gather[1])
        return matmul_ogs(a, w_ekn, None, routing, gather_indx=gather_indx)

    return run


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
    ragged_matmul = _load_matmul_ogs()

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
        if a.shape[0] == max_M and max_M > _OGS_MIN_ROWS:
            # Per-expert row counts from VeOmni's inclusive cumsum (device-side, CUDA-graph safe).
            splits = torch.diff(cumsum_M, prepend=cumsum_M.new_zeros(1)).to(torch.int32)
            return ragged_matmul(a, b.transpose(1, 2), splits)  # [E, K, N] column-major view
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
