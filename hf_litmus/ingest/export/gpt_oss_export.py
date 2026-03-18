#!/usr/bin/env python3
# vim: set filetype=python ts=2 sw=2 et:
"""
GPT-OSS-specific patches for torch.export compatibility.

The GptOssExperts module in some transformers versions uses data-dependent
loops which torch.export cannot handle. This module provides a batched
alternative.
"""

import torch
from moe_ops import moe_router


def _gpt_oss_experts_forward_moe_router(
    self, hidden_states, router_indices, routing_weights
):
    """
    Patched forward for GptOssExperts using moe_router with token-selective routing.

    Uses token-selective MoE routing where each expert only processes the tokens
    that selected it, rather than computing on all tokens and gathering.

    Note: We keep gate_up_proj as a single fused weight tensor and split the
    *output* of the matmul rather than the weights. This avoids creating
    non-contiguous weight views that would require materialization in the
    ingest pipeline.

    Args:
        hidden_states: Input tensor (batch_size, seq_len, hidden_size)
        router_indices: Selected expert indices (num_tokens, top_k)
        routing_weights: Sparse router scores (num_tokens, num_experts) with nonzero
            values only at the top_k indices per token
    """
    if router_indices is None or routing_weights is None:
        raise TypeError(
            "GptOssExperts.forward requires router_indices and routing_weights "
            "tensors, got None"
        )

    # Flatten batch and seq dimensions
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(
        -1, orig_shape[-1]
    )  # [batch_size*seq_len, hidden_size]

    # Keep gate_up_proj as fused weights - don't split with stride 2!
    # The weights are interleaved [g0, u0, g1, u1, ...] but we'll split the
    # matmul output instead, which is much smaller than the weight tensors.
    gate_up_proj = (
        self.gate_up_proj
    )  # [num_experts, in_dim, 2*intermediate_dim]
    gate_up_proj_bias = (
        self.gate_up_proj_bias
    )  # [num_experts, 2*intermediate_dim]

    # router_indices: (num_tokens, top_k) - expert indices for each token
    # routing_weights: (num_tokens, num_experts) - sparse, nonzero only at top_k positions
    # Extract the top_k scores from the sparse routing_weights tensor
    topk_indices = router_indices  # (num_tokens, top_k)
    topk_scores = torch.gather(
        routing_weights, dim=1, index=topk_indices
    )  # (num_tokens, top_k)

    def expert_body(
        idx: torch.Tensor,
        token_indices: torch.Tensor,
        hidden: torch.Tensor,
        gate_up_w: torch.Tensor,
        gate_up_b: torch.Tensor,
        down_w: torch.Tensor,
        down_b: torch.Tensor,
    ) -> torch.Tensor:
        # Gather only the tokens that selected this expert
        hidden_selected = hidden[token_indices]  # (num_selected, hidden_dim)

        # Gather expert-specific weights
        idx_1d = idx.unsqueeze(0)  # Make 1D for index_select
        gate_up_w_i = torch.index_select(gate_up_w, 0, idx_1d).squeeze(0)
        gate_up_b_i = torch.index_select(gate_up_b, 0, idx_1d).squeeze(0)
        down_w_i = torch.index_select(down_w, 0, idx_1d).squeeze(0)
        down_b_i = torch.index_select(down_b, 0, idx_1d).squeeze(0)

        # Single fused matmul for gate+up projection
        # Note: gate_up_w_i shape is [hidden_dim, 2*intermediate_dim]
        gate_up_out = hidden_selected @ gate_up_w_i + gate_up_b_i

        # Deinterleave the output into gate and up projections
        gate_out = gate_up_out[..., ::2]
        up_out = gate_up_out[..., 1::2]

        # Clamp
        gate_out = gate_out.clamp(min=None, max=self.limit)
        up_out = up_out.clamp(min=-self.limit, max=self.limit)

        # GLU computation
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        intermediate = (up_out + 1) * glu
        expert_out = intermediate @ down_w_i + down_b_i

        return expert_out  # (num_selected, out_dim)

    # Token-selective routing with weighted blending inside moe_router
    # moe_router iterates experts in ascending order and accumulates score*output,
    # ensuring deterministic bfloat16 behavior matching the original HuggingFace impl.
    # Output shape: (num_tokens, out_dim) - already blended
    result = moe_router(
        topk_indices,
        topk_scores,
        expert_body,
        hidden_states,
        gate_up_proj,
        gate_up_proj_bias,
        self.down_proj,
        self.down_proj_bias,
    )

    # Restore shape to (batch_size, seq_len, hidden_size)
    return result.reshape(orig_shape)


from model_patch import PatchMethod, register_patch

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    patch = PatchMethod(
        description="GPT-OSS MoE",
        cls=GptOssExperts,
        method="forward",
        func=_gpt_oss_experts_forward_moe_router,
    )
    register_patch(patch)
except ImportError:
    pass  # GPT-OSS not available
