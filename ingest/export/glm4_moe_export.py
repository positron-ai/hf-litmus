#!/usr/bin/env python3
# vim: set filetype=python ts=2 sw=2 et:
"""
GLM-4 MoE specific patches for torch.export compatibility.

The Glm4MoeMoE module uses data-dependent torch.where() which torch.export
cannot handle. This module provides a batched alternative.
"""

import torch


def _glm4_moe_moe_batched(
  self,
  hidden_states: torch.Tensor,
  topk_indices: torch.Tensor,
  topk_weights: torch.Tensor,
) -> torch.Tensor:
  """
  Batched forward pass for Glm4MoeMoE.moe that is torch.export compatible.

  The original implementation uses data-dependent `torch.where(mask)` which
  returns variable-length output. This batched version:
  1. Runs all experts on all tokens (in parallel via stacking)
  2. Uses routing weights to mask/weight the outputs
  3. Sums across experts

  Trade-offs:
      - Pro: Works with torch.export, torch.compile(fullgraph=True)
      - Con: Uses more memory/compute (runs all experts, not just top-k)
  """
  num_tokens, hidden_size = hidden_states.shape
  num_experts = len(self.experts)

  # Run all experts on all tokens
  # Shape: (num_experts, num_tokens, hidden_size)
  all_expert_outputs = torch.stack(
    [expert(hidden_states) for expert in self.experts], dim=0
  )

  # Convert sparse top-k indices/weights to dense routing matrix
  # topk_indices: (num_tokens, top_k)
  # topk_weights: (num_tokens, top_k)
  # routing_weights: (num_tokens, num_experts)
  routing_weights = torch.zeros(
    (num_tokens, num_experts),
    dtype=topk_weights.dtype,
    device=topk_weights.device,
  )
  routing_weights.scatter_(1, topk_indices, topk_weights)

  # Apply routing weights and sum across experts
  # routing_weights: (num_tokens, num_experts) -> (num_experts, num_tokens, 1)
  routing_weights = routing_weights.t().unsqueeze(-1)
  weighted_outputs = all_expert_outputs * routing_weights
  final_hidden_states = weighted_outputs.sum(dim=0)

  return final_hidden_states.type(hidden_states.dtype)


from model_patch import PatchMethod, register_patch

try:
  from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE

  patch = PatchMethod(
    description="GLM4 MoE", cls=Glm4MoeMoE, method="moe", func=_glm4_moe_moe_batched
  )
  register_patch(patch)
except ImportError:
  pass  # glm4_moe not available
