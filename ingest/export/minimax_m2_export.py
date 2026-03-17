#!/usr/bin/env python3
# vim: set filetype=python ts=2 sw=2 et:
"""
MiniMax-M2 specific patches for torch.export compatibility.

The MiniMaxM2SparseMoeBlock uses:
1. Data-dependent loops in MiniMaxM2Experts (iterating over expert_hit
   from nonzero()) which torch.export cannot handle.
2. aten.gather in route_tokens_to_experts to select top-k weights,
   which the downstream Haskell FxTypedFx translator doesn't support.

This module patches the entire MoE block to use a dense routing
approach: scatter + mask + fixed expert loop, avoiding both gather
and data-dependent control flow.
"""

import types

import torch


def _minimax_m2_moe_block_forward(
  self,
  hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Patched forward for MiniMaxM2SparseMoeBlock.

  Replaces the original routing + expert dispatch with a fully
  static, torch.export-compatible implementation that avoids:
  - aten.gather (unsupported in FxTypedFx)
  - Data-dependent loops (unsupported in torch.export)

  Instead of gather, uses scatter_ to build a dense expert mask
  and element-wise multiply to select routing weights.

  Args:
    hidden_states: (batch_size, seq_len, hidden_dim)
  Returns:
    (output, router_logits) where:
      output: (batch_size, seq_len, hidden_dim)
      router_logits: (batch_size * seq_len, num_experts)
  """
  batch_size, seq_len, hidden_dim = hidden_states.shape
  hidden_states_flat = hidden_states.view(-1, hidden_dim)

  # Gate: project to expert scores
  router_logits = self.gate(hidden_states_flat)

  # Routing weights via sigmoid (not softmax)
  routing_weights = torch.sigmoid(router_logits.float())

  # Score correction bias for expert selection
  scores = routing_weights + self.e_score_correction_bias

  # Select top-k experts per token
  _, top_k_index = torch.topk(scores, self.top_k, dim=-1, sorted=False)

  # Build dense binary mask from top_k_index using scatter_
  # instead of gather. This creates a (tokens, num_experts)
  # mask with 1.0 at selected expert positions.
  num_experts = self.experts.num_experts
  mask = torch.zeros_like(routing_weights)
  # Create ones source with matching float dtype for scatter.
  # Avoid zeros_like with dtype= kwarg (unsupported in FxTypedFx).
  # Use slice of routing_weights (already correct dtype/device)
  # multiplied by 0 + 1 to get a ones tensor of shape (tokens, top_k).
  ones_src = routing_weights[:, : self.top_k] * 0.0 + 1.0
  mask.scatter_(1, top_k_index, ones_src)

  # Apply mask to keep only selected expert weights
  selected_weights = routing_weights * mask

  # Normalize so selected weights sum to 1.0 per token
  weight_sum = selected_weights.sum(dim=-1, keepdim=True)
  selected_weights = selected_weights / weight_sum

  # Run all experts with dense weights (fixed loop, export-safe)
  final_hidden = torch.zeros_like(hidden_states_flat)
  for expert_idx in range(num_experts):
    expert = self.experts[expert_idx]
    expert_out = expert(hidden_states_flat)
    weight = selected_weights[:, expert_idx : expert_idx + 1]
    final_hidden = final_hidden + expert_out * weight

  output = final_hidden.reshape(batch_size, seq_len, hidden_dim)
  return output, router_logits


def patch_minimax_m2_for_export(model) -> None:
  """Patch MiniMax-M2 MoE blocks for torch.export compatibility.

  Patches MiniMaxM2SparseMoeBlock.forward to avoid:
  - aten.gather (replaced with scatter_ + mask)
  - Data-dependent expert dispatch (replaced with fixed loop)

  Also converts e_score_correction_bias from buffer to parameter.
  FxTypedFx classifies parameters generically (ParameterTensor)
  but requires buffers to match known semantic categories.
  """
  patched = 0
  for module in model.modules():
    cls_name = type(module).__name__
    if cls_name == "MiniMaxM2SparseMoeBlock":
      module.forward = types.MethodType(_minimax_m2_moe_block_forward, module)
      # Convert e_score_correction_bias from buffer to parameter.
      # FxTypedFx only handles specific buffer types (inv_freq,
      # position_ids). Converting to parameter avoids the
      # guessBufferSemantics error.
      if hasattr(module, "e_score_correction_bias"):
        bias = module.e_score_correction_bias
        del module._buffers["e_score_correction_bias"]
        module.e_score_correction_bias = torch.nn.Parameter(bias, requires_grad=False)
      patched += 1

  if patched:
    print(
      f"Patched {patched} MiniMaxM2SparseMoeBlock modules "
      "for torch.export compatibility"
    )
