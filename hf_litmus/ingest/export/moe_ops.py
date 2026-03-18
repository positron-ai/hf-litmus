"""
MoeRouter combinator.
"""

from collections.abc import Callable

import torch
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    make_fx,
    track_tensor_tree,
)


class MoeRouterOp(HigherOrderOperator):
    """
    Higher-order op for token-selective Mixture-of-Experts routing with blending.

    MoeRouter evaluates body_fn for each unique expert (in ascending index order),
    passing only the tokens that selected it. Results are weighted by topk_scores
    and accumulated in expert order to ensure deterministic bfloat16 behavior.

    Args:
      body_fn: Function taking (expert_idx, token_indices, *body_args) and returning
        a tensor of shape (num_selected, *out_dim) where num_selected = len(token_indices)
      topk_indices: (num_tokens, top_k) tensor of selected expert indices per token
      topk_scores: (num_tokens, top_k) tensor of routing scores for weighted blending
      *body_args: Additional arguments passed to body_fn
      **body_kwargs: Additional keyword arguments passed to body_fn

    Returns:
      (num_tokens, *out_dim) - blended output for each token (weighted sum of expert outputs)
    """

    def __init__(self):
        super().__init__("moe_router")

    def __call__(
        self,
        body_fn: Callable,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        *body_args: torch.Tensor,
        **body_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        return super().__call__(
            body_fn, topk_indices, topk_scores, *body_args, **body_kwargs
        )


moe_router_op = MoeRouterOp()


def _flatten_kwargs(body_args, body_kwargs):
    """Convert kwargs to positional args in sorted key order for determinism.

    Returns (all_body_args, sorted_kwarg_keys) where all_body_args is body_args
    followed by kwarg values in sorted key order.
    """
    sorted_kwarg_keys = sorted(body_kwargs.keys())
    sorted_kwarg_values = tuple(body_kwargs[k] for k in sorted_kwarg_keys)
    return body_args + sorted_kwarg_values, sorted_kwarg_keys


@moe_router_op.py_impl(ProxyTorchDispatchMode)
def moe_router_proxy(
    mode: ProxyTorchDispatchMode,
    body_fn: Callable,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *body_args,
    **body_kwargs,
) -> torch.Tensor:
    """Tracing implementation - captures body_fn as a subgraph.

    When body_fn accesses tensors from its closure (captured tensors), the
    tracing machinery automatically lifts them as inputs to the subgraph.
    These become additional placeholders in the body_graph.

    Note: kwargs are converted to positional args in sorted key order for
    deterministic serialization. The traced body_graph will be called with
    all positional args.
    """
    if mode.enable_tracing:
        # Preserve the outer tracer before entering make_fx
        outer_tracer = mode.tracer

        # Generate unique name while mode.tracer still points to outer tracer
        body_name = unique_graph_id(mode, "moe_body")
        # unique_graph_id returns (graph_id, name) tuple - extract the name
        if isinstance(body_name, tuple):
            body_name = body_name[1]

        num_tokens = topk_indices.shape[0]

        # Create example expert index (scalar) for tracing
        example_idx = torch.tensor(
            0, dtype=torch.long, device=topk_indices.device
        )

        # Create example token indices (1D tensor with symbolic length)
        # For tracing, use all token indices as the example
        example_token_indices = torch.arange(
            num_tokens, dtype=torch.long, device=topk_indices.device
        )

        # Convert kwargs to positional in sorted key order for deterministic tracing
        all_body_args, sorted_kwarg_keys = _flatten_kwargs(
            body_args, body_kwargs
        )

        # Trace the body function into a GraphModule using symbolic tracing
        # Body now receives (expert_idx, token_indices, *body_args, **body_kwargs)
        # We pass kwargs during tracing to handle keyword-only args in body_fn,
        # but use sorted order for deterministic placeholder ordering
        sorted_kwargs = {k: body_kwargs[k] for k in sorted_kwarg_keys}
        body_graph = make_fx(body_fn, tracing_mode="symbolic")(
            example_idx, example_token_indices, *body_args, **sorted_kwargs
        )

        # Register as a submodule using the outer tracer
        outer_tracer.root.register_module(body_name, body_graph)

        # Get proxies for all explicit inputs (args + kwargs flattened)
        topk_indices_proxy = outer_tracer.unwrap_proxy(topk_indices)
        topk_scores_proxy = outer_tracer.unwrap_proxy(topk_scores)
        all_body_args_proxies = [
            outer_tracer.unwrap_proxy(arg) for arg in all_body_args
        ]

        # Collect captured tensors from the body graph's placeholders
        # Placeholders beyond explicit args are captures
        placeholders = [
            n for n in body_graph.graph.nodes if n.op == "placeholder"
        ]
        # expert_idx + token_indices + all_body_args (includes flattened kwargs)
        num_explicit = 2 + len(all_body_args)

        # Get proxies for captured tensors
        # The captures are tensor references stored during tracing
        captured_proxies = []
        for ph in placeholders[num_explicit:]:
            if not (hasattr(ph, "meta") and "example_value" in ph.meta):
                raise RuntimeError(
                    f"moe_router: captured placeholder '{ph.name}' lacks 'example_value' "
                    f"metadata; cannot produce a proxy for it"
                )
            tensor_val = ph.meta["example_value"]
            captured_proxies.append(outer_tracer.unwrap_proxy(tensor_val))

        # Create the proxy call node with all inputs as positional args
        # kwargs are flattened into positional args for consistent ordering with captures
        proxy_args = (
            body_graph,
            topk_indices_proxy,
            topk_scores_proxy,
            *all_body_args_proxies,
            *captured_proxies,
        )
        out_proxy = outer_tracer.create_proxy(
            "call_function",
            moe_router_op,
            proxy_args,
            {},  # No kwargs - all converted to positional
            name="moe_router",
        )

        # Compute output shape: (num_tokens, *body_output_shape)
        # Run body once to get output shape (with all tokens as example)
        with FakeTensorMode(allow_non_fake_inputs=True):
            example_out = body_fn(
                example_idx, example_token_indices, *body_args, **body_kwargs
            )

        # Create output tensor with correct shape
        # Body returns (num_selected, *out_dim), blended output is (num_tokens, *out_dim)
        body_out_dim = example_out.shape[1:]  # everything after num_selected
        out_shape = (num_tokens,) + body_out_dim
        example_output = example_out.new_empty(out_shape)

        # Track the output tensor
        return track_tensor_tree(
            example_output,
            out_proxy,
            constant=None,
            tracer=outer_tracer,
        )
    else:
        return moe_router_op(
            body_fn, topk_indices, topk_scores, *body_args, **body_kwargs
        )


@moe_router_op.py_impl(FakeTensorMode)
def moe_router_fake(
    mode: FakeTensorMode,
    body_fn: Callable,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *body_args,
    **body_kwargs,
) -> torch.Tensor:
    """FakeTensor implementation for shape inference."""
    num_tokens = topk_indices.shape[0]

    # Create example expert index
    example_idx = torch.tensor(0, dtype=torch.long, device=topk_indices.device)
    # Create example token indices (all tokens for shape inference)
    example_token_indices = torch.arange(
        num_tokens, dtype=torch.long, device=topk_indices.device
    )

    # Run body to get output shape
    with mode:
        example_out = body_fn(
            example_idx, example_token_indices, *body_args, **body_kwargs
        )

    # Output shape: (num_tokens, *body_out_dim) - blended result
    # Body returns (num_selected, *out_dim)
    body_out_dim = example_out.shape[1:]  # everything after num_selected
    out_shape = (num_tokens,) + body_out_dim
    return torch.empty(
        out_shape, dtype=example_out.dtype, device=example_out.device
    )


@moe_router_op.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def moe_router_eager(
    body_fn: Callable,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    *body_args,
    **body_kwargs,
) -> torch.Tensor:
    """
    moe_router reference implementation with weighted blending.

    For each unique expert (in ascending index order), finds which tokens selected it,
    calls body with those token indices, and accumulates score-weighted results.
    The ascending expert order ensures deterministic bfloat16 accumulation matching
    the original HuggingFace implementation.
    """
    num_tokens, top_k = topk_indices.shape
    device = topk_indices.device

    # Get unique experts that any token selected (torch.unique returns sorted)
    unique_experts = torch.unique(topk_indices)

    # Initialize result tensor (will accumulate weighted expert outputs)
    result = None

    for expert_idx in unique_experts:
        expert_idx_scalar = expert_idx.item()

        # Find all (token, k) positions where this expert was selected
        mask = topk_indices == expert_idx_scalar
        token_indices, k_indices = torch.where(mask)

        if len(token_indices) == 0:
            continue

        # Call body with expert index and the token indices that selected this expert
        expert_out = body_fn(
            expert_idx, token_indices, *body_args, **body_kwargs
        )

        # Initialize result on first expert
        if result is None:
            out_dim = expert_out.shape[1:]  # everything after num_selected
            result = torch.zeros(
                (num_tokens,) + out_dim, dtype=expert_out.dtype, device=device
            )

        # Accumulate weighted outputs for each (token, k) position
        # This accumulates in expert order (ascending) for deterministic bfloat16 behavior
        for i, (tok, k) in enumerate(
            zip(token_indices.tolist(), k_indices.tolist())
        ):
            score = topk_scores[tok, k]
            result[tok] = result[tok] + score * expert_out[i]

    # Handle edge case where no experts were selected
    if result is None:
        raise ValueError("moe_router: no experts selected by any token")

    return result


# Disable autograd for now
moe_router_op.py_impl(torch._C.DispatchKey.Autograd)(
    autograd_not_implemented(moe_router_op, deferred_error=True)
)


def moe_router(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    body_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *body_args,
    **body_kwargs,
) -> torch.Tensor:
    """
    Execute a Mixture-of-Experts body for selected experts per token with blending.

    This is a higher-order op that traces body_fn as a subgraph. For each unique
    expert (in ascending index order), finds which tokens selected it, calls body_fn
    with those token indices, and accumulates score-weighted results.

    The ascending expert order ensures deterministic bfloat16 accumulation matching
    the original HuggingFace implementation.

    Args:
      topk_indices: (num_tokens, top_k) tensor of selected expert indices per token
      topk_scores: (num_tokens, top_k) tensor of routing scores for weighted blending
      body_fn: Function taking (expert_idx, token_indices, *body_args) ->
        (num_selected, *out_dim) where num_selected = len(token_indices).
        The body should gather its inputs using token_indices.
      *body_args: Additional arguments passed to body_fn
      **body_kwargs: Additional keyword arguments passed to body_fn

    Returns:
      (num_tokens, *out_dim) - blended output for each token (weighted sum of expert outputs)

    Example:
      # topk_indices: which experts each token selected
      topk_values, topk_indices = torch.topk(router_logits, k=4, dim=-1)
      topk_scores = F.softmax(topk_values, dim=-1)

      def expert_body(idx, token_indices, hidden_states, weights):
          # Gather only the tokens that selected this expert
          selected_hidden = hidden_states[token_indices]  # (num_selected, hidden)
          w = weights[idx]
          return selected_hidden @ w.T  # (num_selected, out_dim)

      # Get blended output directly
      blended = moe_router(topk_indices, topk_scores, expert_body, hidden_states, weights)
      # blended: (num_tokens, out_dim)
    """

    return moe_router_op(
        body_fn, topk_indices, topk_scores, *body_args, **body_kwargs
    )
