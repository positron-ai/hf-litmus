"""Export utilities for FX traces and model metadata.

This module provides functions to export PyTorch models to FX traces
in the format expected by the Haskell ingest pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.export
from torch.export.graph_signature import InputKind

if TYPE_CHECKING:
    import transformers


class ExportableModel(torch.nn.Module):
    """Wrapper for torch.export compatibility with HF models.

    Handles both standard attention and sliding window attention
    patterns, as well as encoder-only models that do not use
    use_cache.
    """

    def __init__(self, model, is_encoder_only: bool = False):
        super().__init__()
        self.model = model
        self.is_encoder_only = is_encoder_only

    def forward(self, input_ids, attention_mask, sliding_attention_mask=None):
        if sliding_attention_mask is not None:
            attention_mask = {
                "full_attention": attention_mask,
                "sliding_attention": sliding_attention_mask,
            }

        kwargs: dict = {"attention_mask": attention_mask}
        if not self.is_encoder_only:
            kwargs["use_cache"] = False

        out = self.model(input_ids, **kwargs)
        return out.logits


def get_model_config_safely(model) -> Any:
    """Safely retrieve model config with fallback for models without get_text_config.

    Some models (e.g., multimodal) have get_text_config() while others don't.
    This provides a consistent interface for both cases.
    """
    try:
        if hasattr(model.config, "get_text_config"):
            return model.config.get_text_config()
        return model.config
    except Exception as e:
        print(f"Warning: Could not retrieve text config: {e}")
        return model.config


def export_fx_trace(
    model: torch.nn.Module,
    seq_len: int = 128,
    batch_size_max: int = 1024,
) -> torch.export.ExportedProgram:
    """Export FX trace with dynamic shapes.

    Args:
      model: The model to export
      seq_len: Maximum sequence length
      batch_size_max: Maximum batch size for dynamic dimension

    Returns:
      ExportedProgram containing the FX graph
    """
    batch_size_dim = torch.export.Dim("batch_size", min=1, max=batch_size_max)
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=seq_len)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    BATCH_SZ = 13

    if device.type == "meta":
        input_ids = torch.empty(
            (BATCH_SZ, seq_len), dtype=torch.long, device="meta"
        )
        attention_mask = torch.empty(
            (BATCH_SZ, 1, seq_len, seq_len), dtype=dtype, device="meta"
        )
    else:
        input_ids = torch.zeros(
            (BATCH_SZ, seq_len), dtype=torch.long, device=device
        )
        # Additive causal mask: 0 for attend, -inf for masked (HF 4D convention)
        causal_mask = torch.triu(
            torch.full(
                (seq_len, seq_len), float("-inf"), dtype=dtype, device=device
            ),
            diagonal=1,
        )
        attention_mask = (
            causal_mask.unsqueeze(0).unsqueeze(0).expand(BATCH_SZ, -1, -1, -1)
        )

    dynamic_shapes = (
        {0: batch_size_dim, 1: seq_len_dim},
        {0: batch_size_dim, 2: seq_len_dim, 3: seq_len_dim},
    )

    original_autocast = None
    original_is_autocast_enabled = None
    if device.type == "meta":
        from contextlib import nullcontext

        original_autocast = torch.autocast
        torch.autocast = lambda *args, **kwargs: nullcontext()
        # transformers 5.0+ calls torch.is_autocast_enabled(device_type)
        # which crashes on "meta" device, so patch it to return False
        original_is_autocast_enabled = torch.is_autocast_enabled
        torch.is_autocast_enabled = lambda *args, **kwargs: False

    try:
        wrapper = ExportableModel(model)
        with torch.no_grad():
            exported = torch.export.export(
                wrapper,
                args=(input_ids, attention_mask),
                dynamic_shapes=dynamic_shapes,
            )
        return exported
    finally:
        if original_autocast is not None:
            torch.autocast = original_autocast
        if original_is_autocast_enabled is not None:
            torch.is_autocast_enabled = original_is_autocast_enabled


def _is_moe_weight_transposed(
    target: str, shape: list[int], config: dict
) -> bool:
    """Detect if an MoE expert weight is stored transposed in safetensors.

    See Note [MoE Weight Transpose in Transformers] in FxTypedFx.hs.

    GPT-OSS and similar MoE models store expert weights transposed in safetensors
    for the matmul_ogs kernel, which requires column-major weight layout for mxfp4.

    The Fx graph shows model-format shapes (E, in_features, out_features) from the
    Python model definition, but safetensors stores (E, out_features, in_features).

    Returns True if the parameter is an MoE expert weight that is transposed.
    """
    num_experts = config.get("num_local_experts")
    intermediate_size = config.get("intermediate_size")
    hidden_size = config.get("hidden_size")

    if num_experts is None or intermediate_size is None or hidden_size is None:
        return False

    shape = list(shape)

    # gate_up_proj: Fx shows (E, hidden, 2*inter), safetensors has (E, 2*inter, hidden)
    if "gate_up_proj" in target and "bias" not in target:
        expected_fx = [num_experts, hidden_size, 2 * intermediate_size]
        return len(shape) == 3 and shape == expected_fx

    # down_proj: Fx shows (E, inter, hidden), safetensors has (E, hidden, inter)
    # But when inter == hidden, transpose is a no-op
    if "down_proj" in target and "bias" not in target and "experts" in target:
        if intermediate_size == hidden_size:
            return False  # Transpose is no-op when dimensions equal
        expected_fx = [num_experts, intermediate_size, hidden_size]
        return len(shape) == 3 and shape == expected_fx

    return False


def _parse_stack_trace(trace: str) -> list[dict[str, str | int]]:
    """Parse Python stack trace into structured frames."""
    frames: list[dict[str, str | int]] = []
    for match in re.finditer(
        r'File "([^"]+)", line (\d+), in ([\w<>]+)', trace
    ):
        filepath, line, func = match.groups()
        # Skip deep torch internals (autograd, dispatch, etc.)
        if "/torch/" in filepath and "/nn/" not in filepath:
            continue
        # Keep only transformers model code from site-packages
        if "site-packages" in filepath and "transformers" not in filepath:
            continue
        frames.append({"file": filepath, "line": int(line), "function": func})
    return frames


def extract_node_locations(
    graph: torch.fx.Graph,
    named_submodules: list[tuple[str, torch.fx.Graph]] | None = None,
) -> dict[str, dict]:
    """Extract source location metadata from FX graph nodes.

    Processes the root graph and any submodule graphs so that
    provenance is available for submodule nodes too. Submodule
    node names are prefixed with the submodule path (e.g.
    ``submod_0.add``) to avoid collisions with root graph nodes.
    Root graph entries take priority via setdefault.
    """
    locations: dict[str, dict] = {}
    _extract_graph_node_locations(graph, locations)
    for name, subgraph in named_submodules or []:
        _extract_graph_node_locations(subgraph, locations, prefix=f"{name}.")
    return locations


def _extract_graph_node_locations(
    graph: torch.fx.Graph,
    locations: dict[str, dict],
    prefix: str = "",
) -> None:
    """Extract source locations from a single FX graph into locations dict.

    When prefix is non-empty, node names are qualified to avoid
    collisions between root and submodule graphs.
    """
    for node in graph.nodes:
        if node.op not in (
            "call_function",
            "call_method",
            "call_module",
        ):
            continue
        entry: dict = {}

        st = node.meta.get("stack_trace")
        if st:
            entry["stack_trace"] = st
            entry["stack_frames"] = _parse_stack_trace(st)

        nn_stack = node.meta.get("nn_module_stack")
        if nn_stack:
            entry["nn_module_stack"] = {
                path: qualified_name
                for path, (qualified_name, _cls) in nn_stack.items()
            }

        src_stack = node.meta.get("source_fn_stack")
        if src_stack:
            if isinstance(src_stack, dict):
                entry["source_fn_stack"] = {
                    name: getattr(fn, "__name__", str(fn))
                    for name, fn in src_stack.items()
                }
            else:
                entry["source_fn_stack"] = {
                    name: getattr(fn, "__name__", str(fn))
                    for name, fn in src_stack
                }

        if entry:
            key = f"{prefix}{node.name}" if prefix else node.name
            locations.setdefault(key, entry)


def build_ingest_metadata(
    exported: torch.export.ExportedProgram,
    model_config: transformers.AutoConfig,
    is_encoder_only: bool = False,
) -> dict:
    """Build metadata dict from an exported program and model config.

    Returns a dict suitable for writing as metadata.json in the format
    expected by the Haskell ingest pipeline (Fx.Metadata.load).

    Args:
      exported: The exported program containing the FX graph.
      model_config: HuggingFace model configuration.
      is_encoder_only: If True, relaxes rope_theta requirement and
        adds encoder-only flags to metadata.
    """
    sig = exported.graph_signature

    buffers: list[dict] = []
    for spec in sig.input_specs:
        if spec.kind == InputKind.BUFFER and hasattr(spec.arg, "name"):
            buf = exported.state_dict.get(spec.target)
            if buf is None:
                buf = exported.constants.get(spec.target)
            if buf is not None:
                buffers.append(
                    {
                        "input_name": spec.arg.name,
                        "shape": list(buf.shape),
                        "dtype": str(buf.dtype),
                    }
                )

    parameters: list[dict] = []
    for spec in sig.input_specs:
        if spec.kind == InputKind.PARAMETER and hasattr(spec.arg, "name"):
            param = exported.state_dict.get(spec.target)
            if param is not None:
                parameters.append(
                    {
                        "input_name": spec.arg.name,
                        "target": spec.target.removeprefix("model."),
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                    }
                )

    constants: list[dict] = []
    for spec in sig.input_specs:
        if spec.kind == InputKind.CONSTANT_TENSOR and hasattr(
            spec.arg, "name"
        ):
            const = exported.constants.get(spec.target)
            if const is not None:
                constants.append(
                    {
                        "input_name": spec.arg.name,
                        "shape": list(const.shape),
                        "dtype": str(const.dtype),
                        "value": (
                            float(const)
                            if len(const.shape) == 0
                            else list(const.flatten().tolist())
                        ),
                    }
                )

    input_metadata = {}
    for node in exported.graph.nodes:
        if (
            node.op == "placeholder"
            and hasattr(node, "meta")
            and "val" in node.meta
        ):
            val = node.meta["val"]
            if hasattr(val, "shape") and hasattr(val, "dtype"):
                shape = []
                for dim in val.shape:
                    if isinstance(dim, int):
                        shape.append(dim)
                    else:
                        shape.append(str(dim))
                input_metadata[node.name] = {
                    "shape": shape,
                    "dtype": str(val.dtype),
                }

    user_inputs: list[dict] = []
    for spec in sig.input_specs:
        if spec.kind == InputKind.USER_INPUT and hasattr(spec.arg, "name"):
            meta = input_metadata.get(spec.arg.name, {})
            user_inputs.append(
                {
                    "input_name": spec.arg.name,
                    "shape": meta.get("shape"),
                    "dtype": meta.get("dtype"),
                }
            )

    CONFIG_KEYS = {
        "hidden_size",
        "intermediate_size",  # For MoE transposition detection
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_local_experts",  # For MoE transposition detection
        "vocab_size",
        "head_dim",
    }
    config_dict = {k: getattr(model_config, k, None) for k in CONFIG_KEYS}

    # For encoder-only models, num_key_value_heads equals
    # num_attention_heads (no grouped-query attention) and may
    # not be present as a config attribute.
    if config_dict.get("num_key_value_heads") is None:
        config_dict["num_key_value_heads"] = config_dict["num_attention_heads"]

    # head_dim may be None in some configs; compute from hidden_size
    if config_dict.get("head_dim") is None:
        config_dict["head_dim"] = (
            model_config.hidden_size // model_config.num_attention_heads
        )

    # rope_theta: required for decoder models, optional for
    # encoder-only models (they typically don't use RoPE).
    rope_theta = getattr(model_config, "rope_theta", None)
    # transformers 5.0 moved rope_theta into rope_parameters dict
    rope_params = getattr(model_config, "rope_parameters", None)
    if rope_theta is None and rope_params:
        if "rope_theta" in rope_params:
            rope_theta = rope_params["rope_theta"]

    if rope_theta is None and not is_encoder_only:
        raise ValueError(
            "rope_theta not found in model config or rope_parameters. "
            "The Haskell ingest pipeline requires rope_theta for "
            "decoder models."
        )
    # Encoder-only models without RoPE get a sentinel value.
    config_dict["rope_theta"] = rope_theta if rope_theta is not None else 0.0

    # rope_scaling is optional and may be None or a dict.
    # transformers 5.0 consolidated rope_scaling into rope_parameters.
    rope_scaling = getattr(model_config, "rope_scaling", None)
    if rope_scaling is None and rope_params and "rope_type" in rope_params:
        rope_scaling = dict(rope_params)
        rope_scaling.pop("rope_theta", None)
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)  # copy to avoid mutating config
        if "rope_type" in rope_scaling and "type" not in rope_scaling:
            rope_scaling["type"] = rope_scaling.pop("rope_type")
        config_dict["rope_scaling"] = rope_scaling

    config_dict["is_encoder_only"] = is_encoder_only

    # Additional config fields for model feature tag computation
    # (MoE, sliding window, MLA detection).
    TAG_KEYS = {
        "num_local_experts",
        "sliding_window",
        "kv_lora_rank",
        "model_type",
        "num_experts_per_tok",
    }
    tag_config = {k: getattr(model_config, k, None) for k in TAG_KEYS}

    range_constraints = {
        k.name: (str(v.lower), str(v.upper))
        for k, v in exported.range_constraints.items()
    }

    # Detect sliding window attention from config
    sliding_window = getattr(model_config, "sliding_window", None)
    if is_encoder_only:
        attention_masks = {"attention_mask": {"type": "bidirectional"}}
    elif sliding_window is not None:
        attention_masks = {
            "attention_mask": {
                "type": "sliding_window",
                "window": sliding_window,
            }
        }
    else:
        attention_masks = {"attention_mask": {"type": "full_causal"}}

    named_submodules = [
        (name, mod.graph)
        for name, mod in exported.graph_module.named_children()
        if hasattr(mod, "graph")
    ]
    node_locations = extract_node_locations(exported.graph, named_submodules)

    # Detect transposed MoE expert weights
    # See Note [MoE Weight Transpose in Transformers] in FxTypedFx.hs
    for param in parameters:
        param["transposed"] = _is_moe_weight_transposed(
            param["target"], param["shape"], config_dict
        )

    return {
        "graph": "root.fx",
        "submodules": [],
        "range_constraints": range_constraints,
        "parameters": parameters,
        "user_inputs": user_inputs,
        "buffers": buffers,
        "constants": constants,
        "attention_masks": attention_masks,
        "node_locations": node_locations,
        "config": config_dict,
        "tag_config": tag_config,
    }


def export_to_ingest_directory(
    exported: torch.export.ExportedProgram,
    model_config: transformers.AutoConfig,
    output_dir: Path,
):
    """Export FX trace to directory format expected by Fx.Metadata.load.

    Creates:
      - output_dir/root.fx: The FX graph representation
      - output_dir/metadata.json: Model metadata including parameters, inputs, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "root.fx").write_text(str(exported.graph))
    metadata = build_ingest_metadata(exported, model_config)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
