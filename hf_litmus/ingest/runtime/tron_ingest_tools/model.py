"""Model loading and weight transfer utilities.

This module provides functions for loading TypedFx-generated models
and transferring weights from HuggingFace models.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import torch


class GeneratedModelWrapper(torch.nn.Module):
    """Wrapper for generated TypedFx models with different parameter names.

    Generated models have parameters like p_model_lm_head_weight_1 instead of
    the standard HuggingFace names. This wrapper provides a consistent forward
    interface that handles parameter introspection and buffer injection.
    """

    def __init__(self, model, config=None, hf_buffers=None):
        super().__init__()
        self.model = model
        sig = inspect.signature(model.forward)
        self.param_names = [p for p in sig.parameters.keys() if p != "self"]
        self.hf_buffers = hf_buffers or {}
        if config is not None:
            self.head_dim = getattr(config, "head_dim", None)
            if self.head_dim is None:
                self.head_dim = (
                    config.hidden_size // config.num_attention_heads
                )
            self.rope_theta = getattr(config, "rope_theta", None)
            if self.rope_theta is None:
                rope_params = getattr(config, "rope_parameters", None)
                self.rope_theta = (
                    rope_params.get("rope_theta", 10000.0)
                    if rope_params
                    else 10000.0
                )
        else:
            self.head_dim = 64
            self.rope_theta = 10000.0

    def forward(self, input_ids, attention_mask):
        model_dtype = next(self.model.parameters()).dtype
        device = input_ids.device

        kwargs = {}
        unrecognized = []
        for name in self.param_names:
            if name == "input_ids_1":
                kwargs[name] = input_ids
            elif name == "attention_mask_1":
                kwargs[name] = attention_mask
            elif name.startswith("b_model_model_rotary_emb_"):
                suffix = name.removeprefix("b_model_model_rotary_emb_")
                buf_name = suffix.removesuffix("_1")
                if buf_name in self.hf_buffers:
                    kwargs[name] = self.hf_buffers[buf_name].to(
                        device=device, dtype=model_dtype
                    )
                elif "inv_freq" in buf_name:
                    inv_freq = 1.0 / (
                        self.rope_theta
                        ** (
                            torch.arange(
                                0,
                                self.head_dim,
                                2,
                                dtype=torch.float32,
                                device=device,
                            )
                            / self.head_dim
                        )
                    )
                    kwargs[name] = inv_freq.to(model_dtype)
                else:
                    raise ValueError(
                        f"Rotary buffer '{buf_name}' not found in hf_buffers and cannot "
                        f"be computed from config. Pass it via hf_buffers={{'{buf_name}': tensor}}."
                    )
            else:
                unrecognized.append(name)
        if unrecognized:
            raise ValueError(
                f"Unrecognized forward parameters: {unrecognized}. "
                f"Expected input_ids_1, attention_mask_1, or b_model_model_rotary_emb_*."
            )
        return self.model(**kwargs)


def load_generated_model(
    py_file: Path, model_class_name: str
) -> torch.nn.Module:
    """Dynamically import and instantiate generated model.

    Args:
      py_file: Path to generated Python file
      model_class_name: Name of model class to instantiate

    Returns:
      Instantiated model in eval mode
    """
    spec = importlib.util.spec_from_file_location("generated_model", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec from {py_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_model"] = module
    spec.loader.exec_module(module)

    if hasattr(module, model_class_name):
        model_cls = getattr(module, model_class_name)
    else:
        candidates = [
            name
            for name in dir(module)
            if not name.startswith("_")
            and isinstance(getattr(module, name), type)
            and issubclass(getattr(module, name), torch.nn.Module)
        ]
        if not candidates:
            raise RuntimeError(f"No model class found in {py_file}")
        print(
            f"Warning: Class {model_class_name} not found, using {candidates[0]}"
        )
        model_cls = getattr(module, candidates[0])

    model = model_cls()
    model.eval()
    return model


def smart_reconstruct_hf_name(inner: str) -> str:
    """Reconstruct HuggingFace parameter name from generated model inner name.

    The generated parameter names (after stripping p_model_ prefix and _1 suffix)
    map to HF paths with underscores replaced appropriately.

    Examples:
      lm_head_weight -> lm_head.weight
      model_embed_tokens_weight -> model.embed_tokens.weight
      model_layers_0_self_attn_q_proj_weight ->
        model.layers.0.self_attn.q_proj.weight
    """
    underscore_names = {
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
        "rotary_emb",
        "inv_freq",
    }

    terminators = {"weight", "bias"}

    tokens = inner.split("_")
    result = []
    i = 0

    while i < len(tokens):
        matched = False
        for length in range(4, 0, -1):
            if i + length <= len(tokens):
                candidate = "_".join(tokens[i : i + length])
                if candidate in underscore_names or candidate in terminators:
                    result.append(candidate)
                    i += length
                    matched = True
                    break

        if matched:
            continue

        if tokens[i] == "model":
            result.append("model")
            i += 1
        elif (
            tokens[i] == "layers"
            and i + 1 < len(tokens)
            and tokens[i + 1].isdigit()
        ):
            result.append(f"layers.{tokens[i + 1]}")
            i += 2
        elif tokens[i] == "mlp":
            result.append("mlp")
            i += 1
        elif tokens[i] == "norm":
            result.append("norm")
            i += 1
        else:
            result.append(tokens[i])
            i += 1

    return ".".join(result)


def transfer_weights(
    hf_model: torch.nn.Module,
    generated_model: torch.nn.Module,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> int:
    """Transfer weights from HuggingFace model to generated model.

    Args:
      hf_model: Source HuggingFace model with weights
      generated_model: Target generated model (will be modified)
      device: Target device for generated model
      dtype: Target dtype for generated model

    Returns:
      Count of successfully transferred parameters
    """
    hf_state = hf_model.state_dict()
    gen_state = dict(generated_model.named_parameters())

    transferred = 0
    missing_in_hf = []

    for gen_name, gen_param in gen_state.items():
        if not gen_name.startswith("p_model_") or not gen_name.endswith("_1"):
            print(f"Warning: Unexpected parameter name format: {gen_name}")
            continue

        inner = gen_name[8:-2]
        hf_name = smart_reconstruct_hf_name(inner)

        if hf_name in hf_state:
            hf_weight = hf_state[hf_name]
            if hf_weight.shape == gen_param.shape:
                with torch.no_grad():
                    gen_param.copy_(
                        hf_weight.to(
                            device=gen_param.device, dtype=gen_param.dtype
                        )
                    )
                transferred += 1
            else:
                print(
                    f"Warning: Shape mismatch: {gen_name} has {gen_param.shape}, "
                    f"HF {hf_name} has {hf_weight.shape}"
                )
        else:
            missing_in_hf.append((gen_name, hf_name))

    if missing_in_hf:
        print(
            f"Warning: Missing in HF state dict: {len(missing_in_hf)} parameters"
        )
        for gen_name, hf_name in missing_in_hf[:5]:
            print(f"  {gen_name} -> {hf_name}")

    generated_model.to(device=device, dtype=dtype)

    return transferred
