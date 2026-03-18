#!/usr/bin/env python3
# vim: set filetype=python ts=2 sw=2 et:

import json
import sys
from pathlib import Path
from typing import Any

# Remove nix-provided site-packages from sys.path to prevent
# ABI-incompatible C extensions (e.g. sentencepiece, torchvision)
# from the nix Python 3.13.9 env crashing the uv Python 3.13.5.
# The venv's site-packages at the end of sys.path has compatible
# versions installed by uv.
sys.path = [
    p for p in sys.path if "/nix/store/" not in p or "site-packages" not in p
]

import accelerate
import glm4_moe_export  # noqa: F401 — registers patches at import
import gpt_oss_export  # noqa: F401 — registers patches at import
import model_patch
import torch
import torch._dynamo
import torch.export
import transformers
from minimax_m2_export import patch_minimax_m2_for_export
from torch.export.graph_signature import (
    ArgumentSpec,
    InputKind,
)

# Add runtime/ to path to import tron_ingest_tools
_python_path = Path(__file__).parent.parent / "runtime"
if str(_python_path) not in sys.path:
    sys.path.insert(0, str(_python_path))

from tron_ingest_tools import (
    ExportableModel,
    get_model_config_safely,
)


# Patch ROPE_INIT_FUNCTIONS with 'default' rope type if missing.
# Custom models (e.g. MiniMax-M2.5) expect rope_type='default'
# which was removed/renamed in some transformers versions.
def _patch_rope_init_functions():
    try:
        from transformers.modeling_rope_utils import (
            ROPE_INIT_FUNCTIONS,
        )
    except ImportError:
        return

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _rope_init_default(config, device, **kwargs):
        rotary_dim = getattr(config, "rotary_dim", None)
        if rotary_dim is None:
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            factor = getattr(config, "partial_rotary_factor", 1.0)
            rotary_dim = int(head_dim * factor)

        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        inv_freq = inv_freq.to(device=device)
        # attention_factor is 1.0 for default RoPE
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _rope_init_default


_patch_rope_init_functions()


def _from_config_no_init(auto_cls, config, **kwargs):
    """Create model from config without running post_init.

    Custom models (e.g. MiniMax-M2) may have rotary embeddings
    that are incompatible with transformers' weight initialization
    in post_init. Temporarily disable post_init to avoid crashes.
    """
    original_post_init = transformers.PreTrainedModel.post_init
    try:
        transformers.PreTrainedModel.post_init = lambda self: None
        return auto_cls.from_config(config, **kwargs)
    finally:
        transformers.PreTrainedModel.post_init = original_post_init


def _get_auto_model_class(config) -> type:
    """Select the appropriate AutoModel class based on config.

    Decoder-only (causal LM) models use AutoModelForCausalLM.
    Encoder-only models (BERT, etc.) use AutoModelForMaskedLM.
    """
    architectures = getattr(config, "architectures", []) or []
    for arch in architectures:
        arch_lower = arch.lower()
        if (
            "formaskedlm" in arch_lower
            or "forsequenceclassification" in arch_lower
        ):
            return transformers.AutoModelForMaskedLM
        if "forquestionanswering" in arch_lower:
            return transformers.AutoModelForQuestionAnswering
    # Fallback: check model_type for known encoder-only families
    model_type = getattr(config, "model_type", "")
    ENCODER_ONLY_TYPES = {
        "bert",
        "distilbert",
        "roberta",
        "albert",
        "electra",
        "xlm-roberta",
        "deberta",
        "deberta-v2",
        "camembert",
    }
    if model_type in ENCODER_ONLY_TYPES:
        return transformers.AutoModelForMaskedLM
    return transformers.AutoModelForCausalLM


def _is_encoder_only(config) -> bool:
    """Check if a model config is an encoder-only architecture."""
    return _get_auto_model_class(config) != transformers.AutoModelForCausalLM


def _move_to_meta(module: torch.nn.Module) -> None:
    """Move any non-meta tensors to meta device.

    Some models register buffers outside of init_empty_weights()
    context, causing device mismatches.
    """
    for name, param in list(module.named_parameters(recurse=False)):
        if param.device.type != "meta":
            new_param = torch.nn.Parameter(
                torch.empty(param.shape, dtype=param.dtype, device="meta"),
                requires_grad=param.requires_grad,
            )
            setattr(module, name, new_param)
    for name, buffer in list(module.named_buffers(recurse=False)):
        if buffer.device.type != "meta":
            new_buffer = torch.empty(
                buffer.shape, dtype=buffer.dtype, device="meta"
            )
            module.register_buffer(name, new_buffer)


def init_model_no_weights(
    model_name: str,
    config: transformers.AutoConfig,
    use_meta_device: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    auto_cls = _get_auto_model_class(config)
    if use_meta_device:
        with accelerate.init_empty_weights():
            model = _from_config_no_init(
                auto_cls,
                config,
                dtype=dtype,
                trust_remote_code=True,
            )
        # Fix any tensors that escaped init_empty_weights context
        for module in model.modules():
            _move_to_meta(module)
        print(
            "Using meta device - zero memory footprint for structure export."
        )
        return model

    # Fallback: init_empty_weights then materialize zeros on CPU.
    # Still O(model_size) memory but avoids loading weights.
    # See: https://github.com/pytorch/pytorch/issues/130067
    with accelerate.init_empty_weights():
        model = _from_config_no_init(
            auto_cls,
            config,
            dtype=dtype,
            trust_remote_code=True,
        )

    def materialize_with_zeros(module):
        for name, param in module.named_parameters(recurse=False):
            if param.device.type == "meta":
                new_param = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
                setattr(module, name, new_param)
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device.type == "meta":
                new_buffer = torch.zeros(
                    buffer.shape, dtype=buffer.dtype, device="cpu"
                )
                module.register_buffer(name, new_buffer)

    for module in model.modules():
        materialize_with_zeros(module)

    print(
        "Using zero-initialized weights for structure export. "
        "Use --load-weights to load actual weights."
    )
    return model


def export_model(
    model_name,
    max_seq_length: int = 128,
    load_weights: bool = False,
    offload: bool = False,
    use_meta_device: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    tokenizer_name: str = None,
) -> tuple[torch.export.ExportedProgram, dict[str, Any]]:
    """Export model structure with or without weights.

    Returns (exported_program, metadata) where metadata contains
    attention_masks, config, and tag_config dicts.
    """
    # Patch autocast for meta device - must be active for the
    # entire export because transformers uses autocast in forward
    # passes (e.g., rotary_emb).
    original_autocast = None
    original_is_autocast_enabled = None
    if use_meta_device:
        from contextlib import nullcontext

        original_autocast = torch.autocast
        original_is_autocast_enabled = torch.is_autocast_enabled

        torch.autocast = lambda *args, **kwargs: nullcontext()

        def patched_is_autocast_enabled(device_type=None):
            if device_type == "meta":
                return False
            return original_is_autocast_enabled(device_type)

        torch.is_autocast_enabled = patched_is_autocast_enabled

    try:
        return _export_model_impl(
            model_name,
            max_seq_length,
            load_weights,
            offload,
            use_meta_device,
            dtype,
            tokenizer_name,
        )
    finally:
        if original_autocast is not None:
            torch.autocast = original_autocast
        if original_is_autocast_enabled is not None:
            torch.is_autocast_enabled = original_is_autocast_enabled


def _build_attention_mask(
    model,
    text_config,
    input_ids: torch.Tensor,
    batch_size_dim,
    seq_len_dim,
    batch_sz: int,
    actual_seq_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[tuple, tuple, dict[str, Any]]:
    """Build attention mask, export args, dynamic shapes, metadata.

    Returns (export_args, dynamic_shapes, metadata).
    """
    sliding_window = getattr(text_config, "sliding_window", None)
    target_device = next(model.parameters()).device

    if sliding_window is not None:
        dummy_embeds = torch.zeros(
            (batch_sz, actual_seq_len, text_config.hidden_size),
            dtype=dtype,
            device="cpu",
        )
        attention_masks = transformers.masking_utils.create_masks_for_generate(
            config=model.config,
            input_embeds=dummy_embeds,
            attention_mask=None,
            cache_position=torch.arange(actual_seq_len),
            past_key_values=None,
            position_ids=torch.zeros(
                (batch_sz, actual_seq_len), dtype=torch.long
            ),
        )

        if (
            isinstance(attention_masks, dict)
            and "sliding_attention" in attention_masks
        ):
            full_mask = attention_masks["full_attention"].to(target_device)
            sliding_mask = attention_masks["sliding_attention"].to(
                target_device
            )
            metadata = {
                "attention_masks": {
                    "attention_mask": {"type": "full_causal"},
                    "sliding_attention_mask": {
                        "type": "sliding_window",
                        "window": sliding_window,
                    },
                }
            }
            export_args = (input_ids, full_mask, sliding_mask)
            dynamic_shapes = (
                {0: batch_size_dim, 1: seq_len_dim},
                {0: batch_size_dim, 2: seq_len_dim, 3: seq_len_dim},
                {0: batch_size_dim, 2: seq_len_dim, 3: seq_len_dim},
            )
            return export_args, dynamic_shapes, metadata

        # Single sliding window mask
        attention_mask = attention_masks.to(target_device)
        metadata = {
            "attention_masks": {
                "attention_mask": {
                    "type": "sliding_window",
                    "window": sliding_window,
                },
            }
        }
    else:
        # Full causal mask
        causal_mask = torch.tril(
            torch.ones(actual_seq_len, actual_seq_len, dtype=torch.float32)
        )
        attention_mask = (
            causal_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand((batch_sz, -1, -1, -1))
        )
        attention_mask = attention_mask.to(target_device)
        metadata = {
            "attention_masks": {"attention_mask": {"type": "full_causal"}}
        }

    export_args = (input_ids, attention_mask)
    dynamic_shapes = (
        {0: batch_size_dim, 1: seq_len_dim},
        {0: batch_size_dim, 2: seq_len_dim, 3: seq_len_dim},
    )
    return export_args, dynamic_shapes, metadata


def _replace_dim_static(dynamic_shapes, dim):
    """Replace a specific Dim with None (static) in dynamic_shapes."""
    result = []
    for shape in dynamic_shapes:
        new = {}
        for k, v in shape.items():
            new[k] = None if v is dim else v
        result.append(new)
    return tuple(result)


def _export_model_impl(
    model_name,
    max_seq_length: int,
    load_weights: bool,
    offload: bool,
    use_meta_device: bool,
    dtype: torch.dtype,
    tokenizer_name: str = None,
) -> tuple[torch.export.ExportedProgram, dict[str, Any]]:
    """Internal implementation of export_model."""

    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Multimodal models (e.g. Qwen3.5) have a text sub-config.
    # Extract it so we export only the text/causal LM backbone.
    if hasattr(config, "text_config") and any(
        "ForConditionalGeneration" in a
        for a in getattr(config, "architectures", []) or []
    ):
        print(
            f"Multimodal model detected (type={config.model_type}). "
            "Extracting text backbone for export."
        )
        config = config.text_config

    is_encoder = _is_encoder_only(config)

    if load_weights:
        load_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if offload:
            load_kwargs["offload_state_dict"] = True
            load_kwargs["offload_folder"] = "./offload"

        # For mxfp4-quantized models (e.g., GPT-OSS), request dequantization during
        # loading to ensure weight shapes match the Fx graph.
        #
        # Background: GPT-OSS MoE expert weights are stored transposed in safetensors
        # for HuggingFace's triton matmul_ogs kernel (kernels-community/triton_kernels),
        # which requires column-major weight layout for mxfp4. For example, gate_up_proj
        # is stored as (E, 2*intermediate, hidden) but the model definition and Fx graph
        # expect (E, hidden, 2*intermediate).
        #
        # Without dequantize=True, HuggingFace replaces GptOssExperts with a triton-backed
        # Mxfp4GptOssExperts that "lies" about tensor shapes - it overrides .shape on a
        # wrapper object to present model-format dimensions while the underlying memory
        # remains transposed. This breaks torch.export which sees the fake shapes.
        #
        # With dequantize=True, HuggingFace unpacks mxfp4 to bfloat16 and transposes
        # weights to model format (via convert_moe_packed_tensors in mxfp4.py), keeping
        # the original GptOssExperts module. This gives us real tensors with correct
        # shapes that torch.export can trace properly.
        quant_config = getattr(config, "quantization_config", None)
        if (
            quant_config is not None
            and quant_config.get("quant_method") == "mxfp4"
        ):
            from transformers import Mxfp4Config

            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print(
                "Using mxfp4 dequantization to get model-format weight shapes"
            )

        auto_cls = _get_auto_model_class(config)
        load_kwargs["trust_remote_code"] = True
        model = auto_cls.from_pretrained(model_name, **load_kwargs)
    else:
        model = init_model_no_weights(
            model_name,
            config,
            use_meta_device=use_meta_device,
            dtype=dtype,
        )

    # Patch MoE models for torch.export compatibility
    model_patch.patch_model(model)
    patch_minimax_m2_for_export(model)

    # Patch Qwen3.5 _update_linear_attn_mask: the original checks
    # data-dependent conditions (cache_position[0] > 0, torch.all)
    # which torch.export cannot handle. For prefill export we
    # always pass the mask through.
    import types as _types

    for module in model.modules():
        if hasattr(module, "_update_linear_attn_mask"):

            def _static_linear_attn_mask(self, attention_mask, cache_position):
                return attention_mask

            module._update_linear_attn_mask = _types.MethodType(
                _static_linear_attn_mask, module
            )

    # Build tokenizer — try model path first, fall back to tokenizer_name
    # (HuggingFace ID) when local weights lack tokenizer files.
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    except (OSError, ValueError, TypeError):
        if tokenizer_name:
            print(
                f"Tokenizer not found at {model_name}, downloading from {tokenizer_name}"
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
        else:
            raise
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Adding [PAD] token to tokenizer")
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    model_device = next(model.parameters()).device
    is_meta = model_device.type == "meta"

    # For reasons that remain mysterious, Torch refuses to treat
    # the batch size dimension as dynamic when instantiated at 1.
    # Consequently we expand to 13, a lucky prime.
    BATCH_SZ = 13
    sample_text = BATCH_SZ * ["This is a sample input for model export."]
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )

    batch_size = torch.export.Dim("batch_size", min=1, max=1024)
    seq_len = torch.export.Dim("seq_len", min=1, max=max_seq_length)
    input_ids = inputs["input_ids"]
    _, actual_seq_len = input_ids.shape

    if is_meta:
        input_ids = torch.empty(
            input_ids.shape, dtype=input_ids.dtype, device="meta"
        )

    text_config = get_model_config_safely(model)

    if is_encoder:
        # Encoder-only models (BERT, DistilBERT, etc.) use a
        # simple 2D attention mask [batch_size, seq_len] where
        # 1 = attend, 0 = ignore.
        attention_mask = torch.ones(
            (BATCH_SZ, actual_seq_len),
            dtype=torch.long,
            device=model_device,
        )
        if is_meta:
            attention_mask = torch.empty(
                attention_mask.shape,
                dtype=attention_mask.dtype,
                device="meta",
            )
        export_args = (input_ids, attention_mask)
        dynamic_shapes = (
            {0: batch_size, 1: seq_len},
            {0: batch_size, 1: seq_len},
        )
        metadata: dict[str, Any] = {
            "attention_masks": {"attention_mask": {"type": "bidirectional"}}
        }
    else:
        export_args, dynamic_shapes, metadata = _build_attention_mask(
            model=model,
            text_config=text_config,
            input_ids=input_ids,
            batch_size_dim=batch_size,
            seq_len_dim=seq_len,
            batch_sz=BATCH_SZ,
            actual_seq_len=actual_seq_len,
            dtype=dtype,
        )

    try:
        exported_model = torch.export.export(
            ExportableModel(model, is_encoder_only=is_encoder),
            args=export_args,
            dynamic_shapes=dynamic_shapes,
        )
    except Exception as e:
        err_str = str(e)
        if "Constraints violated" in err_str and "seq_len" in err_str:
            # Models with Conv1d (e.g. Qwen3.5 GatedDeltaNet)
            # specialize seq_len. Retry with static seq_len.
            print(
                "seq_len specialization detected, retrying with static seq_len"
            )
            static_shapes = _replace_dim_static(dynamic_shapes, seq_len)
            exported_model = torch.export.export(
                ExportableModel(model, is_encoder_only=is_encoder),
                args=export_args,
                dynamic_shapes=static_shapes,
            )
        else:
            raise

    # Architectural descriptors for Ingest.
    CONFIG_KEYS = {
        "hidden_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "vocab_size",
        "head_dim",
    }
    metadata["config"] = {
        k: getattr(text_config, k, None) for k in CONFIG_KEYS
    }

    # For encoder-only models, num_key_value_heads equals
    # num_attention_heads and may not be a config attribute.
    if metadata["config"].get("num_key_value_heads") is None:
        metadata["config"]["num_key_value_heads"] = metadata["config"][
            "num_attention_heads"
        ]

    # Additional config for model feature tag computation
    TAG_KEYS = {
        "num_local_experts",
        "sliding_window",
        "kv_lora_rank",
        "model_type",
        "num_experts_per_tok",
    }
    metadata["tag_config"] = {
        k: getattr(text_config, k, None) for k in TAG_KEYS
    }

    if metadata["config"].get("head_dim") is None:
        metadata["config"]["head_dim"] = (
            text_config.hidden_size // text_config.num_attention_heads
        )

    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        if "rope_type" in rope_scaling and "type" not in rope_scaling:
            rope_scaling["type"] = rope_scaling.pop("rope_type")
        metadata["config"]["rope_scaling"] = rope_scaling

    # rope_theta: direct attribute or nested in rope_parameters
    rope_theta = None
    if hasattr(text_config, "rope_theta"):
        rope_theta = text_config.rope_theta
    elif hasattr(text_config, "rope_parameters"):
        rope_params = text_config.rope_parameters
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            rope_theta = rope_params["rope_theta"]

    if rope_theta is None and not is_encoder:
        raise ValueError("failed to identify RoPE theta for decoder model")
    # Encoder-only models without RoPE get a sentinel value.
    metadata["config"]["rope_theta"] = (
        rope_theta if rope_theta is not None else 0.0
    )
    metadata["config"]["is_encoder_only"] = is_encoder

    return exported_model, metadata


def argument_spec_to_dict(x: ArgumentSpec) -> dict:
    import torch.export.graph_signature as gs

    TYPES = {
        gs.ConstantArgument: "constant",
        gs.TensorArgument: "tensor",
        gs.TokenArgument: "token",
    }
    out = {"type": TYPES.get(x.__class__)}
    if hasattr(x, "name"):
        out["name"] = x.name
    if isinstance(x, gs.ConstantArgument):
        out["value"] = x.value

    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="export.fxmodel",
        help="Model output directory",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load actual model weights (slower, more memory)",
    )
    parser.add_argument(
        "--meta-device",
        action="store_true",
        help="Use meta device for zero memory footprint "
        "(experimental, requires PyTorch 2.3+)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Model dtype (bfloat16 required for MoE models using _grouped_mm)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace model ID for tokenizer (when local weights lack tokenizer files)",
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    with torch.no_grad():
        exported, metadata = export_model(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_weights=args.load_weights,
            use_meta_device=args.meta_device,
            dtype=dtype,
            tokenizer_name=args.tokenizer,
        )

    # Note: We do NOT decompose layer_norm or other ops here.
    # Decomposing layer_norm introduces var_mean/prims ops
    # that FxTypedFx also doesn't support. It's simpler to
    # add layer_norm support to FxTypedFx directly.

    sig = exported.graph_signature

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "root.fx").write_text(str(exported.graph))

    # Extract buffer sizes
    buffers = [
        {
            "input_name": spec.arg.name,
            "shape": list(buf.shape),
            "dtype": str(buf.dtype),
        }
        for spec in sig.input_specs
        if spec.kind == InputKind.BUFFER
        if hasattr(spec.arg, "name")
        for buf in [
            exported.state_dict.get(spec.target),
            exported.constants.get(spec.target),
        ]
        if buf is not None
    ]

    # Find parameters (keyed on input name)
    parameters: list[dict] = [
        {
            "input_name": spec.arg.name,
            "target": spec.target.removeprefix("model."),
            "shape": param.shape,
            "dtype": str(param.dtype),
        }
        for spec in sig.input_specs
        if spec.kind == InputKind.PARAMETER
        if hasattr(spec.arg, "name")
        for param in [exported.state_dict[spec.target]]
    ]

    constants: list[dict] = [
        {
            "input_name": spec.arg.name,
            "shape": param.shape,
            "dtype": str(param.dtype),
            "value": (float(param) if len(param.shape) == 0 else list(param)),
        }
        for spec in sig.input_specs
        if spec.kind == InputKind.CONSTANT_TENSOR
        if hasattr(spec.arg, "name")
        for param in [exported.constants[spec.target]]
    ]

    # Build input metadata from graph nodes
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

    user_inputs: list[dict] = [
        {
            "input_name": spec.arg.name,
            "shape": input_metadata.get(spec.arg.name, {}).get("shape"),
            "dtype": input_metadata.get(spec.arg.name, {}).get("dtype"),
        }
        for spec in sig.input_specs
        if spec.kind == InputKind.USER_INPUT
        if hasattr(spec.arg, "name")
    ]

    metadata.update(
        {
            "parameters": parameters,
            "user_inputs": user_inputs,
            "buffers": buffers,
            "constants": constants,
            "range_constraints": {
                k.name: (str(v.lower), str(v.upper))
                for k, v in exported.range_constraints.items()
            },
            "graph": "root.fx",
            "submodules": [],
        }
    )

    # Export submodules recursively
    def export_submodules(module, parents):
        for k, v in module._modules.items():
            new_parents = parents + [k]
            if hasattr(v, "graph"):
                fname = f"{'_'.join(new_parents)}.fx"
                submod_output = args.output / fname
                submod_output.write_text(str(v.graph))
                metadata["submodules"].append((new_parents, fname))

            if hasattr(v, "_modules") and len(v._modules) > 0:
                export_submodules(v, new_parents)

    export_submodules(exported.graph_module, ["root"])

    json.dump(
        metadata,
        (args.output / "metadata.json").open("w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
