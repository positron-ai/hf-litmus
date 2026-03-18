from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_tags_from_hf(model_id: str) -> list[str]:
    """Derive feature tags from a HuggingFace model config.

    Downloads the config from the HF Hub and extracts
    architectural features: attention type (MHA, GQA, MQA,
    MLA), MoE, position encoding (RoPE, YaRN, LinearRoPE),
    and SlidingWindow.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        return []

    return _tags_from_config_attrs(cfg)


def _tags_from_config_attrs(cfg: object) -> list[str]:
    """Extract tags from a config-like object."""
    tags: list[str] = []

    n_heads = getattr(cfg, "num_attention_heads", 0) or 0
    n_kv = getattr(cfg, "num_key_value_heads", 0) or 0

    if getattr(cfg, "kv_lora_rank", None):
        tags.append("MLA")
    elif n_kv and n_kv == 1:
        tags.append("MQA")
    elif n_kv and n_kv < n_heads:
        tags.append("GQA")
    elif n_heads:
        tags.append("MHA")

    if getattr(cfg, "num_local_experts", None):
        tags.append("MoE")

    rope = getattr(cfg, "rope_scaling", None)
    if isinstance(rope, dict):
        rtype = rope.get("type", "").lower()
        if "yarn" in rtype:
            tags.append("YaRN")
        elif "linear" in rtype:
            tags.append("LinearRoPE")
        else:
            tags.append("RoPE")
    elif getattr(cfg, "rope_theta", None):
        tags.append("RoPE")

    if getattr(cfg, "sliding_window", None):
        tags.append("SlidingWindow")

    return sorted(tags)


_TAG_CACHE_NAME = "tag_cache.json"


def load_tag_cache(output_dir: Path) -> dict[str, list[str]]:
    """Load cached model tags from disk."""
    cache_path = output_dir / _TAG_CACHE_NAME
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_tag_cache(output_dir: Path, cache: dict[str, list[str]]) -> None:
    """Save model tag cache to disk."""
    cache_path = output_dir / _TAG_CACHE_NAME
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def ensure_tags(reports: list[dict], output_dir: Path) -> None:
    """Populate model_tags on reports that lack them.

    Uses a disk cache to avoid repeated HF API calls.
    Modifies reports in-place.
    """
    cache = load_tag_cache(output_dir)
    dirty = False

    for r in reports:
        mid = r.get("model_id", "")
        if r.get("model_tags"):
            # Already has tags; update cache if needed
            if mid and mid not in cache:
                cache[mid] = r["model_tags"]
                dirty = True
            continue

        if not mid:
            continue

        # Check cache first
        if mid in cache:
            r["model_tags"] = cache[mid]
            continue

        # Compute from HF config
        tags = compute_tags_from_hf(mid)
        r["model_tags"] = tags
        cache[mid] = tags
        dirty = True
        if tags:
            logger.debug("Tags for %s: %s", mid, tags)

    if dirty:
        save_tag_cache(output_dir, cache)
