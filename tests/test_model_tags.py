"""Tests for model tag extraction logic."""

from __future__ import annotations

from hf_litmus.model_tags import (
    _tags_from_config_attrs,
    load_tag_cache,
    save_tag_cache,
)


class _FakeConfig:
    """Minimal config-like object for testing tag extraction."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        return None


class TestTagsFromConfigAttrs:
    def test_mha(self):
        cfg = _FakeConfig(num_attention_heads=32, num_key_value_heads=32)
        tags = _tags_from_config_attrs(cfg)
        assert "MHA" in tags

    def test_gqa(self):
        cfg = _FakeConfig(num_attention_heads=32, num_key_value_heads=8)
        tags = _tags_from_config_attrs(cfg)
        assert "GQA" in tags

    def test_mqa(self):
        cfg = _FakeConfig(num_attention_heads=32, num_key_value_heads=1)
        tags = _tags_from_config_attrs(cfg)
        assert "MQA" in tags

    def test_mla(self):
        cfg = _FakeConfig(
            num_attention_heads=32, num_key_value_heads=8, kv_lora_rank=256
        )
        tags = _tags_from_config_attrs(cfg)
        assert "MLA" in tags
        assert "GQA" not in tags

    def test_moe(self):
        cfg = _FakeConfig(num_local_experts=8)
        tags = _tags_from_config_attrs(cfg)
        assert "MoE" in tags

    def test_rope_basic(self):
        cfg = _FakeConfig(rope_theta=10000.0)
        tags = _tags_from_config_attrs(cfg)
        assert "RoPE" in tags

    def test_rope_yarn(self):
        cfg = _FakeConfig(rope_scaling={"type": "yarn", "factor": 4.0})
        tags = _tags_from_config_attrs(cfg)
        assert "YaRN" in tags

    def test_rope_linear(self):
        cfg = _FakeConfig(rope_scaling={"type": "linear", "factor": 2.0})
        tags = _tags_from_config_attrs(cfg)
        assert "LinearRoPE" in tags

    def test_sliding_window(self):
        cfg = _FakeConfig(sliding_window=4096)
        tags = _tags_from_config_attrs(cfg)
        assert "SlidingWindow" in tags

    def test_empty_config(self):
        cfg = _FakeConfig()
        tags = _tags_from_config_attrs(cfg)
        assert tags == []

    def test_tags_sorted(self):
        cfg = _FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
            num_local_experts=8,
            rope_theta=10000.0,
            sliding_window=4096,
        )
        tags = _tags_from_config_attrs(cfg)
        assert tags == sorted(tags)


class TestTagCache:
    def test_save_and_load(self, tmp_path):
        cache = {"model/a": ["GQA", "RoPE"], "model/b": ["MHA"]}
        save_tag_cache(tmp_path, cache)
        loaded = load_tag_cache(tmp_path)
        assert loaded == cache

    def test_load_missing(self, tmp_path):
        loaded = load_tag_cache(tmp_path)
        assert loaded == {}

    def test_load_corrupted(self, tmp_path):
        (tmp_path / "tag_cache.json").write_text("{bad")
        loaded = load_tag_cache(tmp_path)
        assert loaded == {}
