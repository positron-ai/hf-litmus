"""Tests for LitmusConfig defaults and construction."""

from __future__ import annotations

from pathlib import Path

from hf_litmus.config import DEFAULT_TRON_URL, LitmusConfig


def test_default_config():
    cfg = LitmusConfig()
    assert cfg.output_dir == Path("./litmus-outputs")
    assert cfg.batch_size == 100
    assert cfg.single_run is True
    assert cfg.deep_analysis is True
    assert cfg.max_jobs == 1
    assert cfg.tron_url == DEFAULT_TRON_URL
    assert cfg.tron_dir == Path("./tron")


def test_custom_config():
    cfg = LitmusConfig(
        output_dir=Path("/tmp/test"),
        batch_size=50,
        single_run=False,
        target_model="meta-llama/Llama-3.1-8B",
    )
    assert cfg.output_dir == Path("/tmp/test")
    assert cfg.batch_size == 50
    assert cfg.single_run is False
    assert cfg.target_model == "meta-llama/Llama-3.1-8B"


def test_default_tron_url():
    assert "positron-ai/tron" in DEFAULT_TRON_URL
    assert DEFAULT_TRON_URL.endswith(".git")
