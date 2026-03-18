from __future__ import annotations


class LitmusError(Exception):
    """Base exception for HF Litmus."""


class ConfigurationError(LitmusError):
    """Invalid configuration or CLI arguments."""


class DependencyError(LitmusError):
    """Missing required dependency (uv, cabal, ghc)."""
