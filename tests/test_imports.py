"""Smoke tests: verify all package modules import cleanly."""

from __future__ import annotations


def test_import_package():
    import hf_litmus

    assert hf_litmus.__version__


def test_import_config():
    from hf_litmus.config import LitmusConfig

    assert LitmusConfig


def test_import_models():
    from hf_litmus.models import (
        FailureClass,
        FailureOrigin,
        FailureStage,
        FurthestStage,
        ModelResult,
        ModelStatus,
        Verdict,
    )

    assert ModelResult
    assert ModelStatus
    assert FailureClass
    assert FailureStage
    assert FailureOrigin
    assert FurthestStage
    assert Verdict


def test_import_error_classifier():
    from hf_litmus.error_classifier import (
        classify_export_error,
        classify_ingest_error,
    )

    assert classify_export_error
    assert classify_ingest_error


def test_import_exceptions():
    from hf_litmus.exceptions import (
        ConfigurationError,
        DependencyError,
        LitmusError,
    )

    assert LitmusError
    assert ConfigurationError
    assert DependencyError


def test_import_state():
    from hf_litmus.state import StateManager

    assert StateManager


def test_import_feature_areas():
    from hf_litmus.feature_areas import OP_FEATURE_MAP

    assert OP_FEATURE_MAP
