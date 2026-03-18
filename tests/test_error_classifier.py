"""Tests for the error classification engine."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from hf_litmus.error_classifier import (
    ClassificationResult,
    _extract_error_summary,
    _extract_missing_ops,
    classify_export_error,
    classify_ingest_error,
)
from hf_litmus.models import FailureClass, FailureOrigin


class TestClassifyExportError:
    def test_timeout(self):
        r = classify_export_error("", "", timed_out=True)
        assert r.failure_class == FailureClass.UNKNOWN
        assert "timed out" in r.error_summary.lower()

    def test_gated_repo(self):
        r = classify_export_error(
            "", "GatedRepoError: access restricted", False
        )
        assert r.failure_origin == FailureOrigin.HF_ACCESS
        assert r.retryable is True

    def test_trust_remote_code(self):
        r = classify_export_error(
            "contains custom code which must be executed", "", False
        )
        assert r.failure_class == FailureClass.TRUST_REMOTE_CODE
        assert r.failure_origin == FailureOrigin.HF_ACCESS
        assert r.retryable is False

    def test_python_infra_error(self):
        r = classify_export_error(
            "", "ModuleNotFoundError: No module named 'foo'"
        )
        assert r.failure_origin == FailureOrigin.PYTHON_INFRA
        assert r.retryable is True

    def test_dynamic_shape(self):
        r = classify_export_error(
            "", "Could not guard on data-dependent expression"
        )
        assert r.failure_class == FailureClass.UNSUPPORTED_DYNAMIC
        assert r.failure_origin == FailureOrigin.TRON_PIPELINE

    def test_memory_error(self):
        r = classify_export_error("", "MemoryError")
        assert r.failure_class == FailureClass.MEMORY_ERROR

    def test_unknown_error(self):
        r = classify_export_error("something went wrong", "no match here")
        assert r.failure_class == FailureClass.UNKNOWN


class TestClassifyIngestError:
    def test_timeout(self):
        r = classify_ingest_error("", "", timed_out=True)
        assert r.failure_class == FailureClass.UNKNOWN
        assert r.failure_origin == FailureOrigin.TRON_PIPELINE

    def test_type_error(self):
        r = classify_ingest_error("Typechecking failed", "")
        assert r.failure_class == FailureClass.TYPE_ERROR
        assert r.failure_origin == FailureOrigin.TRON_PIPELINE

    def test_missing_op(self):
        r = classify_ingest_error("Unknown function: aten.scatter_add", "")
        assert r.failure_class == FailureClass.MISSING_OP
        assert "aten.scatter_add" in r.missing_ops

    def test_unknown_ingest_error(self):
        r = classify_ingest_error("unexpected output", "")
        assert r.failure_class == FailureClass.UNKNOWN
        assert r.failure_origin == FailureOrigin.TRON_PIPELINE


class TestExtractMissingOps:
    def test_unknown_function(self):
        ops = _extract_missing_ops("Unknown function: aten.scatter_add")
        assert "aten.scatter_add" in ops

    def test_unsupported_op(self):
        ops = _extract_missing_ops("Unsupported op: aten.index_put")
        assert "aten.index_put" in ops

    def test_multiple_ops(self):
        text = "Unknown function: scatter\nUnsupported op: gather"
        ops = _extract_missing_ops(text)
        assert len(ops) >= 2

    def test_no_ops(self):
        ops = _extract_missing_ops("everything is fine")
        assert ops == []


class TestExtractErrorSummary:
    def test_finds_error_line(self):
        text = "line1\nline2\nError: something broke\nline4"
        summary = _extract_error_summary(text)
        assert "Error: something broke" in summary

    def test_falls_back_to_tail(self):
        text = "line1\nline2\nline3"
        summary = _extract_error_summary(text, max_lines=2)
        assert "line3" in summary


# -- Hypothesis property tests --


@given(text=st.text(min_size=0, max_size=500))
def test_fuzz_classify_export_never_crashes(text):
    """Property: classify_export_error handles any input without crashing."""
    r = classify_export_error(text, text)
    assert isinstance(r, ClassificationResult)
    assert isinstance(r.failure_class, FailureClass)


@given(text=st.text(min_size=0, max_size=500))
def test_fuzz_classify_ingest_never_crashes(text):
    """Property: classify_ingest_error handles any input without crashing."""
    r = classify_ingest_error(text, text)
    assert isinstance(r, ClassificationResult)
    assert isinstance(r.failure_class, FailureClass)
