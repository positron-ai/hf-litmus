"""Tests for model data structures and serialization."""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given
from hypothesis import strategies as st

from hf_litmus.models import (
    FailureClass,
    FailureOrigin,
    FailureStage,
    FurthestStage,
    GapAnalysisStatus,
    ModelResult,
    ModelStatus,
    Verdict,
)


class TestEnums:
    def test_model_status_values(self):
        assert ModelStatus.SUCCESS.value == "SUCCESS"
        assert ModelStatus.EXPORT_FAIL.value == "EXPORT_FAIL"
        assert ModelStatus.INGEST_FAIL.value == "INGEST_FAIL"
        assert ModelStatus.TIMEOUT.value == "TIMEOUT"
        assert ModelStatus.SKIP.value == "SKIP"

    def test_failure_class_values(self):
        assert FailureClass.MISSING_OP.value == "missing_op"
        assert FailureClass.UNKNOWN.value == "unknown"

    def test_verdict_values(self):
        assert Verdict.PASS.value == "pass"
        assert Verdict.CLOSE.value == "close"
        assert Verdict.FAR.value == "far"
        assert Verdict.BLOCKED.value == "blocked"

    def test_furthest_stage_ordering(self):
        stages = [s.value for s in FurthestStage]
        assert "export" in stages
        assert "success" in stages

    def test_gap_analysis_status(self):
        assert GapAnalysisStatus.NOT_RUN.value == "not_run"
        assert GapAnalysisStatus.COMPLETED.value == "completed"


class TestModelResult:
    def test_minimal_construction(self):
        r = ModelResult(model_id="test/model", status=ModelStatus.SUCCESS)
        assert r.model_id == "test/model"
        assert r.status == ModelStatus.SUCCESS
        assert r.failure_stage is None
        assert r.missing_ops == []

    def test_full_construction(self):
        now = datetime.now(timezone.utc)
        r = ModelResult(
            model_id="org/name",
            status=ModelStatus.EXPORT_FAIL,
            failure_stage=FailureStage.EXPORT,
            failure_class=FailureClass.MISSING_OP,
            missing_ops=["aten.scatter"],
            tested_at=now,
            failure_origin=FailureOrigin.TRON_PIPELINE,
            retryable=False,
            model_tags=["MHA", "RoPE"],
        )
        assert r.failure_stage == FailureStage.EXPORT
        assert r.missing_ops == ["aten.scatter"]
        assert r.model_tags == ["MHA", "RoPE"]

    def test_roundtrip_serialization(self):
        now = datetime.now(timezone.utc)
        original = ModelResult(
            model_id="test/model",
            status=ModelStatus.INGEST_FAIL,
            failure_stage=FailureStage.INGEST,
            failure_class=FailureClass.TYPE_ERROR,
            tested_at=now,
            failure_origin=FailureOrigin.TRON_PIPELINE,
        )
        d = original.to_dict()
        restored = ModelResult.from_dict(d)
        assert restored.model_id == original.model_id
        assert restored.status == original.status
        assert restored.failure_stage == original.failure_stage
        assert restored.failure_class == original.failure_class
        assert restored.failure_origin == original.failure_origin

    def test_to_dict_structure(self):
        r = ModelResult(model_id="a/b", status=ModelStatus.SUCCESS)
        d = r.to_dict()
        assert d["model_id"] == "a/b"
        assert d["status"] == "SUCCESS"
        assert d["failure_stage"] is None
        assert d["missing_ops"] == []
        assert isinstance(d["tested_at"], str)

    def test_from_dict_missing_optional_fields(self):
        d = {
            "model_id": "test/m",
            "status": "SUCCESS",
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }
        r = ModelResult.from_dict(d)
        assert r.model_id == "test/m"
        assert r.failure_stage is None
        assert r.failure_class is None
        assert r.missing_ops == []
        assert r.model_tags == []
        assert r.retryable is False


# -- Hypothesis property tests --


@given(
    model_id=st.text(min_size=1, max_size=100),
    status=st.sampled_from(list(ModelStatus)),
)
def test_fuzz_roundtrip(model_id, status):
    """Property: any ModelResult survives serialization round-trip."""
    r = ModelResult(model_id=model_id, status=status)
    d = r.to_dict()
    restored = ModelResult.from_dict(d)
    assert restored.model_id == r.model_id
    assert restored.status == r.status
