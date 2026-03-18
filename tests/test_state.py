"""Tests for the StateManager persistence layer."""

from __future__ import annotations

from hf_litmus.models import ModelResult, ModelStatus
from hf_litmus.state import StateManager


class TestStateManager:
    def test_load_creates_dir(self, tmp_path):
        out = tmp_path / "subdir"
        sm = StateManager(out)
        sm.load()
        assert out.exists()
        assert sm.count == 0

    def test_save_and_reload(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.load()
        sm.update(ModelResult(model_id="a/b", status=ModelStatus.SUCCESS))
        sm.save()

        sm2 = StateManager(tmp_path)
        sm2.load()
        assert sm2.count == 1
        r = sm2.get_result("a/b")
        assert r is not None
        assert r.status == ModelStatus.SUCCESS

    def test_is_processed(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.load()
        assert not sm.is_processed("x/y")
        sm.update(ModelResult(model_id="x/y", status=ModelStatus.SKIP))
        assert sm.is_processed("x/y")

    def test_get_failed_models(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.load()
        sm.update(ModelResult(model_id="ok", status=ModelStatus.SUCCESS))
        sm.update(ModelResult(model_id="bad", status=ModelStatus.EXPORT_FAIL))
        sm.update(ModelResult(model_id="timeout", status=ModelStatus.TIMEOUT))
        failed = sm.get_failed_models()
        assert "bad" in failed
        assert "timeout" in failed
        assert "ok" not in failed

    def test_flush_if_dirty(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.load()
        sm.update(ModelResult(model_id="d", status=ModelStatus.SUCCESS))
        assert sm._dirty
        sm.flush_if_dirty()
        assert not sm._dirty
        assert sm.state_file.exists()

    def test_corrupted_state_backed_up(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("{invalid json")
        sm = StateManager(tmp_path)
        sm.load()
        assert sm.count == 0
        backups = list(tmp_path.glob("*.bak"))
        assert len(backups) == 1

    def test_all_results_snapshot(self, tmp_path):
        sm = StateManager(tmp_path)
        sm.load()
        sm.update(ModelResult(model_id="a", status=ModelStatus.SUCCESS))
        sm.update(ModelResult(model_id="b", status=ModelStatus.SKIP))
        results = sm.all_results()
        assert len(results) == 2
        ids = {r.model_id for r in results}
        assert ids == {"a", "b"}
