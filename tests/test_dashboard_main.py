from pathlib import Path

import pytest

from hf_litmus import dashboard


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record the output dir _serve_dashboard would serve, without serving."""
    calls: list[Path] = []

    def fake_serve(output_dir: Path, *_args: object, **_kwargs: object) -> int:
        calls.append(output_dir)
        return 0

    monkeypatch.setattr(dashboard, "_serve_dashboard", fake_serve)
    return calls


def test_dashboard_starts_before_first_report(
    tmp_path: Path, served: list[Path]
) -> None:
    out = tmp_path / "litmus-outputs"

    assert dashboard.dashboard_main(["--output-dir", str(out)]) == 0
    assert served == [out]
    assert (out / "reports").is_dir()


def test_dashboard_keeps_existing_reports(
    tmp_path: Path, served: list[Path]
) -> None:
    report = tmp_path / "reports" / "m.json"
    report.parent.mkdir()
    report.write_text('{"model_id": "m", "status": "pass"}')

    assert dashboard.dashboard_main(["--output-dir", str(tmp_path)]) == 0
    assert served == [tmp_path]
    assert report.read_text() == '{"model_id": "m", "status": "pass"}'
