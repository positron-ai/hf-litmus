from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    success: bool
    trace_dir: Optional[Path]
    stdout: str
    stderr: str
    timed_out: bool = False
    error_message: str = ""


class ExportRunner:
    def __init__(
        self,
        ingest_dir: Path,
        max_seq_length: int = 64,
        timeout: int = 600,
    ) -> None:
        self.ingest_dir = ingest_dir
        self.python_torch_dir = ingest_dir / "export"
        self.max_seq_length = max_seq_length
        self.timeout = timeout
        self.uv_path = self._find_uv()

    def _find_uv(self) -> str:
        """Find uv executable."""
        if uv := os.environ.get("UV"):
            return uv
        if uv := shutil.which("uv"):
            return uv
        if Path("/tools/uv/uv").exists():
            return "/tools/uv/uv"
        raise RuntimeError(
            "uv not found. Install from https://docs.astral.sh/uv/"
        )

    def export_model(
        self,
        model_id: str,
        trace_dir: Path,
    ) -> ExportResult:
        """Run torch.export via uv run torch-export."""
        trace_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.uv_path,
            "run",
            "--project",
            str(self.python_torch_dir),
            "torch-export",
            "--model",
            model_id,
            "--output",
            str(trace_dir),
            "--max-seq-length",
            str(self.max_seq_length),
            "--meta-device",
        ]

        logger.info("Exporting %s to %s", model_id, trace_dir)
        logger.debug("Command: %s", " ".join(cmd))

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)

        # When deployed via Nix, the project dir is inside the read-only
        # Nix store.  Point uv at a writable venv location so it doesn't
        # try to create .venv inside the store.
        if not os.access(self.python_torch_dir, os.W_OK):
            venv_dir = Path.home() / ".cache" / "hf-litmus" / "export-venv"
            venv_dir.mkdir(parents=True, exist_ok=True)
            env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=str(self.ingest_dir),
            )

            success = result.returncode == 0
            if success:
                root_fx = trace_dir / "root.fx"
                metadata = trace_dir / "metadata.json"
                if not (root_fx.exists() and metadata.exists()):
                    logger.warning("Export returned 0 but trace files missing")
                    success = False

            return ExportResult(
                success=success,
                trace_dir=trace_dir if success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                error_message=(result.stderr if not success else ""),
            )

        except subprocess.TimeoutExpired as e:
            # TimeoutExpired.stdout/stderr can be bytes even
            # with text=True; decode defensively.
            out = e.stdout or ""
            err = e.stderr or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            if isinstance(err, bytes):
                err = err.decode(errors="replace")
            return ExportResult(
                success=False,
                trace_dir=None,
                stdout=out,
                stderr=err,
                timed_out=True,
                error_message=(f"Export timed out after {self.timeout}s"),
            )
        except Exception as e:
            return ExportResult(
                success=False,
                trace_dir=None,
                stdout="",
                stderr=str(e),
                error_message=str(e),
            )
