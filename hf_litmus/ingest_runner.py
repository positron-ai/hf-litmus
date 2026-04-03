from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def get_ingest_version(ingest_dir: Path) -> str:
    """Get git commit hash for ingest version tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ingest_dir.parent),
        )
        if result.returncode == 0:
            return f"git-{result.stdout.strip()}"
        return "unknown"
    except Exception:
        return "unknown"


@dataclass
class IngestResult:
    success: bool
    stdout: str
    stderr: str
    timed_out: bool = False
    error_message: str = ""
    hpp_generated: bool = False


class IngestRunner:
    def __init__(
        self,
        ingest_dir: Path,
        timeout: int = 300,
        dump_intermediates: bool = False,
    ) -> None:
        self.ingest_dir = ingest_dir
        self.timeout = timeout
        self.dump_intermediates = dump_intermediates
        self.version = get_ingest_version(ingest_dir)
        self.cabal_path = os.environ.get("CABAL_INSTALL", "cabal")
        self.ghc_path = os.environ.get("GHC", "ghc")

    def run_ingest(
        self,
        trace_dir: Path,
        model_name: str,
    ) -> IngestResult:
        """Run cabal run ingest on the trace directory."""
        with tempfile.TemporaryDirectory(
            prefix="litmus_ingest_"
        ) as tmp_output:
            output_dir = Path(tmp_output)

            # When cabal/ghc aren't on PATH (e.g. Nix deployment),
            # wrap in `nix develop` using the Tron clone's flake.
            # Use path: scheme to avoid shallow-clone git errors.
            tron_root = self.ingest_dir.parent
            nix_cmd = shutil.which("nix")
            cabal_available = shutil.which(self.cabal_path) is not None
            if (
                not cabal_available
                and nix_cmd
                and (tron_root / "flake.nix").exists()
            ):
                nix_prefix = [
                    nix_cmd,
                    "develop",
                    f"path:{tron_root}",
                    "--command",
                ]
            else:
                nix_prefix = []

            cmd = [
                *nix_prefix,
                self.cabal_path,
                "run",
                "ingest",
                "-w",
                self.ghc_path,
                "--",
                "--model-name",
                model_name,
                "--output-dir",
                str(output_dir),
                "--torch-trace-directory",
                str(trace_dir),
            ]

            if self.dump_intermediates:
                cmd.append("--dump-all")

            logger.info("Running ingest on %s", trace_dir)
            logger.debug("Command: %s", " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(self.ingest_dir),
                )

                hpp_files = list(output_dir.glob("*.hpp"))
                hpp_generated = len(hpp_files) > 0 and result.returncode == 0

                return IngestResult(
                    success=hpp_generated,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    hpp_generated=hpp_generated,
                    error_message=(result.stderr if not hpp_generated else ""),
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
                return IngestResult(
                    success=False,
                    stdout=out,
                    stderr=err,
                    timed_out=True,
                    error_message=(f"Ingest timed out after {self.timeout}s"),
                )
            except FileNotFoundError as e:
                return IngestResult(
                    success=False,
                    stdout="",
                    stderr=str(e),
                    error_message=(
                        f"Cabal/GHC not found: {e}. "
                        "Run from 'nix develop' or install GHC."
                    ),
                )
