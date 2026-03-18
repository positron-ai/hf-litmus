"""Haskell ingest pipeline runner.

This module provides functions to run the Haskell ingest tool
to generate Python code from FX traces.
"""

import os
import subprocess
from pathlib import Path


def _find_ingest_directory() -> Path:
    """Find the ingest directory containing the Haskell project.

    Checks multiple locations in order:
    1. Relative to this module (when running from source tree)
    2. Current working directory (when installed but run from source checkout)

    Returns:
      Path to the ingest directory containing cabal.project

    Raises:
      RuntimeError: If ingest directory cannot be found
    """
    markers = ["cabal.project", "src", "runtime"]

    # First try: relative to this module (source tree installation)
    # This module is at: ingest/runtime/tron_ingest_tools/ingest.py
    # Need to navigate up to: ingest/
    from_module = Path(__file__).parent.parent.parent
    if all((from_module / marker).exists() for marker in markers):
        return from_module

    # Second try: current working directory (pip installed, run from source checkout)
    cwd = Path.cwd()
    if all((cwd / marker).exists() for marker in markers):
        return cwd

    raise RuntimeError(
        "run_ingest_pipeline requires access to the tron ingest source tree.\n"
        "This function runs 'cabal run ingest' and needs the Haskell source.\n"
        "Either:\n"
        "  1. Run from the ingest directory: cd tron/ingest && python ...\n"
        "  2. Use the pre-generated Python files instead of running ingest"
    )


def run_ingest_pipeline(
    trace_dir: Path,
    model_name: str,
    output_dir: Path,
    dump_name: str = "model",
) -> Path:
    """Run the Haskell ingest tool to generate Python code.

    Args:
      trace_dir: Directory containing root.fx and metadata.json
      model_name: Name for the model (used in generated code)
      output_dir: Directory to write the generated .hpp file
      dump_name: Base name for dump files (Python will be {dump_name}.py)

    Returns:
      Path to generated Python file

    Raises:
      RuntimeError: If not running from source tree or if ingest fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ingest_dir = _find_ingest_directory()

    cabal = os.environ.get("CABAL_INSTALL", "cabal")
    cmd = [
        cabal,
        "run",
        "ingest",
        "--",
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--torch-trace-directory",
        str(trace_dir),
        "--dump-py",
        "--dump-name",
        dump_name,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ingest_dir,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Ingest pipeline failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    generated_py = ingest_dir / f"{dump_name}.py"
    if not generated_py.exists():
        alt_paths = [
            output_dir / f"{dump_name}.py",
            output_dir / f"{model_name}.py",
            Path(f"{dump_name}.py"),
        ]
        for alt in alt_paths:
            if alt.exists():
                generated_py = alt
                break
        else:
            raise RuntimeError(
                f"No Python file generated. Expected {generated_py}. Checked: {alt_paths}"
            )

    return generated_py
