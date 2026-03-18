from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import FailureClass, FailureStage, ModelResult
from .prompts import load_prompt_template, render_template

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
  """Result of deep analysis on a failed model."""
  model_id: str
  analysis_path: Optional[Path] = None
  diff: str = ""
  worktree_path: Optional[Path] = None
  worktree_branch: str = ""
  stages_attempted: list[str] = field(default_factory=list)
  final_blocker: str = ""
  missing_ops: list[str] = field(default_factory=list)
  missing_kernels: list[str] = field(default_factory=list)
  missing_features: list[str] = field(default_factory=list)
  consensus_review: Optional[dict] = None
  error: str = ""


def _normalize_ops(raw_ops: list) -> list[str]:
  """Normalize missing_ops entries to plain strings.

  Gap-summary.json entries may be plain strings or dicts
  with 'op' or 'name' keys.
  """
  result: list[str] = []
  for entry in raw_ops:
    if isinstance(entry, str):
      if entry:
        result.append(entry)
    elif isinstance(entry, dict):
      name = entry.get("op") or entry.get("name") or ""
      if name:
        result.append(name)
  return result


def _sanitize_model_id(model_id: str) -> str:
  """Convert model_id to a filesystem-safe name."""
  return re.sub(
    r"[^a-zA-Z0-9_-]", "-",
    model_id.replace("/", "-"),
  ).lower().strip("-")


# Stage guidance section names keyed by (FailureStage, FailureClass).
_STAGE_SECTION_MAP: dict[
  tuple[Optional[FailureStage], Optional[FailureClass]],
  str,
] = {
  (FailureStage.EXPORT, FailureClass.TRUST_REMOTE_CODE):
    "stage_trust_remote_code",
  (FailureStage.EXPORT, FailureClass.UNSUPPORTED_DYNAMIC):
    "stage_unsupported_dynamic",
}


def _stage_section_name(
  stage: Optional[FailureStage],
  fclass: Optional[FailureClass],
) -> str:
  """Map failure stage/class to a template section name."""
  key = (stage, fclass)
  if key in _STAGE_SECTION_MAP:
    return _STAGE_SECTION_MAP[key]
  if stage == FailureStage.EXPORT:
    return "stage_export_default"
  if stage == FailureStage.INGEST:
    return "stage_ingest"
  return ""


class DeepAnalyzer:
  def __init__(
    self,
    tron_url: str,
    output_dir: Path,
    timeout: int = 900,
    consensus_review: bool = False,
  ) -> None:
    self.tron_url = tron_url
    self.output_dir = output_dir
    self.timeout = timeout
    self.consensus_review = consensus_review
    self._template = load_prompt_template()

  def analyze(
    self,
    model_id: str,
    result: ModelResult,
  ) -> AnalysisResult:
    """Run deep analysis on a failed model.

    Clones Tron into a temporary directory, creates a
    git worktree for the analysis, runs Claude Code,
    collects results, then removes the temp directory.
    """
    sanitized = _sanitize_model_id(model_id)
    branch = f"litmus/{sanitized}"
    analysis_dir = self.output_dir / "analyses" / sanitized

    logger.info(
      "Starting deep analysis of %s", model_id
    )

    tmpdir = tempfile.mkdtemp(prefix="litmus_analysis_")
    tron_path = Path(tmpdir) / "tron"
    worktree_path = Path(tmpdir) / f"hf-litmus-{sanitized}"

    try:
      logger.info(
        "Cloning Tron for analysis of %s", model_id
      )
      clone_result = subprocess.run(
        [
          "git", "clone", "--depth=1",
          self.tron_url, str(tron_path),
        ],
        capture_output=True,
        text=True,
      )
      if clone_result.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return AnalysisResult(
          model_id=model_id,
          error=(
            f"Tron clone failed: {clone_result.stderr}"
          ),
        )
    except Exception as e:
      shutil.rmtree(tmpdir, ignore_errors=True)
      return AnalysisResult(
        model_id=model_id,
        error=f"Tron clone failed: {e}",
      )

    try:
      self._setup_worktree(
        tron_path, worktree_path, branch
      )
    except Exception as e:
      logger.error(
        "Failed to create worktree for %s: %s",
        model_id, e,
      )
      shutil.rmtree(tmpdir, ignore_errors=True)
      return AnalysisResult(
        model_id=model_id,
        error=f"Worktree creation failed: {e}",
      )

    prompt = self._build_prompt(
      model_id, result, worktree_path, analysis_dir,
    )

    try:
      run_error = self._run_claude_code(
        worktree_path, prompt
      )
    except Exception as e:
      logger.error(
        "Claude Code failed for %s: %s",
        model_id, e,
      )
      shutil.rmtree(tmpdir, ignore_errors=True)
      return AnalysisResult(
        model_id=model_id,
        worktree_branch=branch,
        error=f"Claude Code failed: {e}",
      )

    if run_error:
      logger.warning(
        "Claude Code error for %s: %s",
        model_id, run_error,
      )

    ar = self._collect_results(
      model_id, worktree_path, branch, analysis_dir,
    )
    if run_error and not ar.error:
      ar.error = run_error

    # Results are already copied to analysis_dir;
    # clean up the temporary Tron clone.
    shutil.rmtree(tmpdir, ignore_errors=True)
    ar.worktree_path = None

    return ar

  def _setup_worktree(
    self,
    tron_root: Path,
    worktree_path: Path,
    branch: str,
  ) -> None:
    """Create a git worktree from a Tron clone."""
    result = subprocess.run(
      [
        "git", "worktree", "add",
        "-b", branch,
        str(worktree_path), "HEAD",
      ],
      cwd=str(tron_root),
      capture_output=True,
      text=True,
    )

    if result.returncode != 0:
      raise RuntimeError(
        f"git worktree add failed: {result.stderr}"
      )

    logger.info(
      "Created worktree at %s on branch %s",
      worktree_path, branch,
    )

  def _build_prompt(
    self,
    model_id: str,
    result: ModelResult,
    worktree_path: Path,
    analysis_dir: Path,
  ) -> str:
    """Build a detailed prompt for Claude Code.

    Loads the prompt from the Markdown template in
    hf_litmus/prompts/deep_analysis_prompt.md and
    renders it with model-specific variables.
    """
    failure_stage = (
      result.failure_stage.value
      if result.failure_stage else "unknown"
    )
    failure_class = (
      result.failure_class.value
      if result.failure_class else "unknown"
    )

    # Truncate error to avoid overwhelming the prompt
    error = result.error_output
    if len(error) > 8000:
      error = (
        error[:4000]
        + "\n...[truncated]...\n"
        + error[-4000:]
      )
    error = (
      "--- BEGIN ERROR OUTPUT ---\n"
      + error
      + "\n--- END ERROR OUTPUT ---"
    )

    sanitized = _sanitize_model_id(model_id)

    variables = {
      "model_id": model_id,
      "sanitized_model_id": sanitized,
      "failure_stage": failure_stage,
      "failure_class": failure_class,
      "downloads": f"{result.downloads:,}",
      "error": error,
      "analysis_dir": str(analysis_dir),
    }

    # Select and render stage guidance section
    section = _stage_section_name(
      result.failure_stage,
      result.failure_class,
    )
    if section and section in self._template:
      variables["stage_guidance"] = render_template(
        self._template[section], variables,
      )
    else:
      variables["stage_guidance"] = ""

    # Conditionally include consensus review
    if self.consensus_review:
      variables["consensus_section"] = (
        render_template(
          self._template["consensus_review"],
          variables,
        )
      )
    else:
      variables["consensus_section"] = ""

    return render_template(
      self._template["main"], variables,
    )

  _RATE_LIMIT_PATTERNS = [
    r"rate.?limit",
    r"429",
    r"Too Many Requests",
    r"quota",
    r"billing",
  ]

  def _run_claude_code(
    self,
    worktree_path: Path,
    prompt: str,
  ) -> str:
    """Launch Claude Code in the worktree.

    Returns an error string if the subprocess failed,
    empty string on success. Includes 'rate_limit'
    marker when a rate limit is detected.
    """
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
      # Fall back to npx
      claude_cmd_parts = [
        "npx", "@anthropic-ai/claude-code",
      ]
    else:
      claude_cmd_parts = [claude_cmd]

    cmd = [
      *claude_cmd_parts,
      "--dangerously-skip-permissions",
      "-p",
      prompt,
      "--verbose",
    ]

    logger.info(
      "Launching Claude Code in %s",
      worktree_path,
    )

    env = os.environ.copy()

    result = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      timeout=self.timeout,
      cwd=str(worktree_path),
      env=env,
    )

    if result.returncode != 0:
      logger.warning(
        "Claude Code exited with code %d: %s",
        result.returncode,
        result.stderr[:500],
      )
      # Check for rate limit errors
      combined = result.stdout + "\n" + result.stderr
      for pattern in self._RATE_LIMIT_PATTERNS:
        if re.search(
          pattern, combined, re.IGNORECASE
        ):
          return (
            f"rate_limit: Claude Code hit rate limit"
            f" (exit {result.returncode})"
          )
      return (
        f"Claude Code exited with code"
        f" {result.returncode}:"
        f" {result.stderr[:200]}"
      )

    logger.info(
      "Claude Code output:\n%s",
      result.stdout[:2000],
    )
    return ""

  def _collect_results(
    self,
    model_id: str,
    worktree_path: Path,
    branch: str,
    analysis_dir: Path,
  ) -> AnalysisResult:
    """Collect results from the worktree."""
    ar = AnalysisResult(
      model_id=model_id,
      worktree_path=worktree_path,
      worktree_branch=branch,
    )

    # Get the diff from initial commit
    diff_result = subprocess.run(
      [
        "git", "diff", "HEAD~..HEAD",
        "--", ".",
      ],
      capture_output=True,
      text=True,
      cwd=str(worktree_path),
    )
    if diff_result.returncode == 0:
      ar.diff = diff_result.stdout

    # If that didn't work, get all changes from parent
    if not ar.diff:
      diff_result = subprocess.run(
        [
          "git", "log", "--oneline",
          f"HEAD...{branch}@{{upstream}}",
        ],
        capture_output=True,
        text=True,
        cwd=str(worktree_path),
      )
      # Fall back: diff against the worktree base
      diff_result = subprocess.run(
        ["git", "diff", "HEAD~1..HEAD"],
        capture_output=True,
        text=True,
        cwd=str(worktree_path),
      )
      if diff_result.returncode == 0:
        ar.diff = diff_result.stdout

    # Look for analysis.md in various locations
    sanitized = _sanitize_model_id(model_id)
    analysis_paths = [
      analysis_dir / "analysis.md",
      worktree_path / "hf-litmus" / "litmus-output"
      / f"{sanitized}-analysis.md",
      worktree_path / "litmus-output" / "analyses"
      / sanitized / "analysis.md",
      worktree_path / "hf-litmus" / "litmus-output"
      / "analyses" / sanitized / "analysis.md",
    ]
    for p in analysis_paths:
      if p.exists():
        ar.analysis_path = p
        break

    # Copy analysis to our output dir if it's in worktree
    if (
      ar.analysis_path
      and not str(ar.analysis_path).startswith(
        str(self.output_dir)
      )
    ):
      dest = analysis_dir / "analysis.md"
      dest.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ar.analysis_path, dest)
      ar.analysis_path = dest

    # Try to read gap-summary.json
    gap_paths = [
      analysis_dir / "gap-summary.json",
      worktree_path / "hf-litmus" / "litmus-output"
      / "gap-summary.json",
      worktree_path / "litmus-output" / "analyses"
      / sanitized / "gap-summary.json",
      worktree_path / "hf-litmus" / "litmus-output"
      / "analyses" / sanitized / "gap-summary.json",
    ]
    for p in gap_paths:
      if p.exists():
        # Copy to our output dir if it's in worktree
        if not str(p).startswith(str(self.output_dir)):
          dest = analysis_dir / "gap-summary.json"
          dest.parent.mkdir(parents=True, exist_ok=True)
          shutil.copy2(p, dest)
        try:
          data = json.loads(p.read_text())
          raw_ops = data.get("missing_ops", [])
          ar.missing_ops = _normalize_ops(raw_ops)
          ar.missing_kernels = data.get(
            "missing_kernels", []
          )
          ar.missing_features = data.get(
            "missing_patterns", []
          )
          ar.stages_attempted = data.get(
            "fixes_applied", []
          )
          ar.final_blocker = data.get(
            "furthest_stage", ""
          )
          ar.consensus_review = data.get(
            "consensus_review"
          )
          if (
            self.consensus_review
            and not ar.consensus_review
          ):
            logger.warning(
              "Consensus review was requested but"
              " no consensus_review section found"
              " in gap-summary.json for %s",
              model_id,
            )
        except (json.JSONDecodeError, KeyError) as e:
          logger.warning(
            "Failed to parse gap-summary.json: %s", e
          )
        break

    return ar
