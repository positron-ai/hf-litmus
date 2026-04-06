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
from typing import ClassVar, Optional

from .models import ModelResult
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
    return (
        re.sub(
            r"[^a-zA-Z0-9_-]",
            "-",
            model_id.replace("/", "-"),
        )
        .lower()
        .strip("-")
    )


class DeepAnalyzer:
    def __init__(
        self,
        tron_url: str,
        output_dir: Path,
        timeout: int = 900,
        consensus_review: bool = False,
        tron_dir: Path | None = None,
    ) -> None:
        self.tron_url = tron_url
        self.output_dir = output_dir
        self.timeout = timeout
        self.consensus_review = consensus_review
        self.tron_dir = tron_dir or Path("./tron")
        self._addendum = load_prompt_template()

    def analyze(
        self,
        model_id: str,
        result: ModelResult,
    ) -> AnalysisResult:
        """Run deep analysis on a failed model.

        Uses a persistent Tron clone at ``tron_dir/.repo`` and
        creates a per-model git worktree at
        ``tron_dir/<sanitized-model-id>``.  Worktrees are
        preserved after analysis so the user can inspect the
        intermediate modifications that Claude Code made.
        """
        sanitized = _sanitize_model_id(model_id)
        branch = f"litmus/{sanitized}"
        analysis_dir = self.output_dir / "analyses" / sanitized

        # Clear stale artifacts so a failed re-run doesn't
        # surface results from a previous attempt.
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir, ignore_errors=True)

        logger.info("Starting deep analysis of %s", model_id)

        try:
            tron_path = self._ensure_tron_clone()
        except Exception as e:
            return AnalysisResult(
                model_id=model_id,
                error=f"Tron clone failed: {e}",
            )

        worktree_path = self.tron_dir / sanitized

        try:
            self._prepare_worktree(tron_path, worktree_path, branch)
        except Exception as e:
            logger.error(
                "Failed to create worktree for %s: %s",
                model_id,
                e,
            )
            return AnalysisResult(
                model_id=model_id,
                error=f"Worktree creation failed: {e}",
            )

        prompt = self._build_prompt(
            model_id,
            result,
            worktree_path,
            analysis_dir,
        )

        try:
            run_error = self._run_claude_code(worktree_path, prompt)
        except Exception as e:
            logger.error(
                "Claude Code failed for %s: %s",
                model_id,
                e,
            )
            return AnalysisResult(
                model_id=model_id,
                worktree_path=worktree_path,
                worktree_branch=branch,
                error=f"Claude Code failed: {e}",
            )

        if run_error:
            logger.warning(
                "Claude Code error for %s: %s",
                model_id,
                run_error,
            )

        ar = self._collect_results(
            model_id,
            worktree_path,
            branch,
            analysis_dir,
        )
        if run_error and not ar.error:
            ar.error = run_error

        # Worktree is intentionally preserved for inspection.
        logger.info(
            "Worktree preserved at %s (branch %s)",
            worktree_path,
            branch,
        )

        return ar

    def _ensure_tron_clone(self) -> Path:
        """Clone or update the persistent Tron checkout.

        The clone lives at ``tron_dir/.repo``.  On first call
        it is created with ``git clone --depth=1``.  On
        subsequent calls the existing clone is fetched and
        reset to the latest remote HEAD so new worktrees
        branch from current Tron code.
        """
        clone_path = self.tron_dir / ".repo"
        self.tron_dir.mkdir(parents=True, exist_ok=True)

        if (clone_path / ".git").is_dir():
            logger.info("Updating existing Tron clone at %s", clone_path)
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--depth=1",
                    "origin",
                ],
                cwd=str(clone_path),
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "reset", "--hard", "FETCH_HEAD"],
                cwd=str(clone_path),
                capture_output=True,
                text=True,
            )
        else:
            logger.info("Cloning Tron to %s", clone_path)
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    self.tron_url,
                    str(clone_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Tron clone failed: {result.stderr}")

        return clone_path

    def _prepare_worktree(
        self,
        tron_root: Path,
        worktree_path: Path,
        branch: str,
    ) -> None:
        """Create a fresh git worktree, removing any prior one.

        If a worktree or branch from a previous analysis
        already exists, they are cleaned up first so the new
        analysis starts from a clean Tron HEAD.
        """
        # Remove existing worktree if present
        if worktree_path.exists():
            logger.info(
                "Removing previous worktree at %s",
                worktree_path,
            )
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_path),
                ],
                cwd=str(tron_root),
                capture_output=True,
                text=True,
            )
            # Belt-and-suspenders: directory may linger if
            # the worktree metadata was already gone.
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)

        # Prune stale worktree bookkeeping
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(tron_root),
            capture_output=True,
            text=True,
        )

        # Delete the branch so it can be recreated from HEAD
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=str(tron_root),
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "-b",
                branch,
                str(worktree_path),
                "HEAD",
            ],
            cwd=str(tron_root),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {result.stderr}")

        logger.info(
            "Created worktree at %s on branch %s",
            worktree_path,
            branch,
        )

    @staticmethod
    def _read_ingest_command(worktree_path: Path) -> str:
        """Read the /ingest command template from the Tron worktree.

        Returns the command body with YAML front matter stripped.
        Raises FileNotFoundError if the command file is missing.
        """
        cmd_path = worktree_path / ".claude" / "commands" / "ingest.md"
        if not cmd_path.exists():
            raise FileNotFoundError(
                f"Tron /ingest command not found at {cmd_path}. "
                "Ensure the Tron clone includes "
                ".claude/commands/ingest.md"
            )
        text = cmd_path.read_text(encoding="utf-8")
        # Strip YAML front matter (---\n...\n---)
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3 :].lstrip()
        return text

    def _build_prompt(
        self,
        model_id: str,
        result: ModelResult,
        worktree_path: Path,
        analysis_dir: Path,
    ) -> str:
        """Build a prompt that combines the Tron /ingest command
        with the HF-Litmus addendum.

        Reads .claude/commands/ingest.md from the Tron worktree,
        substitutes the model ID, then appends the litmus-specific
        addendum with failure context and output requirements.
        """
        # Read and populate the Tron /ingest command
        ingest_cmd = self._read_ingest_command(worktree_path)
        ingest_cmd = ingest_cmd.replace("$ARGUMENTS", model_id)

        # Build variables for the litmus addendum
        failure_stage = (
            result.failure_stage.value if result.failure_stage else "unknown"
        )
        failure_class = (
            result.failure_class.value if result.failure_class else "unknown"
        )

        # Truncate error to avoid overwhelming the prompt
        error = result.error_output
        if len(error) > 8000:
            error = error[:4000] + "\n...[truncated]...\n" + error[-4000:]
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

        # Conditionally include consensus review
        if self.consensus_review:
            variables["consensus_section"] = render_template(
                self._addendum["consensus_review"],
                variables,
            )
        else:
            variables["consensus_section"] = ""

        addendum = render_template(
            self._addendum["main"],
            variables,
        )

        return ingest_cmd + "\n\n" + addendum

    _RATE_LIMIT_PATTERNS: ClassVar[list[str]] = [
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
                "npx",
                "@anthropic-ai/claude-code",
            ]
        else:
            claude_cmd_parts = [claude_cmd]

        # Wrap in `nix develop` so Tron build tools (cabal, ghc,
        # etc.) are available to Claude Code during analysis.
        # Use path: scheme so Nix treats the worktree as a plain
        # directory — the shallow clone is missing git objects that
        # git+file:// would try to resolve from flake.lock.
        nix_cmd = shutil.which("nix")
        if nix_cmd and (worktree_path / "flake.nix").exists():
            nix_prefix = [
                nix_cmd,
                "develop",
                f"path:{worktree_path}",
                "--command",
            ]
        else:
            nix_prefix = []

        cmd = [
            *nix_prefix,
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
            # stderr may be empty; stdout often has the real error
            detail = result.stderr.strip() or result.stdout.strip()
            logger.warning(
                "Claude Code exited with code %d:\n%s",
                result.returncode,
                detail[:1000],
            )
            # Check for rate limit errors
            combined = result.stdout + "\n" + result.stderr
            for pattern in self._RATE_LIMIT_PATTERNS:
                if re.search(pattern, combined, re.IGNORECASE):
                    return (
                        f"rate_limit: Claude Code hit rate limit"
                        f" (exit {result.returncode})"
                    )
            return (
                f"Claude Code exited with code"
                f" {result.returncode}:"
                f" {detail[:500]}"
            )

        logger.info(
            "Claude Code output:\n%s",
            result.stdout[:2000],
        )
        return ""

    @staticmethod
    def _find_ingest_workdirs(model_id: str) -> list[Path]:
        """Find /tmp/ingest_* work directories for this model.

        The Tron /ingest command writes artifacts to
        /tmp/ingest_TRON_NAME/ where TRON_NAME is derived
        from the model ID with underscores.
        """
        tmp = Path(tempfile.gettempdir())
        if not tmp.exists():
            return []
        # Try both the sanitized name and the underscore variant
        # that the /ingest command uses for TRON_NAME.
        candidates: list[Path] = []
        for d in tmp.iterdir():
            if d.is_dir() and d.name.startswith("ingest_"):
                candidates.append(d)
        if not candidates:
            return []
        # Filter to directories that plausibly match this model
        model_lower = model_id.lower()
        # e.g. "meta-llama/Llama-3.2-1B" -> keywords: llama, 3, 2, 1b
        parts = re.split(r"[/\-_. ]+", model_lower)
        keywords = [p for p in parts if len(p) >= 2]
        matched: list[Path] = []
        for d in candidates:
            name = d.name.lower()
            if all(kw in name for kw in keywords):
                matched.append(d)
        return matched

    def _collect_results(
        self,
        model_id: str,
        worktree_path: Path,
        branch: str,
        analysis_dir: Path,
    ) -> AnalysisResult:
        """Collect results from the worktree and ingest work dirs."""
        ar = AnalysisResult(
            model_id=model_id,
            worktree_path=worktree_path,
            worktree_branch=branch,
        )

        # Get the diff from initial commit
        diff_result = subprocess.run(
            [
                "git",
                "diff",
                "HEAD~..HEAD",
                "--",
                ".",
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
                    "git",
                    "log",
                    "--oneline",
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

        # Discover /tmp/ingest_* work directories from the
        # Tron /ingest command for additional artifact sources.
        ingest_workdirs = self._find_ingest_workdirs(model_id)

        # Look for analysis.md in various locations
        sanitized = _sanitize_model_id(model_id)
        analysis_paths = [
            analysis_dir / "analysis.md",
            worktree_path
            / "hf-litmus"
            / "litmus-outputs"
            / f"{sanitized}-analysis.md",
            worktree_path
            / "litmus-outputs"
            / "analyses"
            / sanitized
            / "analysis.md",
            worktree_path
            / "hf-litmus"
            / "litmus-outputs"
            / "analyses"
            / sanitized
            / "analysis.md",
        ]
        # Also check ingest work directories
        for wd in ingest_workdirs:
            analysis_paths.append(wd / "analysis.md")
        for p in analysis_paths:
            if p.exists():
                ar.analysis_path = p
                break

        # Copy analysis to our output dir if it's in worktree
        if ar.analysis_path and not str(ar.analysis_path).startswith(
            str(self.output_dir)
        ):
            dest = analysis_dir / "analysis.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ar.analysis_path, dest)
            ar.analysis_path = dest

        # Try to read gap-summary.json
        gap_paths = [
            analysis_dir / "gap-summary.json",
            worktree_path
            / "hf-litmus"
            / "litmus-outputs"
            / "gap-summary.json",
            worktree_path
            / "litmus-outputs"
            / "analyses"
            / sanitized
            / "gap-summary.json",
            worktree_path
            / "hf-litmus"
            / "litmus-outputs"
            / "analyses"
            / sanitized
            / "gap-summary.json",
        ]
        # Also check ingest work directories
        for wd in ingest_workdirs:
            gap_paths.append(wd / "gap-summary.json")
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
                    ar.missing_kernels = data.get("missing_kernels", [])
                    ar.missing_features = data.get("missing_patterns", [])
                    ar.stages_attempted = data.get("fixes_applied", [])
                    ar.final_blocker = data.get("furthest_stage", "")
                    ar.consensus_review = data.get("consensus_review")
                    if self.consensus_review and not ar.consensus_review:
                        logger.warning(
                            "Consensus review was requested but"
                            " no consensus_review section found"
                            " in gap-summary.json for %s",
                            model_id,
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to parse gap-summary.json: %s", e)
                break

        # Copy the ingest run ledger if available
        for wd in ingest_workdirs:
            ledger = wd / "ledger.md"
            if ledger.exists():
                dest = analysis_dir / "ledger.md"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ledger, dest)
                break

        return ar
