from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Optional

from .config import LitmusConfig
from .deep_analysis import _sanitize_model_id
from .error_classifier import (
    classify_export_error,
    classify_ingest_error,
)
from .exceptions import DependencyError
from .export_runner import ExportRunner
from .hf_enumeration import HFModelEnumerator, filter_by_state
from .ingest_runner import IngestRunner
from .model_tags import compute_tags_from_hf
from .models import (
    FailureOrigin,
    FailureStage,
    ModelResult,
    ModelStatus,
)
from .report_generator import (
    generate_model_metadata,
    generate_report,
)
from .state import StateManager
from .summary_generator import generate_summary

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps
_deep_analyzer = None


def check_dependencies() -> None:
    """Verify required tools are available.

    Only uv and git are hard requirements. cabal/ghc are
    checked at runtime and ingest is skipped if absent.
    """
    errors: list[str] = []

    if not (
        os.environ.get("UV")
        or shutil.which("uv")
        or Path("/tools/uv/uv").exists()
    ):
        errors.append("uv not found. Install from https://docs.astral.sh/uv/")

    if not shutil.which("git"):
        errors.append(
            "git not found. Install git to enable Tron"
            " cloning for ingest and deep analysis."
        )

    if errors:
        raise DependencyError("\n".join(errors))


class LitmusOrchestrator:
    def __init__(self, config: LitmusConfig) -> None:
        self.config = config
        self.state = StateManager(config.output_dir)
        self.enumerator = HFModelEnumerator(
            token=config.hf_token,
            sort=config.sort_mode,
        )

        # Export always uses the bundled ingest/export/ scripts.
        bundled_ingest = Path(__file__).resolve().parent / "ingest"
        self.export_runner = ExportRunner(
            ingest_dir=bundled_ingest,
            timeout=config.export_timeout,
        )

        self._shutdown = False
        self._state_lock = threading.Lock()
        self._retry_lock = threading.Lock()
        self._retry_set: set[str] = set()
        self._retry_path = config.output_dir / "retry.txt"

        # Persistent Tron clone for ingest (shared with deep analysis).
        self._tron_clone_lock = threading.Lock()
        self._tron_clone_path: Path | None = None

        self._deep_analyzer = None
        if config.deep_analysis:
            from .deep_analysis import DeepAnalyzer

            self._deep_analyzer = DeepAnalyzer(
                tron_url=config.tron_url,
                output_dir=config.output_dir,
                timeout=config.deep_analysis_timeout,
                consensus_review=config.consensus_review,
                tron_dir=config.tron_dir,
            )

        self._notion_publisher = None
        if config.notion_mcp_url and config.notion_parent_page_id:
            try:
                from .notion_publisher import (
                    NotionPublisher,
                )

                self._notion_publisher = NotionPublisher(
                    mcp_url=config.notion_mcp_url,
                    parent_page_id=(config.notion_parent_page_id),
                )
                logger.info(
                    "Notion publishing enabled: %s",
                    config.notion_mcp_url,
                )
            except ImportError:
                logger.warning(
                    "Notion publishing requested but"
                    " mcp package not installed."
                    " Install with:"
                    " pip install 'hf-litmus[notion]'"
                )

    def _get_tron_clone(self) -> Path:
        """Return the persistent Tron clone, creating it on first call.

        The clone lives at ``config.tron_dir/.repo`` and is
        shared with the deep-analysis subsystem.
        """
        with self._tron_clone_lock:
            if self._tron_clone_path is None:
                clone_path = self.config.tron_dir / ".repo"
                self.config.tron_dir.mkdir(parents=True, exist_ok=True)
                if (clone_path / ".git").is_dir():
                    logger.info(
                        "Using existing Tron clone at %s",
                        clone_path,
                    )
                else:
                    logger.info(
                        "Cloning Tron from %s...",
                        self.config.tron_url,
                    )
                    result = subprocess.run(
                        [
                            "git",
                            "clone",
                            "--depth=1",
                            self.config.tron_url,
                            str(clone_path),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"Failed to clone Tron: {result.stderr}"
                        )
                    logger.info("Tron cloned to %s", clone_path)
                self._tron_clone_path = clone_path
            return self._tron_clone_path

    def cleanup(self) -> None:
        """Reset in-memory state.

        The persistent Tron clone and per-model worktrees in
        ``tron_dir`` are intentionally preserved so users can
        inspect intermediate analysis results.
        """
        with self._tron_clone_lock:
            self._tron_clone_path = None

    def _compute_model_tags(self, trace_dir: Path) -> list[str]:
        """Derive model feature tags from metadata."""
        tags: list[str] = []
        meta_path = trace_dir / "metadata.json"
        if not meta_path.exists():
            return tags
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return tags

        cfg = meta.get("config", {})
        tag_cfg = meta.get("tag_config", {})

        # Attention type
        n_heads = cfg.get("num_attention_heads", 0) or 0
        n_kv = cfg.get("num_key_value_heads", 0) or 0
        if tag_cfg.get("kv_lora_rank"):
            tags.append("MLA")
        elif n_kv and n_kv == 1:
            tags.append("MQA")
        elif n_kv and n_kv < n_heads:
            tags.append("GQA")
        elif n_heads:
            tags.append("MHA")

        # MoE
        if tag_cfg.get("num_local_experts"):
            tags.append("MoE")

        # Position encoding
        rope = cfg.get("rope_scaling")
        if isinstance(rope, dict):
            rtype = rope.get("type", "").lower()
            if "yarn" in rtype:
                tags.append("YaRN")
            elif "linear" in rtype:
                tags.append("LinearRoPE")
            else:
                tags.append("RoPE")
        elif cfg.get("rope_theta"):
            tags.append("RoPE")

        # Sliding window
        if tag_cfg.get("sliding_window"):
            tags.append("SlidingWindow")

        return sorted(tags)

    def _compute_model_tags_from_hf(self, model_id: str) -> list[str]:
        """Fallback tag extraction from HF config."""
        return compute_tags_from_hf(model_id)

    def _log_retry(self, model_id: str) -> None:
        """Append a model ID to retry.txt for later rerun."""
        with self._retry_lock:
            if model_id not in self._retry_set:
                self._retry_set.add(model_id)
                with self._retry_path.open("a") as f:
                    f.write(model_id + "\n")
                logger.info(
                    "Logged %s for retry in %s",
                    model_id,
                    self._retry_path,
                )

    def setup_signal_handlers(self) -> None:
        """Register graceful shutdown handlers."""

        def handler(signum: int, frame: object) -> None:
            if self._shutdown:
                # Second signal: force exit
                logger.info("Forced shutdown")
                raise SystemExit(1)
            logger.info(
                "Shutdown requested, finishing current model... (repeat to force)"
            )
            self._shutdown = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def process_model(
        self,
        model_id: str,
        downloads: int = 0,
        likes: int = 0,
        pipeline_tag: str = "",
    ) -> ModelResult:
        """Process a single model through export+ingest."""
        logger.info("Processing %s", model_id)

        result = ModelResult(
            model_id=model_id,
            status=ModelStatus.SUCCESS,
            pipeline_tag=pipeline_tag,
            downloads=downloads,
            likes=likes,
        )

        with tempfile.TemporaryDirectory(prefix="litmus_trace_") as trace_tmp:
            trace_dir = Path(trace_tmp) / "trace"

            # Step 1: Export
            export_result = self.export_runner.export_model(
                model_id, trace_dir
            )

            if not export_result.success:
                cls = classify_export_error(
                    export_result.stdout,
                    export_result.stderr,
                    export_result.timed_out,
                )
                result.status = (
                    ModelStatus.TIMEOUT
                    if export_result.timed_out
                    else ModelStatus.EXPORT_FAIL
                )
                result.failure_stage = FailureStage.EXPORT
                result.failure_class = cls.failure_class
                result.failure_origin = cls.failure_origin
                result.retryable = cls.retryable
                result.missing_ops = cls.missing_ops
                result.error_output = export_result.stderr
                result.model_tags = self._compute_model_tags_from_hf(model_id)
                return result

            # Extract model feature tags from metadata
            result.model_tags = self._compute_model_tags(trace_dir)

            # Step 2: Ingest (requires Tron clone with cabal/ghc)
            try:
                tron_path = self._get_tron_clone()
            except Exception as e:
                logger.warning(
                    "Skipping ingest for %s: %s",
                    model_id,
                    e,
                )
                result.status = ModelStatus.SUCCESS
                return result

            ingest_runner = IngestRunner(
                ingest_dir=tron_path / "ingest",
                timeout=self.config.ingest_timeout,
                dump_intermediates=self.config.dump_intermediates,
            )
            result.ingest_version = ingest_runner.version

            model_name = (
                model_id.replace("/", "_").replace(".", "_").replace("-", "_")
            )
            model_name = f"litmus_{model_name}"

            ingest_result = ingest_runner.run_ingest(trace_dir, model_name)

            if not ingest_result.success:
                cls = classify_ingest_error(
                    ingest_result.stdout,
                    ingest_result.stderr,
                    ingest_result.timed_out,
                )
                result.status = (
                    ModelStatus.TIMEOUT
                    if ingest_result.timed_out
                    else ModelStatus.INGEST_FAIL
                )
                result.failure_stage = FailureStage.INGEST
                result.failure_class = cls.failure_class
                result.failure_origin = cls.failure_origin
                result.retryable = cls.retryable
                result.missing_ops = cls.missing_ops
                result.error_output = ingest_result.stderr
                return result

        result.status = ModelStatus.SUCCESS
        return result

    def _run_deep_analysis(
        self,
        result: ModelResult,
    ) -> None:
        """Run deep analysis if enabled and model failed."""
        if not self._deep_analyzer:
            return
        if result.status == ModelStatus.SUCCESS:
            return

        logger.info(
            "Running deep analysis for %s",
            result.model_id,
        )
        try:
            ar = self._deep_analyzer.analyze(
                result.model_id,
                result,
            )
            if ar.analysis_path:
                result.analysis_path = str(ar.analysis_path)
            if ar.worktree_branch:
                result.analysis_branch = ar.worktree_branch
            if ar.missing_ops:
                seen = set(result.missing_ops)
                for op in ar.missing_ops:
                    if op not in seen:
                        result.missing_ops.append(op)
                        seen.add(op)
            if ar.error:
                result.deep_analysis_error = ar.error
                if "rate_limit" in ar.error:
                    result.failure_origin = FailureOrigin.DEEP_ANALYSIS
                    result.retryable = True
            logger.info(
                "Deep analysis complete for %s: %s",
                result.model_id,
                ar.final_blocker or "see analysis",
            )
        except Exception:
            logger.exception(
                "Deep analysis failed for %s",
                result.model_id,
            )

    def _load_gap_data(
        self,
        model_id: str,
    ) -> dict | None:
        """Load gap-summary.json for a model if available."""
        sanitized = _sanitize_model_id(model_id)
        gap_path = (
            self.config.output_dir
            / "analyses"
            / sanitized
            / "gap-summary.json"
        )
        if gap_path.exists():
            try:
                return json.loads(gap_path.read_text())
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _publish_to_notion(
        self,
        result: ModelResult,
    ) -> None:
        """Publish result to Notion if enabled."""
        if not self._notion_publisher:
            return

        # Sync database ID from state
        if (
            self.state.notion_database_id
            and not self._notion_publisher.database_id
        ):
            self._notion_publisher.database_id = self.state.notion_database_id

        # Carry over cached page ID from previous run
        prev = self.state.get_result(result.model_id)
        if prev and prev.notion_page_id:
            result.notion_page_id = prev.notion_page_id

        # Read analysis artifacts if available
        analysis_md = None
        gap_data = None
        if result.analysis_path:
            ap = Path(result.analysis_path)
            try:
                if ap.exists():
                    analysis_md = ap.read_text()
            except Exception:
                pass

            # Look for gap-summary.json next to it
            try:
                gap_path = ap.parent / "gap-summary.json"
                if gap_path.exists():
                    gap_data = json.loads(gap_path.read_text())
            except Exception:
                pass

        try:
            page = self._notion_publisher.publish_result(
                result,
                analysis_md=analysis_md,
                gap_data=gap_data,
            )
            if page:
                result.notion_page_id = page.page_id
                logger.info(
                    "Published %s to Notion: %s",
                    result.model_id,
                    page.url or page.page_id,
                )
            # Persist database ID back to state
            if self._notion_publisher.database_id:
                self.state.notion_database_id = (
                    self._notion_publisher.database_id
                )
        except Exception:
            logger.warning(
                "Notion publish failed for %s",
                result.model_id,
                exc_info=True,
            )

    def run_batch(self, target_model: Optional[str] = None) -> int:
        """Process a batch of models. Returns count."""
        self.state.load()

        if target_model:
            downloads = 0
            likes = 0
            pipeline_tag = ""
            try:
                info = self.enumerator.get_model_info(target_model)
                downloads = info.downloads or 0
                likes = info.likes or 0
                pipeline_tag = info.pipeline_tag or ""
            except Exception:
                logger.warning(
                    "Could not fetch model info for %s,"
                    " proceeding without metadata",
                    target_model,
                )
            result = self.process_model(
                target_model,
                downloads=downloads,
                likes=likes,
                pipeline_tag=pipeline_tag,
            )
            self._run_deep_analysis(result)
            self.state.update(result)
            self.state.save()
            generate_report(
                result,
                self.config.output_dir,
                result.error_output,
            )
            generate_model_metadata(
                result,
                self.config.output_dir,
                self._load_gap_data(result.model_id),
            )
            self._publish_to_notion(result)
            generate_summary(self.state, self.config.output_dir)
            return 1

        models = self.enumerator.enumerate_models()
        models = filter_by_state(models, self.state, self.config.retest_mode)

        processed = 0
        for model in models:
            if self._shutdown:
                logger.info("Shutdown, stopping batch")
                break
            if processed >= self.config.batch_size:
                break

            try:
                result = self.process_model(
                    model.id,
                    downloads=model.downloads or 0,
                    likes=model.likes or 0,
                    pipeline_tag=model.pipeline_tag or "",
                )
            except Exception:
                logger.exception("Failed to process %s", model.id)
                result = ModelResult(
                    model_id=model.id,
                    status=ModelStatus.EXPORT_FAIL,
                    error_output="Unexpected error",
                )

            self._run_deep_analysis(result)
            self.state.update(result)
            self.state.flush_if_dirty()
            generate_report(
                result,
                self.config.output_dir,
                result.error_output,
            )
            generate_model_metadata(
                result,
                self.config.output_dir,
                self._load_gap_data(result.model_id),
            )
            self._publish_to_notion(result)
            processed += 1
            logger.info(
                "[%d/%d] %s: %s",
                processed,
                self.config.batch_size,
                model.id,
                result.status.value,
            )

        generate_summary(self.state, self.config.output_dir)
        return processed

    def _process_single_model(
        self,
        model_id: str,
    ) -> ModelResult:
        """Process one model end-to-end (thread-safe).

        Runs export, ingest, deep analysis, then
        acquires lock for state/report updates.
        """
        downloads = 0
        likes = 0
        pipeline_tag = ""
        try:
            info = self.enumerator.get_model_info(model_id)
            downloads = info.downloads or 0
            likes = info.likes or 0
            pipeline_tag = info.pipeline_tag or ""
        except Exception:
            logger.warning(
                "Could not fetch model info for %s,"
                " proceeding without metadata",
                model_id,
            )

        result = self.process_model(
            model_id,
            downloads=downloads,
            likes=likes,
            pipeline_tag=pipeline_tag,
        )

        self._run_deep_analysis(result)

        with self._state_lock:
            self.state.update(result)
            self.state.flush_if_dirty()

        # Report/metadata writes are per-model files,
        # safe without lock.
        try:
            generate_report(
                result,
                self.config.output_dir,
                result.error_output,
            )
            generate_model_metadata(
                result,
                self.config.output_dir,
                self._load_gap_data(result.model_id),
            )
        except Exception:
            logger.exception(
                "Report generation failed for %s",
                result.model_id,
            )

        try:
            with self._state_lock:
                self._publish_to_notion(result)
        except Exception:
            logger.exception(
                "Notion publish failed for %s",
                result.model_id,
            )

        return result

    def run_model_list(
        self,
        model_ids: list[str],
        max_jobs: int = 1,
    ) -> int:
        """Process a list of models, up to max_jobs
        in parallel. Returns count of models processed.
        """
        self.state.load()
        self.setup_signal_handlers()
        # Clear retry file and set from previous run
        self._retry_set.clear()
        if self._retry_path.exists():
            self._retry_path.unlink()
        total = len(model_ids)

        if max_jobs <= 1:
            # Sequential path
            processed = 0
            for i, model_id in enumerate(model_ids, 1):
                if self._shutdown:
                    logger.info("Shutdown requested, stopping")
                    # Log remaining models for retry
                    for mid in model_ids[i - 1 :]:
                        self._log_retry(mid)
                    break
                logger.info(
                    "[%d/%d] %s",
                    i,
                    total,
                    model_id,
                )
                processed += 1
                try:
                    result = self._process_single_model(model_id)
                    logger.info(
                        "[%d/%d] %s: %s",
                        i,
                        total,
                        model_id,
                        result.status.value,
                    )
                    if result.retryable:
                        self._log_retry(model_id)
                except Exception:
                    logger.exception(
                        "[%d/%d] Failed: %s",
                        i,
                        total,
                        model_id,
                    )
                    self._log_retry(model_id)
            generate_summary(self.state, self.config.output_dir)
            return processed

        # Parallel path
        completed = 0
        with ThreadPoolExecutor(
            max_workers=max_jobs,
            thread_name_prefix="litmus",
        ) as pool:
            futures = {}
            for model_id in model_ids:
                if self._shutdown:
                    break
                fut = pool.submit(
                    self._process_single_model,
                    model_id,
                )
                futures[fut] = model_id

            if self._shutdown:
                # Cancel any pending futures
                for fut, mid in futures.items():
                    if fut.cancel():
                        self._log_retry(mid)

            for fut in as_completed(futures):
                model_id = futures[fut]
                completed += 1
                try:
                    result = fut.result()
                    logger.info(
                        "[%d/%d] %s: %s",
                        completed,
                        total,
                        model_id,
                        result.status.value,
                    )
                    if result.retryable:
                        self._log_retry(model_id)
                except Exception:
                    logger.exception(
                        "[%d/%d] Failed: %s",
                        completed,
                        total,
                        model_id,
                    )
                    self._log_retry(model_id)

        generate_summary(self.state, self.config.output_dir)
        if self._retry_path.exists():
            logger.info(
                "Models needing retry logged to %s",
                self._retry_path,
            )
        return completed

    def run_daemon(self) -> None:
        """Run the continuous daemon loop."""
        self.setup_signal_handlers()
        logger.info(
            "Starting HF Litmus daemon (batch=%d, interval=%dm)",
            self.config.batch_size,
            self.config.interval_minutes,
        )

        while not self._shutdown:
            start = time.time()

            try:
                n = self.run_batch()
                logger.info("Batch complete: %d models", n)
            except Exception:
                logger.exception("Batch failed")

            elapsed = time.time() - start
            sleep_s = max(
                0,
                self.config.interval_minutes * 60 - elapsed,
            )

            if sleep_s > 0 and not self._shutdown:
                logger.info(
                    "Sleeping %.1f min until next batch",
                    sleep_s / 60,
                )
                deadline = time.monotonic() + sleep_s
                while time.monotonic() < deadline and not self._shutdown:
                    time.sleep(1)

        logger.info("HF Litmus daemon stopped")
