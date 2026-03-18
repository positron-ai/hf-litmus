from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import DEFAULT_TRON_URL, LitmusConfig


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "HF Litmus: Continuous model compatibility testing for Tron"
        ),
        formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./litmus-output"),
        help="Directory for output reports and state",
    )
    parser.add_argument(
        "--sort",
        choices=["trending", "lastModified"],
        default="trending",
        help="Model sort order from HF Hub",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of models per batch",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Minutes between batch runs",
    )
    parser.add_argument(
        "--once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run a single batch then exit"
            " (enabled by default; use --no-once"
            " or --daemon to run continuously)"
        ),
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help=("Run continuously in daemon mode (equivalent to --no-once)"),
    )
    parser.add_argument(
        "--retest",
        action="store_true",
        help="Re-test previously failed models",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Test a specific model by HF ID",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help=(
            "File containing model IDs to test,"
            " one per line (blank lines and lines"
            " starting with # are ignored)"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of models to process in parallel"
            " (requires --model-file or batch mode;"
            " each job uses a separate git worktree)"
        ),
    )
    parser.add_argument(
        "--dump-intermediates",
        action="store_true",
        help="Keep intermediate trace files",
    )
    parser.add_argument(
        "--deep-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On failure, launch Claude Code in a git"
            " worktree to attempt fixing the model,"
            " producing a detailed gap analysis of"
            " missing ops, kernels, and features"
            " (enabled by default; use"
            " --no-deep-analysis to disable)"
        ),
    )
    parser.add_argument(
        "--deep-analysis-timeout",
        type=int,
        default=7200,
        help=("Timeout for deep analysis Claude Code session in seconds"),
    )
    parser.add_argument(
        "--consensus-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During deep analysis, use PAL MCP to get"
            " multi-model consensus from Gemini 3 Pro"
            " and GPT 5.2 Pro via an agent team with"
            " UX, architecture, and devil's advocate"
            " perspectives (enabled by default; use"
            " --no-consensus-review to disable)"
        ),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated models",
    )
    parser.add_argument(
        "--notion-mcp-url",
        type=str,
        default=None,
        help=(
            "Notion MCP server URL for publishing"
            " reports (e.g. https://mcp.notion.com/mcp"
            " or http://localhost:8787/mcp)"
        ),
    )
    parser.add_argument(
        "--notion-parent-id",
        type=str,
        default=None,
        help=("Notion page ID under which to create litmus result pages"),
    )
    parser.add_argument(
        "--export-timeout",
        type=int,
        default=600,
        help="Timeout for torch.export in seconds",
    )
    parser.add_argument(
        "--ingest-timeout",
        type=int,
        default=300,
        help="Timeout for Haskell ingest in seconds",
    )
    parser.add_argument(
        "--tron-url",
        type=str,
        default=None,
        help=(
            "URL of the Tron git repository."
            " Overrides LITMUS_TRON_URL env var."
            " Tron is cloned on demand when ingest"
            " or deep analysis is needed."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _resolve_tron_url(arg: str | None) -> str:
    """Resolve tron_url from arg, env, or default."""
    if arg is not None:
        return arg
    return os.environ.get("LITMUS_TRON_URL") or DEFAULT_TRON_URL


def config_from_args(
    args: argparse.Namespace,
) -> LitmusConfig:
    # --daemon overrides --once
    single_run = args.once and not args.daemon
    return LitmusConfig(
        output_dir=args.output_dir,
        sort_mode=args.sort,
        batch_size=args.batch_size,
        interval_minutes=args.interval,
        single_run=single_run,
        retest_mode=args.retest,
        target_model=args.model,
        model_file=args.model_file,
        max_jobs=args.jobs,
        dump_intermediates=args.dump_intermediates,
        deep_analysis=args.deep_analysis,
        deep_analysis_timeout=args.deep_analysis_timeout,
        consensus_review=args.consensus_review,
        export_timeout=args.export_timeout,
        ingest_timeout=args.ingest_timeout,
        hf_token=args.hf_token,
        notion_mcp_url=args.notion_mcp_url,
        notion_parent_page_id=args.notion_parent_id,
        verbose=args.verbose,
        tron_url=_resolve_tron_url(args.tron_url),
    )


def _read_model_file(path: Path) -> list[str]:
    """Read model IDs from a text file.

    Blank lines and lines starting with # are ignored.
    """
    models: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append(line)
    return models


def setup_logging(output_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "litmus.log"
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def main() -> int:
    # Dispatch to dashboard subcommand if requested
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        from .dashboard import dashboard_main

        return dashboard_main(sys.argv[2:])

    parser = create_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.output_dir, config.verbose)

    from .exceptions import DependencyError, LitmusError
    from .orchestrator import (
        LitmusOrchestrator,
        check_dependencies,
    )

    try:
        check_dependencies()
    except DependencyError as e:
        logging.error("Missing dependencies:\n%s", e)
        return 1

    orchestrator = LitmusOrchestrator(config)
    try:
        if config.model_file:
            models = _read_model_file(config.model_file)
            if not models:
                logging.error(
                    "No models found in %s",
                    config.model_file,
                )
                return 1
            logging.info(
                "Processing %d models from %s (jobs=%d)",
                len(models),
                config.model_file,
                config.max_jobs,
            )
            orchestrator.run_model_list(models, max_jobs=config.max_jobs)
        elif config.target_model:
            orchestrator.run_batch(target_model=config.target_model)
        elif config.single_run:
            orchestrator.run_batch()
        else:
            orchestrator.run_daemon()
    except LitmusError as e:
        logging.error("Error: %s", e)
        return 1
    except (KeyboardInterrupt, SystemExit):
        logging.info("Interrupted")
        return 130
    finally:
        orchestrator.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
