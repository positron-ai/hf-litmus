# hf-litmus

Continuous model compatibility testing for [Tron](https://github.com/positron-ai/tron),
Positron's hardware compiler. hf-litmus automatically evaluates HuggingFace models
through the Tron ingest pipeline (torch.export + Haskell IR compilation), identifies
compatibility gaps, and optionally launches Claude Code for deep failure analysis.

## Features

- **Batch enumeration** of HuggingFace Hub models by trending/recent activity
- **Export testing** via `torch.export` with bundled export scripts
- **Full pipeline testing** through Haskell ingest (Tron is cloned on demand)
- **Deep analysis** using Claude Code in git worktrees with multi-model consensus review
- **Interactive dashboard** with live retry, model search, and analysis submission
- **Report generation** per-model and aggregate summary with failure classification
- **Notion publishing** for sharing results via MCP integration

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (for managing export script dependencies)
- [git](https://git-scm.com/) (for on-demand Tron cloning during ingest and deep analysis)

For full pipeline testing (export + Haskell ingest), the Tron clone must have
GHC and Cabal available (i.e., run inside `nix develop` from the Tron repo).

For deep analysis:
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/positron-ai/hf-litmus.git
cd hf-litmus

# Install the package
uv pip install -e .
# Or with Notion integration:
uv pip install -e '.[notion]'
```

## Quick Start

### Test a single model

```bash
hf-litmus --model meta-llama/Llama-3.1-8B
```

This runs the full pipeline: torch.export (using bundled scripts) then Haskell
ingest (Tron is cloned automatically from GitHub). On first ingest, expect a
short delay while Tron is cloned.

### Batch processing

```bash
# Test top 100 trending models
hf-litmus --batch-size 100

# Process a list of models in parallel
hf-litmus --model-file models.txt -j 4

# Continuous daemon mode
hf-litmus --no-once --interval 30
```

### Dashboard

```bash
hf-litmus dashboard --serve --output-dir ./litmus-output
```

## Configuration

### Tron URL

hf-litmus clones Tron on demand whenever ingest or deep analysis is needed.
By default it clones from `https://github.com/positron-ai/tron.git`. Override with:

```bash
# CLI flag
hf-litmus --tron-url https://github.com/your-fork/tron.git --model my/model

# Environment variable
export LITMUS_TRON_URL=https://github.com/your-fork/tron.git
hf-litmus --model my/model
```

A shallow clone (`--depth=1`) is used. For ingest, one clone is reused across
all models in a session and cleaned up on exit. For deep analysis, each model
gets its own fresh clone (with an isolated git worktree) that is removed after
results are collected.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `./litmus-output` | Directory for reports and state |
| `--tron-url` | GitHub URL | Tron repository URL for on-demand cloning |
| `--model` | — | Test a specific model by HF ID |
| `--model-file` | — | File with model IDs (one per line) |
| `--batch-size` | 100 | Models per batch |
| `-j, --jobs` | 1 | Parallel workers |
| `--once/--no-once` | `--once` | Single batch vs continuous |
| `--daemon` | off | Continuous mode (alias for `--no-once`) |
| `--deep-analysis/--no-deep-analysis` | on | Claude Code failure analysis |
| `--deep-analysis-timeout` | 7200 | Seconds for Claude Code session |
| `--consensus-review/--no-consensus-review` | on | Multi-model review during analysis |
| `--hf-token` | — | HuggingFace token for gated models |
| `--export-timeout` | 600 | Seconds for torch.export |
| `--ingest-timeout` | 300 | Seconds for Haskell ingest |
| `-v, --verbose` | off | Debug logging |

## Repository Structure

```
hf-litmus/
├── hf_litmus/          # Main Python package
│   ├── cli.py          # CLI entry point
│   ├── orchestrator.py # Core pipeline orchestration
│   ├── dashboard.py    # Web dashboard + HTTP server
│   ├── deep_analysis.py# Claude Code integration
│   ├── ingest/         # Bundled export tooling
│   │   ├── export/     # torch.export scripts + uv project
│   │   └── runtime/    # Shared Python utilities
│   └── ...
├── litmus-outputs/     # Submodule: persistent results repo
├── pyproject.toml
└── README.md
```

## Output

Results are written to `--output-dir` (default `./litmus-output/`):

- `state.json` — Persistent state tracking all tested models
- `reports/<model>.md` — Per-model Markdown reports
- `analyses/<model>/` — Deep analysis artifacts (analysis.md, gap-summary.json)
- `summary.json` — Aggregate statistics
- `litmus.log` — Rotating log file

## License

Internal use only. Copyright Positron AI.
