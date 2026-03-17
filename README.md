# hf-litmus

Continuous model compatibility testing for [Tron](https://github.com/positron-ai/tron),
Positron's hardware compiler. hf-litmus automatically evaluates HuggingFace models
through the Tron ingest pipeline (torch.export + Haskell IR compilation), identifies
compatibility gaps, and optionally launches Claude Code for deep failure analysis.

## Features

- **Batch enumeration** of HuggingFace Hub models by trending/recent activity
- **Export testing** via `torch.export` with bundled export scripts (no Tron checkout needed)
- **Full pipeline testing** through Haskell ingest when a Tron checkout is available
- **Deep analysis** using Claude Code in git worktrees with multi-model consensus review
- **Interactive dashboard** with live retry, model search, and analysis submission
- **Report generation** per-model and aggregate summary with failure classification
- **Notion publishing** for sharing results via MCP integration

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (for managing export script dependencies)

For full pipeline testing (export + Haskell ingest):
- A [Tron](https://github.com/positron-ai/tron) checkout
- GHC and Cabal (available via `nix develop` from the Tron repo)

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

### Export-only mode (no Tron checkout required)

Test whether a model can be exported via `torch.export`:

```bash
hf-litmus --model meta-llama/Llama-3.1-8B
```

### Full pipeline mode (with Tron checkout)

Test export + Haskell ingest:

```bash
# Option 1: Explicit path
hf-litmus --model meta-llama/Llama-3.1-8B --tron-root ~/tron

# Option 2: Environment variable
export LITMUS_TRON_ROOT=~/tron
hf-litmus --model meta-llama/Llama-3.1-8B

# Option 3: Auto-detect (run from within a Tron checkout)
cd ~/tron
hf-litmus --model meta-llama/Llama-3.1-8B
```

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
# Launch the web dashboard
hf-litmus dashboard --serve --output-dir ./litmus-output

# With Tron integration (enables retry and analysis features)
hf-litmus dashboard --serve --output-dir ./litmus-output --tron-root ~/tron
```

## Configuration

### Tron Root Resolution

The `--tron-root` argument tells hf-litmus where to find the Tron repository.
Resolution precedence:

1. `--tron-root` CLI argument
2. `LITMUS_TRON_ROOT` environment variable
3. Auto-detection: walks up from the current directory looking for `ingest/cabal.project`
4. None (export-only mode; Haskell ingest and deep analysis are disabled)

When `--tron-root` is provided, hf-litmus uses the Tron checkout's `ingest/export/`
scripts and `cabal run ingest` for the Haskell pipeline. When unavailable, it falls
back to the bundled export scripts in `ingest/export/` within this repository.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `./litmus-output` | Directory for reports and state |
| `--tron-root` | auto-detect | Path to Tron repository |
| `--model` | — | Test a specific model by HF ID |
| `--model-file` | — | File with model IDs (one per line) |
| `--batch-size` | 100 | Models per batch |
| `-j, --jobs` | 1 | Parallel workers |
| `--once/--no-once` | `--once` | Single batch vs continuous |
| `--daemon` | off | Continuous mode (alias for `--no-once`) |
| `--deep-analysis/--no-deep-analysis` | on | Claude Code failure analysis |
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
│   └── ...
├── ingest/             # Bundled export tooling (from Tron)
│   ├── export/         # torch.export scripts + uv project
│   └── runtime/        # Shared Python utilities
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
