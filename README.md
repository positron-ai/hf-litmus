# hf-litmus

I got tired of manually checking whether HuggingFace models could make it
through [Tron](https://github.com/positron-ai/tron)'s compilation pipeline. There are thousands of
models on the Hub, new ones every day, and I wanted a tool that would just grind
through them -- `torch.export`, then Haskell IR ingestion -- and tell me exactly
where each one breaks (or doesn't).

That's what hf-litmus does. Point it at a model and it runs the full pipeline.
Point it at the Hub and it'll chew through the top trending models in batch.
When something fails, it can spin up Claude Code in an isolated git worktree to
do a deep analysis of the failure, complete with a multi-model consensus review.

## Getting started

You'll need Python 3.10+, [uv](https://docs.astral.sh/uv/), and git. For
ingest testing (the Haskell stage), Tron gets cloned automatically -- you don't
need a local checkout.

```bash
git clone --recurse-submodules https://github.com/positron-ai/hf-litmus.git
cd hf-litmus
nix develop          # recommended: sets up everything
uv pip install -e .  # or install directly
```

### Test a single model

```bash
hf-litmus --model meta-llama/Llama-3.1-8B
```

This runs the full pipeline: `torch.export` first (using bundled scripts), then
Haskell ingest (Tron is shallow-cloned from GitHub the first time). Expect a
short delay on the first run while Tron is fetched.

### Batch processing

```bash
hf-litmus --batch-size 100             # top 100 trending models
hf-litmus --model-file models.txt -j 4 # parallel from a list
hf-litmus --no-once --interval 30      # continuous daemon mode
```

### Dashboard

```bash
hf-litmus dashboard --serve --output-dir ./litmus-output
```

## Configuration

hf-litmus clones Tron on demand. By default it uses
`https://github.com/positron-ai/tron.git`. You can override this:

```bash
hf-litmus --tron-url https://github.com/your-fork/tron.git --model my/model
# or
export LITMUS_TRON_URL=https://github.com/your-fork/tron.git
```

A shallow clone (`--depth=1`) is used. For ingest, one clone is reused across
all models in a session and cleaned up on exit. For deep analysis, each model
gets its own fresh clone with an isolated worktree.

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `./litmus-output` | Reports and state directory |
| `--tron-url` | GitHub URL | Tron repo URL for on-demand cloning |
| `--model` | -- | Test a specific model by HF ID |
| `--model-file` | -- | File with model IDs, one per line |
| `--batch-size` | 100 | Models per batch |
| `-j, --jobs` | 1 | Parallel workers |
| `--once/--no-once` | `--once` | Single batch vs continuous |
| `--deep-analysis/--no-deep-analysis` | on | Claude Code failure analysis |
| `--consensus-review/--no-consensus-review` | on | Multi-model review |
| `--hf-token` | -- | HuggingFace token for gated models |
| `--export-timeout` | 600 | Seconds for `torch.export` |
| `--ingest-timeout` | 300 | Seconds for Haskell ingest |
| `-v, --verbose` | off | Debug logging |

## Development

```bash
nix develop  # enters a shell with Python, ruff, pytest, lefthook
lefthook install
```

Pre-commit hooks run formatting, linting, tests, and coverage checks in
parallel. You can also run them individually:

```bash
ruff check .                         # lint
ruff format --check .                # formatting
python -m pytest tests/ -x -q        # tests
nix flake check                      # everything (lint + format + tests + coverage + build)
```

## Project layout

```
hf-litmus/
  hf_litmus/           Main package
    cli.py             CLI entry point
    orchestrator.py    Pipeline orchestration
    dashboard.py       Web dashboard + HTTP server
    deep_analysis.py   Claude Code integration
    error_classifier.py Error pattern matching
    ingest/            Bundled torch.export scripts
  tests/               Unit and property-based tests
  flake.nix            Nix development shell and CI checks
  lefthook.yml         Pre-commit hooks
```

## Output

Results go to `--output-dir` (default `./litmus-output/`):

- `state.json` -- persistent state tracking all tested models
- `reports/<model>.md` -- per-model Markdown reports
- `analyses/<model>/` -- deep analysis artifacts
- `summary.json` -- aggregate statistics
- `litmus.log` -- rotating log

## License

BSD 3-Clause. See [LICENSE.md](LICENSE.md).
