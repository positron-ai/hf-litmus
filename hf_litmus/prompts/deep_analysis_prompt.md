<!-- Deep Analysis Prompt Template for HF Litmus
     =============================================

     This file contains the prompt sent to Claude Code during deep analysis
     of HuggingFace model compatibility with the Tron FPGA pipeline.

     EDITING NOTES:
     - Sections are delimited by HTML comments: <!-- SECTION: name -->
     - Variables use $-style substitution: $variable_name
     - Use $$ for a literal dollar sign (e.g., shell variables)
     - JSON examples use normal {} braces (no escaping needed)
     - The Python loader selects one stage guidance section based on
       the failure type; all other stage sections are ignored.
     - The consensus_review section is only included when the
       - -consensus-review flag is passed.

     AVAILABLE VARIABLES:
     - $model_id          HuggingFace model identifier (e.g., "meta-llama/Llama-3-8B")
     - $sanitized_model_id Filesystem-safe model name (e.g., "meta-llama-llama-3-8b")
     - $failure_stage      Pipeline stage that failed (export, ingest, unknown)
     - $failure_class      Error classification (missing_op, type_error, etc.)
     - $downloads          Model download count (formatted with commas)
     - $error              Truncated error output from the failed stage
     - $analysis_dir       Path to the analysis output directory
     - $stage_guidance     (auto-injected) Selected stage guidance section
     - $consensus_section  (auto-injected) Consensus review section or empty
-->

<!-- SECTION: main -->
You are performing a deep compatibility analysis of the HuggingFace model `$model_id` against the Tron FPGA inference pipeline. Your goal is to push this model as far through the pipeline as possible, fixing issues along the way, and then produce a comprehensive gap analysis of what remains.

## Pipeline Overview

The Tron ingest pipeline converts HuggingFace transformer models into C++ plugins for FPGA-based inference. The pipeline has these stages:

1. **torch.export** (Python) - Traces the model forward pass into a static FX graph.
   - Key file: `ingest/export/torch_export.py`
   - Output: `root.fx` (FX IR text) + `metadata.json` (config, shapes, params)
   - Run: `cd ingest && uv run --project export torch-export \
       --model MODEL --output TRACE_DIR --max-seq-length 64 --meta-device`

2. **FxTypedFx** (Haskell) - Parses FX IR text into a typed IR.
   - Key file: `ingest/src/FxTypedFx.hs`
   - Supported aten ops include: add, sub, mul, div, neg, matmul, mm, bmm, linear, embedding, softmax, silu, gelu, tanh, sigmoid, sin, cos, rsqrt, reshape, view, transpose, permute (transpose-only), unsqueeze, squeeze, cat, slice, expand, arange, mean, sum, topk, max, clamp, scatter, index_select, zeros, zeros_like, rms_norm, layer_norm, pow (square only), chunk (n=2 only), to.dtype, contiguous, copy_, full
   - Unsupported: arbitrary permute, chunk n!=2, GetAttr, bool tensors, higher-order ops (autocast)

3. **TypedFxTypedFx / EqSat** (Haskell) - Equality saturation rewrites that recognize high-level patterns.
   - Key files: `ingest/src/TypedFxTypedFx.hs`, `ingest/src/RewriteFxRewriteFx/EqSat.hs`
   - Recognized patterns: RoPE, SDPA (scaled dot-product attention), RMSNorm, RMSNorm+Mul, SwiGLU, LayerNorm

4. **TypedFxBulk** (Haskell) - Lowers TypedFx to hardware-aware Bulk IR.
   - Key file: `ingest/src/TypedFxBulk.hs`
   - Many ops still TODO: rsqrt, pow, neg, sin, cos, swish, tanh, gelu, sigmoid, sub, mul, div, clamp, IndexSelect, Expand, Slice, Cat, Scatter, standalone Matmul (non-SDPA)

5. **BulkLoopy -> LoopyTron -> TronCpp** - Final lowering to C++ plugin source.
   - Key files: `ingest/src/BulkLoopy.hs`, `ingest/src/LoopyTron.hs`, `ingest/src/TronCpp.hs`

**Metadata requirements** (`ingest/src/Fx/Metadata.hs`):
Required config keys in metadata.json: `hidden_size`, `max_position_embeddings`, `num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`, `rope_theta`, `vocab_size`, `head_dim`, and `rope_scaling` type.

These are extracted by `torch_export.py` from the HuggingFace model config. Models using non-standard config attribute names (e.g. `num_layers` instead of `num_hidden_layers`) require mapping logic in the export code.

**Build commands** (all require `nix develop`):
- Rebuild Haskell ingest: `cd ingest && cabal build`
- Run export: see stage 1 above
- Run ingest: `cd ingest && cabal run ingest -w ghc -- \
    --model-name NAME --output-dir OUT --torch-trace-directory TRACE`
- Run with intermediate dumps: add `--dump-all` to ingest command

## Current Failure

- **Model**: $model_id
- **Stage**: $failure_stage
- **Classification**: $failure_class
- **Downloads**: $downloads

### Error Output

```
$error
```

## Your Instructions

$stage_guidance

### General Approach

1. **Read** the error output carefully and identify the root cause.
2. **Fix** the issue in the source code (Python export or Haskell ingest).
3. **Re-run** the failing stage under `nix develop`:
   - Export: `nix develop --command bash -c 'cd ingest && uv run \
       --project export torch-export --model $model_id \
       --output /tmp/litmus-trace-$sanitized_model_id \
       --max-seq-length 64 --meta-device'`
   - Ingest: `nix develop --command bash -c 'cd ingest && cabal run \
       ingest -w ghc -- --model-name $sanitized_model_id \
       --output-dir /tmp/litmus-ingest-$sanitized_model_id \
       --torch-trace-directory /tmp/litmus-trace-$sanitized_model_id \
       --dump-all'`
4. **Repeat** steps 1-3 for each new error until you either:
   - Successfully generate a C++ plugin (full success!), or
   - Hit a blocker you cannot fix (document it precisely)
5. **Document** everything (see below).
6. **Commit** all changes to the worktree. Do NOT push.

### What to Fix vs What to Document

**Fix these** (they're straightforward):
- `trust_remote_code=True` missing
- Config attribute name mapping (e.g. `num_layers` -> `num_hidden_layers`)
- Missing Python dependencies (add to `pyproject.toml`)
- Simple monkey-patches for data-dependent control flow
- Adding `trust_remote_code=True` to from_pretrained/from_config calls

**Document but don't try to fully implement**:
- New Haskell op translations in FxTypedFx.hs (document which ops are missing)
- New Haskell IR lowerings in TypedFxBulk.hs (document which are needed)
- New EqSat rewrite patterns (document the attention/RoPE pattern)
- New C++ kernels (document what would be needed)
- Major architectural changes (sequence-first layout, non-standard attention)

### Output

Create the directory `$analysis_dir` and write `analysis.md` there with:

1. **Model Overview**: Name, architecture class, parameters, config details
2. **Pipeline Progression**: A table showing each stage attempted, whether it passed/failed, and what error occurred
3. **Fixes Applied**: For each fix, show the diff and explain what it does
4. **Remaining Blockers**: Organized by pipeline stage, with:
   - The exact error message
   - Which file and function is involved
   - What would need to be implemented
   - Severity (critical/high/medium/low)
5. **Missing Operations**: A structured list of aten ops that the FxTypedFx translator doesn't handle for this model
6. **Missing Kernels**: Operations that TypedFxBulk can't lower to hardware
7. **Missing Patterns**: High-level patterns (attention, RoPE, norm) that EqSat doesn't recognize
8. **Recommendations**: Prioritized list of what to implement to support this model

Also create `gap-summary.json` in the same directory with:
```json
{
  "model_id": "$model_id",
  "furthest_stage": "export|fxtypedfx|rewrite|bulk|loopy|tron|cpp|success",
  "fixes_applied": ["description of each fix"],
  "missing_ops": ["aten.op_name", ...],
  "missing_kernels": ["kernel_name", ...],
  "missing_patterns": ["pattern description", ...],
  "blockers": [
    {
      "stage": "stage_name",
      "severity": "critical|high|medium|low",
      "description": "what's missing",
      "file": "path/to/file.hs",
      "effort": "small|medium|large"
    }
  ]
}
```
$consensus_section

<!-- SECTION: stage_trust_remote_code -->
### Stage-Specific Guidance: trust_remote_code

The model uses custom code. Add `trust_remote_code=True` to all `from_pretrained` and `from_config` calls in `ingest/export/torch_export.py`.

Also check if the tokenizer needs additional dependencies (sentencepiece, tiktoken, etc.) and add them to `ingest/export/pyproject.toml`.

After fixing, re-run export and continue to the next failure.

<!-- SECTION: stage_unsupported_dynamic -->
### Stage-Specific Guidance: Data-Dependent Control Flow

The model has data-dependent branching that `torch.export` can't trace. Look at the traceback to find the exact line in the model's `modeling_*.py`.

Consider writing a monkey-patch that removes the data-dependent branch. For example, if the model checks `attention_mask.all()`, replace the conditional with the always-mask path.

Check `ingest/export/` for existing model-specific patches like `gpt_oss_export.py` for reference.

If monkey-patching is not feasible, document exactly which control flow is problematic and what a fix would look like.

<!-- SECTION: stage_export_default -->
### Stage-Specific Guidance: Export Failure

Read the full traceback to understand the root cause. Common issues:
- Missing config attributes in metadata extraction
- Unsupported model architecture class
- Memory issues (try --meta-device)
- Missing tokenizer dependencies

<!-- SECTION: stage_ingest -->
### Stage-Specific Guidance: Ingest Failure

Export succeeded, so focus on the Haskell ingest pipeline. Read the error to determine which stage failed:

- **FxTypedFx**: Unknown aten op. Check `ingest/src/FxTypedFx.hs` for the op name. Document which op is missing.
- **TypedFxTypedFx/EqSat**: Pattern not recognized. The model's attention or norm pattern doesn't match the rewrite rules.
- **TypedFxBulk**: Op not lowered to Bulk IR. Check `ingest/src/TypedFxBulk.hs` for TODO markers.
- **BulkLoopy/LoopyTron/TronCpp**: Late-stage failure in hardware lowering or C++ codegen.

For missing ops, don't try to implement them in Haskell. Just document precisely which ops are needed and where they'd go.

<!-- SECTION: consensus_review -->

## Multi-Model Consensus Review

After completing your initial analysis and writing `analysis.md` and `gap-summary.json`, you MUST perform a consensus review using the PAL MCP server. This ensures your findings are validated by multiple AI models from different perspectives.

### Step 1: Create an Agent Team

Use the `TeamCreate` tool to create a team called `litmus-review-$sanitized_model_id`. Then spawn three teammates using the `Task` tool with `team_name`:

1. **ux-reviewer** (subagent_type: `general-purpose`): Evaluates the quality of your analysis report. Focuses on:
   - Is the analysis.md clear and actionable for a developer?
   - Are recommendations prioritized sensibly?
   - Is the gap-summary.json schema complete and well-structured?
   - Would a developer reading this know exactly what to do next?

2. **arch-reviewer** (subagent_type: `general-purpose`): Deep-dives on technical architecture. Focuses on:
   - Are the identified missing ops actually needed, or can they be decomposed into supported ops?
   - Is the furthest_stage assessment accurate?
   - Are effort estimates realistic given the Tron pipeline architecture?
   - Are there alternative code paths or workarounds not considered?

3. **devils-advocate** (subagent_type: `general-purpose`): Challenges your assumptions. Focuses on:
   - What failure modes might you have missed?
   - Are any of your "fixes" actually introducing new problems?
   - Is the verdict too optimistic or pessimistic?
   - Are there edge cases in the model architecture that could cause issues downstream even if ingest succeeds?

### Step 2: PAL Consensus

After the team review, use the PAL MCP `consensus` tool to get independent validation from external models. Call `consensus` with:
- `models`: `[{"model": "gemini-3-pro-preview", "stance": "neutral"}, {"model": "gpt-5.2-pro", "stance": "neutral"}]`
- Present the full gap-summary.json content and your key findings
- Ask them to validate: (a) correctness of missing op identification, (b) accuracy of effort estimates, (c) whether the verdict is appropriate

### Step 3: Merge Review into Artifacts

Add a `consensus_review` section to your `gap-summary.json`:
```json
{
  "consensus_review": {
    "ux_findings": ["list of UX review findings"],
    "arch_findings": ["list of architecture review findings"],
    "devils_advocate_findings": ["list of challenges raised"],
    "external_consensus": {
      "models_consulted": ["gemini-3-pro-preview", "gpt-5.2-pro"],
      "agreed_verdict": "the consensus verdict",
      "disagreements": ["any points of disagreement"],
      "additional_risks": ["risks identified by external models"]
    },
    "final_verdict_adjusted": false,
    "adjustment_reason": ""
  }
}
```

If the team review or external consensus causes you to change your verdict or effort estimate, update those fields in gap-summary.json and set `final_verdict_adjusted` to true with the reason.

After the review is complete, shut down the team with `shutdown_request` messages and `TeamDelete`.
