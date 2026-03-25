<!-- HF-Litmus Addendum Template
     ============================

     This template is appended to the Tron /ingest command prompt to provide
     prior failure context and specify output artifact requirements for
     automated HF-Litmus runs.

     EDITING NOTES:
     - Sections are delimited by HTML comments: <!-- SECTION: name -->
     - Variables use $-style substitution: $variable_name
     - Use $$ for a literal dollar sign
     - JSON examples use normal {} braces (no escaping needed)

     AVAILABLE VARIABLES:
     - $model_id          HuggingFace model identifier
     - $failure_stage      Pipeline stage that failed (export, ingest, unknown)
     - $failure_class      Error classification
     - $downloads          Model download count (formatted with commas)
     - $error              Truncated error output from the failed stage
     - $analysis_dir       Path to the analysis output directory
     - $consensus_section  (auto-injected) Consensus review section or empty
-->

<!-- SECTION: main -->

# HF-Litmus Automation Context

This ingestion is being run automatically by HF-Litmus, a continuous model
compatibility tester. The model has already been through a preliminary pipeline
run and **failed**. Use the failure information below to guide your investigation.

## Prior Failure Information

- **Model**: `$model_id`
- **Failed Stage**: $failure_stage
- **Classification**: $failure_class
- **Downloads**: $downloads

### Error Output from Preliminary Run

```
$error
```

## Automation Constraints

- This is a **non-interactive automated run**. Do NOT prompt for user input.
- Skip Phase 8 (FPGA testing) -- there is no operator present.
- Make reasonable decisions autonomously at every step.
- Limit wall-clock time: focus on diagnosing and fixing the known failure, then
  push as far through the pipeline as possible.
- **Commit** all changes to the worktree. Do NOT push.

## Required Output Artifacts

Write the following files to `$analysis_dir`:

### `analysis.md`

A detailed report covering:

1. **Model Overview**: Name, architecture class, parameters, config details
2. **Pipeline Progression**: Table of each stage attempted (pass/fail/error)
3. **Fixes Applied**: Diffs and explanations for each fix
4. **Remaining Blockers**: By stage, with error message, file, what's needed, severity
5. **Missing Operations**: Aten ops not handled by FxTypedFx
6. **Missing Kernels**: Operations that TypedFxBulk can't lower
7. **Missing Patterns**: High-level patterns EqSat doesn't recognize
8. **Recommendations**: Prioritized implementation list

### `gap-summary.json`

```json
{
  "model_id": "$model_id",
  "furthest_stage": "export|fxtypedfx|rewrite|bulk|loopy|tron|cpp|success",
  "fixes_applied": ["description of each fix"],
  "missing_ops": ["aten.op_name"],
  "missing_kernels": ["kernel_name"],
  "missing_patterns": ["pattern description"],
  "blockers": [
    {
      "stage": "stage_name",
      "severity": "critical|high|medium|low",
      "description": "what is missing",
      "file": "path/to/file",
      "effort": "small|medium|large"
    }
  ]
}
```
$consensus_section

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
