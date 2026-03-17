from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from .feature_areas import (
  classify_op_area,
  compute_model_assessment,
)
from .models import ModelResult, ModelStatus

METADATA_SCHEMA_VERSION = 3


def generate_report(
  result: ModelResult,
  output_dir: Path,
  error_output: str = "",
) -> Path:
  """Generate a Markdown report for a model result."""
  model_id = result.model_id
  if "/" in model_id:
    org, name = model_id.split("/", 1)
  else:
    org, name = "_local", model_id

  report_dir = output_dir / "reports" / org
  report_dir.mkdir(parents=True, exist_ok=True)
  report_path = report_dir / f"{name}.md"

  frontmatter = {
    "model_id": result.model_id,
    "status": result.status.value,
    "tested_at": result.tested_at.isoformat(),
    "failure_stage": (
      result.failure_stage.value
      if result.failure_stage
      else None
    ),
    "failure_class": (
      result.failure_class.value
      if result.failure_class
      else None
    ),
    "missing_ops": result.missing_ops,
    "ingest_version": result.ingest_version,
    "pipeline_tag": result.pipeline_tag,
    "model_downloads": result.downloads,
    "model_likes": result.likes,
    "failure_origin": (
      result.failure_origin.value
      if result.failure_origin else None
    ),
    "retryable": result.retryable,
    "model_tags": result.model_tags,
  }

  lines = [
    "---",
    yaml.dump(
      frontmatter, default_flow_style=False
    ).strip(),
    "---",
    "",
    f"# Ingest Report: {result.model_id}",
    "",
    "## Model Information",
    "",
    "| Field | Value |",
    "|-------|-------|",
    f"| Model ID | {result.model_id} |",
    f"| Pipeline Tag | {result.pipeline_tag} |",
    f"| Downloads | {result.downloads:,} |",
    f"| Likes | {result.likes:,} |",
    (
      "| Tested At | "
      f"{result.tested_at:%Y-%m-%d %H:%M:%S UTC} |"
    ),
    f"| Ingest Version | {result.ingest_version} |",
  ]
  if result.model_tags:
    lines.append(
      f"| Model Tags | {', '.join(result.model_tags)} |"
    )
  if result.failure_origin:
    lines.append(
      f"| Failure Origin | {result.failure_origin.value} |"
    )
  if result.retryable:
    lines.append("| Retryable | Yes |")
  lines.append("")

  if result.status == ModelStatus.SUCCESS:
    lines.extend([
      "## Result: SUCCESS",
      "",
      "Tron ingest successfully generated a C++ plugin"
      " for this model.",
    ])
  else:
    stage = (
      result.failure_stage.value
      if result.failure_stage
      else "unknown"
    )
    fclass = (
      result.failure_class.value
      if result.failure_class
      else "unknown"
    )
    lines.extend([
      f"## Result: {result.status.value}",
      "",
      "### Failure Classification",
      "",
      f"**Stage:** {stage}",
      f"**Class:** {fclass}",
    ])

    if result.missing_ops:
      ops = ", ".join(
        f"`{op}`" for op in result.missing_ops
      )
      lines.append(f"**Missing Operations:** {ops}")

    if error_output:
      lines.extend([
        "",
        "### Error Output",
        "",
        "```",
        error_output.strip()[:5000],
        "```",
      ])

    lines.extend([
      "",
      "### Analysis",
      "",
      _generate_analysis(result),
    ])

    if result.analysis_path:
      ap = str(result.analysis_path)
      if "analyses/" in ap:
        rel = ap.split("analyses/", 1)[-1]
        url = f"/analyses/{rel}"
        display = f"analyses/{rel}"
      else:
        url = ap
        display = ap
      lines.extend([
        "",
        "### Deep Analysis",
        "",
        "A detailed gap analysis was performed"
        " by Claude Code in a git worktree.",
        "",
        f"- **Analysis**: [{display}]"
        f"({url})",
      ])
      if result.analysis_branch:
        lines.append(
          f"- **Branch**: `{result.analysis_branch}`"
        )

  report_path.write_text("\n".join(lines) + "\n")
  return report_path


def _generate_analysis(result: ModelResult) -> str:
  """Generate human-readable analysis."""
  if result.failure_class is None:
    return "No specific analysis available."

  fc = result.failure_class.value
  analyses = {
    "missing_op": (
      f"The ingest pipeline encountered"
      f" {len(result.missing_ops)} unsupported"
      f" operation(s): {', '.join(result.missing_ops)}."
      " These need implementation in the FxTypedFx"
      " translation layer."
    ),
    "type_error": (
      "Tensor shapes or types are incompatible"
      " with the TypedFx type checker. This may"
      " indicate unusual architecture or dynamic"
      " shape patterns."
    ),
    "unsupported_dynamic": (
      "The model uses data-dependent control flow"
      " that torch.export cannot trace statically."
      " This is a PyTorch limitation."
    ),
    "aten_fallback": (
      "An unsupported ATen operator was encountered"
      " during export. The operator needs a"
      " decomposition or custom implementation."
    ),
    "shape_mismatch": (
      "A shape or configuration mismatch was"
      " detected. The model config may have"
      " unexpected or missing fields."
    ),
    "memory_error": (
      "The process ran out of memory."
      " Try reducing --max-seq-length or"
      " running on a machine with more RAM."
    ),
    "trust_remote_code": (
      "The model requires trust_remote_code=True"
      " because it uses custom code hosted in the"
      " HuggingFace repository. The export tool"
      " needs to be updated to pass this flag to"
      " transformers API calls."
    ),
  }

  return analyses.get(
    fc,
    f"Classified as '{fc}'."
    " Manual investigation may be needed.",
  )


def generate_model_metadata(
  result: ModelResult,
  output_dir: Path,
  gap_data: Optional[dict] = None,
) -> Path:
  """Generate a per-model JSON metadata file.

  Produces a machine-readable overview of the model's
  compatibility status, capabilities, requirements, and
  platform support gaps. Omits verbose error output and
  explanatory text.
  """
  model_id = result.model_id
  if "/" in model_id:
    org, name = model_id.split("/", 1)
  else:
    org, name = "_local", model_id

  report_dir = output_dir / "reports" / org
  report_dir.mkdir(parents=True, exist_ok=True)
  meta_path = report_dir / f"{name}.json"

  assessment = compute_model_assessment(
    result, gap_data
  )

  # Map missing ops to feature areas
  ops_by_area: dict[str, list[str]] = {}
  for op in result.missing_ops:
    area = classify_op_area(op)
    ops_by_area.setdefault(area, []).append(op)

  # Build platform compatibility section
  all_areas = set(ops_by_area.keys())
  if gap_data:
    for p in gap_data.get("missing_patterns", []):
      if isinstance(p, str):
        all_areas.add(p)
    for k in gap_data.get("missing_kernels", []):
      if isinstance(k, str):
        all_areas.add(k)
  all_areas.discard("other")

  blockers = []
  if gap_data:
    for b in gap_data.get("blockers", []):
      blockers.append({
        "description": b.get("description", ""),
        "severity": b.get("severity", "unknown"),
        "effort": b.get("effort", "unknown"),
      })

  metadata: dict = {
    "metadata_schema_version": METADATA_SCHEMA_VERSION,
    "model_id": result.model_id,
    "pipeline_tag": result.pipeline_tag,
    "downloads": result.downloads,
    "likes": result.likes,
    "tested_at": result.tested_at.isoformat(),
    "ingest_version": result.ingest_version,
    "status": result.status.value,
    "failure_stage": (
      result.failure_stage.value
      if result.failure_stage
      else None
    ),
    "failure_class": (
      result.failure_class.value
      if result.failure_class
      else None
    ),
    "failure_origin": (
      result.failure_origin.value
      if result.failure_origin else None
    ),
    "retryable": result.retryable,
    "model_tags": result.model_tags,
    "deep_analysis_error": (
      result.deep_analysis_error or None
    ),
    "assessment": assessment,
    "platform_compatibility": {
      "missing_ops": result.missing_ops,
      "missing_ops_by_area": ops_by_area,
      "missing_feature_areas": sorted(all_areas),
      "blockers": blockers,
    },
  }

  if gap_data:
    metadata["gap_analysis"] = {
      "status": "completed",
      "furthest_stage": gap_data.get(
        "furthest_stage", None
      ),
      "missing_kernels": gap_data.get(
        "missing_kernels", []
      ),
      "missing_patterns": gap_data.get(
        "missing_patterns", []
      ),
      "fixes_applied": gap_data.get(
        "fixes_applied", []
      ),
      "consensus_review": gap_data.get(
        "consensus_review", None
      ),
    }
  else:
    metadata["gap_analysis"] = {
      "status": "not_run",
      "furthest_stage": None,
      "missing_kernels": [],
      "missing_patterns": [],
      "fixes_applied": [],
      "consensus_review": None,
    }

  meta_path.write_text(
    json.dumps(metadata, indent=2, default=str)
    + "\n"
  )
  return meta_path
