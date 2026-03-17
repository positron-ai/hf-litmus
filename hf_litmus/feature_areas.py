from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from .models import (
  FailureClass,
  FurthestStage,
  ModelResult,
  ModelStatus,
  Verdict,
)

logger = logging.getLogger(__name__)

_VERDICT_VALUES = {v.value for v in Verdict}
_STAGE_VALUES = {s.value for s in FurthestStage}

# Curated mapping of aten ops to feature areas.
# When a missing op matches a key prefix, it maps to
# the corresponding feature area.
OP_FEATURE_MAP: dict[str, str] = {
  "aten.scatter": "MoE",
  "aten.gather": "MoE",
  "aten.index_put": "MoE",
  "aten.repeat": "attention_variants",
  "aten.repeat_interleave": "attention_variants",
  "aten.expand": "attention_variants",
  "aten.new_zeros": "tensor_construction",
  "aten.new_ones": "tensor_construction",
  "aten.new_full": "tensor_construction",
  "aten.new_empty": "tensor_construction",
  "aten.zeros": "tensor_construction",
  "aten.ones": "tensor_construction",
  "aten.full": "tensor_construction",
  "aten.empty": "tensor_construction",
  "aten.arange": "tensor_construction",
  "aten.linspace": "tensor_construction",
  "aten.where": "conditional_ops",
  "aten.masked_fill": "conditional_ops",
  "aten.masked_select": "conditional_ops",
  "aten.index_select": "indexing",
  "aten.index": "indexing",
  "aten.embedding": "embedding",
  "aten.one_hot": "embedding",
  "aten.conv1d": "convolution",
  "aten.conv2d": "convolution",
  "aten.conv3d": "convolution",
  "aten.batch_norm": "normalization",
  "aten.group_norm": "normalization",
  "aten.instance_norm": "normalization",
  "aten.layer_norm": "normalization",
  "aten.rms_norm": "normalization",
  "aten.softmax": "activation",
  "aten.log_softmax": "activation",
  "aten.gelu": "activation",
  "aten.silu": "activation",
  "aten.relu": "activation",
  "aten.sigmoid": "activation",
  "aten.tanh": "activation",
  "aten.mish": "activation",
  "aten.cumsum": "scan_ops",
  "aten.cumprod": "scan_ops",
  "aten.sort": "sorting",
  "aten.topk": "sorting",
  "aten.argsort": "sorting",
  "aten.unique": "deduplication",
  "aten.nonzero": "dynamic_shape",
  "aten.item": "dynamic_shape",
  "aten.roll": "position_encoding",
  "aten.tril": "causal_mask",
  "aten.triu": "causal_mask",
  "aten.baddbmm": "attention_variants",
  "aten.addmm": "linear",
  "aten.mm": "linear",
  "aten.bmm": "linear",
  "aten.matmul": "linear",
  "aten.linear": "linear",
  "operator.mul": "python_ops",
  "operator.add": "python_ops",
  "operator.sub": "python_ops",
  "operator.truediv": "python_ops",
  "operator.getitem": "python_ops",
}

# Failure class to verdict/effort heuristic mapping.
# Used when deep analysis is not available.
FAILURE_CLASS_HEURISTICS: dict[str, dict] = {
  "missing_op": {
    "small_ops_threshold": 2,
    "medium_ops_threshold": 5,
  },
  "type_error": {
    "verdict": "far",
    "effort": "medium",
  },
  "unsupported_dynamic": {
    "verdict": "blocked",
    "effort": "large",
  },
  "aten_fallback": {
    "verdict": "far",
    "effort": "medium",
  },
  "shape_mismatch": {
    "verdict": "far",
    "effort": "medium",
  },
  "memory_error": {
    "verdict": "blocked",
    "effort": "large",
  },
  "trust_remote_code": {
    "verdict": "close",
    "effort": "small",
  },
  "unknown": {
    "verdict": "far",
    "effort": "medium",
  },
}


def _normalize_op(op: object) -> str:
  """Extract op name string from gap-summary entries.

  Gap-summary.json missing_ops entries may be plain
  strings or dicts with 'op' or 'name' keys.
  """
  if isinstance(op, str):
    return op
  if isinstance(op, dict):
    return op.get("op") or op.get("name") or ""
  return ""


def classify_op_area(op: object) -> str:
  """Map an op name to its feature area."""
  name = _normalize_op(op)
  if not name:
    return "other"
  for prefix, area in OP_FEATURE_MAP.items():
    if name.startswith(prefix):
      return area
  return "other"


def compute_model_assessment(
  result: ModelResult,
  gap_data: Optional[dict] = None,
) -> dict:
  """Compute a concise assessment for a single model.

  Returns a dict with: verdict, furthest_stage,
  blocker_count, effort_estimate, confidence, missing_areas.
  """
  if result.status == ModelStatus.SUCCESS:
    return {
      "verdict": "pass",
      "verdict_detail": None,
      "furthest_stage": "success",
      "blocker_count": 0,
      "missing_ops_count": 0,
      "effort_estimate": "none",
      "confidence": "high",
      "missing_areas": [],
    }

  # If deep analysis gap data is available, use it
  if gap_data:
    return _assessment_from_gap(result, gap_data)

  # Heuristic fallback
  return _assessment_heuristic(result)


def _classify_prose_verdict(prose: str) -> str:
  """Classify a prose verdict string into a keyword."""
  lower = prose.lower()
  if "blocked" in lower or "not supported" in lower:
    return "blocked"
  if "far" in lower or "significant" in lower:
    return "far"
  if (
    "close" in lower
    or "minor" in lower
    or "pass" in lower
  ):
    return "close"
  return "far"


def _assessment_from_gap(
  result: ModelResult,
  gap_data: dict,
) -> dict:
  """Build assessment from gap-summary.json data."""
  blockers = gap_data.get("blockers", [])
  critical = [
    b for b in blockers
    if isinstance(b, dict)
    and b.get("severity") in ("critical", "high")
  ]

  missing_ops = gap_data.get("missing_ops", [])
  missing_kernels = gap_data.get("missing_kernels", [])
  missing_patterns = gap_data.get("missing_patterns", [])

  total_blockers = len(critical)

  # Determine effort from blocker efforts
  efforts = [
    b.get("effort", "medium") for b in blockers
    if isinstance(b, dict)
  ]
  if any(e == "large" for e in efforts):
    effort = "large"
  elif len(efforts) > 3:
    effort = "large"
  elif efforts:
    effort = "medium"
  else:
    effort = "small"

  # Determine verdict
  if total_blockers == 0 and not missing_ops:
    verdict = "close"
  elif total_blockers <= 2:
    verdict = "close"
  elif total_blockers <= 5:
    verdict = "far"
  else:
    verdict = "blocked"

  # Check for fundamental blockers
  for b in blockers:
    if not isinstance(b, dict):
      continue
    desc = b.get("description", "").lower()
    if any(
      kw in desc
      for kw in [
        "architecture", "fundamentally",
        "not supported", "dynamic control",
      ]
    ):
      verdict = "blocked"
      effort = "large"
      break

  # Compute missing areas from gap data
  areas: set[str] = set()
  for op in missing_ops:
    name = _normalize_op(op)
    if name:
      areas.add(classify_op_area(name))
  for p in missing_patterns:
    if isinstance(p, str):
      areas.add(p)
  for k in missing_kernels:
    if isinstance(k, str):
      areas.add(k)
  areas.discard("other")

  confidence = "high"
  verdict_detail: Optional[str] = None

  # If consensus review is present, incorporate it
  consensus = gap_data.get("consensus_review")
  if isinstance(consensus, dict):
    confidence = "consensus"
    ext = consensus.get("external_consensus")
    if isinstance(ext, dict):
      agreed = ext.get("agreed_verdict", "")
      if agreed and isinstance(agreed, str):
        if agreed in _VERDICT_VALUES:
          verdict = agreed
        else:
          verdict_detail = agreed
          verdict = _classify_prose_verdict(agreed)
      additional = ext.get("additional_risks", [])
      if isinstance(additional, list):
        areas.update(
          r for r in additional
          if isinstance(r, str)
        )

  # Validate furthest_stage against enum
  raw_stage = gap_data.get(
    "furthest_stage", "unknown"
  )
  stage = (
    raw_stage if raw_stage in _STAGE_VALUES
    else "unknown"
  )

  return {
    "verdict": verdict,
    "verdict_detail": verdict_detail,
    "furthest_stage": stage,
    "blocker_count": total_blockers,
    "missing_ops_count": len(result.missing_ops),
    "effort_estimate": effort,
    "confidence": confidence,
    "missing_areas": sorted(areas),
  }


def _assessment_heuristic(
  result: ModelResult,
) -> dict:
  """Build assessment from ModelResult only."""
  fc = (
    result.failure_class.value
    if result.failure_class
    else "unknown"
  )
  heuristic = FAILURE_CLASS_HEURISTICS.get(
    fc, FAILURE_CLASS_HEURISTICS["unknown"]
  )

  if fc == "missing_op":
    n = len(result.missing_ops)
    thresh = heuristic
    if n <= thresh["small_ops_threshold"]:
      verdict = "close"
      effort = "small"
    elif n <= thresh["medium_ops_threshold"]:
      verdict = "far"
      effort = "medium"
    else:
      verdict = "far"
      effort = "large"
  else:
    verdict = heuristic.get("verdict", "far")
    effort = heuristic.get("effort", "medium")

  # Compute missing areas from ops
  areas: set[str] = set()
  for op in result.missing_ops:
    areas.add(classify_op_area(op))
  areas.discard("other")

  raw_stage = (
    result.failure_stage.value
    if result.failure_stage
    else "unknown"
  )
  stage = (
    raw_stage if raw_stage in _STAGE_VALUES
    else "unknown"
  )

  return {
    "verdict": verdict,
    "verdict_detail": None,
    "furthest_stage": stage,
    "blocker_count": 0,
    "missing_ops_count": len(result.missing_ops),
    "effort_estimate": effort,
    "confidence": "heuristic",
    "missing_areas": sorted(areas),
  }


def load_gap_summaries(
  output_dir: Path,
) -> dict[str, dict]:
  """Scan analyses directory for gap-summary.json files.

  Returns a dict mapping model_id to gap data.
  """
  analyses_dir = output_dir / "analyses"
  if not analyses_dir.exists():
    return {}

  gap_summaries: dict[str, dict] = {}
  for gap_path in analyses_dir.glob(
    "*/gap-summary.json"
  ):
    try:
      data = json.loads(gap_path.read_text())
      model_id = data.get("model_id", "")
      if model_id:
        gap_summaries[model_id] = data
    except (json.JSONDecodeError, KeyError) as e:
      logger.warning(
        "Failed to parse %s: %s", gap_path, e
      )

  return gap_summaries


def compute_salient_points(
  results: list[ModelResult],
  gap_summaries: Optional[dict[str, dict]] = None,
) -> dict:
  """Compute the top-level salient_points section.

  Returns a dict with:
  - missing_feature_areas: sorted list of areas
  - missing_ops: flat deduplicated list
  - systemic_issues: list of systemic problems
  - observed_passing_architectures: pipeline_tags
      of successful models
  - sources: provenance of this data
  """
  if gap_summaries is None:
    gap_summaries = {}

  # Collect all missing ops across failures
  all_missing: Counter[str] = Counter()
  all_areas: Counter[str] = Counter()
  systemic: set[str] = set()
  passing_tags: set[str] = set()

  for r in results:
    if r.status == ModelStatus.SUCCESS:
      if r.pipeline_tag:
        passing_tags.add(r.pipeline_tag)
      continue

    # Collect missing ops
    for op in r.missing_ops:
      all_missing[op] += 1
      area = classify_op_area(op)
      if area != "other":
        all_areas[area] += 1

    # Detect systemic issues
    if r.failure_class == FailureClass.TRUST_REMOTE_CODE:
      systemic.add("trust_remote_code not supported")
    elif (
      r.failure_class
      == FailureClass.UNSUPPORTED_DYNAMIC
    ):
      systemic.add(
        "data-dependent control flow"
      )
    elif r.failure_class == FailureClass.MEMORY_ERROR:
      systemic.add("memory limits")

  # Enrich from gap summaries
  for model_id, gap in gap_summaries.items():
    for p in gap.get("missing_patterns", []):
      if isinstance(p, str):
        all_areas[p] += 1
    for k in gap.get("missing_kernels", []):
      if isinstance(k, str):
        all_areas[k] += 1

  sources = ["aggregate"]
  if gap_summaries:
    sources.append("gap_summary")

  return {
    "missing_feature_areas": [
      area
      for area, _ in all_areas.most_common()
    ],
    "missing_ops": [
      op for op, _ in all_missing.most_common()
    ],
    "systemic_issues": sorted(systemic),
    "observed_passing_architectures": sorted(
      passing_tags
    ),
    "sources": sources,
  }
