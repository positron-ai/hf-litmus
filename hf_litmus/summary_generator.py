from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .feature_areas import (
    classify_op_area,
    compute_model_assessment,
    compute_salient_points,
    load_gap_summaries,
)
from .models import ModelResult, ModelStatus

if TYPE_CHECKING:
    from .state import StateManager

SCHEMA_VERSION = 3


def generate_summary(state: StateManager, output_dir: Path) -> None:
    """Generate summary.md and summary.json."""
    results = state.all_results()
    if not results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    gap_summaries = load_gap_summaries(output_dir)
    summary_data = _compute_summary(results, output_dir, gap_summaries)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary_data, indent=2, default=str))

    md_path = output_dir / "summary.md"
    md_path.write_text(_format_markdown(summary_data))


def _compute_summary(
    results: list[ModelResult],
    output_dir: Path,
    gap_summaries: dict[str, dict],
) -> dict:
    """Compute aggregate statistics with enrichment."""
    total = len(results)

    status_counts = Counter(r.status.value for r in results)
    passed = status_counts.get(ModelStatus.SUCCESS.value, 0)

    failure_classes = Counter(
        r.failure_class.value for r in results if r.failure_class is not None
    )

    ops_counter: Counter[str] = Counter()
    for r in results:
        for op in r.missing_ops:
            ops_counter[op] += 1

    top_ops = ops_counter.most_common(50)

    failed = [r for r in results if r.status != ModelStatus.SUCCESS]
    high_impact = sorted(failed, key=lambda r: r.downloads, reverse=True)[:20]

    # Compute per-model assessments
    high_impact_entries = []
    for r in high_impact:
        gap = gap_summaries.get(r.model_id)
        assessment = compute_model_assessment(r, gap)
        high_impact_entries.append(
            {
                "model_id": r.model_id,
                "downloads": r.downloads,
                "failure_class": (
                    r.failure_class.value if r.failure_class else "unknown"
                ),
                "missing_ops": r.missing_ops[:5],
                "assessment": assessment,
            }
        )

    # Compute salient points
    salient = compute_salient_points(results, gap_summaries)

    # Failure origin distribution
    origin_counts: Counter[str] = Counter()
    retryable_count = 0
    for r in results:
        if r.status != ModelStatus.SUCCESS:
            origin = r.failure_origin.value if r.failure_origin else "unknown"
            origin_counts[origin] += 1
            if r.retryable:
                retryable_count += 1

    # Model tag distribution
    tag_counts: Counter[str] = Counter()
    for r in results:
        for tag in r.model_tags:
            tag_counts[tag] += 1

    # Build per-model metadata entries
    model_entries = []
    for r in results:
        gap = gap_summaries.get(r.model_id)
        assessment = compute_model_assessment(r, gap)

        ops_by_area: dict[str, list[str]] = {}
        for op in r.missing_ops:
            area = classify_op_area(op)
            ops_by_area.setdefault(area, []).append(op)

        missing_areas: set[str] = set(ops_by_area.keys())
        if gap:
            for p in gap.get("missing_patterns", []):
                if isinstance(p, str):
                    missing_areas.add(p)
            for k in gap.get("missing_kernels", []):
                if isinstance(k, str):
                    missing_areas.add(k)
        missing_areas.discard("other")

        entry: dict = {
            "model_id": r.model_id,
            "pipeline_tag": r.pipeline_tag,
            "downloads": r.downloads,
            "status": r.status.value,
            "failure_class": (
                r.failure_class.value if r.failure_class else None
            ),
            "failure_origin": (
                r.failure_origin.value if r.failure_origin else None
            ),
            "retryable": r.retryable,
            "model_tags": r.model_tags,
            "assessment": assessment,
            "missing_ops": r.missing_ops,
            "missing_feature_areas": sorted(missing_areas),
        }
        model_entries.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_models": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (round(passed / total * 100, 2) if total > 0 else 0),
        "status_breakdown": dict(status_counts),
        "failure_class_breakdown": dict(failure_classes),
        "failure_origin_breakdown": dict(origin_counts),
        "retryable_count": retryable_count,
        "tag_distribution": dict(tag_counts.most_common()),
        "salient_points": salient,
        "top_missing_ops": [
            {
                "op": op,
                "count": count,
                "percentage": round(count / total * 100, 2),
            }
            for op, count in top_ops
        ],
        "high_impact_failures": high_impact_entries,
        "models": model_entries,
    }


def _format_markdown(data: dict) -> str:
    """Format summary data as Markdown."""
    lines = [
        "# HF Litmus Summary Report",
        "",
        f"*Generated: {data['generated_at']}*",
        "",
    ]

    # Executive summary from salient points
    salient = data.get("salient_points", {})
    if salient:
        lines.extend(_format_executive_summary(data, salient))

    lines.extend(
        [
            "## Overall Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Models Tested | {data['total_models']:,} |",
            f"| Passed | {data['passed']:,} |",
            f"| Failed | {data['failed']:,} |",
            f"| Pass Rate | {data['pass_rate']}% |",
            "",
            "## Status Breakdown",
            "",
            "| Status | Count |",
            "|--------|-------|",
        ]
    )

    for status, count in sorted(data["status_breakdown"].items()):
        lines.append(f"| {status} | {count:,} |")

    lines.extend(
        [
            "",
            "## Failure Classification",
            "",
            "| Class | Count |",
            "|-------|-------|",
        ]
    )

    for cls, count in sorted(
        data["failure_class_breakdown"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        lines.append(f"| {cls} | {count:,} |")

    # Failure origin breakdown
    origin_bd = data.get("failure_origin_breakdown", {})
    if origin_bd:
        lines.extend(
            [
                "",
                "## Failure Origin Breakdown",
                "",
                "| Origin | Count |",
                "|--------|-------|",
            ]
        )
        for origin, count in sorted(
            origin_bd.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"| {origin} | {count:,} |")

    retryable = data.get("retryable_count", 0)
    if retryable:
        lines.append("")
        lines.append(f"**Retryable failures:** {retryable:,}")

    # Tag distribution
    tag_dist = data.get("tag_distribution", {})
    if tag_dist:
        lines.extend(
            [
                "",
                "## Model Feature Tags",
                "",
                "| Tag | Count |",
                "|-----|-------|",
            ]
        )
        for tag, count in sorted(
            tag_dist.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"| {tag} | {count:,} |")

    lines.extend(
        [
            "",
            "## Top Missing Operations",
            "",
            "Operations ranked by models affected:",
            "",
            "| Rank | Operation | Count | % |",
            "|------|-----------|-------|---|",
        ]
    )

    for i, op in enumerate(data["top_missing_ops"][:20], 1):
        lines.append(
            f"| {i} | `{op['op']}` | {op['count']} | {op['percentage']}% |"
        )

    lines.extend(
        [
            "",
            "## High-Impact Failures",
            "",
            "Top downloaded models that fail:",
            "",
            ("| Model | Downloads | Class | Verdict | Effort | Blockers |"),
            ("|-------|-----------|------|---------|--------|----------|"),
        ]
    )

    for item in data["high_impact_failures"]:
        blockers = ", ".join(f"`{op}`" for op in item["missing_ops"])
        assessment = item.get("assessment", {})
        verdict = assessment.get("verdict", "?")
        effort = assessment.get("effort_estimate", "?")
        lines.append(
            f"| {item['model_id']} |"
            f" {item['downloads']:,} |"
            f" {item['failure_class']} |"
            f" {verdict} |"
            f" {effort} |"
            f" {blockers or 'N/A'} |"
        )

    # Consensus review summary (if any models had it)
    consensus_models = [
        item
        for item in data["high_impact_failures"]
        if item.get("assessment", {}).get("confidence") == "consensus"
    ]
    if consensus_models:
        lines.extend(
            [
                "",
                "## Consensus-Reviewed Models",
                "",
                "Models with multi-model consensus"
                " validation (Gemini 3 Pro + GPT 5.2 Pro):",
                "",
                "| Model | Verdict | Confidence |",
                "|-------|---------|------------|",
            ]
        )
        for item in consensus_models:
            a = item.get("assessment", {})
            lines.append(
                f"| {item['model_id']}"
                f" | {a.get('verdict', '?')}"
                f" | {a.get('confidence', '?')} |"
            )

    return "\n".join(lines) + "\n"


def _format_executive_summary(data: dict, salient: dict) -> list[str]:
    """Format the executive summary section."""
    lines = [
        "## Executive Summary",
        "",
    ]

    # Missing feature areas
    areas = salient.get("missing_feature_areas", [])
    if areas:
        lines.append("**Missing feature areas:**")
        for area in areas:
            lines.append(f"- {area}")
        lines.append("")

    # Missing ops (just the list)
    ops = salient.get("missing_ops", [])
    if ops:
        lines.append(f"**Missing ops** ({len(ops)} total):")
        for op in ops[:15]:
            lines.append(f"- `{op}`")
        if len(ops) > 15:
            lines.append(f"- ... and {len(ops) - 15} more")
        lines.append("")

    # Systemic issues
    issues = salient.get("systemic_issues", [])
    if issues:
        lines.append("**Systemic issues:**")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    # Passing architectures
    passing = salient.get("observed_passing_architectures", [])
    if passing:
        lines.append(
            "**Observed passing architectures:** " + ", ".join(passing)
        )
        lines.append("")

    lines.append("")
    return lines
