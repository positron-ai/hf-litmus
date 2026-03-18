from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Model ID must be org/name with no leading/trailing dots
# and no consecutive dots per segment.
_MODEL_ID_RE = re.compile(
  r"[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*"
  r"/"
  r"[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*"
)

# Maximum POST body size (64 KB)
_MAX_BODY = 64 * 1024

# Maximum retry models per request
_MAX_RETRY_MODELS = 50

# Maximum analysis tracker output lines
_MAX_ANALYSIS_LINES = 10_000

# Maximum completed retry jobs to keep
_MAX_COMPLETED_JOBS = 200



def create_dashboard_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="hf-litmus dashboard",
    description=("Summarize model compatibility from JSON reports"),
    formatter_class=(argparse.ArgumentDefaultsHelpFormatter),
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("./litmus-output"),
    help="Directory containing reports/",
  )
  parser.add_argument(
    "--format",
    choices=["terminal", "csv", "json", "html"],
    default="terminal",
    help="Output format",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=None,
    help="Write output to file (default: stdout)",
  )
  parser.add_argument(
    "--sort",
    choices=["verdict", "name", "downloads", "ops"],
    default="verdict",
    help="Sort order for failing models",
  )
  parser.add_argument(
    "--serve",
    action="store_true",
    default=False,
    help=(
      "Start a local HTTP server to view the"
      " dashboard in a browser; rescans reports"
      " every --refresh-interval minutes"
    ),
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8150,
    help="Port for the HTTP server (--serve)",
  )
  parser.add_argument(
    "--refresh-interval",
    type=int,
    default=15,
    help=("Minutes between report rescans when serving (--serve)"),
  )
  return parser


def dashboard_main(argv: list[str]) -> int:
  parser = create_dashboard_parser()
  args = parser.parse_args(argv)

  reports_dir = args.output_dir / "reports"
  if not reports_dir.exists():
    print(
      f"No reports directory found at {reports_dir}",
      file=sys.stderr,
    )
    return 1

  if args.serve:
    return _serve_dashboard(
      args.output_dir,
      args.sort,
      args.port,
      args.refresh_interval,
    )

  reports = _load_reports(reports_dir)
  if not reports:
    print("No JSON reports found.", file=sys.stderr)
    return 1

  data = _compute_dashboard(reports, args.sort, args.output_dir)

  if args.format == "csv":
    text = _render_csv(data)
  elif args.format == "json":
    text = _render_json(data)
  elif args.format == "html":
    text = _render_html(data)
  else:
    text = _render_terminal(data)

  if args.output:
    args.output.write_text(text)
    print(f"Wrote {args.format} to {args.output}")
  else:
    print(text)

  return 0


def _load_reports(reports_dir: Path) -> list[dict]:
  """Load all per-model JSON metadata files."""
  reports: list[dict] = []
  for json_path in sorted(reports_dir.rglob("*.json")):
    try:
      data = json.loads(json_path.read_text())
      if "model_id" in data and "status" in data:
        reports.append(data)
    except (json.JSONDecodeError, KeyError):
      continue
  return reports


# Verdict ordering for sorting (lower = closer)
_VERDICT_ORDER = {
  "pass": 0,
  "close": 1,
  "far": 2,
  "blocked": 3,
}

_EFFORT_ORDER = {
  "none": 0,
  "small": 1,
  "medium": 2,
  "large": 3,
}

_KNOWN_VERDICTS = {"pass", "close", "far", "blocked"}


def _normalize_verdict(assessment: dict) -> str:
  """Extract a short verdict label from assessment.

  With v2 schema, verdict is always a keyword. Keeps
  fallback inference for any straggling v1 reports.
  """
  raw = assessment.get("verdict", "unknown")
  if raw in _KNOWN_VERDICTS:
    return raw

  # v1 fallback: infer from numeric fields
  blockers = assessment.get("blocker_count", 0)
  effort = assessment.get("effort_estimate", "medium")
  if blockers == 0 and effort in ("none", "small"):
    return "close"
  elif blockers <= 2 and effort != "large":
    return "close"
  elif blockers <= 5:
    return "far"
  else:
    return "blocked"


def _sort_key(report: dict, sort: str):
  """Return a sort key for a report entry."""
  assessment = report.get("assessment", {})
  verdict = _normalize_verdict(assessment)
  if sort == "name":
    return report.get("model_id", "")
  elif sort == "downloads":
    return -report.get("downloads", 0)
  elif sort == "ops":
    compat = report.get("platform_compatibility", {})
    return len(compat.get("missing_ops", []))
  else:
    # verdict: group by verdict, then by effort, then
    # by missing ops count ascending
    compat = report.get("platform_compatibility", {})
    return (
      _VERDICT_ORDER.get(verdict, 9),
      _EFFORT_ORDER.get(assessment.get("effort_estimate", "large"), 9),
      len(compat.get("missing_ops", [])),
    )


def _compute_dashboard(
  reports: list[dict],
  sort: str,
  output_dir: Path | None = None,
) -> dict:
  """Aggregate report data into dashboard."""
  # Backfill model feature tags from HF configs
  if output_dir is not None:
    from .model_tags import ensure_tags

    ensure_tags(reports, output_dir)

  passing = []
  failing = []

  for r in reports:
    if r.get("status") == "SUCCESS":
      passing.append(r)
    else:
      failing.append(r)

  failing.sort(key=lambda r: _sort_key(r, sort))

  # Count feature areas across all failures
  area_counts: Counter[str] = Counter()
  for r in failing:
    compat = r.get("platform_compatibility", {})
    for area in compat.get("missing_feature_areas", []):
      area_counts[area] += 1

  # Count missing ops across all failures
  op_counts: Counter[str] = Counter()
  for r in failing:
    compat = r.get("platform_compatibility", {})
    for op in compat.get("missing_ops", []):
      op_counts[op] += 1

  # Verdict distribution
  verdict_counts: Counter[str] = Counter()
  for r in failing:
    assessment = r.get("assessment", {})
    verdict = _normalize_verdict(assessment)
    verdict_counts[verdict] += 1

  retryable_count = 0
  for r in failing:
    if r.get("retryable"):
      retryable_count += 1

  # Model tag distribution
  tag_counts: Counter[str] = Counter()
  all_tags: set[str] = set()
  for r in reports:
    for tag in r.get("model_tags", []):
      tag_counts[tag] += 1
      all_tags.add(tag)

  return {
    "total": len(reports),
    "passing": passing,
    "failing": failing,
    "pass_count": len(passing),
    "fail_count": len(failing),
    "area_counts": area_counts,
    "op_counts": op_counts,
    "verdict_counts": verdict_counts,
    "retryable_count": retryable_count,
    "tag_counts": tag_counts,
    "all_tags": sorted(all_tags),
  }


def _bar(count: int, max_count: int, width: int = 30) -> str:
  """Render an ASCII bar."""
  if max_count == 0:
    return ""
  filled = round(count / max_count * width)
  return "\u2588" * filled


def _render_terminal(data: dict) -> str:
  """Render dashboard as a terminal-friendly table."""
  lines: list[str] = []
  total = data["total"]
  pass_count = data["pass_count"]
  fail_count = data["fail_count"]

  pct = f"{pass_count / total * 100:.1f}%" if total > 0 else "N/A"
  lines.append("HF Litmus Dashboard")
  lines.append("=" * 60)
  lines.append("")
  lines.append(f"  Models tested:  {total}")
  lines.append(f"  Supported:      {pass_count}  ({pct})")
  lines.append(f"  Unsupported:    {fail_count}")
  lines.append("")

  # Verdict distribution
  vd = data["verdict_counts"]
  if vd:
    lines.append("  Verdict breakdown:")
    for v in ["close", "far", "blocked"]:
      if vd.get(v, 0) > 0:
        lines.append(f"    {v:10s}  {vd[v]}")
    lines.append("")

  # Supported models
  if data["passing"]:
    lines.append("SUPPORTED MODELS")
    lines.append("-" * 60)

    for r in sorted(
      data["passing"],
      key=lambda x: -x.get("downloads", 0),
    ):
      mid = r.get("model_id", "?")
      dl = r.get("downloads", 0)
      tag = r.get("pipeline_tag", "")
      lines.append(f"  + {mid:42s} {tag:18s} {dl:>10,} dl")
    lines.append("")

  # Failing models table
  if data["failing"]:
    lines.append("UNSUPPORTED MODELS")
    lines.append("-" * 60)

    # Header
    lines.append(
      f"  {'Model':<40s} {'Verdict':>8s}"
      f" {'Effort':>7s} {'Ops':>4s}"
      f" {'Areas':>5s}  Feature Areas"
    )
    lines.append("  " + "-" * 105)

    for r in data["failing"]:
      mid = r.get("model_id", "?")
      assessment = r.get("assessment", {})
      verdict = _normalize_verdict(assessment)
      effort = assessment.get("effort_estimate", "?")
      compat = r.get("platform_compatibility", {})
      n_ops = len(compat.get("missing_ops", []))
      feat_areas = compat.get("missing_feature_areas", [])
      n_areas = len(feat_areas)

      # Truncate feature areas for display
      areas_str = ", ".join(a for a in feat_areas[:4] if len(a) < 30)
      if len(feat_areas) > 4:
        areas_str += f", +{len(feat_areas) - 4}"

      lines.append(
        f"  {mid:<40s} {verdict:>8s}"
        f" {effort:>7s} {n_ops:>4d}"
        f" {n_areas:>5d}  {areas_str}"
      )
    lines.append("")

  # Feature area bar chart
  area_counts = data["area_counts"]
  if area_counts:
    lines.append("MISSING FEATURE AREAS (models affected)")
    lines.append("-" * 60)

    # Filter to short area names for the chart;
    # long ones (from gap analysis) are too verbose
    short_areas = {k: v for k, v in area_counts.items() if len(k) < 30}
    if short_areas:
      max_count = max(short_areas.values())
      for area, count in sorted(
        short_areas.items(),
        key=lambda x: -x[1],
      ):
        bar = _bar(count, max_count)
        lines.append(f"  {area:<25s} {bar} {count}")
      lines.append("")

    # Show verbose areas (from deep analysis)
    verbose_areas = {k: v for k, v in area_counts.items() if len(k) >= 30}
    if verbose_areas:
      lines.append("  Additional gaps (from deep analysis):")
      for area, count in sorted(
        verbose_areas.items(),
        key=lambda x: -x[1],
      ):
        lines.append(f"    [{count}] {area}")
      lines.append("")

  # Top missing ops
  op_counts = data["op_counts"]
  if op_counts:
    lines.append("TOP MISSING OPS (models affected)")
    lines.append("-" * 60)
    max_op = max(op_counts.values())
    for op, count in op_counts.most_common(15):
      bar = _bar(count, max_op, width=20)
      lines.append(f"  {op:<35s} {bar} {count}")
    lines.append("")

  return "\n".join(lines)


def _render_csv(data: dict) -> str:
  """Render dashboard data as CSV."""
  import io

  buf = io.StringIO()
  writer = csv.writer(buf)
  writer.writerow(
    [
      "model_id",
      "pipeline_tag",
      "downloads",
      "status",
      "verdict",
      "effort_estimate",
      "failure_origin",
      "retryable",
      "model_tags",
      "missing_ops_count",
      "missing_feature_areas_count",
      "blocker_count",
      "confidence",
      "feature_areas",
    ]
  )

  all_models = data["passing"] + data["failing"]
  all_models.sort(key=lambda r: r.get("model_id", ""))

  for r in all_models:
    assessment = r.get("assessment", {})
    compat = r.get("platform_compatibility", {})
    ops = compat.get("missing_ops", [])
    areas = compat.get("missing_feature_areas", [])
    writer.writerow(
      [
        r.get("model_id", ""),
        r.get("pipeline_tag", ""),
        r.get("downloads", 0),
        r.get("status", ""),
        _normalize_verdict(assessment),
        assessment.get("effort_estimate", ""),
        r.get("failure_origin", ""),
        r.get("retryable", False),
        "; ".join(r.get("model_tags", [])),
        len(ops),
        len(areas),
        assessment.get("blocker_count", 0),
        assessment.get("confidence", ""),
        "; ".join(areas),
      ]
    )

  return buf.getvalue()


def _render_json(data: dict) -> str:
  """Render dashboard data as JSON."""
  models = []
  all_models = data["passing"] + data["failing"]
  all_models.sort(key=lambda r: r.get("model_id", ""))

  for r in all_models:
    assessment = r.get("assessment", {})
    compat = r.get("platform_compatibility", {})
    models.append(
      {
        "model_id": r.get("model_id", ""),
        "pipeline_tag": r.get("pipeline_tag", ""),
        "downloads": r.get("downloads", 0),
        "status": r.get("status", ""),
        "verdict": _normalize_verdict(assessment),
        "effort_estimate": assessment.get("effort_estimate", ""),
        "failure_origin": r.get("failure_origin", None),
        "retryable": r.get("retryable", False),
        "model_tags": r.get("model_tags", []),
        "missing_ops_count": len(compat.get("missing_ops", [])),
        "missing_feature_areas_count": len(compat.get("missing_feature_areas", [])),
        "blocker_count": assessment.get("blocker_count", 0),
        "confidence": assessment.get("confidence", ""),
        "feature_areas": compat.get("missing_feature_areas", []),
      }
    )

  output = {
    "total_models": data["total"],
    "supported": data["pass_count"],
    "unsupported": data["fail_count"],
    "pass_rate_pct": round(data["pass_count"] / data["total"] * 100, 1)
    if data["total"] > 0
    else 0,
    "verdict_breakdown": dict(data["verdict_counts"]),
    "feature_area_frequency": dict(data["area_counts"].most_common()),
    "missing_op_frequency": dict(data["op_counts"].most_common()),
    "models": models,
  }

  return json.dumps(output, indent=2) + "\n"


def _esc(text: str) -> str:
  """Escape HTML special characters."""
  return (
    text.replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
    .replace("'", "&#39;")
  )


def _report_href(model_id: str) -> str:
  """Build URL path to a model's markdown report."""
  if "/" in model_id:
    org, name = model_id.split("/", 1)
  else:
    org, name = "_local", model_id
  return (
    f"/reports/"
    f"{urllib.parse.quote(org, safe='')}"
    f"/{urllib.parse.quote(name, safe='')}.md"
  )


def _model_link(model_id: str) -> str:
  """Render a model_id as an <a> link to its report."""
  href = _report_href(model_id)
  return f"<a href='{href}' class='model-link'>{_esc(model_id)}</a>"


_VERDICT_COLORS = {
  "pass": "#22c55e",
  "close": "#eab308",
  "far": "#f97316",
  "blocked": "#ef4444",
}

_EFFORT_COLORS = {
  "none": "#22c55e",
  "small": "#84cc16",
  "medium": "#eab308",
  "large": "#ef4444",
}


def _render_html(
  data: dict,
  refresh_seconds: int = 0,
  auth_token: str = "",
) -> str:
  """Render dashboard as self-contained HTML."""
  total = data["total"]
  pass_count = data["pass_count"]
  fail_count = data["fail_count"]
  pct = round(pass_count / total * 100, 1) if total > 0 else 0
  now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

  meta_refresh = ""
  if refresh_seconds > 0:
    meta_refresh = f'<meta http-equiv="refresh" content="{refresh_seconds}">'

  # Build verdict pie data
  vd = data["verdict_counts"]
  pie_parts = []
  offset = 0
  for v in ["close", "far", "blocked"]:
    c = vd.get(v, 0)
    if c == 0:
      continue
    pct_v = c / fail_count * 100 if fail_count else 0
    color = _VERDICT_COLORS.get(v, "#999")
    pie_parts.append(f"{color} {offset}% {offset + pct_v}%")
    offset += pct_v

  # Supported models rows
  supported_parts: list[str] = []
  for r in sorted(
    data["passing"],
    key=lambda x: -x.get("downloads", 0),
  ):
    mid = r.get("model_id", "?")
    link = _model_link(mid)
    tag = _esc(r.get("pipeline_tag", ""))
    dl = r.get("downloads", 0)
    mtags = r.get("model_tags", [])
    tags_str = ", ".join(_esc(t) for t in mtags)
    data_tags = _esc(",".join(mtags))
    mid_esc = _esc(mid)
    supported_parts.append(
      f"<tr data-tags='{data_tags}'"
      f" data-model='{mid_esc}'>"
      f"<td>{link}</td>"
      f"<td>{tag}</td>"
      f"<td class='num'>{dl:,}</td>"
      f"<td class='tags'>{tags_str}</td></tr>\n"
    )
  supported_rows = "".join(supported_parts)

  # Collect all observed tags for filter panel
  all_tags = data.get("all_tags", [])

  # Unsupported models rows
  unsupported_parts: list[str] = []
  for r in data["failing"]:
    mid = r.get("model_id", "?")
    mid_esc = _esc(mid)
    link = _model_link(mid)
    assessment = r.get("assessment", {})
    verdict = _normalize_verdict(assessment)
    effort = assessment.get("effort_estimate", "?")
    retryable = r.get("retryable", False)
    mtags = r.get("model_tags", [])
    compat = r.get("platform_compatibility", {})
    n_ops = len(compat.get("missing_ops", []))
    feat_areas = compat.get("missing_feature_areas", [])
    n_areas = len(feat_areas)
    short = [_esc(a) for a in feat_areas[:3] if len(a) < 40]
    areas_str = ", ".join(short)
    if len(feat_areas) > 3:
      areas_str += f", +{len(feat_areas) - 3}"
    vc = _VERDICT_COLORS.get(verdict, "#999")
    ec = _EFFORT_COLORS.get(effort, "#999")
    tags_str = ", ".join(_esc(t) for t in mtags)
    data_tags = _esc(",".join(mtags))
    status = _esc(r.get("status", ""))
    unsupported_parts.append(
      f"<tr data-verdict='{_esc(verdict)}'"
      f" data-effort='{_esc(effort)}'"
      f" data-retryable='{'true' if retryable else 'false'}'"
      f" data-tags='{data_tags}'"
      f" data-status='{status}'"
      f" data-model='{mid_esc}'>"
      f"<td>{link}</td>"
      f"<td><span class='badge'"
      f" style='background:{vc}'>"
      f"{_esc(verdict)}</span></td>"
      f"<td><span class='badge'"
      f" style='background:{ec}'>"
      f"{_esc(effort)}</span></td>"
      f"<td class='num'>{n_ops}</td>"
      f"<td class='num'>{n_areas}</td>"
      f"<td class='tags'>{tags_str}</td>"
      f"<td class='areas'>{areas_str}</td>"
      f"<td><button class='retry-btn'"
      f" data-model='{mid_esc}'"
      f" title='Re-run analysis'>"
      f"&#x21bb;</button></td>"
      f"</tr>\n"
    )
  unsupported_rows = "".join(unsupported_parts)

  # Feature area bars
  area_counts = data["area_counts"]
  short_areas = {k: v for k, v in area_counts.items() if len(k) < 30}
  area_bars = ""
  if short_areas:
    max_a = max(short_areas.values())
    for area, count in sorted(
      short_areas.items(),
      key=lambda x: -x[1],
    ):
      w = round(count / max_a * 100)
      area_bars += (
        f"<div class='bar-row'>"
        f"<span class='bar-label'>"
        f"{_esc(area)}</span>"
        f"<div class='bar-track'>"
        f"<div class='bar-fill'"
        f" style='width:{w}%'></div></div>"
        f"<span class='bar-val'>{count}</span>"
        f"</div>\n"
      )

  # Deep analysis gaps
  verbose_areas = {k: v for k, v in area_counts.items() if len(k) >= 30}
  deep_gaps = ""
  if verbose_areas:
    deep_gaps = "<h3>Additional Gaps (from deep analysis)</h3><ul class='gap-list'>"
    for area, count in sorted(
      verbose_areas.items(),
      key=lambda x: -x[1],
    ):
      deep_gaps += f"<li><span class='gap-count'>{count}</span> {_esc(area)}</li>"
    deep_gaps += "</ul>"

  # Filter panel tag buttons
  tag_buttons = ""
  for tag in all_tags:
    tag_buttons += (
      f"<button class='tag-btn' data-tag='{_esc(tag)}'>{_esc(tag)}</button> "
    )

  # Top missing ops bars
  op_counts = data["op_counts"]
  op_bars = ""
  if op_counts:
    max_o = max(op_counts.values())
    for op, count in op_counts.most_common(15):
      w = round(count / max_o * 100)
      op_bars += (
        f"<div class='bar-row'>"
        f"<span class='bar-label mono'>"
        f"{_esc(op)}</span>"
        f"<div class='bar-track'>"
        f"<div class='bar-fill op-fill'"
        f" style='width:{w}%'></div></div>"
        f"<span class='bar-val'>{count}</span>"
        f"</div>\n"
      )

  return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
  content="width=device-width, initial-scale=1">
{meta_refresh}
<title>HF Litmus Dashboard</title>
<style>
:root {{
  --bg: #0f172a; --surface: #1e293b;
  --border: #334155; --text: #e2e8f0;
  --muted: #94a3b8; --accent: #38bdf8;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont,
    'Segoe UI', system-ui, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.5; padding: 2rem;
  max-width: 1200px; margin: 0 auto;
}}
h1 {{
  font-size: 1.75rem; font-weight: 700;
  margin-bottom: .25rem;
}}
.subtitle {{
  color: var(--muted); font-size: .85rem;
  margin-bottom: 2rem;
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit,
    minmax(180px, 1fr));
  gap: 1rem; margin-bottom: 2rem;
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: .75rem; padding: 1.25rem;
}}
.card .label {{
  color: var(--muted); font-size: .75rem;
  text-transform: uppercase; letter-spacing: .05em;
}}
.card .value {{
  font-size: 2rem; font-weight: 700;
  margin-top: .25rem;
}}
.card .value.pass {{ color: #22c55e; }}
.card .value.fail {{ color: #f97316; }}
.card .value.pct {{ color: var(--accent); }}
section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: .75rem; padding: 1.5rem;
  margin-bottom: 1.5rem;
}}
h2 {{
  font-size: 1.1rem; font-weight: 600;
  margin-bottom: 1rem;
  padding-bottom: .5rem;
  border-bottom: 1px solid var(--border);
}}
h3 {{
  font-size: .95rem; font-weight: 600;
  margin: 1.25rem 0 .75rem;
  color: var(--muted);
}}
table {{
  width: 100%; border-collapse: collapse;
  font-size: .85rem;
}}
th {{
  text-align: left; color: var(--muted);
  font-weight: 500; padding: .5rem .75rem;
  border-bottom: 1px solid var(--border);
  font-size: .75rem; text-transform: uppercase;
  letter-spacing: .04em;
}}
td {{
  padding: .5rem .75rem;
  border-bottom: 1px solid
    color-mix(in srgb, var(--border) 50%,
      transparent);
}}
tr:hover td {{ background: rgba(56,189,248,.04); }}
.num {{ text-align: right; font-variant-numeric:
  tabular-nums; }}
.badge {{
  display: inline-block; padding: .15rem .5rem;
  border-radius: .35rem; font-size: .75rem;
  font-weight: 600; color: #fff;
}}
.areas {{
  font-size: .78rem; color: var(--muted);
  max-width: 350px;
}}
.bar-row {{
  display: flex; align-items: center;
  gap: .75rem; margin-bottom: .4rem;
}}
.bar-label {{
  min-width: 180px; font-size: .82rem;
  text-align: right; flex-shrink: 0;
}}
.bar-label.mono {{ font-family: 'SF Mono',
  'Cascadia Code', monospace; font-size: .78rem;
  min-width: 260px; }}
.bar-track {{
  flex: 1; height: 18px; background: var(--border);
  border-radius: 3px; overflow: hidden;
}}
.bar-fill {{
  height: 100%; background: var(--accent);
  border-radius: 3px;
  transition: width .3s ease;
}}
.bar-fill.op-fill {{ background: #a78bfa; }}
.bar-val {{
  min-width: 2rem; font-size: .82rem;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
}}
.legend {{
  display: flex; gap: 1.5rem;
  margin-bottom: 1rem; flex-wrap: wrap;
}}
.legend-item {{
  display: flex; align-items: center;
  gap: .35rem; font-size: .8rem;
}}
.legend-dot {{
  width: 10px; height: 10px;
  border-radius: 50%; flex-shrink: 0;
}}
.gap-list {{
  list-style: none; font-size: .82rem;
  columns: 1;
}}
.gap-list li {{
  padding: .25rem 0;
  border-bottom: 1px solid
    color-mix(in srgb, var(--border) 40%,
      transparent);
}}
.gap-count {{
  display: inline-block; min-width: 1.5rem;
  text-align: center; font-weight: 600;
  color: var(--accent); margin-right: .35rem;
}}
.tags {{
  font-size: .75rem; color: var(--accent);
}}
.model-link {{
  color: #93c5fd; text-decoration: none;
}}
.model-link:hover {{
  color: #bfdbfe; text-decoration: underline;
}}
.filters {{
  display: flex; flex-wrap: wrap; gap: .75rem;
  align-items: flex-start;
  margin-bottom: 1rem;
}}
.filter-group {{
  display: flex; flex-direction: column;
  gap: .25rem;
}}
.filter-group .filter-label {{
  font-size: .7rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted);
  font-weight: 500;
}}
.filter-group .filter-opts {{
  display: flex; flex-wrap: wrap; gap: .25rem;
}}
.filter-cb {{
  display: flex; align-items: center;
  gap: .25rem; font-size: .78rem; cursor: pointer;
}}
.filter-cb input {{ cursor: pointer; }}
.tag-btn {{
  background: var(--border); border: none;
  color: var(--text); padding: .2rem .5rem;
  border-radius: .3rem; font-size: .72rem;
  cursor: pointer; transition: background .15s;
}}
.tag-btn:hover {{ background: var(--accent);
  color: #0f172a; }}
.tag-btn.active {{ background: var(--accent);
  color: #0f172a; font-weight: 600; }}
.filter-search {{
  background: var(--bg); border: 1px solid
  var(--border); color: var(--text);
  padding: .35rem .6rem; border-radius: .35rem;
  font-size: .82rem; width: 200px;
}}
.filter-search::placeholder {{
  color: var(--muted); }}
.filter-counter {{
  font-size: .82rem; color: var(--muted);
  margin-bottom: .5rem;
}}
th.sortable {{ cursor: pointer;
  user-select: none; }}
th.sortable:hover {{ color: var(--accent); }}
th.sortable::after {{ content: ' \u2195';
  font-size: .7rem; }}
th.sort-asc::after {{ content: ' \u2191'; }}
th.sort-desc::after {{ content: ' \u2193'; }}
.retry-btn {{
  background: transparent; border: 1px solid
    var(--border); color: var(--muted);
  width: 1.6rem; height: 1.6rem;
  border-radius: .3rem; cursor: pointer;
  font-size: .85rem; display: inline-flex;
  align-items: center; justify-content: center;
  transition: all .15s;
}}
.retry-btn:hover {{ border-color: var(--accent);
  color: var(--accent); }}
.retry-btn.running {{
  animation: spin 1s linear infinite;
  color: var(--accent); border-color: var(--accent);
  pointer-events: none;
}}
.retry-btn.done {{ color: #22c55e;
  border-color: #22c55e; pointer-events: none; }}
.retry-btn.error {{ color: #ef4444;
  border-color: #ef4444; }}
@keyframes spin {{
  from {{ transform: rotate(0deg); }}
  to {{ transform: rotate(360deg); }}
}}
.retry-visible-btn {{
  background: var(--border); border: none;
  color: var(--text); padding: .3rem .75rem;
  border-radius: .35rem; font-size: .78rem;
  cursor: pointer; margin-left: 1rem;
  transition: background .15s;
}}
.retry-visible-btn:hover {{
  background: var(--accent); color: #0f172a; }}
.retry-visible-btn:disabled {{
  opacity: .5; cursor: not-allowed; }}
/* Analysis submission */
.analyze-section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: .75rem; padding: 1.25rem;
  margin-bottom: 1.5rem;
}}
.analyze-form {{
  display: flex; gap: .5rem;
  align-items: flex-start;
  position: relative;
}}
.analyze-input-wrap {{
  position: relative; flex: 1;
  max-width: 500px;
}}
.analyze-input {{
  width: 100%; background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text); padding: .5rem .75rem;
  border-radius: .35rem; font-size: .9rem;
  font-family: 'SF Mono', 'Cascadia Code',
    monospace;
}}
.analyze-input::placeholder {{
  color: var(--muted); font-family: inherit;
}}
.analyze-input:focus {{
  outline: none; border-color: var(--accent);
}}
.ac-dropdown {{
  position: absolute; top: 100%;
  left: 0; right: 0; z-index: 100;
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 .35rem .35rem;
  max-height: 250px; overflow-y: auto;
  display: none;
}}
.ac-dropdown.open {{ display: block; }}
.ac-item {{
  padding: .4rem .75rem; cursor: pointer;
  font-size: .85rem; font-family: 'SF Mono',
    'Cascadia Code', monospace;
}}
.ac-item:hover, .ac-item.active {{
  background: rgba(56, 189, 248, .12);
  color: var(--accent);
}}
.analyze-btn {{
  background: var(--accent); border: none;
  color: #0f172a; padding: .5rem 1.25rem;
  border-radius: .35rem; font-size: .85rem;
  font-weight: 600; cursor: pointer;
  white-space: nowrap;
  transition: opacity .15s;
}}
.analyze-btn:hover {{ opacity: .85; }}
.analyze-btn:disabled {{
  opacity: .5; cursor: not-allowed;
}}
.analyze-status {{
  font-size: .8rem; color: var(--muted);
  margin-top: .5rem;
}}
.analyze-terminal {{
  background: #0d1117; border: 1px solid
    var(--border); border-radius: .5rem;
  padding: .75rem 1rem; margin-top: .75rem;
  max-height: 350px; overflow-y: auto;
  font-family: 'SF Mono', 'Cascadia Code',
    monospace; font-size: .78rem;
  line-height: 1.5; color: #8b949e;
  display: none;
}}
.analyze-terminal.visible {{
  display: block;
}}
.analyze-terminal .line-done {{
  color: #22c55e;
}}
.analyze-terminal .line-err {{
  color: #ef4444;
}}
@media (max-width: 700px) {{
  body {{ padding: 1rem; }}
  .bar-label {{ min-width: 120px; }}
  .bar-label.mono {{ min-width: 160px; }}
  .areas {{ max-width: 200px; }}
  .analyze-form {{ flex-direction: column; }}
  .analyze-input-wrap {{ max-width: 100%; }}
}}
</style>
</head>
<body>

<h1>HF Litmus Dashboard</h1>
<p class="subtitle">Generated {now}
{
    " &middot; auto-refreshing every " + str(refresh_seconds // 60) + " min"
    if refresh_seconds > 0
    else ""
  }</p>

<div class="analyze-section">
  <div class="analyze-form">
    <div class="analyze-input-wrap">
      <input type="text" class="analyze-input"
        id="analyze-model" autocomplete="off"
        placeholder="provider/model (e.g. meta-llama/Llama-3.3-70B-Instruct)">
      <div class="ac-dropdown" id="ac-dropdown">
      </div>
    </div>
    <button class="analyze-btn" id="analyze-btn">
      Analyze Model</button>
  </div>
  <div class="analyze-status" id="analyze-status">
  </div>
  <div class="analyze-terminal" id="analyze-term">
  </div>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Models Tested</div>
    <div class="value">{total}</div>
  </div>
  <div class="card">
    <div class="label">Supported</div>
    <div class="value pass">{pass_count}</div>
  </div>
  <div class="card">
    <div class="label">Unsupported</div>
    <div class="value fail">{fail_count}</div>
  </div>
  <div class="card">
    <div class="label">Pass Rate</div>
    <div class="value pct">{pct}%</div>
  </div>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Close</div>
    <div class="value"
      style="color:#eab308">{vd.get("close", 0)}</div>
  </div>
  <div class="card">
    <div class="label">Far</div>
    <div class="value"
      style="color:#f97316">{vd.get("far", 0)}</div>
  </div>
  <div class="card">
    <div class="label">Blocked</div>
    <div class="value"
      style="color:#ef4444">{vd.get("blocked", 0)}</div>
  </div>
</div>

<section>
<h2>Supported Models</h2>
<div class="filter-counter" id="supported-counter">
  {pass_count} models
</div>
<table id="supported-table">
<thead>
<tr><th>Model</th><th>Pipeline</th>
<th style="text-align:right">Downloads</th>
<th>Tags</th></tr>
</thead>
<tbody>
{supported_rows}
</tbody>
</table>
</section>

<section>
<h2>Unsupported Models</h2>
<div class="filters">
  <div class="filter-group">
    <span class="filter-label">Verdict</span>
    <div class="filter-opts">
      <label class="filter-cb">
        <input type="checkbox" data-filter="verdict"
          value="close" checked> close</label>
      <label class="filter-cb">
        <input type="checkbox" data-filter="verdict"
          value="far" checked> far</label>
      <label class="filter-cb">
        <input type="checkbox" data-filter="verdict"
          value="blocked" checked> blocked</label>
    </div>
  </div>
  <div class="filter-group">
    <span class="filter-label">Effort</span>
    <div class="filter-opts">
      <label class="filter-cb">
        <input type="checkbox" data-filter="effort"
          value="small" checked> small</label>
      <label class="filter-cb">
        <input type="checkbox" data-filter="effort"
          value="medium" checked> medium</label>
      <label class="filter-cb">
        <input type="checkbox" data-filter="effort"
          value="large" checked> large</label>
    </div>
  </div>
  <div class="filter-group">
    <span class="filter-label">Retryable</span>
    <div class="filter-opts">
      <label class="filter-cb">
        <input type="checkbox" id="retryable-only">
          Only retryable</label>
    </div>
  </div>
  <div class="filter-group">
    <span class="filter-label">Tags</span>
    <div class="filter-opts" id="tag-filters">
      {tag_buttons}
    </div>
  </div>
  <div class="filter-group">
    <span class="filter-label">Search</span>
    <input type="text" class="filter-search"
      id="model-search"
      placeholder="Filter by model name...">
  </div>
</div>
<div class="filter-counter" id="filter-counter">
  <span id="filter-counter-text">Showing {fail_count} of {fail_count} models</span>
  <button class="retry-visible-btn"
    id="retry-visible" title="Re-run analysis for all
    visible models">&#x21bb; Retry Visible</button>
</div>
<table id="unsupported-table">
<thead>
<tr>
  <th class="sortable" data-col="model">Model</th>
  <th class="sortable" data-col="verdict">Verdict</th>
  <th class="sortable" data-col="effort">Effort</th>
  <th class="sortable" data-col="ops"
    style="text-align:right">Ops</th>
  <th class="sortable" data-col="areas"
    style="text-align:right">Areas</th>
  <th>Tags</th>
  <th>Feature Areas</th>
  <th style="width:3rem"></th>
</tr>
</thead>
<tbody>
{unsupported_rows}
</tbody>
</table>
</section>

<section>
<h2>Missing Feature Areas
  <span style="color:var(--muted);
    font-weight:400;font-size:.85rem">
    (models affected)</span></h2>
{area_bars}
{deep_gaps}
</section>

<section>
<h2>Top Missing Ops
  <span style="color:var(--muted);
    font-weight:400;font-size:.85rem">
    (models affected)</span></h2>
{op_bars}
</section>

<script>var LITMUS_TOKEN = '{auth_token}';</script>
<script>
(function() {{
  var table = document.getElementById(
    'unsupported-table'
  );
  if (!table) return;
  var tbody = table.querySelector('tbody');
  var rows = Array.from(
    tbody.querySelectorAll('tr')
  );
  var counter = document.getElementById(
    'filter-counter-text'
  );
  var total = rows.length;

  // Supported table filtering
  var supTable = document.getElementById(
    'supported-table'
  );
  var supTbody = supTable
    ? supTable.querySelector('tbody') : null;
  var supRows = supTbody
    ? Array.from(supTbody.querySelectorAll('tr'))
    : [];
  var supCounter = document.getElementById(
    'supported-counter'
  );
  var supTotal = supRows.length;

  function getChecked(name) {{
    var cbs = document.querySelectorAll(
      'input[data-filter="' + name + '"]:checked'
    );
    return Array.from(cbs).map(function(c) {{
      return c.value;
    }});
  }}

  var activeTags = new Set();

  function matchTagsAndSearch(row, search) {{
    var m = row.getAttribute('data-model') || '';
    var t = (
      row.getAttribute('data-tags') || ''
    ).split(',').filter(Boolean);
    if (search && m.toLowerCase().indexOf(
      search
    ) === -1) return false;
    if (activeTags.size > 0) {{
      var hasAll = true;
      activeTags.forEach(function(tag) {{
        if (t.indexOf(tag) === -1) hasAll = false;
      }});
      if (!hasAll) return false;
    }}
    return true;
  }}

  function applyFilters() {{
    var verdicts = new Set(getChecked('verdict'));
    var efforts = new Set(getChecked('effort'));
    var retryOnly = document.getElementById(
      'retryable-only'
    ).checked;
    var search = document.getElementById(
      'model-search'
    ).value.toLowerCase();

    // Filter unsupported models
    var shown = 0;
    rows.forEach(function(row) {{
      var v = row.getAttribute('data-verdict');
      var e = row.getAttribute('data-effort');
      var r = row.getAttribute('data-retryable');

      var show = verdicts.has(v)
        && efforts.has(e);
      if (retryOnly && r !== 'true') show = false;
      if (show) show = matchTagsAndSearch(
        row, search
      );

      row.style.display = show ? '' : 'none';
      if (show) shown++;
    }});

    if (counter) {{
      counter.textContent = (
        'Showing ' + shown + ' of ' + total
        + ' models'
      );
    }}

    // Filter supported models
    var supShown = 0;
    supRows.forEach(function(row) {{
      var show = matchTagsAndSearch(row, search);
      row.style.display = show ? '' : 'none';
      if (show) supShown++;
    }});
    if (supCounter) {{
      supCounter.textContent = (
        supShown === supTotal
        ? supTotal + ' models'
        : supShown + ' of ' + supTotal + ' models'
      );
    }}

    saveHash();
  }}

  // Bind filter events
  document.querySelectorAll(
    'input[data-filter], #retryable-only'
  ).forEach(function(cb) {{
    cb.addEventListener('change', applyFilters);
  }});
  document.getElementById(
    'model-search'
  ).addEventListener('input', applyFilters);

  // Tag buttons
  document.querySelectorAll('.tag-btn').forEach(
    function(btn) {{
      btn.addEventListener('click', function() {{
        var tag = btn.getAttribute('data-tag');
        if (activeTags.has(tag)) {{
          activeTags.delete(tag);
          btn.classList.remove('active');
        }} else {{
          activeTags.add(tag);
          btn.classList.add('active');
        }}
        applyFilters();
      }});
    }}
  );

  // Column sorting
  var sortCol = null;
  var sortAsc = true;

  var colExtractors = {{
    model: function(r) {{
      return (r.getAttribute('data-model')
        || '').toLowerCase();
    }},
    verdict: function(r) {{
      var order = {{close:1, far:2, blocked:3}};
      return order[
        r.getAttribute('data-verdict')
      ] || 9;
    }},
    effort: function(r) {{
      var order = {{
        none:0, small:1, medium:2, large:3
      }};
      return order[
        r.getAttribute('data-effort')
      ] || 9;
    }},
    ops: function(r) {{
      return parseInt(
        r.cells[3].textContent, 10
      ) || 0;
    }},
    areas: function(r) {{
      return parseInt(
        r.cells[4].textContent, 10
      ) || 0;
    }},
  }};

  document.querySelectorAll(
    'th.sortable'
  ).forEach(function(th) {{
    th.addEventListener('click', function() {{
      var col = th.getAttribute('data-col');
      if (sortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        sortAsc = true;
      }}

      document.querySelectorAll(
        'th.sortable'
      ).forEach(function(h) {{
        h.classList.remove('sort-asc', 'sort-desc');
      }});
      th.classList.add(
        sortAsc ? 'sort-asc' : 'sort-desc'
      );

      var extract = colExtractors[col];
      if (!extract) return;

      rows.sort(function(a, b) {{
        var va = extract(a);
        var vb = extract(b);
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
      }});

      rows.forEach(function(r) {{
        tbody.appendChild(r);
      }});
    }});
  }});

  // Retry functionality
  function retryModels(modelIds) {{
    return fetch('/api/retry', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json',
        'X-Litmus-Token': LITMUS_TOKEN,
      }},
      body: JSON.stringify({{models: modelIds}}),
    }}).then(function(r) {{ return r.json(); }});
  }}

  function pollRetryStatus() {{
    fetch('/api/retry/status')
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        var active = 0;
        Object.keys(data).forEach(function(mid) {{
          var st = data[mid];
          var btns = document.querySelectorAll(
            '.retry-btn[data-model="'
            + CSS.escape(mid) + '"]'
          );
          btns.forEach(function(btn) {{
            btn.classList.remove(
              'running', 'done', 'error'
            );
            if (st === 'running') {{
              btn.classList.add('running');
              active++;
            }} else if (st === 'done') {{
              btn.classList.add('done');
              btn.innerHTML = '&#x2713;';
            }} else if (st === 'error') {{
              btn.classList.add('error');
              btn.innerHTML = '&#x2717;';
            }}
          }});
        }});
        if (active > 0) {{
          setTimeout(pollRetryStatus, 3000);
        }}
      }});
  }}

  // Per-row retry buttons
  document.querySelectorAll('.retry-btn').forEach(
    function(btn) {{
      btn.addEventListener('click', function() {{
        var mid = btn.getAttribute('data-model');
        btn.classList.add('running');
        btn.innerHTML = '&#x21bb;';
        retryModels([mid]).then(function() {{
          setTimeout(pollRetryStatus, 2000);
        }});
      }});
    }}
  );

  // Retry all visible
  var retryVisBtn = document.getElementById(
    'retry-visible'
  );
  if (retryVisBtn) {{
    retryVisBtn.addEventListener(
      'click', function() {{
        var ids = [];
        rows.forEach(function(row) {{
          if (row.style.display !== 'none') {{
            var mid = row.getAttribute(
              'data-model'
            );
            if (mid) ids.push(mid);
          }}
        }});
        if (ids.length === 0) return;
        if (!confirm(
          'Re-run analysis for ' + ids.length
          + ' models?'
        )) return;
        retryVisBtn.disabled = true;
        retryVisBtn.textContent = (
          'Retrying ' + ids.length + '...'
        );
        retryModels(ids).then(function() {{
          setTimeout(pollRetryStatus, 2000);
          setTimeout(function() {{
            retryVisBtn.disabled = false;
            retryVisBtn.innerHTML = (
              '&#x21bb; Retry Visible'
            );
          }}, 5000);
        }});
      }}
    );
  }}

  // URL hash persistence
  function saveHash() {{
    var state = {{}};
    ['verdict', 'effort'].forEach(
      function(name) {{
        state[name] = getChecked(name);
      }}
    );
    state.retryOnly = document.getElementById(
      'retryable-only'
    ).checked;
    state.search = document.getElementById(
      'model-search'
    ).value;
    state.tags = Array.from(activeTags);
    location.hash = encodeURIComponent(
      JSON.stringify(state)
    );
  }}

  function loadHash() {{
    if (!location.hash) return;
    try {{
      var state = JSON.parse(
        decodeURIComponent(
          location.hash.substring(1)
        )
      );
      ['verdict', 'effort'].forEach(
        function(name) {{
          var vals = new Set(state[name] || []);
          document.querySelectorAll(
            'input[data-filter="' + name + '"]'
          ).forEach(function(cb) {{
            cb.checked = vals.has(cb.value);
          }});
        }}
      );
      if (state.retryOnly) {{
        document.getElementById(
          'retryable-only'
        ).checked = true;
      }}
      if (state.search) {{
        document.getElementById(
          'model-search'
        ).value = state.search;
      }}
      if (state.tags) {{
        state.tags.forEach(function(tag) {{
          activeTags.add(tag);
          var btn = document.querySelector(
            '.tag-btn[data-tag="' + tag + '"]'
          );
          if (btn) btn.classList.add('active');
        }});
      }}
      applyFilters();
    }} catch(e) {{}}
  }}

  loadHash();
}})();

// -- Model analysis submission & live progress --
(function() {{
  var input = document.getElementById('analyze-model');
  var dropdown = document.getElementById('ac-dropdown');
  var btn = document.getElementById('analyze-btn');
  var statusEl = document.getElementById(
    'analyze-status'
  );
  var term = document.getElementById('analyze-term');
  if (!input || !btn) return;

  var debounceTimer = null;
  var acIndex = -1;
  var acItems = [];

  function renderDropdown(items) {{
    acItems = items;
    acIndex = -1;
    if (!items.length) {{
      dropdown.classList.remove('open');
      dropdown.innerHTML = '';
      return;
    }}
    dropdown.innerHTML = '';
    items.forEach(function(id, i) {{
      var div = document.createElement('div');
      div.className = 'ac-item';
      div.setAttribute('data-idx', i);
      div.textContent = id;
      div.addEventListener('mousedown', function(e) {{
        e.preventDefault();
        input.value = div.textContent;
        dropdown.classList.remove('open');
      }});
      dropdown.appendChild(div);
    }});
    dropdown.classList.add('open');
  }}

  function highlightItem(idx) {{
    dropdown.querySelectorAll('.ac-item').forEach(
      function(el, i) {{
        el.classList.toggle('active', i === idx);
      }}
    );
  }}

  input.addEventListener('input', function() {{
    clearTimeout(debounceTimer);
    var q = input.value.trim();
    if (q.length < 2) {{
      dropdown.classList.remove('open');
      return;
    }}
    debounceTimer = setTimeout(function() {{
      fetch('/api/models/search?q='
        + encodeURIComponent(q))
        .then(function(r) {{ return r.json(); }})
        .then(renderDropdown)
        .catch(function() {{ renderDropdown([]); }});
    }}, 300);
  }});

  input.addEventListener('keydown', function(e) {{
    if (!dropdown.classList.contains('open')) {{
      if (e.key === 'Enter') {{
        e.preventDefault();
        btn.click();
      }}
      return;
    }}
    if (e.key === 'ArrowDown') {{
      e.preventDefault();
      acIndex = Math.min(
        acIndex + 1, acItems.length - 1
      );
      highlightItem(acIndex);
    }} else if (e.key === 'ArrowUp') {{
      e.preventDefault();
      acIndex = Math.max(acIndex - 1, 0);
      highlightItem(acIndex);
    }} else if (e.key === 'Enter') {{
      e.preventDefault();
      if (acIndex >= 0 && acIndex < acItems.length) {{
        input.value = acItems[acIndex];
      }}
      dropdown.classList.remove('open');
      btn.click();
    }} else if (e.key === 'Escape') {{
      dropdown.classList.remove('open');
    }}
  }});

  input.addEventListener('blur', function() {{
    setTimeout(function() {{
      dropdown.classList.remove('open');
    }}, 200);
  }});

  // Submit analysis
  var polling = false;
  var lineOffset = 0;

  btn.addEventListener('click', function() {{
    var model = input.value.trim();
    if (!model || model.indexOf('/') === -1) {{
      statusEl.textContent = (
        'Enter a valid provider/model name.'
      );
      return;
    }}
    btn.disabled = true;
    btn.textContent = 'Starting...';
    statusEl.textContent = '';
    term.innerHTML = '';
    term.classList.add('visible');
    lineOffset = 0;

    fetch('/api/analyze', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json',
        'X-Litmus-Token': LITMUS_TOKEN,
      }},
      body: JSON.stringify({{model: model}}),
    }}).then(function(r) {{
      if (r.status === 409) {{
        statusEl.textContent = (
          'An analysis is already running.'
        );
        btn.disabled = false;
        btn.textContent = 'Analyze Model';
        return;
      }}
      return r.json();
    }}).then(function(data) {{
      if (!data) return;
      statusEl.textContent = (
        'Analyzing ' + data.model + '...'
      );
      startPolling();
    }}).catch(function(err) {{
      statusEl.textContent = 'Error: ' + err;
      btn.disabled = false;
      btn.textContent = 'Analyze Model';
    }});
  }});

  function startPolling() {{
    if (polling) return;
    polling = true;
    pollStatus();
  }}

  function pollStatus() {{
    fetch('/api/analyze/status?offset=' + lineOffset)
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        if (data.lines && data.lines.length) {{
          data.lines.forEach(function(line) {{
            var div = document.createElement('div');
            div.textContent = line;
            term.appendChild(div);
          }});
          lineOffset = data.total_lines;
          term.scrollTop = term.scrollHeight;
        }}
        if (data.status === 'running') {{
          setTimeout(pollStatus, 1000);
        }} else {{
          polling = false;
          btn.disabled = false;
          btn.textContent = 'Analyze Model';
          if (data.status === 'done') {{
            var safeModel = document.createElement('strong');
            safeModel.textContent = data.model;
            statusEl.textContent = '';
            statusEl.appendChild(
              document.createTextNode('Analysis complete for ')
            );
            statusEl.appendChild(safeModel);
            statusEl.appendChild(
              document.createTextNode('. Results committed and pushed.')
            );
            var last = term.lastElementChild;
            if (last) last.className = 'line-done';
          }} else if (data.status === 'error') {{
            var safeModel2 = document.createElement('strong');
            safeModel2.textContent = data.model;
            statusEl.textContent = '';
            statusEl.appendChild(
              document.createTextNode('Analysis failed for ')
            );
            statusEl.appendChild(safeModel2);
            statusEl.appendChild(document.createTextNode('.'));
            var last2 = term.lastElementChild;
            if (last2) last2.className = 'line-err';
          }}
        }}
      }})
      .catch(function() {{
        polling = false;
        btn.disabled = false;
        btn.textContent = 'Analyze Model';
      }});
  }}

  // Check if analysis is already running on load
  fetch('/api/analyze/status?offset=0')
    .then(function(r) {{ return r.json(); }})
    .then(function(data) {{
      if (data.status === 'running') {{
        btn.disabled = true;
        btn.textContent = 'Running...';
        statusEl.textContent = (
          'Analyzing ' + data.model + '...'
        );
        term.classList.add('visible');
        if (data.lines && data.lines.length) {{
          data.lines.forEach(function(line) {{
            var div = document.createElement('div');
            div.textContent = line;
            term.appendChild(div);
          }});
          lineOffset = data.total_lines;
          term.scrollTop = term.scrollHeight;
        }}
        startPolling();
      }}
    }}).catch(function() {{}});
}})();
</script>

</body>
</html>
"""


# -- Markdown viewer -----------------------------------------------

_MD_VIEWER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
  content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com\
/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<style>
:root {{
  --bg: #0f172a; --surface: #1e293b;
  --text: #e2e8f0; --muted: #94a3b8;
  --border: #334155; --accent: #38bdf8;
  --link: #93c5fd;
}}
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0;
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont,
    'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  line-height: 1.7;
}}
.top-bar {{
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: .75rem 2rem;
  display: flex; align-items: center; gap: 1rem;
}}
.top-bar a {{
  color: var(--accent); text-decoration: none;
  font-size: .875rem; font-weight: 500;
}}
.top-bar a:hover {{ text-decoration: underline; }}
.top-bar .title {{
  color: var(--muted); font-size: .875rem;
  margin-left: auto;
}}
.container {{
  max-width: 52rem; margin: 2rem auto;
  padding: 0 1.5rem;
}}
article {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: .75rem;
  padding: 2.5rem;
}}
h1 {{
  font-size: 1.75rem; font-weight: 700;
  color: #f1f5f9; margin: 0 0 1.5rem 0;
  padding-bottom: .75rem;
  border-bottom: 2px solid var(--accent);
}}
h2 {{
  font-size: 1.25rem; font-weight: 600;
  color: #f1f5f9; margin: 2rem 0 .75rem 0;
}}
h3 {{
  font-size: 1.05rem; font-weight: 600;
  color: var(--accent); margin: 1.5rem 0 .5rem 0;
}}
p {{ margin: .75rem 0; }}
a {{ color: var(--link); }}
a:hover {{ color: #bfdbfe; }}
code {{
  font-family: 'JetBrains Mono', 'Fira Code',
    'Cascadia Code', monospace;
  font-size: .875em;
  background: rgba(56, 189, 248, 0.1);
  padding: .15em .35em; border-radius: .25rem;
  color: var(--accent);
}}
pre {{
  background: #0d1117 !important;
  border: 1px solid var(--border);
  border-radius: .5rem; padding: 1.25rem;
  overflow-x: auto; margin: 1rem 0;
}}
pre code {{
  background: none; padding: 0;
  color: var(--text); font-size: .8125rem;
  line-height: 1.6;
}}
table {{
  width: 100%; border-collapse: collapse;
  margin: 1rem 0; font-size: .875rem;
}}
th {{
  text-align: left; padding: .6rem .75rem;
  background: rgba(56, 189, 248, 0.08);
  border-bottom: 2px solid var(--border);
  color: var(--accent); font-weight: 600;
  font-size: .75rem; text-transform: uppercase;
  letter-spacing: .05em;
}}
td {{
  padding: .5rem .75rem;
  border-bottom: 1px solid var(--border);
}}
tr:hover td {{ background: rgba(255,255,255,.02); }}
blockquote {{
  border-left: 3px solid var(--accent);
  margin: 1rem 0; padding: .5rem 1rem;
  background: rgba(56, 189, 248, 0.05);
  color: var(--muted);
}}
strong {{ color: #f1f5f9; }}
hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 2rem 0;
}}
ul, ol {{ padding-left: 1.5rem; }}
li {{ margin: .25rem 0; }}
.frontmatter {{
  display: none;
}}
</style>
</head>
<body>
<div class="top-bar">
  <a href="/">&larr; Dashboard</a>
  <span class="title">{title}</span>
</div>
<div class="container">
<article id="content">Loading&hellip;</article>
</div>
<script src="https://cdnjs.cloudflare.com\
/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com\
/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com\
/ajax/libs/dompurify/3.2.4/purify.min.js"></script>
<script>
marked.setOptions({{
  highlight: function(code, lang) {{
    if (lang && hljs.getLanguage(lang))
      return hljs.highlight(code, {{language: lang}}).value;
    return hljs.highlightAuto(code).value;
  }}
}});
var md = {markdown_json};
// Strip YAML frontmatter
md = md.replace(/^---[\\s\\S]*?---\\n*/, '');
document.getElementById('content').innerHTML =
  DOMPurify.sanitize(marked.parse(md));
document.querySelectorAll('#content a').forEach(function(a) {{
  var href = a.getAttribute('href');
  if (href && href.startsWith('litmus-output/analyses/')) {{
    a.setAttribute('href',
      '/' + href.replace('litmus-output/', ''));
  }}
}});
</script>
</body>
</html>
"""


def _render_markdown_page(
  title: str,
  markdown_content: str,
) -> str:
  """Wrap raw markdown in a styled HTML viewer."""
  # Use HTML-safe JSON: escape </script> and other HTML
  # special sequences to prevent script block breakout.
  safe_json = (
    json.dumps(markdown_content)
    .replace("<", "\\u003c")
    .replace(">", "\\u003e")
  )
  safe_title = _esc(title).replace("{", "{{").replace(
    "}", "}}"
  )
  return _MD_VIEWER_HTML.format(
    title=safe_title,
    markdown_json=safe_json,
  )


# -- HTTP server ---------------------------------------------------


class _RetryTracker:
  """Track background retry jobs."""

  def __init__(
    self,
    output_dir: Path,
  ) -> None:
    self.output_dir = output_dir
    self._lock = threading.Lock()
    self._jobs: dict[str, str] = {}
    self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retry")

  def submit(self, model_id: str) -> None:
    with self._lock:
      if self._jobs.get(model_id) == "running":
        return
      self._prune_completed()
      self._jobs[model_id] = "running"
    self._pool.submit(self._run, model_id)

  def status(self) -> dict[str, str]:
    with self._lock:
      return dict(self._jobs)

  def _prune_completed(self) -> None:
    """Remove oldest completed/errored jobs when limit exceeded."""
    terminal = [
      k for k, v in self._jobs.items()
      if v in ("done", "error")
    ]
    excess = len(terminal) - _MAX_COMPLETED_JOBS
    if excess > 0:
      for k in terminal[:excess]:
        del self._jobs[k]

  def _run(self, model_id: str) -> None:
    try:
      out_dir = str(self.output_dir.resolve())
      cmd = [
        sys.executable,
        "-m",
        "hf_litmus.cli",
        "--model",
        model_id,
        "--output-dir",
        out_dir,
      ]
      logger.info("Retry started: %s", model_id)
      result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,
      )
      if result.returncode == 0:
        with self._lock:
          self._jobs[model_id] = "done"
        logger.info("Retry succeeded: %s", model_id)
      else:
        with self._lock:
          self._jobs[model_id] = "error"
        logger.warning(
          "Retry failed: %s (rc=%d)\n%s",
          model_id,
          result.returncode,
          result.stderr[-500:] if result.stderr else "",
        )
    except subprocess.TimeoutExpired:
      with self._lock:
        self._jobs[model_id] = "error"
      logger.warning("Retry timed out: %s", model_id)
    except Exception as e:
      with self._lock:
        self._jobs[model_id] = "error"
      logger.warning("Retry error: %s: %s", model_id, e)


class _AnalysisTracker:
  """Track a single background analysis job with
  live output streaming."""

  _TIMEOUT = 7200  # 2 hours

  def __init__(
    self,
    output_dir: Path,
  ) -> None:
    self.output_dir = output_dir
    self._lock = threading.Lock()
    self._model: str = ""
    self._status: str = "idle"  # idle/running/done/error
    self._lines: list[str] = []
    self._proc: subprocess.Popen | None = None

  def submit(self, model_id: str) -> bool:
    """Start analysis for a model. Returns False if
    already running."""
    with self._lock:
      if self._status == "running":
        return False
      self._model = model_id
      self._status = "running"
      self._lines = []  # reset on new submission
    t = threading.Thread(
      target=self._run,
      args=(model_id,),
      daemon=True,
    )
    t.start()
    return True

  def status(
    self,
    offset: int = 0,
  ) -> dict:
    with self._lock:
      return {
        "model": self._model,
        "status": self._status,
        "lines": self._lines[offset:],
        "total_lines": len(self._lines),
      }

  def _run(self, model_id: str) -> None:
    try:
      def log_line(msg: str) -> None:
        with self._lock:
          if len(self._lines) < _MAX_ANALYSIS_LINES:
            self._lines.append(msg)

      out_dir = str(self.output_dir.resolve())
      cmd = [
        sys.executable,
        "-m",
        "hf_litmus.cli",
        "--model",
        model_id,
        "--output-dir",
        out_dir,
      ]
      logger.info("Analysis started: %s", model_id)
      with self._lock:
        self._lines.append(f"Starting analysis for {model_id}...")
      proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
      )
      self._proc = proc
      deadline = time.monotonic() + self._TIMEOUT
      for line in proc.stdout:
        with self._lock:
          if len(self._lines) < _MAX_ANALYSIS_LINES:
            self._lines.append(line.rstrip("\n"))
        if time.monotonic() > deadline:
          proc.kill()
          with self._lock:
            self._lines.append(
              f"Analysis timed out after {self._TIMEOUT}s"
            )
          break
      proc.wait(timeout=30)
      if proc.returncode == 0:
        with self._lock:
          self._lines.append("Analysis completed successfully.")
          self._lines.append("Committing results...")
        self._commit_results(model_id)
        with self._lock:
          self._status = "done"
          self._lines.append("Done.")
        logger.info("Analysis succeeded: %s", model_id)
      else:
        with self._lock:
          self._status = "error"
          self._lines.append(f"Analysis failed (exit code {proc.returncode})")
        logger.warning(
          "Analysis failed: %s (rc=%d)",
          model_id,
          proc.returncode,
        )
    except Exception as e:
      with self._lock:
        self._status = "error"
        self._lines.append(f"Error: {e}")
      logger.warning("Analysis error: %s: %s", model_id, e)
    finally:
      self._proc = None

  def _commit_results(
    self,
    model_id: str,
  ) -> None:
    """Commit and push results in litmus-output."""
    out = self.output_dir.resolve()
    try:
      # Stage only report/analysis artifacts, not logs or state
      paths_to_add = ["reports", "analyses", "summary.json", "summary.md"]
      subprocess.run(
        ["git", "add", "--"] + paths_to_add,
        cwd=str(out),
        check=True,
        capture_output=True,
        text=True,
      )
      result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(out),
        capture_output=True,
        text=True,
      )
      if not result.stdout.strip():
        with self._lock:
          self._lines.append("No changes to commit.")
        return
      subprocess.run(
        ["git", "commit", "-m", f"Add analysis for {model_id}"],
        cwd=str(out),
        check=True,
        capture_output=True,
        text=True,
      )
      with self._lock:
        self._lines.append("Committed. Pushing...")
      subprocess.run(
        ["git", "push"],
        cwd=str(out),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
      )
      with self._lock:
        self._lines.append("Pushed successfully.")
    except subprocess.CalledProcessError as e:
      with self._lock:
        self._lines.append(f"Git error: {e.stderr or e.stdout or e}")
      logger.warning(
        "Git commit/push failed for %s: %s",
        model_id,
        e,
      )
    except subprocess.TimeoutExpired:
      with self._lock:
        self._lines.append("Git push timed out.")


class _DashboardHandler(BaseHTTPRequestHandler):
  """Serves the live dashboard HTML."""

  # Set by _serve_dashboard before starting
  html_cache: str = ""
  json_cache: str = ""
  csv_cache: str = ""
  reports_dir: Path = Path()
  analyses_dir: Path = Path()
  retry_tracker: _RetryTracker | None = None
  analysis_tracker: _AnalysisTracker | None = None
  auth_token: str = ""

  def do_GET(self) -> None:  # noqa: N802
    if self.path == "/api/data.json":
      self._respond(200, "application/json", self.json_cache)
    elif self.path == "/api/data.csv":
      self._respond(200, "text/csv", self.csv_cache)
    elif self.path in ("/", "/index.html"):
      self._respond(200, "text/html", self.html_cache)
    elif self.path == "/api/retry/status":
      data = self.retry_tracker.status() if self.retry_tracker else {}
      self._respond(200, "application/json", json.dumps(data))
    elif self.path.startswith("/api/models/search"):
      self._handle_model_search()
    elif self.path.startswith("/api/analyze/status"):
      self._handle_analyze_status()
    elif self.path.startswith("/analyses/"):
      self._serve_analysis()
    elif self.path.startswith("/reports/"):
      self._serve_report()
    else:
      self._respond(404, "text/plain", "Not found")

  def do_POST(self) -> None:  # noqa: N802
    if self.auth_token:
      provided = self.headers.get("X-Litmus-Token", "")
      if provided != self.auth_token:
        self._respond(
          403,
          "application/json",
          '{"error":"invalid or missing X-Litmus-Token"}',
        )
        return
    if self.path == "/api/retry":
      self._handle_retry()
    elif self.path == "/api/analyze":
      self._handle_analyze_submit()
    else:
      self._respond(404, "text/plain", "Not found")

  def _read_body(self) -> str | None:
    """Read POST body with size cap. Returns None on error."""
    length = max(0, int(self.headers.get("Content-Length", 0)))
    if length > _MAX_BODY:
      self._respond(
        413,
        "application/json",
        '{"error":"request too large"}',
      )
      return None
    return self.rfile.read(length).decode("utf-8")

  def _handle_retry(self) -> None:
    """Queue models for re-analysis."""
    if not self.retry_tracker:
      self._respond(
        503,
        "application/json",
        '{"error":"retry not available"}',
      )
      return
    body = self._read_body()
    if body is None:
      return
    try:
      data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
      self._respond(
        400,
        "application/json",
        '{"error":"invalid json"}',
      )
      return
    models = data.get("models", [])
    if not isinstance(models, list) or not models:
      self._respond(
        400,
        "application/json",
        '{"error":"models list required"}',
      )
      return
    if len(models) > _MAX_RETRY_MODELS:
      self._respond(
        400,
        "application/json",
        f'{{"error":"max {_MAX_RETRY_MODELS} models per request"}}',
      )
      return
    queued = []
    for mid in models:
      if isinstance(mid, str) and _MODEL_ID_RE.fullmatch(mid.strip()):
        self.retry_tracker.submit(mid.strip())
        queued.append(mid.strip())
    self._respond(
      200,
      "application/json",
      json.dumps(
        {
          "status": "queued",
          "models": queued,
        }
      ),
    )

  def _handle_model_search(self) -> None:
    """Proxy model search to HuggingFace Hub API."""
    parsed = urllib.parse.urlparse(self.path)
    qs = urllib.parse.parse_qs(parsed.query)
    query = qs.get("q", [""])[0].strip()
    if not query or len(query) < 2:
      self._respond(200, "application/json", "[]")
      return
    try:
      url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(
        {
          "search": query,
          "limit": "20",
          "sort": "downloads",
          "direction": "-1",
        }
      )
      req = urllib.request.Request(
        url,
        headers={"User-Agent": "hf-litmus"},
      )
      with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
      results = [m.get("modelId", m.get("id", "")) for m in data if isinstance(m, dict)]
      self._respond(
        200,
        "application/json",
        json.dumps(results),
      )
    except Exception as e:
      logger.warning("HF search error: %s", e)
      self._respond(200, "application/json", "[]")

  def _handle_analyze_submit(self) -> None:
    """Start a model analysis job."""
    if not self.analysis_tracker:
      self._respond(
        503,
        "application/json",
        '{"error":"analysis not available"}',
      )
      return
    body = self._read_body()
    if body is None:
      return
    try:
      data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
      self._respond(
        400,
        "application/json",
        '{"error":"invalid json"}',
      )
      return
    model = data.get("model", "").strip()
    if not model or not _MODEL_ID_RE.fullmatch(model):
      self._respond(
        400,
        "application/json",
        '{"error":"model must be provider/name"}',
      )
      return
    ok = self.analysis_tracker.submit(model)
    if not ok:
      self._respond(
        409,
        "application/json",
        '{"error":"analysis already running"}',
      )
      return
    self._respond(
      200,
      "application/json",
      json.dumps(
        {
          "status": "started",
          "model": model,
        }
      ),
    )

  def _handle_analyze_status(self) -> None:
    """Return current analysis status and output."""
    if not self.analysis_tracker:
      self._respond(
        200,
        "application/json",
        json.dumps(
          {
            "model": "",
            "status": "idle",
            "lines": [],
            "total_lines": 0,
          }
        ),
      )
      return
    parsed = urllib.parse.urlparse(self.path)
    qs = urllib.parse.parse_qs(parsed.query)
    offset = max(0, int(qs.get("offset", ["0"])[0]))
    data = self.analysis_tracker.status(offset)
    self._respond(200, "application/json", json.dumps(data))

  def _serve_report(self) -> None:
    """Serve a markdown report as styled HTML."""
    rel = urllib.parse.unquote(self.path.lstrip("/"))
    report_path = (
      self.reports_dir / rel.split("reports/", 1)[-1]
    ).resolve()
    # Verify path stays within reports_dir
    try:
      report_path.relative_to(self.reports_dir.resolve())
    except ValueError:
      self._respond(403, "text/plain", "Forbidden")
      return
    if not report_path.is_file():
      self._respond(
        404,
        "text/plain",
        "Not found",
      )
      return
    md_content = report_path.read_text(encoding="utf-8")
    title = report_path.stem
    if report_path.parent != self.reports_dir.resolve():
      title = f"{report_path.parent.name}/{title}"
    html = _render_markdown_page(title, md_content)
    self._respond(200, "text/html", html)

  def _serve_analysis(self) -> None:
    """Serve an analysis markdown as styled HTML."""
    rel = urllib.parse.unquote(self.path.lstrip("/"))
    analysis_path = (
      self.analyses_dir / rel.split("analyses/", 1)[-1]
    ).resolve()
    # Verify path stays within analyses_dir
    try:
      analysis_path.relative_to(self.analyses_dir.resolve())
    except ValueError:
      self._respond(403, "text/plain", "Forbidden")
      return
    if not analysis_path.is_file():
      self._respond(
        404,
        "text/plain",
        "Not found",
      )
      return
    md_content = analysis_path.read_text(encoding="utf-8")
    title = analysis_path.parent.name
    html = _render_markdown_page(title, md_content)
    self._respond(200, "text/html", html)

  def _respond(self, code: int, ctype: str, body: str) -> None:
    encoded = body.encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(encoded)))
    self.end_headers()
    self.wfile.write(encoded)

  def log_message(self, format: str, *args: object) -> None:
    pass  # suppress per-request logging


def _serve_dashboard(
  output_dir: Path,
  sort: str,
  port: int,
  refresh_minutes: int,
) -> int:
  """Run a local HTTP server with auto-refresh."""
  reports_dir = output_dir / "reports"
  _DashboardHandler.reports_dir = reports_dir
  _DashboardHandler.analyses_dir = output_dir / "analyses"
  _DashboardHandler.retry_tracker = _RetryTracker(output_dir)
  _DashboardHandler.analysis_tracker = _AnalysisTracker(output_dir)
  auth_token = secrets.token_urlsafe(32)
  _DashboardHandler.auth_token = auth_token
  refresh_secs = refresh_minutes * 60

  def refresh() -> None:
    reports = _load_reports(reports_dir)
    if not reports:
      _DashboardHandler.html_cache = (
        "<html><body><h1>No reports found</h1></body></html>"
      )
      _DashboardHandler.json_cache = "{}"
      _DashboardHandler.csv_cache = ""
      return
    data = _compute_dashboard(reports, sort, output_dir)
    _DashboardHandler.html_cache = _render_html(
      data, refresh_seconds=refresh_secs,
      auth_token=auth_token,
    )
    _DashboardHandler.json_cache = _render_json(data)
    _DashboardHandler.csv_cache = _render_csv(data)

  # Initial load
  refresh()

  # Background refresh thread
  stop_event = threading.Event()

  def _refresh_loop() -> None:
    while not stop_event.wait(refresh_secs):
      try:
        refresh()
      except Exception as e:
        print(f"Refresh error: {e}", file=sys.stderr)

  t = threading.Thread(target=_refresh_loop, daemon=True)
  t.start()

  server = HTTPServer(("127.0.0.1", port), _DashboardHandler)
  print(f"Serving dashboard at http://127.0.0.1:{port}")
  print(f"Auth token: {auth_token}")
  print(f"Rescanning {reports_dir} every {refresh_minutes} min")
  print("Press Ctrl+C to stop")

  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down")
    stop_event.set()
    server.shutdown()

  return 0
