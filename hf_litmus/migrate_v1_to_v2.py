"""One-time migration of JSON reports from schema v1 to v2.

Changes applied:
- Split prose verdicts into keyword verdict + verdict_detail
- Add missing_ops_count from platform_compatibility.missing_ops
- Fix blocker_count: set to 0 for heuristic reports
- Add gap_analysis section with status defaults if missing
- Validate furthest_stage against enum, fallback to "unknown"
- Bump metadata_schema_version to 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hf_litmus.feature_areas import _classify_prose_verdict

_VERDICT_VALUES = {"pass", "close", "far", "blocked"}

_STAGE_VALUES = {
    "export",
    "fxtypedfx",
    "rewrite",
    "bulk",
    "loopy",
    "tron",
    "cpp",
    "success",
    "unknown",
}


def migrate_report(data: dict) -> bool:
    """Migrate a single report dict in place.

    Returns True if the report was modified.
    """
    version = data.get("metadata_schema_version", 1)
    if version >= 2:
        return False

    assessment = data.get("assessment", {})

    # Split prose verdicts
    raw_verdict = assessment.get("verdict", "far")
    if raw_verdict in _VERDICT_VALUES:
        assessment["verdict_detail"] = None
    else:
        assessment["verdict_detail"] = raw_verdict
        assessment["verdict"] = _classify_prose_verdict(raw_verdict)

    # Add missing_ops_count
    compat = data.get("platform_compatibility", {})
    missing_ops = compat.get("missing_ops", [])
    assessment["missing_ops_count"] = len(missing_ops)

    # Fix blocker_count for heuristic reports
    confidence = assessment.get("confidence", "heuristic")
    if confidence == "heuristic":
        assessment["blocker_count"] = 0

    # Validate furthest_stage
    raw_stage = assessment.get("furthest_stage", "unknown")
    if raw_stage not in _STAGE_VALUES:
        assessment["furthest_stage"] = "unknown"

    data["assessment"] = assessment

    # Ensure gap_analysis section exists
    if "gap_analysis" not in data:
        data["gap_analysis"] = {
            "status": "not_run",
            "furthest_stage": None,
            "missing_kernels": [],
            "missing_patterns": [],
            "fixes_applied": [],
            "consensus_review": None,
        }
    else:
        gap = data["gap_analysis"]
        gap.setdefault("status", "completed")
        gap.setdefault("consensus_review", None)
        # Validate gap furthest_stage too
        gs = gap.get("furthest_stage")
        if gs and gs not in _STAGE_VALUES:
            gap["furthest_stage"] = "unknown"

    data["metadata_schema_version"] = 2
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate hf-litmus JSON reports v1 -> v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./litmus-outputs"),
        help="Directory containing reports/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    args = parser.parse_args(argv)

    reports_dir = args.output_dir / "reports"
    if not reports_dir.exists():
        print(
            f"No reports directory at {reports_dir}",
            file=sys.stderr,
        )
        return 1

    migrated = 0
    skipped = 0
    errors = 0

    for json_path in sorted(reports_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ERROR reading {json_path}: {e}")
            errors += 1
            continue

        if "model_id" not in data or "status" not in data:
            skipped += 1
            continue

        if migrate_report(data):
            if not args.dry_run:
                json_path.write_text(
                    json.dumps(data, indent=2, default=str) + "\n"
                )
            migrated += 1
        else:
            skipped += 1

    action = "Would migrate" if args.dry_run else "Migrated"
    print(f"{action} {migrated} reports (skipped {skipped}, errors {errors})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
