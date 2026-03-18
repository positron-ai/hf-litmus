from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .models import FailureClass, FailureOrigin

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    failure_class: FailureClass
    failure_origin: FailureOrigin = FailureOrigin.UNKNOWN
    retryable: bool = False
    missing_ops: list[str] = field(default_factory=list)
    error_summary: str = ""


PYTHON_INFRA_PATTERNS: list[str] = [
    r"ModuleNotFoundError",
    r"ImportError",
    r"No module named",
    r"pip install",
    r"FileNotFoundError.*\.py",
    r"SyntaxError",
]

HF_ACCESS_PATTERNS: list[tuple[str, bool]] = [
    # (pattern, retryable) - gated repos are retryable
    # after approval; trust_remote_code is not
    (r"GatedRepoError", True),
    (r"403.*gated", True),
    (r"Access to model.*is restricted", True),
    (r"you must be authenticated", True),
    (r"repository is gated", True),
    (r"trust_remote_code", False),
    (r"contains custom code which must be executed", False),
]

EXPORT_PATTERNS: list[tuple[FailureClass, list[str]]] = [
    (
        FailureClass.TRUST_REMOTE_CODE,
        [
            r"trust_remote_code",
            r"contains custom code which must be executed",
        ],
    ),
    (
        FailureClass.UNSUPPORTED_DYNAMIC,
        [
            r"torch\._dynamo.*guard failure",
            r"data-dependent control flow",
            r"Could not guard on data-dependent expression",
        ],
    ),
    (
        FailureClass.ATEN_FALLBACK,
        [
            r"operator.*not supported",
            r"Unsupported: Operator",
        ],
    ),
    (
        FailureClass.SHAPE_MISMATCH,
        [
            r"KeyError:.*config",
            r"AttributeError.*config",
            r"missing.*required.*field",
        ],
    ),
    (
        FailureClass.MEMORY_ERROR,
        [
            r"MemoryError",
            r"CUDA out of memory",
            r"Killed",
            r"SIGKILL",
        ],
    ),
]

INGEST_PATTERNS: list[tuple[FailureClass, list[str]]] = [
    (
        FailureClass.TYPE_ERROR,
        [
            r"Typechecking failed",
            r"Type error",
            r"shape mismatch",
            r"expected.*got",
        ],
    ),
    (
        FailureClass.MISSING_OP,
        [
            r"Unknown function",
            r"Unsupported op",
            r"not implemented",
            r"pattern match failure",
        ],
    ),
]

MISSING_OP_PATTERNS: list[str] = [
    r"Unknown function:\s*([\w.]+)",
    r"Unsupported op:\s*([\w.]+)",
    r"not implemented:\s*([\w.]+)",
    r"pattern match failure.*aten\.([\w]+)",
    r"(aten\.[\w._]+).*not supported",
]


def classify_export_error(
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> ClassificationResult:
    """Classify an export-stage failure."""
    combined = stdout + "\n" + stderr

    if timed_out:
        return ClassificationResult(
            failure_class=FailureClass.UNKNOWN,
            error_summary="Export timed out",
        )

    # Check HF access patterns first (gated models,
    # trust_remote_code)
    for pattern, retryable in HF_ACCESS_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            # Determine failure class
            fc = FailureClass.TRUST_REMOTE_CODE
            if retryable:
                fc = FailureClass.UNKNOWN
            return ClassificationResult(
                failure_class=fc,
                failure_origin=FailureOrigin.HF_ACCESS,
                retryable=retryable,
                error_summary=_extract_error_summary(combined),
            )

    # Check Python infrastructure patterns
    for pattern in PYTHON_INFRA_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return ClassificationResult(
                failure_class=FailureClass.UNKNOWN,
                failure_origin=FailureOrigin.PYTHON_INFRA,
                retryable=True,
                error_summary=_extract_error_summary(combined),
            )

    # Check Tron pipeline patterns
    for failure_class, patterns in EXPORT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return ClassificationResult(
                    failure_class=failure_class,
                    failure_origin=FailureOrigin.TRON_PIPELINE,
                    error_summary=_extract_error_summary(combined),
                )

    return ClassificationResult(
        failure_class=FailureClass.UNKNOWN,
        error_summary=_extract_error_summary(combined),
    )


def classify_ingest_error(
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> ClassificationResult:
    """Classify an ingest-stage failure.

    All ingest failures are Tron pipeline failures since
    we got past export successfully.
    """
    combined = stdout + "\n" + stderr

    if timed_out:
        return ClassificationResult(
            failure_class=FailureClass.UNKNOWN,
            failure_origin=FailureOrigin.TRON_PIPELINE,
            error_summary="Ingest timed out",
        )

    missing_ops = _extract_missing_ops(combined)
    if missing_ops:
        return ClassificationResult(
            failure_class=FailureClass.MISSING_OP,
            failure_origin=FailureOrigin.TRON_PIPELINE,
            missing_ops=missing_ops,
            error_summary=_extract_error_summary(combined),
        )

    for failure_class, patterns in INGEST_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return ClassificationResult(
                    failure_class=failure_class,
                    failure_origin=FailureOrigin.TRON_PIPELINE,
                    error_summary=_extract_error_summary(combined),
                )

    return ClassificationResult(
        failure_class=FailureClass.UNKNOWN,
        failure_origin=FailureOrigin.TRON_PIPELINE,
        error_summary=_extract_error_summary(combined),
    )


def _extract_missing_ops(text: str) -> list[str]:
    """Extract operation names from error text."""
    ops: set[str] = set()
    for pattern in MISSING_OP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            op = match.group(1)
            if not op.startswith("aten."):
                op = f"aten.{op}"
            ops.add(op)
    return sorted(ops)


def _extract_error_summary(text: str, max_lines: int = 10) -> str:
    """Extract the most relevant error lines."""
    lines = text.strip().split("\n")
    for i, line in enumerate(lines):
        if "Error:" in line or "Exception:" in line:
            start = max(0, i - 2)
            end = min(len(lines), i + max_lines)
            return "\n".join(lines[start:end])
    return "\n".join(lines[-max_lines:])
