from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ModelStatus(Enum):
  SUCCESS = "SUCCESS"
  EXPORT_FAIL = "EXPORT_FAIL"
  INGEST_FAIL = "INGEST_FAIL"
  TIMEOUT = "TIMEOUT"
  SKIP = "SKIP"


class FailureStage(Enum):
  EXPORT = "export"
  INGEST = "ingest"


class FailureClass(Enum):
  MISSING_OP = "missing_op"
  TYPE_ERROR = "type_error"
  UNSUPPORTED_DYNAMIC = "unsupported_dynamic"
  ATEN_FALLBACK = "aten_fallback"
  SHAPE_MISMATCH = "shape_mismatch"
  MEMORY_ERROR = "memory_error"
  TRUST_REMOTE_CODE = "trust_remote_code"
  UNKNOWN = "unknown"


class FailureOrigin(Enum):
  TRON_PIPELINE = "tron_pipeline"
  PYTHON_INFRA = "python_infra"
  HF_ACCESS = "hf_access"
  DEEP_ANALYSIS = "deep_analysis"
  UNKNOWN = "unknown"


class Verdict(Enum):
  PASS = "pass"
  CLOSE = "close"
  FAR = "far"
  BLOCKED = "blocked"


class FurthestStage(Enum):
  EXPORT = "export"
  FXTYPEDFX = "fxtypedfx"
  REWRITE = "rewrite"
  BULK = "bulk"
  LOOPY = "loopy"
  TRON = "tron"
  CPP = "cpp"
  SUCCESS = "success"
  UNKNOWN = "unknown"


class GapAnalysisStatus(Enum):
  NOT_RUN = "not_run"
  COMPLETED = "completed"


@dataclass
class ModelResult:
  model_id: str
  status: ModelStatus
  failure_stage: Optional[FailureStage] = None
  failure_class: Optional[FailureClass] = None
  missing_ops: list[str] = field(default_factory=list)
  tested_at: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc)
  )
  ingest_version: str = ""
  pipeline_tag: str = ""
  downloads: int = 0
  likes: int = 0
  error_output: str = ""
  analysis_path: str = ""
  analysis_branch: str = ""
  notion_page_id: str = ""
  failure_origin: Optional[FailureOrigin] = None
  retryable: bool = False
  deep_analysis_error: str = ""
  model_tags: list[str] = field(default_factory=list)

  def to_dict(self) -> dict:
    """Serialize to JSON-compatible dict."""
    d = {
      "model_id": self.model_id,
      "status": self.status.value,
      "failure_stage": (
        self.failure_stage.value if self.failure_stage else None
      ),
      "failure_class": (
        self.failure_class.value if self.failure_class else None
      ),
      "missing_ops": self.missing_ops,
      "tested_at": self.tested_at.isoformat(),
      "ingest_version": self.ingest_version,
      "pipeline_tag": self.pipeline_tag,
      "downloads": self.downloads,
      "likes": self.likes,
      "error_output": self.error_output,
      "analysis_path": self.analysis_path,
      "analysis_branch": self.analysis_branch,
      "notion_page_id": self.notion_page_id,
      "failure_origin": (
        self.failure_origin.value
        if self.failure_origin else None
      ),
      "retryable": self.retryable,
      "deep_analysis_error": self.deep_analysis_error,
      "model_tags": self.model_tags,
    }
    return d

  @classmethod
  def from_dict(cls, d: dict) -> ModelResult:
    """Deserialize from dict."""
    return cls(
      model_id=d["model_id"],
      status=ModelStatus(d["status"]),
      failure_stage=(
        FailureStage(d["failure_stage"])
        if d.get("failure_stage")
        else None
      ),
      failure_class=(
        FailureClass(d["failure_class"])
        if d.get("failure_class")
        else None
      ),
      missing_ops=d.get("missing_ops", []),
      tested_at=datetime.fromisoformat(d["tested_at"]),
      ingest_version=d.get("ingest_version", ""),
      pipeline_tag=d.get("pipeline_tag", ""),
      downloads=d.get("downloads", 0),
      likes=d.get("likes", 0),
      error_output=d.get("error_output", ""),
      analysis_path=d.get("analysis_path", ""),
      analysis_branch=d.get("analysis_branch", ""),
      notion_page_id=d.get("notion_page_id", ""),
      failure_origin=(
        FailureOrigin(d["failure_origin"])
        if d.get("failure_origin")
        else None
      ),
      retryable=d.get("retryable", False),
      deep_analysis_error=d.get(
        "deep_analysis_error", ""
      ),
      model_tags=d.get("model_tags", []),
    )
