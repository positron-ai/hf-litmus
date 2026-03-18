from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TRON_URL = "https://github.com/positron-ai/tron.git"


@dataclass
class LitmusConfig:
  output_dir: Path = Path("./litmus-output")
  sort_mode: str = "trending"
  batch_size: int = 100
  interval_minutes: int = 15
  single_run: bool = True
  retest_mode: bool = False
  target_model: str | None = None
  model_file: Path | None = None
  max_jobs: int = 1
  dump_intermediates: bool = False
  deep_analysis: bool = True
  deep_analysis_timeout: int = 7200
  consensus_review: bool = True
  export_timeout: int = 600
  ingest_timeout: int = 300
  hf_token: str | None = None
  notion_mcp_url: str | None = None
  notion_parent_page_id: str | None = None
  verbose: bool = False
  tron_url: str = DEFAULT_TRON_URL
