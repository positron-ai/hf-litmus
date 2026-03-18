from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock

from .models import ModelResult, ModelStatus

logger = logging.getLogger(__name__)


class StateManager:
    STATE_VERSION = 2

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.state_file = output_dir / "state.json"
        self.lock_file = output_dir / "state.json.lock"
        self.lock = FileLock(str(self.lock_file), timeout=10)
        self._state: dict[str, ModelResult] = {}
        self._dirty = False
        self.notion_database_id: str = ""

    def load(self) -> None:
        """Load state from disk, creating if doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.state_file.exists():
                try:
                    data = json.loads(self.state_file.read_text())
                    version = data.get("version", 1)
                    if version != self.STATE_VERSION:
                        logger.warning(
                            "State version %d != expected %d",
                            version,
                            self.STATE_VERSION,
                        )
                        data = self._migrate_state(version, data)
                    self._state = {
                        k: ModelResult.from_dict(v)
                        for k, v in data.get("models", {}).items()
                    }
                    self.notion_database_id = data.get(
                        "notion_database_id", ""
                    )
                    logger.info(
                        "Loaded %d models from state", len(self._state)
                    )
                except json.JSONDecodeError:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    logger.error(
                        "Corrupted state file, backing up and starting fresh"
                    )
                    backup = self.state_file.with_suffix(f".{ts}.json.bak")
                    self.state_file.rename(backup)
                    self._state = {}
            else:
                self._state = {}
        self._dirty = False

    def save(self) -> None:
        """Atomically save state to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            temp_file = self.state_file.with_suffix(".tmp")
            data = {
                "version": self.STATE_VERSION,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "model_count": len(self._state),
                "notion_database_id": self.notion_database_id,
                "models": {k: v.to_dict() for k, v in self._state.items()},
            }
            temp_file.write_text(json.dumps(data, indent=2, default=str))
            temp_file.rename(self.state_file)
            self._dirty = False

    def all_results(self) -> list[ModelResult]:
        """Return a snapshot of all results (thread-safe)."""
        with self.lock:
            return list(self._state.values())

    def is_processed(self, model_id: str) -> bool:
        return model_id in self._state

    def get_result(self, model_id: str) -> Optional[ModelResult]:
        return self._state.get(model_id)

    def update(self, result: ModelResult) -> None:
        self._state[result.model_id] = result
        self._dirty = True

    def get_failed_models(self) -> list[str]:
        """Return model IDs with failed status for retest mode."""
        return [
            k
            for k, v in self._state.items()
            if v.status
            in (
                ModelStatus.EXPORT_FAIL,
                ModelStatus.INGEST_FAIL,
                ModelStatus.TIMEOUT,
            )
        ]

    def flush_if_dirty(self) -> None:
        if self._dirty:
            self.save()

    def _migrate_state(self, old_version: int, data: dict) -> dict:
        """Migrate state from older versions."""
        logger.info(
            "Migrating state from version %d to %d",
            old_version,
            self.STATE_VERSION,
        )
        if old_version == 1:
            # New fields (failure_origin, retryable,
            # deep_analysis_error, model_tags) have defaults
            # in ModelResult.from_dict(), just bump version.
            data["version"] = 2
        return data

    @property
    def count(self) -> int:
        return len(self._state)
