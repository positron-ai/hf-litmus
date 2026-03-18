from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.utils import HfHubHTTPError

if TYPE_CHECKING:
    from .state import StateManager

from .models import ModelStatus

logger = logging.getLogger(__name__)


class HFModelEnumerator:
    PIPELINE_TAG = "text-generation"
    COMPATIBLE_LIBRARIES = {"transformers", "pytorch"}
    MAX_RETRIES = 5
    BASE_BACKOFF = 1.0

    def __init__(
        self,
        token: Optional[str] = None,
        sort: str = "trending_score",
    ) -> None:
        self.api = HfApi(token=token)
        self.sort = sort

    def enumerate_models(self, limit: int = 0) -> Iterator[ModelInfo]:
        """Yield text-generation models from HF Hub.

        Args:
          limit: Max models to yield (0 = unlimited).
        """
        retries = 0
        yielded = 0

        while True:
            try:
                models = self.api.list_models(
                    filter=self.PIPELINE_TAG,
                    sort=self.sort,
                )
                for model in models:
                    if (
                        model.library_name
                        and model.library_name not in self.COMPATIBLE_LIBRARIES
                    ):
                        logger.debug(
                            "Skipping %s: library %s",
                            model.id,
                            model.library_name,
                        )
                        continue

                    if model.gated and not self.api.token:
                        logger.debug(
                            "Skipping gated model %s: no token",
                            model.id,
                        )
                        continue

                    yield model
                    yielded += 1
                    retries = 0
                    if limit and yielded >= limit:
                        return
                return

            except HfHubHTTPError as e:
                status = getattr(
                    getattr(e, "response", None),
                    "status_code",
                    None,
                )
                if status == 429:
                    retries += 1
                    if retries > self.MAX_RETRIES:
                        raise RuntimeError(
                            "Max retries exceeded for HF API rate limit"
                        ) from e
                    backoff = self.BASE_BACKOFF * (2**retries)
                    logger.warning(
                        "Rate limited, sleeping %.1fs (retry %d/%d)",
                        backoff,
                        retries,
                        self.MAX_RETRIES,
                    )
                    time.sleep(backoff)
                else:
                    raise

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get detailed info for a specific model."""
        return self.api.model_info(model_id)


def filter_by_state(
    models: Iterator[ModelInfo],
    state: StateManager,
    retest_mode: bool = False,
) -> Iterator[ModelInfo]:
    """Filter models based on processing state.

    Normal mode: skip already-processed models.
    Retest mode: only yield previously failed models.
    """
    for model in models:
        if retest_mode:
            result = state.get_result(model.id)
            if result and result.status not in (
                ModelStatus.SUCCESS,
                ModelStatus.SKIP,
            ):
                yield model
        else:
            if not state.is_processed(model.id):
                yield model
