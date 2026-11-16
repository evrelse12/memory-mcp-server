# memory_mcp/services/nli.py
"""NLI classifier for hypothesis verification (shadow mode).

Uses mDeBERTa-v3-base-xnli-multilingual-nli (ONNX) for cross-lingual entailment
scoring. Lazy-loads on first use, unloads after nightly run to free VRAM/RAM.

Shadow mode: logs scores but doesn't decide (thresholds calibrated in Phase 4).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from memory_mcp.config.settings import settings

logger = logging.getLogger(__name__)

NLILabel = Literal["entailment", "neutral", "contradiction"]


class NLIError(RuntimeError):
    """NLI model unavailable or inference failed."""


class NLIClassifier:
    """Cross-lingual NLI via ONNX runtime (lazy-load, explicit unload)."""

    def __init__(self, model_dir: Path | None = None):
        self._model_dir = model_dir or (settings.data_dir / "models" / "nli")
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            raise NLIError(
                "onnxruntime or transformers not installed; "
                "run: pip install onnxruntime transformers"
            ) from e

        self._model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_dir / "model.onnx"

        if not model_path.exists():
            logger.info("Downloading NLI model to %s", self._model_dir)
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli",
                    filename="onnx/model.onnx",
                    local_dir=self._model_dir,
                )
                logger.info("NLI model downloaded")
            except Exception as e:
                raise NLIError(f"failed to download NLI model: {e}") from e

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli",
                local_files_only=False,
            )
            logger.info("NLI model loaded")
        except Exception as e:
            raise NLIError(f"failed to load NLI model: {e}") from e

    def unload(self) -> None:
        """Explicitly release model resources."""
        if self._session is not None:
            del self._session
            self._session = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        logger.info("NLI model unloaded")

    async def classify(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        """Score (premise, hypothesis) pair -> (label, confidence).

        Label: entailment/neutral/contradiction; confidence in [0, 1].
        """
        self._ensure_loaded()
        try:
            inputs = self._tokenizer(
                premise, hypothesis, return_tensors="np",
                truncation=True, max_length=512, padding=True
            )
            outputs = self._session.run(
                None,
                {"input_ids": inputs["input_ids"],
                 "attention_mask": inputs["attention_mask"]},
            )
            logits = outputs[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            label_idx = int(np.argmax(probs))
            labels: list[NLILabel] = ["entailment", "neutral", "contradiction"]
            return labels[label_idx], float(probs[label_idx])
        except Exception as e:
            raise NLIError(f"NLI inference failed: {e}") from e

    async def classify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[NLILabel, float]]:
        """Classify multiple (premise, hypothesis) pairs."""
        return [await self.classify(p, h) for p, h in pairs]
