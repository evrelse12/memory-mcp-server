# memory_mcp/services/nli.py
"""NLI classifier for hypothesis verification (shadow mode).

Cross-lingual entailment scoring via ONNX (mDeBERTa-v3-XNLI by default).
Lazy-loads on first use, unloads after the nightly run to free RAM.
Best-effort: if no valid ONNX export is available, Reflection runs critic-only.
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
        self._model_dir = model_dir or settings.nli_models_path
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
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
            from huggingface_hub import hf_hub_download
            candidates = [settings.nli_model_file, "model.onnx",
                          "onnx/model.onnx", "model_quantized.onnx"]
            last = None
            for fname in candidates:
                try:
                    got = hf_hub_download(repo_id=settings.nli_model_repo,
                                          filename=fname,
                                          local_dir=self._model_dir)
                    model_path = Path(got)
                    break
                except Exception as e:
                    last = e
            else:
                raise NLIError(f"failed to download NLI model: {last}")
            logger.info("NLI model downloaded")

        try:
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
            self._tokenizer = AutoTokenizer.from_pretrained(settings.nli_model_repo)
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
        """Score (premise, hypothesis) -> (label, confidence in [0,1])."""
        self._ensure_loaded()
        try:
            inputs = self._tokenizer(
                premise, hypothesis, return_tensors="np",
                truncation=True, max_length=512, padding=True)
            feeds = {k: inputs[k] for k in ("input_ids", "attention_mask")
                     if k in inputs}
            names = {i.name for i in self._session.get_inputs()}
            if "token_type_ids" in inputs and "token_type_ids" in names:
                feeds["token_type_ids"] = inputs["token_type_ids"]
            logits = self._session.run(None, feeds)[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            idx = int(np.argmax(probs))
            labels: list[NLILabel] = ["entailment", "neutral", "contradiction"]
            return labels[idx], float(probs[idx])
        except NLIError:
            raise
        except Exception as e:
            raise NLIError(f"NLI inference failed: {e}") from e

    async def classify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[NLILabel, float]]:
        return [await self.classify(p, h) for p, h in pairs]
