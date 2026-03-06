import logging
from typing import Any, Tuple

import numpy as np
from paddleocr import PaddleOCR

from config import settings

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self):
        self._ocr = self._build_engine()

    @staticmethod
    def _build_engine() -> PaddleOCR:
        try:
            # Vietnamese model handles mixed VN/EN transfer content in practice.
            return PaddleOCR(use_angle_cls=settings.ocr_use_angle_cls, lang="vi", show_log=False)
        except Exception as exc:
            logger.warning("PaddleOCR vi model load failed, fallback to en: %s", exc)
            return PaddleOCR(use_angle_cls=settings.ocr_use_angle_cls, lang="en", show_log=False)

    def extract_document(self, image: np.ndarray) -> tuple[str, list[str], float, list[dict[str, Any]]]:
        result = self._ocr.ocr(image, cls=settings.ocr_use_angle_cls)
        if not result:
            return "", [], 0.0, []

        lines: list[str] = []
        confidences: list[float] = []
        items: list[dict[str, Any]] = []

        for block in result:
            if not block:
                continue
            for item in block:
                if len(item) < 2:
                    continue
                box = item[0]
                text = str(item[1][0]).strip()
                score = float(item[1][1])
                if text:
                    lines.append(text)
                    confidences.append(score)
                    items.append({"text": text, "score": score, "box": box})

        raw_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return raw_text, lines, avg_conf, items

    def extract_text(self, image: np.ndarray) -> Tuple[str, list[str], float]:
        # Backward-compatible helper retained for existing call sites.
        raw_text, lines, avg_conf, _ = self.extract_document(image)
        return raw_text, lines, avg_conf


ocr_engine = OCREngine()
