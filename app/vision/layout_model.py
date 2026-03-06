import logging
import re
from typing import Any

from transformers import AutoModel, AutoProcessor

from config import settings

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Document layout detector wrapper.

    The HuggingFace model is loaded lazily for POC compatibility. The current
    implementation combines OCR line boxes with semantic keyword matching to
    assign transfer-related regions.
    """

    def __init__(self, model_name: str = settings.layout_model_name):
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
        except Exception as exc:
            logger.warning("Layout model unavailable (%s). Using OCR-box heuristics only.", exc)
            self._model = False

    @staticmethod
    def _bbox_from_poly(poly: list[list[float]]) -> list[int]:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]

    def detect_regions(self, ocr_items: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
        self._ensure_model()

        regions = {
            "bank": None,
            "account_number": None,
            "amount": None,
            "description": None,
        }

        amount_pattern = re.compile(r"(?<!\d)(\d{1,3}(?:[.,\s]\d{3})+|\d{4,15})(?!\d)")

        for item in ocr_items:
            text = item.get("text", "")
            if not text:
                continue
            upper = text.upper()
            bbox = self._bbox_from_poly(item.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]]))
            payload = {"text": text, "bbox": bbox, "score": item.get("score", 0.0)}

            if regions["bank"] is None and any(
                k in upper for k in ["MB", "MB BANK", "VIETCOMBANK", "TECHCOMBANK", "BIDV", "AGRIBANK", "ACB", "SACOMBANK"]
            ):
                regions["bank"] = payload

            if regions["account_number"] is None and re.search(r"(?<!\d)\d{8,20}(?!\d)", text):
                regions["account_number"] = payload

            if regions["amount"] is None and amount_pattern.search(text):
                regions["amount"] = payload

            if regions["description"] is None and any(k in upper for k in ["NOI DUNG", "DIEN GIAI", "DESCRIPTION", "REMARK", "NOTE"]):
                regions["description"] = payload

        return regions


layout_detector = LayoutDetector()
