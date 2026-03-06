import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.extraction.entity_extractor import extract_entities
from app.intent.intent_detector import detect_intent
from app.nlp.ner_model import ner_model
from app.ocr.ocr_engine import ocr_engine
from app.planner.action_planner import plan_action
from app.preprocessing.image_preprocess import preprocess_image
from app.schemas.response import ExtractResponse
from app.storage.db import get_db
from app.storage.image_store import LocalImageStore
from app.storage.models import ImageRecord
from app.vision.layout_model import layout_detector

logger = logging.getLogger(__name__)
router = APIRouter()
store = LocalImageStore()


def _merge_entities(ner_entities: dict, regex_entities: dict) -> dict:
    merged = dict(regex_entities)
    for key, value in ner_entities.items():
        if value not in (None, "", []):
            merged[key] = value

    # Keep regex amount when NER predicts an account-like amount token.
    if (
        merged.get("amount") is not None
        and merged.get("accountNumber")
        and str(merged["amount"]) == str(merged["accountNumber"])
        and regex_entities.get("amount") is not None
    ):
        merged["amount"] = regex_entities["amount"]

    # Prefer concise regex description if NER returns an over-long block.
    if (
        merged.get("description")
        and regex_entities.get("description")
        and len(str(merged["description"])) > 80
    ):
        merged["description"] = regex_entities["description"]
    return merged


@router.post("/extract", response_model=ExtractResponse)
async def extract_transfer_info(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    try:
        file_bytes = await file.read()
        image_path = store.save_upload(file, file_bytes)

        preprocessed = preprocess_image(file_bytes)
        raw_text, lines, confidence, ocr_items = ocr_engine.extract_document(preprocessed)

        # New pipeline stage: layout region detection from OCR boxes.
        layout_regions = layout_detector.detect_regions(ocr_items=ocr_items)

        # NER-first extraction using text from detected regions when available.
        ner_text_parts = [raw_text]
        for region in layout_regions.values():
            if region and region.get("text"):
                ner_text_parts.append(region["text"])
        ner_entities = ner_model.extract("\n".join(ner_text_parts))

        # Existing regex/heuristic extraction retained as fallback.
        regex_entities = extract_entities(raw_text=raw_text, lines=lines)
        entities = _merge_entities(ner_entities=ner_entities, regex_entities=regex_entities)

        intent = detect_intent(
            account_number=entities.get("accountNumber"),
            amount=entities.get("amount"),
        )
        action = plan_action(intent=intent, entities=entities)

        record = ImageRecord(
            image_path=image_path,
            raw_text=raw_text,
            intent=intent,
            account_number=entities.get("accountNumber"),
            bank=entities.get("bank"),
            amount=entities.get("amount"),
        )
        db.add(record)
        db.commit()

        return ExtractResponse(
            intent=intent,
            accountNumber=entities.get("accountNumber"),
            accountName=entities.get("accountName"),
            bank=entities.get("bank"),
            amount=entities.get("amount"),
            amountCandidates=entities.get("amountCandidates", []),
            description=entities.get("description"),
            rawText=raw_text,
            confidence=round(confidence, 4),
            action=action,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("extract failed")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"extract failed: {exc}") from exc
