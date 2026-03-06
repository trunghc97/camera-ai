import logging

from fastapi import FastAPI

from app.api.extract_api import router as extract_router
from app.storage.db import init_db
from config import settings


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title=settings.app_name)
app.include_router(extract_router, prefix="", tags=["extract"])


@app.on_event("startup")
def on_startup() -> None:
    # Ensure table exists before serving traffic.
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
