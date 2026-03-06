import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from config import settings


class LocalImageStore:
    def __init__(self, base_dir: str = settings.image_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file: UploadFile, file_bytes: bytes) -> str:
        # Use date partitioning to keep file system manageable.
        day_dir = self.base_dir / datetime.utcnow().strftime("%Y/%m/%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename or "image.jpg").suffix or ".jpg"
        filename = f"{uuid4().hex}{ext.lower()}"
        target = day_dir / filename

        with open(target, "wb") as f:
            f.write(file_bytes)

        return os.fspath(target)
