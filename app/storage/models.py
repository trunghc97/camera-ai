import uuid

from sqlalchemy import BIGINT, TIMESTAMP, Column, String, Text, func
from sqlalchemy.dialects.postgresql import UUID

from app.storage.db import Base


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_path = Column(Text, nullable=False)
    raw_text = Column(Text, nullable=True)
    intent = Column(String(32), nullable=False)
    account_number = Column(String(32), nullable=True)
    bank = Column(String(64), nullable=True)
    amount = Column(BIGINT, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
