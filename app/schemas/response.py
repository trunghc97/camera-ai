from typing import Any, Optional

from pydantic import BaseModel, Field


class ActionResponse(BaseModel):
    screen: str
    fields: dict[str, Any]


class ExtractResponse(BaseModel):
    intent: str = Field(default="UNKNOWN")
    accountNumber: Optional[str] = None
    accountName: Optional[str] = None
    bank: Optional[str] = None
    amount: Optional[int] = None
    amountCandidates: list[int] = Field(default_factory=list)
    description: Optional[str] = None
    rawText: str = ""
    confidence: float = 0.0
    action: Optional[ActionResponse] = None
