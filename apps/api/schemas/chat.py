from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    llm_provider: str = Field(default="vllm", pattern="^(vllm|openai)$")


class SourceChunk(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
