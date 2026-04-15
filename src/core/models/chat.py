from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str  # "user" или "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    use_web_search: bool = False


class SourceInfo(BaseModel):
    document_name: str
    chunk_text: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
