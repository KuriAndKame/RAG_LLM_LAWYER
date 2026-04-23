from typing import Optional, List
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    use_web_search: Optional[bool] = None   # теперь Optional
    use_vector_db: Optional[bool] = None    # теперь Optional


class SourceInfo(BaseModel):
    document_name: str
    chunk_text: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
