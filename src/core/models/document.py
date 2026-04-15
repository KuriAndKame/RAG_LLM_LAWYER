from pydantic import BaseModel
from typing import List, Optional
import uuid


class DocumentChunk(BaseModel):
    id: str = str(uuid.uuid4())
    document_id: str
    text: str
    metadata: dict
    chunk_index: int


class Document(BaseModel):
    id: str = str(uuid.uuid4())
    filename: str
    source_path: str
    chunks: List[DocumentChunk] = []
