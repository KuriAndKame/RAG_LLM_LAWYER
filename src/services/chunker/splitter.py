from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from src.core.models.document import Document, DocumentChunk
from config.settings import settings


class Chunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_document(self, document: Document, raw_text: str) -> Document:
        chunks_text = self.splitter.split_text(raw_text)
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk = DocumentChunk(
                document_id=document.id,
                text=chunk_text,
                metadata={"filename": document.filename,
                          "source": document.source_path},
                chunk_index=i
            )
            chunks.append(chunk)
        document.chunks = chunks
        return document
