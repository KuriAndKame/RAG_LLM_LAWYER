# Файл: src/services/rag/hybrid_retriever.py
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.services.embeddings.client import get_embedding_model
from src.core.models.document import DocumentChunk
from src.core.models.chat import SourceInfo
from config.settings import settings


class HybridRetriever:
    def __init__(self, vector_store: FAISSVectorStore, chunks: List[DocumentChunk]):
        self.vector_store = vector_store
        self.embedding_model = get_embedding_model()
        self.tokenized_corpus = [chunk.text.split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> List[SourceInfo]:
        top_k = top_k or settings.HYBRID_RETRIEVER_TOP_K

        # Используем только семантический (dense) поиск
        query_embedding = self.embedding_model.encode_query(query)
        dense_results = self.vector_store.search(query_embedding, k=top_k)

        final_sources = []
        for chunk, score in dense_results:
            final_sources.append(SourceInfo(
                document_name=chunk.metadata.get("filename", "Неизвестно"),
                chunk_text=chunk.text,
                relevance_score=float(score)
            ))
        return final_sources
