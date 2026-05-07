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
        k_constant = 60  # Константа сглаживания для RRF (обычно 60)

        # 1. Семантический поиск (Dense)
        query_embedding = self.embedding_model.encode_query(query)
        dense_results = self.vector_store.search(query_embedding, k=top_k)

        # 2. Лексический поиск (Sparse / BM25)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        # 3. Объединение через Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # Считаем RRF для Dense
        for rank, (chunk, _) in enumerate(dense_results):
            chunk_id = chunk.metadata.get(
                "id_chunk", chunk.text[:50])  # Нужен уникальный ID
            rrf_scores[chunk_id] = rrf_scores.get(
                chunk_id, 0) + (1.0 / (k_constant + rank + 1))

        # Считаем RRF для BM25
        for rank, idx in enumerate(bm25_top_indices):
            chunk = self.chunks[idx]
            chunk_id = chunk.metadata.get("id_chunk", chunk.text[:50])
            rrf_scores[chunk_id] = rrf_scores.get(
                chunk_id, 0) + (1.0 / (k_constant + rank + 1))

        # 4. Сортируем и выбираем финальные чанки
        sorted_rrf = sorted(rrf_scores.items(),
                            key=lambda x: x[1], reverse=True)
        final_sources = []

        for chunk_id, rrf_score in sorted_rrf[:top_k]:
            # Находим сам чанк по ID (или тексту)
            original_chunk = next(c for c in self.chunks if c.metadata.get(
                "id_chunk", c.text[:50]) == chunk_id)
            final_sources.append(SourceInfo(
                document_name=original_chunk.metadata.get(
                    "filename", "Неизвестно"),
                chunk_text=original_chunk.text,
                relevance_score=float(rrf_score)
            ))

        return final_sources
