from src.services.embeddings.client import get_embedding_model
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.core.models.chat import SourceInfo
from typing import List


class Retriever:
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.embedding_model = get_embedding_model()

    def retrieve(self, query: str, top_k: int = None) -> List[SourceInfo]:
        query_embedding = self.embedding_model.encode_query(query)
        results = self.vector_store.search(query_embedding, k=top_k)
        sources = []
        for chunk, score in results:
            sources.append(SourceInfo(
                document_name=chunk.metadata.get("filename", "Неизвестно"),
                chunk_text=chunk.text,
                relevance_score=score
            ))
        return sources
