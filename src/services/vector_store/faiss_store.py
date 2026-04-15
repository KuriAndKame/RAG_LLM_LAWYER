import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from src.core.models.document import DocumentChunk
from config.settings import settings


class FAISSVectorStore:
    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[DocumentChunk] = []

    def _ensure_index(self, dim: int):
        """Создаёт индекс, если он ещё не создан, с указанной размерностью."""
        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatIP(dim)
            print(f"FAISS index created with dimension {dim}")

    def add_embeddings(self, embeddings: List[List[float]], chunks: List[DocumentChunk]):
        if not embeddings:
            return
        vectors = np.array(embeddings).astype('float32')
        dim = vectors.shape[1]
        self._ensure_index(dim)

        # Проверка совпадения размерности
        if dim != self.dimension:
            raise ValueError(
                f"Embedding dimension {dim} does not match index dimension {self.dimension}")

        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks_metadata.extend(chunks)

    def search(self, query_embedding: List[float], k: int = None) -> List[Tuple[DocumentChunk, float]]:
        k = k or settings.TOP_K_RETRIEVAL
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.chunks_metadata[idx], float(score)))
        return results

    def save(self, path: str = None):
        if self.index is None:
            return
        path = path or settings.VECTOR_STORE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "chunks_metadata": self.chunks_metadata,
                "dimension": self.dimension
            }, f)

    def load(self, path: str = None):
        path = path or settings.VECTOR_STORE_PATH
        index_path = f"{path}.index"
        pkl_path = f"{path}.pkl"
        if os.path.exists(index_path) and os.path.exists(pkl_path):
            self.index = faiss.read_index(index_path)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.chunks_metadata = data["chunks_metadata"]
                self.dimension = data["dimension"]
        return self
