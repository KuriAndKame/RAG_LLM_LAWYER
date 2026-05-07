# evaluation/local_embeddings.py
import asyncio
from typing import List
from ragas.embeddings import BaseRagasEmbeddings
from sentence_transformers import SentenceTransformer
from pydantic import PrivateAttr


class LocalRagasEmbeddings(BaseRagasEmbeddings):
    """Обёртка над SentenceTransformer для RAGAS (совместимая с Pydantic/v0.2+)."""

    # Указываем Pydantic, что это приватный атрибут и его не нужно валидировать как строку
    _transformer: SentenceTransformer = PrivateAttr()

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        # RAGAS ожидает, что self.model — это строка (название модели)
        self.model = model_name
        self._transformer = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Синхронный метод для генерации эмбеддингов документов"""
        return self._transformer.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        """Синхронный метод для генерации эмбеддинга запроса (исправлено с async)"""
        embedding = self._transformer.encode(
            [query], normalize_embeddings=True)
        return embedding[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Асинхронная версия для документов"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, query: str) -> List[float]:
        """Асинхронная версия для запроса"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, query)
