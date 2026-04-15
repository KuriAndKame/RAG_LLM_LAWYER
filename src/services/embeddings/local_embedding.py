from sentence_transformers import SentenceTransformer
from typing import List
from .model import EmbeddingModel
from config.settings import settings
import torch


class LocalEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=settings.EMBEDDING_DEVICE,
            # device='cpu',
            # model_kwargs={
            # "torch_dtype": torch.bfloat16,
            # "attn_implementation": "eager"
            # }
        )
        # Для модели Giga-Embeddings-instruct может потребоваться специальный prompt
        # self.instruction = "Представьте текст для поиска релевантных документов: "
        self.instruction = "query: "

    def encode(self, texts: List[str]) -> List[List[float]]:
        # Модель принимает список текстов
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        query_with_instruction = self.instruction + query
        embedding = self.model.encode(
            query_with_instruction, normalize_embeddings=True)
        return embedding.tolist()
