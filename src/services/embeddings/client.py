from config.settings import settings
from .model import EmbeddingModel
from .local_embedding import LocalEmbedding


def get_embedding_model() -> EmbeddingModel:
    return LocalEmbedding()
