
from config.settings import settings
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.services.rag.pipeline import RAGPipeline
from src.services.rag.advanced_pipeline import AdvancedRAGPipeline
from src.services.document_loader.loader import DocumentLoader
from src.services.chunker.splitter import Chunker
from src.core.models.document import DocumentChunk
from typing import List
from src.services.rag.modular.modular_pipeline import ModularRAGPipeline
import pickle

_vector_store = None
_chunks = None
_naive_pipeline = None
_advanced_pipeline = None
_modular_pipeline = None


def get_modular_pipeline():
    global _modular_pipeline
    if _modular_pipeline is None:
        print("[INFO] Инициализация ModularRAGPipeline...")
        # Используем правильное имя функции: _load_all_chunks()
        _modular_pipeline = ModularRAGPipeline(
            get_vector_store(), _load_all_chunks())
    return _modular_pipeline


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
        try:
            _vector_store.load()
        except:
            pass
    return _vector_store


def _load_all_chunks() -> List[DocumentChunk]:
    """Загружаем все чанки из сохранённого состояния FAISS для BM25."""
    global _chunks
    if _chunks is None:
        try:
            with open(settings.VECTOR_STORE_PATH + ".pkl", "rb") as f:
                data = pickle.load(f)
                _chunks = data["chunks_metadata"]
        except:
            _chunks = []
    return _chunks


def get_naive_pipeline():
    global _naive_pipeline
    if _naive_pipeline is None:
        _naive_pipeline = RAGPipeline(get_vector_store())
    return _naive_pipeline


def get_advanced_pipeline():
    global _advanced_pipeline
    if _advanced_pipeline is None:
        chunks = _load_all_chunks()
        _advanced_pipeline = AdvancedRAGPipeline(get_vector_store(), chunks)
    return _advanced_pipeline


def get_current_pipeline():
    """Возвращает пайплайн в зависимости от настроек в .env."""
    if settings.RAG_PIPELINE_TYPE.lower() == "advanced":
        return get_advanced_pipeline()
    else:
        return get_naive_pipeline()
