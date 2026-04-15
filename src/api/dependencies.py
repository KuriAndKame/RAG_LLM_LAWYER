from src.services.vector_store.faiss_store import FAISSVectorStore
from src.services.rag.pipeline import RAGPipeline
from config.settings import settings

# Глобальные объекты (можно улучшить через lifespan)
vector_store = None
rag_pipeline = None


def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = FAISSVectorStore()
        try:
            vector_store.load()
        except:
            # Если индекса ещё нет, создаём пустой
            pass
    return vector_store


def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(get_vector_store())
    return rag_pipeline
