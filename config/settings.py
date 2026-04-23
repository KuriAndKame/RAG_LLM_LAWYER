import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # LM Studio LLM
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:1234")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "lm-studio")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "local-model")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1024))

    # Embeddings
    EMBEDDING_PROVIDER: str = os.getenv(
        "EMBEDDING_PROVIDER", "local")
    EMBEDDING_BASE_URL: str = os.getenv(
        "EMBEDDING_BASE_URL", "http://localhost:1235/v1")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "lm-studio")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 1024))

    # Vector store
    VECTOR_STORE_PATH: str = os.getenv(
        "VECTOR_STORE_PATH", "data/vector_store/faiss_index")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", 5))

    # Data
    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "data/raw")
    PROCESSED_DATA_PATH: str = os.getenv(
        "PROCESSED_DATA_PATH", "data/processed")

    RERANKER_MODEL_NAME: str = os.getenv(
        "RERANKER_MODEL_NAME", "DiTy/cross-encoder-russian-msmarco")

    RAG_PIPELINE_TYPE: str = os.getenv(
        "RAG_PIPELINE_TYPE", "naive")  # naive / advanced

    HYBRID_RETRIEVER_TOP_K: int = 20   # сколько кандидатов берём от гибридного поиска
    FINAL_TOP_K: int = 5               # сколько остаётся после реранкинга

    # Поиск в сети
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
