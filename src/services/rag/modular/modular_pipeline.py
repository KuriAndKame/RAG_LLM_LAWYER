# src/services/rag/modular/modular_pipeline.py

from typing import List
from src.services.llm.client import LLMClient
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.services.rag.hybrid_retriever import HybridRetriever
from src.services.rag.reranker import Reranker
from src.services.rag.generator import Generator
from src.services.web_search.service import WebSearchService
from src.core.models.chat import ChatResponse, ChatMessage, SourceInfo
from src.core.models.document import DocumentChunk
from config.settings import settings
from .query_rewriter import QueryRewriter
from .router import Router
from .memory import MemoryManager


class ModularRAGPipeline:
    def __init__(self, vector_store: FAISSVectorStore, chunks: List[DocumentChunk]):
        self.llm_client = LLMClient()
        self.rewriter = QueryRewriter(self.llm_client)
        self.router = Router()
        self.memory = MemoryManager(max_history=5)
        self.retriever = HybridRetriever(vector_store, chunks)
        self.reranker = None
        if settings.RERANKER_MODEL_NAME:
            try:
                self.reranker = Reranker(
                    model_name=settings.RERANKER_MODEL_NAME)
                print(f"Загружен реранкер: {settings.RERANKER_MODEL_NAME}")
            except Exception as e:
                print(
                    f"Не удалось загрузить реранкер: {e}. Работаем без реранкера.")
        self.generator = Generator(self.llm_client)
        self.web_search = WebSearchService()

    def query(self, user_message: str, history: List[ChatMessage] = None,
              use_vector_db: bool = None, use_web_search: bool = None) -> ChatResponse:
        # 1. Загружаем историю
        history = self.memory.get_context(history)

        # 2. Определяем стратегию: приоритет у флагов, иначе — маршрутизатор
        if use_vector_db is not None or use_web_search is not None:
            decision = {"use_vector_db": bool(use_vector_db), "use_web": bool(
                use_web_search), "rewrite": False}
        else:
            decision = self.router.decide(user_message)

        # 3. Рерайт запроса, если нужно
        query = self.rewriter.rewrite(
            user_message) if decision["rewrite"] else user_message

        # 4. Поиск в нескольких источниках
        sources = []
        if decision["use_vector_db"]:
            sources.extend(self.retriever.retrieve(query, top_k=20))
        if decision["use_web"]:
            web_results = self.web_search.search(query)
            for r in web_results:
                sources.append(SourceInfo(
                    document_name=r.get('title', 'Неизвестный источник'),
                    chunk_text=r.get('snippet', ''),
                    relevance_score=1.0
                ))

        # 5. Ре-ранкинг и лимитирование
        if self.reranker and sources:
            sources = self.reranker.rerank(query, sources, top_k=5)
        else:
            sources = sources[:5]

        # 6. Подготовка контекста
        MAX_CONTEXT_CHARS = 2000
        context_chunks = []
        total_len = 0
        for s in sources:
            if total_len + len(s.chunk_text) > MAX_CONTEXT_CHARS:
                break
            context_chunks.append(s.chunk_text)
            total_len += len(s.chunk_text)

        history_dicts = [{"role": msg.role, "content": msg.content}
                         for msg in history] if history else []

        answer = self.generator.generate(query, context_chunks, history_dicts)
        return ChatResponse(answer=answer, sources=sources)
