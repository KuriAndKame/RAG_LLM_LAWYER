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
        history_context = self.memory.get_context(history)

        # 2. Получаем базовое решение от роутера
        router_decision = self.router.decide(user_message)

        # 3. Формируем итоговое решение с учетом флагов (тестов)
        decision = {
            "use_vector_db": use_vector_db if use_vector_db is not None else router_decision["use_vector_db"],
            "use_web": use_web_search if use_web_search is not None else router_decision["use_web"],
            # Рерайт всегда зависит от логики роутера!
            "rewrite": router_decision["rewrite"]
        }

        # 4. Рерайт запроса, если нужно
        if decision["rewrite"]:
            print(f"[MODULAR] Выполняю рерайт запроса: '{user_message}'")
            query = self.rewriter.rewrite(user_message)
            print(f"[MODULAR] Новый запрос: '{query}'")
        else:
            query = user_message

        # 5. Поиск в нескольких источниках
        sources = []
        if decision["use_vector_db"]:
            sources.extend(self.retriever.retrieve(query, top_k=20))

        # Ищем в вебе, только если база пуста (или если нужно искать везде)
        if decision["use_web"] and not sources:
            print("[MODULAR] Выполняется поиск в интернете...")
            web_results = self.web_search.search(query)
            for r in web_results:
                sources.append(SourceInfo(
                    document_name=r.get('title', 'Неизвестный источник'),
                    chunk_text=r.get('snippet', ''),
                    relevance_score=1.0
                ))

        # 6. Ре-ранкинг и лимитирование
        if self.reranker and sources:
            sources = self.reranker.rerank(query, sources, top_k=5)
        else:
            sources = sources[:5]

        # 7. Подготовка контекста (без обрезки)
        context_chunks = [s.chunk_text for s in sources]

        history_dicts = [{"role": msg.role, "content": msg.content}
                         for msg in history_context] if history_context else []

        # 8. Генерация ответа
        answer = self.generator.generate(query, context_chunks, history_dicts)
        return ChatResponse(answer=answer, sources=sources)
