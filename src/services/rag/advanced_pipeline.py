# src/services/rag/advanced_pipeline.py
from typing import List
from src.services.llm.client import LLMClient
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.services.rag.hybrid_retriever import HybridRetriever
from src.services.rag.reranker import Reranker
from src.services.rag.generator import Generator
from src.core.models.chat import ChatResponse, ChatMessage, SourceInfo
from src.core.models.document import DocumentChunk
from src.services.web_search.service import WebSearchService
from config.settings import settings


class AdvancedRAGPipeline:
    def __init__(self, vector_store: FAISSVectorStore, chunks: List[DocumentChunk]):
        self.retriever = HybridRetriever(vector_store, chunks)
        # Возвращаем ре-ранкер с новой моделью
        self.reranker = Reranker(model_name=settings.RERANKER_MODEL_NAME)
        self.generator = Generator(LLMClient())
        self.web_search = WebSearchService()

    def query(self, user_message: str, history: List[ChatMessage] = None, use_vector_db: bool = True) -> ChatResponse:
        user_message = user_message.strip()
        if not user_message:
            return ChatResponse(answer="Пожалуйста, введите ваш вопрос.", sources=[])

        initial_sources = []

        # 1. Поиск в векторной базе (если включён)
        if use_vector_db:
            initial_sources = self.retriever.retrieve(user_message, top_k=20)

        # 2. Если результатов нет (или БД отключена), ищем в интернете
        if not initial_sources:
            print("Выполняется поиск в интернете...")
            web_results = self.web_search.search(user_message)
            if web_results:
                initial_sources = []
                for result in web_results:
                    initial_sources.append(SourceInfo(
                        document_name=result.get(
                            'title', 'Неизвестный источник'),
                        chunk_text=result.get('snippet', ''),
                        relevance_score=1.0
                    ))

        # 3. Ре-ранкинг (если активирован) и ограничение контекста
        if self.reranker and initial_sources:
            sources = self.reranker.rerank(
                user_message, initial_sources, top_k=5)
        else:
            sources = initial_sources[:5]

        MAX_CONTEXT_CHARS = 2000
        context_chunks = []
        total_len = 0
        for s in sources:
            if total_len + len(s.chunk_text) > MAX_CONTEXT_CHARS:
                break
            context_chunks.append(s.chunk_text)
            total_len += len(s.chunk_text)

        history_dicts = []
        if history:
            for msg in history:
                history_dicts.append(
                    {"role": msg.role, "content": msg.content})

        answer = self.generator.generate(
            user_message, context_chunks, history_dicts)
        return ChatResponse(answer=answer, sources=sources)
