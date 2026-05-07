# src/services/rag/pipeline.py
from src.services.rag.retriever import Retriever
from src.services.rag.generator import Generator
from src.services.llm.client import LLMClient
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.core.models.chat import ChatResponse, ChatMessage
from src.services.web_search.service import WebSearchService
from typing import List
from src.core.models.chat import SourceInfo


class RAGPipeline:
    def __init__(self, vector_store: FAISSVectorStore):
        self.retriever = Retriever(vector_store)
        self.generator = Generator(LLMClient())
        self.web_search = WebSearchService()

    def query(self, user_message: str, history: List[ChatMessage] = None,
              use_vector_db: bool = True, use_web_search: bool = False) -> ChatResponse:
        user_message = user_message.strip()
        if not user_message:
            return ChatResponse(answer="Пожалуйста, введите ваш вопрос.", sources=[])

        sources = []

        # 1. Поиск в векторной базе (если включён)
        if use_vector_db:
            sources = self.retriever.retrieve(user_message, top_k=5)

        # 2. Если результатов нет и поиск разрешен, ищем в интернете
        if use_web_search and not sources:
            print("Выполняется поиск в интернете...")
            web_results = self.web_search.search(user_message)
            if web_results:
                sources = []
                for result in web_results:
                    sources.append(SourceInfo(
                        document_name=result.get(
                            'title', 'Неизвестный источник'),
                        chunk_text=result.get('snippet', ''),
                        relevance_score=1.0
                    ))
                print(f"Найдено {len(sources)} результатов в интернете.")

        # 3. Формирование контекста (передаем найденные 5 чанков целиком)
        context_chunks = [s.chunk_text for s in sources]

        history_dicts = []
        if history:
            for msg in history:
                history_dicts.append(
                    {"role": msg.role, "content": msg.content})

        answer = self.generator.generate(
            user_message, context_chunks, history_dicts)
        return ChatResponse(answer=answer, sources=sources)
