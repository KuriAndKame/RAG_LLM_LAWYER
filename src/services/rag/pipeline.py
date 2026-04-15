from src.services.rag.retriever import Retriever
from src.services.rag.generator import Generator
from src.services.llm.client import LLMClient
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.core.models.chat import ChatResponse, ChatMessage
from typing import List


class RAGPipeline:
    def __init__(self, vector_store: FAISSVectorStore):
        self.retriever = Retriever(vector_store)
        self.generator = Generator(LLMClient())

    def query(self, user_message: str, history: List[ChatMessage] = None) -> ChatResponse:
        sources = self.retriever.retrieve(user_message)

        # Ограничиваем количество чанков и их общую длину
        MAX_CONTEXT_CHARS = 2000  # примерно 500 токенов
        context_chunks = []
        total_len = 0
        for s in sources:
            if total_len + len(s.chunk_text) > MAX_CONTEXT_CHARS:
                break
            context_chunks.append(s.chunk_text)
            total_len += len(s.chunk_text)

        # ... остальной код
        # Поиск релевантных чанков
        sources = self.retriever.retrieve(user_message)
        context_chunks = [s.chunk_text for s in sources]

        # Приведение истории к простым словарям
        history_dicts = []
        if history:
            for msg in history:
                if isinstance(msg, ChatMessage):
                    # Pydantic модель -> словарь
                    history_dicts.append(
                        {"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    # Уже словарь — убедимся, что только нужные ключи
                    history_dicts.append(
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")})
                else:
                    # Неизвестный тип — пропускаем или преобразуем
                    try:
                        history_dicts.append(
                            {"role": msg.role, "content": msg.content})
                    except:
                        pass

        # Генерация ответа
        answer = self.generator.generate(
            user_message, context_chunks, history_dicts)

        return ChatResponse(answer=answer, sources=sources)
