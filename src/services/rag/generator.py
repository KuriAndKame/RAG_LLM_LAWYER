from src.services.llm.client import LLMClient
from src.services.llm.prompts import SYSTEM_PROMPT, build_rag_prompt
from typing import List, Dict


class Generator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, query: str, context_chunks: List[str], history: List[Dict] = None) -> str:
        # Объединяем system prompt и rag_prompt в одно сообщение
        combined_content = SYSTEM_PROMPT + "\n\n" + \
            build_rag_prompt(query, context_chunks)
        messages = [{"role": "user", "content": combined_content}]
        # Историю пока не добавляем, чтобы минимизировать переменные
        return self.llm_client.generate(messages)
