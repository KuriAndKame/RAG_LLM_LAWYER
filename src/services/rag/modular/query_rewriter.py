# src/services/rag/modular/query_rewriter.py

class QueryRewriter:
    """Переформулирует запрос в юридический контекст с помощью LLM."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def rewrite(self, query: str) -> str:
        prompt = (
            "Переформулируй следующий запрос в лаконичный юридический вопрос на русском языке, "
            "связанный с законодательством РФ. Используй только русский язык, без перевода:\n\n"
            f"Запрос: {query}\n\n"
            "Юридическая формулировка:"
        )
        try:
            rewritten = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return rewritten.strip()
        except Exception as e:
            print(f"Не удалось переформулировать запрос: {e}")
            return query
