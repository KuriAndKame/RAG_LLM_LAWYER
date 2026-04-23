# src/services/rag/modular/memory.py

class MemoryManager:
    """Управляет краткосрочной памятью диалога."""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history

    def get_context(self, history: list) -> list:
        if not history:
            return []
        return history[-self.max_history:]
