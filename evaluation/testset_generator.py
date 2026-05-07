# evaluation/testset_generator.py
"""
Генератор синтетического тестового набора вопросов на русском языке.

Основан на чанках из вашей базы знаний (FAISS) и локальной LLM.
"""

import json
import random
from typing import List, Dict, Optional
from pathlib import Path

from tqdm import tqdm

from evaluation.config import config as eval_config
from src.services.llm.client import LLMClient  # ваш собственный клиент
import sys
from pathlib import Path

# Добавляем корень проекта (где лежит папка src, evaluation, .env и т.д.) в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSetGenerator:
    """Генерирует синтетические вопросы и эталонные ответы из чанков."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.max_chunk_chars = 1500  # обрезаем слишком длинные чанки

    def _load_random_chunks(self, n: int = 20) -> List[Dict]:
        """Загружает случайные чанки из FAISS (через pickle-файл)."""
        import pickle
        from config.settings import settings
        pkl_path = Path(settings.VECTOR_STORE_PATH).with_suffix(".pkl")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Файл с чанками не найден: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            chunks_metadata = data["chunks_metadata"]

        # Выбираем случайные чанки
        selected = random.sample(chunks_metadata, min(n, len(chunks_metadata)))
        return [{"text": ch.text, "document": ch.metadata.get("filename", "unknown")} for ch in selected]

    def generate_question(self, chunk: str) -> Optional[str]:
        chunk_trimmed = chunk[:self.max_chunk_chars]
        prompt = f"""Ты — эксперт по законодательству РФ. Прочитай фрагмент правового документа и придумай ОДИН короткий вопрос, на который можно ответить, используя ТОЛЬКО этот фрагмент.

    Фрагмент документа:
    {chunk_trimmed}

    Вопрос (на русском языке):"""
        try:
            question = self.llm.generate(
                [{"role": "user", "content": prompt}], max_tokens=100)
            return question.strip().rstrip("?").strip()
        except Exception as e:
            print(f"Ошибка генерации вопроса: {e}")
            return None

    def generate_answer(self, chunk: str, question: str) -> Optional[str]:
        chunk_trimmed = chunk[:self.max_chunk_chars]
        prompt = f"""Ты — юрист. Ответь на вопрос кратко, используя ТОЛЬКО предоставленный фрагмент документа.

    Фрагмент:
    {chunk_trimmed}

    Вопрос: {question}

    Ответ (2-3 предложения):"""
        try:
            answer = self.llm.generate(
                [{"role": "user", "content": prompt}], max_tokens=200)
            return answer.strip()
        except Exception as e:
            print(f"Ошибка генерации ответа: {e}")
            return None

    def generate_testset(self, num_questions: int = 20, output_path: Optional[str] = None) -> List[Dict]:
        """
        Генерирует тестовый набор вопросов и эталонных ответов.

        Аргументы:
            num_questions: количество вопросов
            output_path: путь для сохранения JSON (по умолчанию evaluation/test_questions.json)

        Возвращает:
            список словарей с ключами question, ground_truth, use_web_search, use_vector_db
        """
        chunks = self._load_random_chunks(num_questions)
        testset = []

        for ch in tqdm(chunks, desc="Генерация вопросов"):
            question = self.generate_question(ch["text"])
            if not question:
                continue
            answer = self.generate_answer(ch["text"], question)
            if not answer:
                continue
            testset.append({
                "question": question,
                "ground_truth": answer,
                "use_web_search": False,
                "use_vector_db": True
            })

        if output_path is None:
            output_path = Path(eval_config.results_path) / \
                "test_questions.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(testset, f, ensure_ascii=False, indent=2)

        print(
            f"Сгенерирован тестовый набор из {len(testset)} вопросов, сохранён в {output_path}")
        return testset


if __name__ == "__main__":
    generator = TestSetGenerator()
    questions = generator.generate_testset(
        num_questions=eval_config.testset_size)
