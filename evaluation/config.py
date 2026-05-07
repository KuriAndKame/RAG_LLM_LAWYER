# evaluation/config.py
"""
Централизованная конфигурация для оценки RAG-пайплайнов с помощью RAGAS.
Позволяет использовать любые локальные модели, доступные через LM Studio,
и гибко настраивать параметры оценки.
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# evaluation/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Находим путь к .env относительно этого файла (поднимаемся на уровень выше в корень проекта)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()


class EvaluationConfig(BaseSettings):
    """
    Настройки для запуска оценки RAGAS.
    Все параметры могут быть переопределены через переменные окружения
    или в файле .env с префиксом EVAL_.
    """

    # --- Параметры LLM-оценщика (Judge) ---
    # URL LM Studio, где запущена LLM
    eval_llm_base_url: str = os.getenv(
        "EVAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"
    )
    # API-ключ (для LM Studio подойдёт любая непустая строка)
    eval_llm_api_key: str = os.getenv("EVAL_LLM_API_KEY", "lm-studio")
    # Имя модели, загруженной в LM Studio (должно соответствовать названию в LM Studio)
    eval_llm_model_name: str = os.getenv(
        "EVAL_LLM_MODEL_NAME", "local-model"
    )
    # Температура генерации для оценки (0.0 → максимальная детерминированность)
    eval_llm_temperature: float = float(
        os.getenv("EVAL_LLM_TEMPERATURE", "0.0")
    )
    # Максимальное число токенов в ответе оценщика
    eval_llm_max_tokens: int = int(
        os.getenv("EVAL_LLM_MAX_TOKENS", "512")
    )

    # --- Параметры эмбеддера (если понадобится для метрик) ---
    # Какую embedding-модель использовать для вычисления семантического сходства
    eval_embedding_provider: str = os.getenv(
        "EVAL_EMBEDDING_PROVIDER", "local"
    )  # "local" или "lmstudio"
    eval_embedding_model_name: str = os.getenv(
        "EVAL_EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large"
    )
    eval_embedding_device: str = os.getenv(
        "EVAL_EMBEDDING_DEVICE", "cpu"
    )

    # --- Общие настройки RAGAS ---
    # Количество вопросов для генерации синтетического тестового набора
    testset_size: int = int(os.getenv("EVAL_TESTSET_SIZE", "20"))
    # Путь для сохранения результатов оценки
    results_path: str = os.getenv(
        "EVAL_RESULTS_PATH", "evaluation/results"
    )

    class Config:
        env_file = ".env"
        env_prefix = "EVAL_"  # все переменные окружения с префиксом EVAL_
        extra = "ignore"      # игнорируем неописанные переменные


# Синглтон для использования во всем пакете evaluation
config = EvaluationConfig()
