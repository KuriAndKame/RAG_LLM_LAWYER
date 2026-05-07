#!/usr/bin/env python
"""
Script to run RAGAS evaluation of RAG pipelines.
Использует конфигурацию из evaluation/config.py и .env файла.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from evaluation.evaluator import run_evaluation


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAGAS Evaluation Script")
    print("Запуск оценки RAG-пайплайнов с помощью RAGAS")
    print("=" * 70 + "\n")
    
    print("Убедитесь, что:")
    print("1. LM Studio запущен локально на http://127.0.0.1:1234")
    print("2. Модель YandexGPT-5-Lite-8B-instruct загружена в LM Studio")
    print("3. RAG API сервер работает на http://127.0.0.1:8000")
    print("\nНажмите Enter для начала оценки...")
    # input()
    
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\n[INFO] Оценка прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Ошибка при запуске оценки: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
