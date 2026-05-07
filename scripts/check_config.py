#!/usr/bin/env python
"""
Скрипт для проверки конфигурации перед запуском RAGAS оценки.
Проверяет:
1. Подключение к LM Studio
2. Подключение к RAG API
3. Наличие файла с вопросами
4. Загрузку embedding модели
"""

import sys
from pathlib import Path
import json

# Добавляем корневую директорию в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from evaluation.config import config as eval_config


def check_lm_studio():
    """Проверяет подключение к LM Studio"""
    print("\n" + "=" * 60)
    print("1. Проверка LM Studio")
    print("=" * 60)
    
    try:
        from openai import OpenAI
        
        base_url = eval_config.eval_llm_base_url
        api_key = eval_config.eval_llm_api_key
        model_name = eval_config.eval_llm_model_name
        
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model_name}")
        print(f"   API Key: {api_key}")
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        print("\n   Отправляю тестовый запрос...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=50,
            temperature=0.0
        )
        
        print(f"   ✓ LM Studio ОК")
        print(f"   Ответ: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"   ✗ ОШИБКА: {e}")
        print(f"\n   💡 Решение:")
        print(f"      1. Откройте LM Studio")
        print(f"      2. Загрузите модель yandex/YandexGPT-5-Lite-8B-instruct")
        print(f"      3. Откройте вкладку 'Local Server'")
        print(f"      4. Нажмите кнопку запуска")
        print(f"      5. Убедитесь, что сервер работает на {eval_config.eval_llm_base_url}")
        return False


def check_rag_api():
    """Проверяет подключение к RAG API"""
    print("\n" + "=" * 60)
    print("2. Проверка RAG API сервера")
    print("=" * 60)
    
    try:
        import requests
        
        url = "http://127.0.0.1:8000/health"
        print(f"   Проверяю: {url}")
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print(f"   ✓ RAG API ОК")
        print(f"   Status: {data}")
        return True
        
    except Exception as e:
        print(f"   ✗ ОШИБКА: {e}")
        print(f"\n   💡 Решение:")
        print(f"      Запустите RAG API сервер:")
        print(f"      python src/api/main.py")
        return False


def check_test_questions():
    """Проверяет наличие файла с вопросами"""
    print("\n" + "=" * 60)
    print("3. Проверка файла с тестовыми вопросами")
    print("=" * 60)
    
    try:
        # Пытаемся найти файл в разных местах
        paths_to_try = [
            Path("evaluation/test_questions.json"),
            Path(eval_config.results_path) / "test_questions.json",
            root_dir / "evaluation/test_questions.json",
        ]
        
        file_path = None
        for p in paths_to_try:
            if p.exists():
                file_path = p
                break
        
        if not file_path:
            print(f"   ✗ Файл не найден в:")
            for p in paths_to_try:
                print(f"      - {p}")
            
            print(f"\n   💡 Решение:")
            print(f"      Создайте файл evaluation/test_questions.json:")
            print(f"      [")
            print(f"        {{")
            print(f'          "question": "Ваш вопрос?"')
            print(f"        }}")
            print(f"      ]")
            return False
        
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        print(f"   ✓ Файл найден: {file_path}")
        print(f"   Количество вопросов: {len(questions)}")
        if questions:
            print(f"   Первый вопрос: {questions[0].get('question', 'N/A')[:60]}...")
        
        return len(questions) > 0
        
    except json.JSONDecodeError as e:
        print(f"   ✗ ОШИБКА в JSON: {e}")
        return False
    except Exception as e:
        print(f"   ✗ ОШИБКА: {e}")
        return False


def check_embedding_model():
    """Проверяет загрузку embedding модели"""
    print("\n" + "=" * 60)
    print("4. Проверка embedding модели")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = eval_config.eval_embedding_model_name
        device = eval_config.eval_embedding_device
        
        print(f"   Модель: {model_name}")
        print(f"   Устройство: {device}")
        print(f"   Загружаю модель...")
        
        model = SentenceTransformer(model_name, device=device)
        
        # Тестируем encode
        test_text = ["Hello world"]
        embeddings = model.encode(test_text, normalize_embeddings=True)
        
        print(f"   ✓ Embedding модель ОК")
        print(f"   Размер эмбеддинга: {len(embeddings[0])}")
        return True
        
    except Exception as e:
        print(f"   ✗ ОШИБКА: {e}")
        print(f"\n   💡 Решение:")
        print(f"      Попробуйте скачать модель вручную:")
        print(f"      python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{eval_config.eval_embedding_model_name}')\"")
        return False


def check_ragas():
    """Проверяет наличие RAGAS"""
    print("\n" + "=" * 60)
    print("5. Проверка RAGAS библиотеки")
    print("=" * 60)
    
    try:
        import ragas
        print(f"   ✓ RAGAS установлен")
        print(f"   Версия: {ragas.__version__}")
        return True
    except ImportError:
        print(f"   ✗ RAGAS не установлен")
        print(f"\n   💡 Решение:")
        print(f"      pip install ragas")
        return False


def main():
    print("\n" + "🔍 " * 30)
    print("ПРОВЕРКА КОНФИГУРАЦИИ РАGAS ОЦЕНКИ")
    print("🔍 " * 30)
    
    checks = [
        ("LM Studio", check_lm_studio),
        ("RAG API", check_rag_api),
        ("Test Questions", check_test_questions),
        ("Embedding Model", check_embedding_model),
        ("RAGAS", check_ragas),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n   ✗ Критическая ошибка: {e}")
            results.append((name, False))
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"   {status} {name}")
    
    all_ok = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ВСЁ ГОТОВО! Можно запустить оценку:")
        print("\n   python scripts/run_evaluation.py")
    else:
        print("✗ ЕСТЬ ПРОБЛЕМЫ! Исправьте ошибки выше перед запуском.")
    print("=" * 60 + "\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
