# evaluation/generate_answers.py
import json
import requests
import time
from pathlib import Path

API_BASE_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/"


def query_pipeline(pipeline_type: str, question: str, use_web_search: bool = False, use_vector_db: bool = True):
    headers = {"accept": "application/json",
               "Content-Type": "application/json"}
    payload = {
        "message": question,
        "history": [],
        "use_web_search": use_web_search,
        "use_vector_db": use_vector_db
    }
    params = {"pipeline_type": pipeline_type}
    try:
        resp = requests.post(CHAT_ENDPOINT, params=params,
                             json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] {pipeline_type} на вопросе '{question[:30]}...' : {e}")
        return None


def main():
    questions_path = Path("evaluation/test_questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    pipeline_types = ["naive", "advanced", "modular"]
    all_datasets = {}

    for pipeline_type in pipeline_types:
        print(f"\n--- ГЕНЕРАЦИЯ ОТВЕТОВ: {pipeline_type.upper()} ---")
        dataset = []
        for idx, q in enumerate(questions):
            print(
                f"[{pipeline_type}] Вопрос {idx+1}/{len(questions)}: {q['question'][:50]}...")
            result = query_pipeline(pipeline_type, q["question"], q.get(
                "use_web_search", False), q.get("use_vector_db", True))

            if result and "answer" in result:
                contexts = [src.get("chunk_text", "")
                            for src in result.get("sources", [])]
                if not contexts:
                    contexts = [""]
                dataset.append({
                    "question": q.get("question", ""),
                    "answer": result.get("answer", ""),
                    "contexts": contexts,
                    "ground_truth": q.get("ground_truth", "")
                })
            time.sleep(25)
        all_datasets[pipeline_type] = dataset

    # Сохраняем сгенерированные ответы
    output_path = Path("evaluation/results/generated_answers.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_datasets, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Генерация завершена. Ответы сохранены в {output_path}")


if __name__ == "__main__":
    main()
