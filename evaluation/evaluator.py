# evaluation/evaluator.py
import json
from pathlib import Path
from datasets import Dataset as HfDataset
from ragas import evaluate
from openai import OpenAI
from ragas.llms import llm_factory

from evaluation.metrics import get_ragas_metrics
from evaluation.config import config as eval_config
from evaluation.local_embeddings import LocalRagasEmbeddings
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


def create_eval_llm():
    key = eval_config.eval_llm_api_key
    print(f"[DEBUG] Используемый ключ: {key[:4]}...{key[-4:]}")

    # Используем Langchain обертку, чтобы обойти конфликт с LM Studio
    chat_llm = ChatOpenAI(
        base_url=eval_config.eval_llm_base_url,
        api_key=key,
        model=eval_config.eval_llm_model_name,
        temperature=eval_config.eval_llm_temperature,
        # Принудительно отключаем json_object, чтобы не злить LM Studio
        model_kwargs={"response_format": {"type": "text"}}
    )

    return LangchainLLMWrapper(chat_llm)


def run_evaluation():
    print("=" * 60)
    print("ЗАПУСК ОЦЕНКИ RAG-ПАЙПЛАЙНОВ (ОФЛАЙН РЕЖИМ)")
    print("=" * 60)

    answers_path = Path("evaluation/results/generated_answers.json")
    if not answers_path.exists():
        print("[CRITICAL] Файл generated_answers.json не найден. Сначала запустите generate_answers.py!")
        return

    with open(answers_path, "r", encoding="utf-8") as f:
        all_datasets = json.load(f)

    eval_llm = create_eval_llm()
    raw_embeddings = LocalRagasEmbeddings(
        model_name=eval_config.eval_embedding_model_name,
        device=eval_config.eval_embedding_device
    )
    metrics = get_ragas_metrics()
    all_results = {}

    for pipeline_type, data_list in all_datasets.items():
        if not data_list:
            continue
        print(f"\n--- ОЦЕНКА ПАЙПЛАЙНА: {pipeline_type.upper()} ---")

        hf_dataset = HfDataset.from_list(data_list)
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=raw_embeddings
        )

        df = result.to_pandas()
        numeric_scores = df.select_dtypes(include='number').mean().to_dict()

        print(f"\n[DONE] Результаты {pipeline_type}:")
        for m, s in numeric_scores.items():
            print(f"  {m}: {s:.4f}")

        all_results[pipeline_type] = {
            "metrics": numeric_scores,
            "num_samples": len(data_list)
        }

    output_dir = Path(eval_config.results_path)
    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(
        f"\n[DONE] Итоговый отчёт сохранен в {output_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    run_evaluation()
