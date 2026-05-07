# evaluation/report.py
"""
Генератор читаемого отчёта сравнения RAG-пайплайнов.

Принимает на вход JSON-файл, созданный evaluator.py,
и выводит таблицу с метриками для каждого пайплайна.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from evaluation.config import config as eval_config
import sys
from pathlib import Path

# Добавляем корень проекта (где лежит папка src, evaluation, .env и т.д.) в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_report(
    json_path: Optional[str] = None,
    output_format: str = "markdown"
) -> str:
    """
    Читает evaluation_report.json и возвращает строку с таблицей сравнения.

    Аргументы:
        json_path: путь к JSON-отчёту (по умолчанию evaluation/results/evaluation_report.json)
        output_format: "markdown" (для .md) или "html" (простая таблица)

    Возвращает:
        строку с отчётом
    """
    if json_path is None:
        json_path = Path(eval_config.results_path) / "evaluation_report.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Собираем все имена метрик (из первого доступного пайплайна)
    metric_names = set()
    for pipeline, info in data.items():
        if "metrics" in info:
            metric_names.update(info["metrics"].keys())

    if not metric_names:
        return "Нет данных для отчёта."

    # Заголовки таблицы
    header = ["Pipeline"] + sorted(metric_names) + ["Samples"]

    # Строки данных
    rows = []
    for pipeline in ["naive", "advanced", "modular"]:
        info = data.get(pipeline, {})
        metrics = info.get("metrics", {})
        samples = info.get("num_samples", 0)
        row = [pipeline.capitalize()]
        for metric in sorted(metric_names):
            val = metrics.get(metric)
            if val is not None:
                row.append(f"{val:.4f}")
            else:
                row.append("N/A")
        row.append(str(samples))
        rows.append(row)

    # Формируем вывод
    if output_format == "markdown":
        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    elif output_format == "html":
        html = "<table border='1'><thead><tr>"
        for h in header:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"
        for row in rows:
            html += "<tr>" + \
                "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        html += "</tbody></table>"
        return html
    else:
        raise ValueError("output_format должен быть 'markdown' или 'html'")


if __name__ == "__main__":
    report = generate_report()
    print(report)
    # Сохраняем в файл
    out_path = Path(eval_config.results_path) / "comparison_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nОтчёт сохранён в {out_path}")
