import re
import json
import fitz  # PyMuPDF
import os


def extract_text_from_pdf(pdf_path):
    """Извлекает сырой текст из PDF файла"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text


def clean_and_parse_law(pdf_path, output_json_path, doc_name):
    print(f"Обработка документа: {doc_name}...")

    # 1. Извлекаем текст
    text = extract_text_from_pdf(pdf_path)

    # 2. ОЧИСТКА ШУМА (Специфично для Гаранта и PDF)
    noise_patterns = [
        r"--- PAGE \d+ ---",                                    # Номера страниц
        r"Дата актуализации:.*?\n",                             # Даты
        r"Актуальную версию смотрите на сайте.*?\n",            # Ссылки
        r"WWW\.GARANT\.RU.*?\n",
        # Копирайты (с учетом кривых пробелов)
        r"© ООО \"НПП \"ГАР\s*АНТ-СЕРВИС-УНИВЕРСИТЕТ\".*?\n",
        r"© ООО \"НПП \"ГАРАНТ-СЕРВИС-УНИВЕРСИТЕТ\".*?\n",
        r"Система ГАР\s*АНТ выпускается с 1990г\..*?\n",
        r"Система ГАРАНТ выпускается с 1990г\..*?\n",
        r"по состоянию на.*?\n"
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Убираем множественные пустые строки, оставляем максимум двойные
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 3. СЕМАНТИЧЕСКОЕ ЧАНКОВАНИЕ
    # Разбиваем текст ровно перед словом "Статья" (учитываем статьи с точками, например 159.1)
    chunks = re.split(r'(?=\n\s*Статья\s+\d+\.?\d*)', text)

    parsed_articles = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("Статья"):
            continue  # Пропускаем оглавления, разделы и преамбулы

        # Вытаскиваем номер статьи
        match = re.search(r'Статья\s+(\d+\.?\d*)', chunk)
        article_num = match.group(1) if match else "Неизвестно"

        # Формируем объект чанка
        article_data = {
            "metadata": {
                "document_name": doc_name,
                "article": f"Статья {article_num}"
            },
            "text": chunk
        }
        parsed_articles.append(article_data)

    # 4. Сохранение результата
    # Убедимся, что папка существует
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_articles, f, ensure_ascii=False, indent=4)

    print(
        f"✅ Успешно! Документ '{doc_name}' разбит на {len(parsed_articles)} статей.")
    print(f"📁 Сохранено в: {output_json_path}\n")


if __name__ == "__main__":
    # Прописываем пути к нашим файлам
    # Используем относительные пути от корня проекта
    uk_pdf = "data/raw/garant_ugolovny_kodeks_rf.pdf"
    tk_pdf = "data/raw/Trudovoj-kodeks-Rossijskoj-Federatsii.pdf"

    uk_json = "data/processed/uk_rf_clean.json"
    tk_json = "data/processed/tk_rf_clean.json"

    # Запускаем парсинг
    if os.path.exists(uk_pdf):
        clean_and_parse_law(uk_pdf, uk_json, "Уголовный кодекс РФ")
    else:
        print(f"Файл {uk_pdf} не найден!")

    if os.path.exists(tk_pdf):
        clean_and_parse_law(tk_pdf, tk_json, "Трудовой кодекс РФ")
    else:
        print(f"Файл {tk_pdf} не найден!")
