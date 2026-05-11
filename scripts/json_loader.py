import json
from typing import List
from src.core.models.document import DocumentChunk


def load_json_to_chunks(json_path: str) -> List[DocumentChunk]:
    """Читает распарщенный JSON и преобразует его в список DocumentChunk"""
    chunks = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            chunk = DocumentChunk(
                text=item["text"],
                metadata=item["metadata"]
            )
            chunks.append(chunk)

        print(f"Успешно загружено {len(chunks)} чанков из {json_path}")
        return chunks
    except FileNotFoundError:
        print(f"Ошибка: Файл {json_path} не найден.")
        return []
    except Exception as e:
        print(f"Ошибка при чтении JSON: {e}")
        return []
