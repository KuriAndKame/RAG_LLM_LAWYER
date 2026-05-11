from src.core.models.document import DocumentChunk
from src.services.embeddings.client import get_embedding_model
from src.services.vector_store.faiss_store import FAISSVectorStore
from config.settings import settings
import os
import sys
import glob
import json

# Добавляем корень проекта в пути, чтобы работали импорты
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_index_from_json(data_folder: str = "data/processed"):
    """
    Строит FAISS индекс из идеально распарщенных JSON-файлов (статей).
    """
    files = glob.glob(os.path.join(data_folder, "*.json"))
    if not files:
        print(f"JSON файлы не найдены в папке {data_folder}.")
        return

    vector_store = FAISSVectorStore()
    embed_model = get_embedding_model()

    total_chunks_added = 0

    for file_path in files:
        print(f"Обработка файла {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Генерируем ID документа из названия файла (например, uk_rf_clean)
            doc_id = os.path.splitext(os.path.basename(file_path))[0]

            chunks = []
            # Используем enumerate, чтобы получить порядковый номер (idx)
            for idx, item in enumerate(data):
                chunk = DocumentChunk(
                    document_id=doc_id,      # <-- ИСПРАВЛЕНО: передаем ID документа
                    chunk_index=idx,         # <-- ИСПРАВЛЕНО: передаем порядковый номер
                    text=item["text"],
                    metadata=item["metadata"]
                )
                chunks.append(chunk)

            print(
                f"Сформировано {len(chunks)} чанков. Генерация эмбеддингов...")

            texts = [chunk.text for chunk in chunks]
            embeddings = embed_model.encode(texts)

            vector_store.add_embeddings(embeddings, chunks)
            print(
                f"✅ Успешно проиндексировано {len(chunks)} статей из {file_path}")
            total_chunks_added += len(chunks)

        except Exception as e:
            import traceback
            print(f"❌ Ошибка при обработке {file_path}: {e}")
            traceback.print_exc()

    # Сохраняем базу только если мы успешно добавили векторы
    if total_chunks_added > 0:
        vector_store.save()
        print(
            f"\n🎉 Индекс успешно построен! Всего векторов в базе: {vector_store.index.ntotal}")
    else:
        print("\n⚠️ Векторная база не была сохранена, так как не удалось обработать чанки.")


if __name__ == "__main__":
    build_index_from_json()
