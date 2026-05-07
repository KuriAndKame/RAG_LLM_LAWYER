from src.services.embeddings.local_embedding import LocalEmbedding

# Глобальная переменная для хранения загруженной модели
_embedding_model_instance = None


def get_embedding_model():
    global _embedding_model_instance

    # Если модель еще не загружена в память — загружаем
    if _embedding_model_instance is None:
        print("[INFO] Инициализация Embedding-модели (загрузка в RAM)...")
        _embedding_model_instance = LocalEmbedding()

    # Возвращаем уже загруженную модель
    return _embedding_model_instance
