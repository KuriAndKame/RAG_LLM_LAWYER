from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from src.services.document_loader.loader import DocumentLoader
from src.services.chunker.splitter import Chunker
from src.services.embeddings.client import get_embedding_model
from src.api.dependencies import get_vector_store
from src.services.vector_store.faiss_store import FAISSVectorStore
import os
import tempfile
from config.settings import settings

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    vector_store: FAISSVectorStore = Depends(get_vector_store)
):
    # Сохраняем файл временно
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Загрузка документа
        loader = DocumentLoader()
        document, raw_text = loader.load(tmp_path)

        # Чанкование
        chunker = Chunker()
        document = chunker.split_document(document, raw_text)

        # Эмбеддинги
        embed_model = get_embedding_model()
        texts = [chunk.text for chunk in document.chunks]
        embeddings = embed_model.encode(texts)

        # Добавление в векторное хранилище
        vector_store.add_embeddings(embeddings, document.chunks)
        vector_store.save()

        return {"message": f"Document {file.filename} indexed successfully", "chunks": len(document.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
