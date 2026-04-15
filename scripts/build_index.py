import glob
from config.settings import settings
from src.services.vector_store.faiss_store import FAISSVectorStore
from src.services.embeddings.client import get_embedding_model
from src.services.chunker.splitter import Chunker
from src.services.document_loader.loader import DocumentLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_index(data_folder: str = None):
    data_folder = data_folder or settings.RAW_DATA_PATH
    files = glob.glob(os.path.join(data_folder, "*.*"))
    if not files:
        print("No files found.")
        return

    vector_store = FAISSVectorStore()
    embed_model = get_embedding_model()
    loader = DocumentLoader()
    chunker = Chunker()

   # scripts/build_index.py (фрагмент)
    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            document, raw_text = loader.load(file_path)
            print(f"Loaded text length: {len(raw_text)} chars")
            document = chunker.split_document(document, raw_text)
            texts = [chunk.text for chunk in document.chunks]
            embeddings = embed_model.encode(texts)
            vector_store.add_embeddings(embeddings, document.chunks)
            print(f"Indexed {len(document.chunks)} chunks from {file_path}")
        except Exception as e:
            import traceback
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()

    vector_store.save()
    print(f"Index built with {vector_store.index.ntotal} vectors.")


if __name__ == "__main__":
    build_index()
