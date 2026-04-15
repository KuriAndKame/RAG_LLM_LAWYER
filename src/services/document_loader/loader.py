from typing import List
import os
from src.services.document_loader.pdf_loader import PDFLoader
from src.services.document_loader.docx_loader import DocxLoader
from src.services.document_loader.text_loader import TextLoader
from src.core.models.document import Document


class DocumentLoader:
    def __init__(self):
        self.loaders = {
            '.pdf': PDFLoader(),
            '.docx': DocxLoader(),
            '.txt': TextLoader(),
            '.odt': TextLoader(),  # odfpy будет обрабатывать аналогично
        }

    def load(self, file_path: str) -> Document:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file extension: {ext}")
        loader = self.loaders[ext]
        return loader.load(file_path)
