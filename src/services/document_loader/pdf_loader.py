import pypdf
from src.core.models.document import Document


class PDFLoader:
    def load(self, file_path: str) -> Document:
        text = ""
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        doc = Document(
            filename=file_path.split('/')[-1],
            source_path=file_path
        )
        # Пока чанки не добавляем, это будет позже
        return doc, text  # Возвращаем и документ, и сырой текст
