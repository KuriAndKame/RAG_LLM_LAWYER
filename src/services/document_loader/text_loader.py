from src.core.models.document import Document
# import odfpy  # если установлен
from odf import text, teletype
from odf.opendocument import load


class TextLoader:
    def load(self, file_path: str) -> tuple[Document, str]:
        if file_path.endswith('.odt'):
            # Используем odfpy для извлечения текста
            doc = load(file_path)
            all_paras = doc.getElementsByType(text.P)
            content = "\n".join(teletype.extractText(p) for p in all_paras)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        document = Document(
            filename=file_path.split('/')[-1],
            source_path=file_path
        )
        return document, content
