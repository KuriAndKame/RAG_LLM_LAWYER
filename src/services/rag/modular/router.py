# src/services/rag/modular/router.py

class Router:
    """Принимает решение об источниках поиска и необходимости рерайта."""

    LEGAL_KEYWORDS = [
        "статья", "ук рф", "гк рф", "коап", "закон", "право", "суд",
        "уголовн", "административ", "гражданск", "процессуальн", "кодекс",
        "наказание", "ответственность", "штраф", "лишение свободы",
        "взрыв", "заминирова", "террор", "хулиган", "вандал", "поджог",
        "кража", "грабёж", "разбой", "убийств", "телесные", "ущерб",
        "заведомо ложн", "сообщение", "акт", "теракт", "туалет"
    ]

    def decide(self, query: str):
        query_lower = query.lower()
        has_legal_terms = any(
            word in query_lower for word in self.LEGAL_KEYWORDS)

        if has_legal_terms:
            # Явно юридический запрос: ищем в базе, рерайт не обязателен
            return {"use_vector_db": True, "use_web": False, "rewrite": False}
        else:
            # Неочевидный запрос: ищем в интернете с предварительным рерайтом
            return {"use_vector_db": False, "use_web": True, "rewrite": True}
