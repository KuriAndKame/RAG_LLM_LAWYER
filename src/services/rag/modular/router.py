# src/services/rag/modular/router.py

class Router:
    """Принимает решение об источниках поиска и необходимости рерайта."""

    # Слова, указывающие, что запрос УЖЕ сформулирован юридически грамотно
    FORMAL_KEYWORDS = [
        "статья", "ук рф", "гк рф", "коап", "тк рф", "федеральный закон",
        "пункт", "часть", "кодекс"
    ]

    def decide(self, query: str):
        query_lower = query.lower()

        # Проверяем, есть ли формальные юридические ссылки
        is_formal = any(word in query_lower for word in self.FORMAL_KEYWORDS)

        # Если запрос слишком короткий (меньше 4 слов), ему точно нужен рерайт для обогащения контекста
        is_short = len(query.split()) < 4

        if is_formal and not is_short:
            # Запрос уже грамотный и подробный: ищем в базе, рерайт не нужен
            return {"use_vector_db": True, "use_web": False, "rewrite": False}
        else:
            # Запрос обывательский или короткий: делаем рерайт, чтобы вытащить юр. термины
            # Ищем сначала в БД, веб оставляем как запасной вариант
            return {"use_vector_db": True, "use_web": True, "rewrite": True}
