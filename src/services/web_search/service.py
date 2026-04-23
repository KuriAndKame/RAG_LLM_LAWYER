from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from config.settings import settings


class WebSearchService:
    """Сервис для выполнения поисковых запросов в интернете."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.WEB_SEARCH_PROVIDER
        self.client = None
        self._init_client()

    def _init_client(self):
        pass

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Синхронный поиск в интернете."""
        if self.provider == 'tavily':
            return self._search_tavily(query, max_results)
        elif self.provider == 'duckduckgo':
            return self._search_duckduckgo(query, max_results)
        else:
            raise ValueError(f"Неизвестный провайдер поиска: {self.provider}")

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
