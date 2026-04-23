from typing import List
from sentence_transformers import CrossEncoder
from src.core.models.chat import SourceInfo
from config.settings import settings


class Reranker:
    def __init__(self, model_name: str = None):
        model_name = model_name or settings.RERANKER_MODEL_NAME
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, sources: List[SourceInfo], top_k: int = None) -> List[SourceInfo]:
        if not sources:
            return []
        top_k = top_k or settings.FINAL_TOP_K
        pairs = [[query, source.chunk_text] for source in sources]
        scores = self.model.predict(pairs)
        scored_sources = sorted(zip(sources, scores),
                                key=lambda x: x[1], reverse=True)
        final_sources = []
        for source, score in scored_sources[:top_k]:
            source.relevance_score = float(score)
            final_sources.append(source)
        return final_sources
