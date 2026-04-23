from fastapi import APIRouter, Depends, Query
from src.core.models.chat import ChatRequest, ChatResponse
from src.api.dependencies import get_current_pipeline, get_naive_pipeline, get_advanced_pipeline, get_modular_pipeline
from src.services.rag.pipeline import RAGPipeline
from src.services.rag.advanced_pipeline import AdvancedRAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline_type: str = Query(
        None, description="'naive', 'advanced' или 'modular'")
):
    if pipeline_type == "naive":
        pipeline = get_naive_pipeline()
    elif pipeline_type == "advanced":
        pipeline = get_advanced_pipeline()
    elif pipeline_type == "modular":
        pipeline = get_modular_pipeline()
    else:
        pipeline = get_current_pipeline()

    response = pipeline.query(
        request.message,
        request.history,
        use_vector_db=request.use_vector_db,
        use_web_search=request.use_web_search
    )
    return response
