from fastapi import APIRouter, Depends
from src.core.models.chat import ChatRequest, ChatResponse
from src.api.dependencies import get_rag_pipeline
from src.services.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    response = pipeline.query(request.message, request.history)
    return response
