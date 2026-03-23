from fastapi import APIRouter, Depends

from apps.api.dependencies.auth import get_current_user
from apps.api.dependencies.services import get_generator, get_retriever
from apps.api.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from apps.core.logging_config import get_logger

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

logger = get_logger(__name__)
@router.post("/", response_model=ChatResponse)
def chat(
        request: ChatRequest,
        user: dict = Depends(get_current_user)
):
    logger.info(f"Chat request: {request}")
    retriever = get_retriever()
    generator = get_generator()

    results = retriever.retrieve(request.query,
                                 top_k=request.top_k)

    answer = generator.generate(query=request.query,
                                contexts=[result["text"] for result in results])

    return ChatResponse(
        answer=answer,
        sources=[SourceChunk(**result) for result in results]
    )
