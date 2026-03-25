from functools import lru_cache

from apps.core.config import get_settings
from apps.core.llm.base import BaseLLM
from apps.ingestion_service.provider.embedding_provider import EmbeddingService
from apps.ingestion_service.services.embedding_service import embed_chunks
from apps.ingestion_service.services.ingest_service import ingest_pdf
from apps.llm_service.generator import Generator
from apps.llm_service.openai_service import OpenAILLM
from apps.llm_service.vllm_service import VLLMLLM
from apps.retrieval_service.reranker import Reranker
from apps.retrieval_service.retriever import Retriever
from apps.retrieval_service.vector_store.qdrant_store import QdrantVectorStore


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(model_name=settings.embedding_model,
                            device=settings.embedding_device)


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    return QdrantVectorStore(
        collection_name=settings.qdrant_collection_name,
        host=settings.qdrant_host,
        port=settings.qdrant_port)


def get_llm(provider: str = "vllm") -> BaseLLM:
    settings = get_settings()
    if provider == "openai":
        return OpenAILLM(model=settings.openai_model)
    return VLLMLLM(
        model=settings.vllm_model,
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key
    )


def get_generator(provider: str = "vllm") -> Generator:
    return Generator(llm=get_llm(provider))


def get_retriever() -> Retriever:
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    documents = ingest_pdf(settings.demo_document_path)
    documents = embed_chunks(documents, embedding_service)

    retriever = Retriever(
        documents=documents,
        vector_store=vector_store,
        embedding_service=embedding_service,
        reranker=Reranker(
            model_name=settings.reranker_model,
            device=settings.reranker_device
        )
    )
    return retriever
