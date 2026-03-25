import os
from apps.ingestion_service.domain.entities import dataclass


@dataclass
class Settings:
    api_title: str = os.getenv("API_TITLE", "LLM RAG Platform")
    api_version: str = os.getenv("API_VERSION", "0.1.0")

    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    demo_username: str = os.getenv("DEMO_USERNAME", "admin")
    demo_password: str = os.getenv("DEMO_PASSWORD", "admin")

    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "docs")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    llm_provider: str = os.getenv("LLM_PROVIDER", "vllm")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local-dev-key")
    vllm_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen3-0.6B")

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_token: str = os.getenv("OPENAI_TOKEN", "")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", 384))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
    embedding_normalize: bool = os.getenv("EMBEDDING_NORMALIZE", True)
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    reranker_device: str = os.getenv("RERANKER_DEVICE", "cpu")

    rag_default_top_k: int = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
    rag_candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "20"))

    demo_document_path: str = os.getenv(
        "DEMO_DOCUMENT_PATH",
        os.path.join(os.getcwd(), "data", "raw", "_10-K-2025-As-Filed.pdf"),
    )


def get_settings() -> Settings:
    return Settings()
