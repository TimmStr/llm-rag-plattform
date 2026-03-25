from apps.core.config import get_settings
from apps.ingestion_service.provider.embedding_provider import EmbeddingService
from apps.ingestion_service.services.indexing_service import docs_to_vectorstore
from apps.ingestion_service.services.embedding_service import embed_chunks
from apps.ingestion_service.services.ingest_service import ingest_pdf
from apps.llm_service.generator import Generator
from apps.llm_service.vllm_service import VLLMLLM
from apps.retrieval_service.retriever import Retriever
from apps.retrieval_service.vector_store.qdrant_store import QdrantVectorStore

if __name__ == "__main__":
    settings = get_settings()

    embedding_service = EmbeddingService(settings.embedding_model, settings.embedding_device)
    vector_store = QdrantVectorStore(settings.qdrant_collection_name)

    # Ingest
    chunks = ingest_pdf(settings.demo_document_path)
    chunks = embed_chunks(chunks, embedding_service)

    # Index
    docs_to_vectorstore(chunks, vector_store)

    # Retriever
    retriever = Retriever(
        documents=chunks,
        vector_store=vector_store,
        embedding_service=embedding_service
    )

    # Query
    query = "European Commission State Aid Decision?"
    results = retriever.retrieve(
        query,
        top_k=5
    )

    for r in results:
        print(r["text"][:200])
        print("Score:", r["score"])
        print("-" * 30)

    vllm = VLLMLLM()
    llm_generator = Generator(vllm)
    print(llm_generator.generate(query, [r["text"] for r in results]))
