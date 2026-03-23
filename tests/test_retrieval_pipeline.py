from os import path, getcwd

from apps.ingestion_service.embedding import EmbeddingService
from apps.ingestion_service.ingest import ingest_pdf, embed_chunks, index_chunks
from apps.retrieval_service.retriever import Retriever
from apps.retrieval_service.vector_store.qdrant_store import QdrantVectorStore

if __name__ == "__main__":
    file_path = path.join(getcwd(), "data", "raw", "_10-K-2025-As-Filed.pdf")

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore(collection_name="docs")

    # Ingest
    chunks = ingest_pdf(file_path)
    chunks = embed_chunks(chunks, embedding_service)

    # Index
    index_chunks(chunks, vector_store)

    # Retriever
    retriever = Retriever(
        documents=chunks,
        vector_store=vector_store,
        embedding_service=embedding_service
    )

    # Query
    results = retriever.retrieve(
        "European Commission State Aid Decision?",
        top_k=5
    )

    for r in results:
        print(r["text"][:200])
        print("Score:", r["score"])
        print("-" * 30)
