# Production-Grade Rag Platform

## Features
- Scalable RAG System (kubernetes)
- LLM Observability (Prometheus + Grafana)
- Evaluation pipeline (RAG metrics)
- CI/CD Deployment

## Architecture
![architecutre-diagramm.png](architecture-diagram.png)

## Tech Stack
- FastAPI
- Qdrant
- MLflow
- Kubernetes
- Prometheus


## Docker compose
### docker compose -f infra/docker/docker-compose.yml up --build