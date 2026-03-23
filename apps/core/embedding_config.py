from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"
