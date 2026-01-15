import os
import pickle
import numpy as np
from typing import List, Optional
from pathlib import Path


class EmbeddingCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.pkl"
    
    def exists(self, name: str) -> bool:
        return self.get_cache_path(name).exists()
    
    def save(self, name: str, embeddings: np.ndarray, metadata: Optional[dict] = None):
        data = {"embeddings": embeddings, "metadata": metadata}
        with open(self.get_cache_path(name), "wb") as f:
            pickle.dump(data, f)
    
    def load(self, name: str) -> tuple:
        with open(self.get_cache_path(name), "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data.get("metadata")
    
    def list_cached(self) -> List[str]:
        return [p.stem for p in self.cache_dir.glob("*.pkl")]
