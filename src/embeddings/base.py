import numpy as np
import torch
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
from sklearn.decomposition import PCA


class BaseEmbeddingExtractor(ABC):
    """Abstract base class for embedding extractors."""
    
    def __init__(self, target_dim: int = 768):
        self.target_dim = target_dim
        self.pca = None
        self.native_dim = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def _load_model(self):
        """Load the embedding model."""
        pass
    
    @abstractmethod
    def _extract_raw(self, texts: List[str]) -> np.ndarray:
        """Extract raw embeddings without dimension standardization."""
        pass
    
    def encode(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Encode text(s) into embeddings with dimension standardization.
        
        Args:
            texts: Single text or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            embeddings: numpy array of shape (n, target_dim) or (target_dim,)
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        raw_embs = self._extract_raw(texts)
        
        standardized = self._standardize_dim(raw_embs)
        
        if is_single:
            return standardized[0]
        return standardized
    
    def _standardize_dim(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to target_dim using PCA or padding."""
        current_dim = embeddings.shape[-1]
        
        if current_dim == self.target_dim:
            return embeddings
        
        elif current_dim > self.target_dim:
            if self.pca is None:
                raise ValueError(
                    f"PCA not fitted! Native dim ({current_dim}) > target dim ({self.target_dim}). "
                    f"Call fit_pca() on representative sample data first."
                )
            
            return self.pca.transform(embeddings)
        
        else:
            padding = np.zeros((embeddings.shape[0], self.target_dim - current_dim))
            return np.concatenate([embeddings, padding], axis=1)
    
    def fit_pca(self, texts: List[str], n_samples: int = 1000):
        """
        Fit PCA on a sample of embeddings from the dataset.
        Call this once before encoding the full dataset.
        """
        import random
        
        if len(texts) > n_samples:
            texts = random.sample(texts, n_samples)
        
        print(f"Fitting PCA on {len(texts)} samples...")
        raw_embs = self._extract_raw(texts)
        
        current_dim = raw_embs.shape[-1]
        self.native_dim = current_dim
        
        if current_dim > self.target_dim:
            self.pca = PCA(n_components=self.target_dim)
            self.pca.fit(raw_embs)
            print(f"PCA fitted: {current_dim} → {self.target_dim} dims")
        else:
            print(f"No PCA needed: {current_dim} ≤ {self.target_dim}")
    
    def save_pca(self, path: str):
        """Save fitted PCA transformer."""
        if self.pca is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.pca, f)
    
    def load_pca(self, path: str):
        """Load fitted PCA transformer."""
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)


class EmbeddingCache:
    """Cache embeddings to disk to avoid recomputation."""
    
    def __init__(self, cache_dir: str = 'data/embeddings'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_path(self, model_name: str, dataset_name: str) -> Path:
        return self.cache_dir / f"{model_name}_{dataset_name}.pkl"
    
    def load(self, model_name: str, dataset_name: str) -> Optional[dict]:
        cache_path = self.get_cache_path(model_name, dataset_name)
        if cache_path.exists():
            print(f"Loading cached embeddings: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save(self, embeddings: dict, model_name: str, dataset_name: str):
        cache_path = self.get_cache_path(model_name, dataset_name)
        print(f"Saving embeddings to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def exists(self, model_name: str, dataset_name: str) -> bool:
        return self.get_cache_path(model_name, dataset_name).exists()
