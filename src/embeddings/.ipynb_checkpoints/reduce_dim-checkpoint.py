"""
Dimensionality reduction utilities for embeddings.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def reduce_embeddings_pca(
        embeddings_dict: Dict[str, np.ndarray],
        n_components: int = 50,
        random_state: int = 42) -> Tuple[Dict[str, np.ndarray], PCA]:
    """
    Apply PCA to reduce embedding dimensionality.

    Args:
        embeddings_dict: Dictionary mapping ids to embeddings
        n_components: Target dimensionality
        random_state: Random seed for reproducibility

    Returns:
        reduced_embeddings: Dictionary with reduced embeddings
        pca_model: Fitted PCA model
    """
    ids = list(embeddings_dict.keys())
    X = np.array([embeddings_dict[id_] for id_ in ids])

    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    print(f"PCA: {X.shape[1]} → {n_components} dims")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    reduced_dict = {id_: X_reduced[i] for i, id_ in enumerate(ids)}
    return reduced_dict, pca


def reduce_embeddings_umap(
        embeddings_dict: Dict[str, np.ndarray],
        n_components: int = 10,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
) -> Tuple[Dict[str, np.ndarray], Optional[object]]:
    """
    Apply UMAP to reduce embedding dimensionality.

    Args:
        embeddings_dict: Dictionary mapping ids to embeddings
        n_components: Target dimensionality
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed

    Returns:
        reduced_embeddings: Dictionary with reduced embeddings
        umap_model: Fitted UMAP model (or None if unavailable)
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP not installed. Install with: pip install umap-learn")

    ids = list(embeddings_dict.keys())
    X = np.array([embeddings_dict[id_] for id_ in ids])

    reducer = umap.UMAP(n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        random_state=random_state)
    X_reduced = reducer.fit_transform(X)

    print(f"UMAP: {X.shape[1]} → {n_components} dims")

    reduced_dict = {id_: X_reduced[i] for i, id_ in enumerate(ids)}
    return reduced_dict, reducer


def reduce_embeddings(embeddings_dict: Dict[str, np.ndarray],
                      method: str = 'pca',
                      n_components: Optional[int] = None,
                      **kwargs) -> Tuple[Dict[str, np.ndarray], object]:
    """
    Unified interface for dimensionality reduction.

    Args:
        embeddings_dict: Dictionary mapping ids to embeddings
        method: 'pca' or 'umap'
        n_components: Target dimensions (default: 50 for PCA, 10 for UMAP)
        **kwargs: Additional arguments for specific method

    Returns:
        reduced_embeddings: Dictionary with reduced embeddings
        model: Fitted reduction model
    """
    if method == 'pca':
        n_components = n_components or 50
        return reduce_embeddings_pca(embeddings_dict, n_components, **kwargs)
    elif method == 'umap':
        n_components = n_components or 10
        return reduce_embeddings_umap(embeddings_dict, n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'")
