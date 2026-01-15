import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List


def uniform_sample(n_items: int, k: int = 500, replace: bool = False) -> np.ndarray:
    return np.random.choice(n_items, size=min(k, n_items), replace=replace)


def coverage_metric_rho(embeddings: np.ndarray, k: int, num_trials: int = 100) -> Tuple[float, float]:
    n = len(embeddings)
    if k >= n:
        return 0.0, 0.0
    
    distances = []
    
    for _ in range(num_trials):
        sample_indices = np.random.choice(n, size=k, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        remaining_indices = np.setdiff1d(np.arange(n), sample_indices)
        if len(remaining_indices) == 0:
            distances.append(0.0)
            continue
        
        remaining_embeddings = embeddings[remaining_indices]
        
        dist_matrix = cdist(remaining_embeddings, sample_embeddings, metric='cosine')
        min_distances = dist_matrix.min(axis=1)
        distances.append(min_distances.mean())
    
    return float(np.mean(distances)), float(np.std(distances))


def coverage_metric_rho_sampled(full_embeddings: np.ndarray, 
                                 sampled_indices: np.ndarray) -> float:
    if len(sampled_indices) == 0 or len(sampled_indices) >= len(full_embeddings):
        return 0.0
    
    sample_embeddings = full_embeddings[sampled_indices]
    
    remaining_indices = np.setdiff1d(np.arange(len(full_embeddings)), sampled_indices)
    if len(remaining_indices) == 0:
        return 0.0
    
    remaining_embeddings = full_embeddings[remaining_indices]
    
    dist_matrix = cdist(remaining_embeddings, sample_embeddings, metric='cosine')
    min_distances = dist_matrix.min(axis=1)
    
    return float(min_distances.mean())


def batch_coverage_analysis(embeddings: np.ndarray, 
                            k_values: List[int], 
                            num_trials: int = 50) -> dict:
    results = {}
    
    for k in k_values:
        rho_mean, rho_std = coverage_metric_rho(embeddings, k, num_trials)
        results[k] = {"mean": rho_mean, "std": rho_std}
    
    return results


def compute_embedding_uniformity(embeddings: np.ndarray, sample_size: int = 1000) -> float:
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -np.inf)
    
    uniformity = np.log(np.exp(2 * sim_matrix).mean())
    return float(uniformity)


def compute_embedding_alignment(embeddings: np.ndarray, positive_pairs: np.ndarray) -> float:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    
    alignments = []
    for i, j in positive_pairs:
        sim = (embeddings[i] * embeddings[j]).sum()
        alignments.append(1 - sim)
    
    return float(np.mean(alignments)) if alignments else 0.0


def compute_isotropy(embeddings: np.ndarray) -> float:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    
    centered = embeddings - embeddings.mean(axis=0)
    
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    
    normalized_sv = singular_values / singular_values.sum()
    isotropy = -np.sum(normalized_sv * np.log(normalized_sv + 1e-10))
    
    max_entropy = np.log(len(singular_values))
    
    return float(isotropy / max_entropy) if max_entropy > 0 else 0.0
