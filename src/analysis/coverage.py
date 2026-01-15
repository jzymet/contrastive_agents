import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def compute_coverage_metric(
    embeddings: np.ndarray, 
    k_values: List[int] = [10, 50, 100, 500, 1000], 
    n_trials: int = 100
) -> Dict[int, float]:
    """
    Compute coverage metric ρ(k,K).
    
    ρ(k,K) = Expected minimum distance from unsampled items to k-sample.
    Lower ρ = better coverage = uniformity.
    
    Args:
        embeddings: (K, d) All item embeddings
        k_values: List of sample sizes to test
        n_trials: Number of Monte Carlo trials per k
        
    Returns:
        rho_curves: Dict mapping k -> ρ(k,K)
    """
    K = len(embeddings)
    rho_curves = {}
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-10)
    
    for k in k_values:
        if k >= K:
            rho_curves[k] = 0.0
            continue
        
        distances_all_trials = []
        
        for trial in range(n_trials):
            sample_indices = np.random.choice(K, k, replace=False)
            sample_embs = embeddings_normalized[sample_indices]
            
            remaining_indices = np.setdiff1d(np.arange(K), sample_indices)
            remaining_embs = embeddings_normalized[remaining_indices]
            
            sample_size = min(500, len(remaining_indices))
            if sample_size < len(remaining_indices):
                sample_remaining = np.random.choice(len(remaining_indices), sample_size, replace=False)
                remaining_embs = remaining_embs[sample_remaining]
            
            dist_matrix = cdist(remaining_embs, sample_embs, metric='cosine')
            min_distances = dist_matrix.min(axis=1)
            distances_all_trials.extend(min_distances.tolist())
        
        rho_curves[k] = float(np.mean(distances_all_trials))
    
    return rho_curves


def analyze_coverage_all_models(
    embedding_dict: Dict[str, np.ndarray],
    k_values: List[int] = [10, 50, 100, 500],
    n_trials: int = 50
) -> Dict[str, Dict[int, float]]:
    """
    Compute coverage metrics for all embedding types.
    """
    results = {}
    
    for model_name, embeddings in embedding_dict.items():
        print(f"Computing coverage for {model_name}...")
        rho = compute_coverage_metric(embeddings, k_values, n_trials)
        results[model_name] = rho
        
        print(f"  {model_name}: ρ(100) = {rho.get(100, 'N/A'):.4f}")
    
    return results


def plot_coverage_curves(
    coverage_results: Dict[str, Dict[int, float]],
    save_path: str = None
):
    """
    Plot coverage metric curves for all models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'bert': '#e41a1c',
        'roberta': '#377eb8',
        'llama3': '#4daf4a',
        'simcse': '#984ea3',
        'jina': '#ff7f00',
        'llm2vec': '#a65628'
    }
    
    anisotropic = ['bert', 'roberta', 'llama3']
    
    for model_name, rho_dict in coverage_results.items():
        k_values = sorted(rho_dict.keys())
        rho_vals = [rho_dict[k] for k in k_values]
        
        style = '--' if model_name in anisotropic else '-'
        color = colors.get(model_name, 'gray')
        marker = 'o' if model_name in anisotropic else 's'
        
        ax.plot(k_values, rho_vals, marker=marker, linestyle=style,
               label=model_name, linewidth=2, color=color, markersize=8)
    
    ax.set_xlabel('Sample Size (k)', fontsize=12)
    ax.set_ylabel('Coverage ρ(k,K)', fontsize=12)
    ax.set_title('Exploration Coverage: Lower is Better\n(Dashed = Anisotropic, Solid = Contrastive)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved coverage plot to {save_path}")
    
    return fig


def generate_demo_coverage_data() -> Dict[str, Dict[int, float]]:
    """Generate demo coverage data for visualization."""
    np.random.seed(42)
    
    k_values = [10, 50, 100, 500, 1000]
    results = {}
    
    for model_name in ['bert', 'roberta', 'llama3']:
        results[model_name] = {}
        for k in k_values:
            base_rho = 0.8 / np.sqrt(k)
            noise = np.random.rand() * 0.02
            results[model_name][k] = base_rho + noise + 0.05
    
    for model_name in ['simcse', 'jina', 'llm2vec']:
        results[model_name] = {}
        for k in k_values:
            base_rho = 0.5 / np.sqrt(k)
            noise = np.random.rand() * 0.01
            results[model_name][k] = base_rho + noise
    
    return results
