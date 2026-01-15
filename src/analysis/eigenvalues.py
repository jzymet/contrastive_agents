import numpy as np
from scipy.linalg import eigh
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


def compute_eigenvalue_spectrum(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of embedding covariance matrix.
    
    Args:
        embeddings: (N, d) array where N = number of items, d = embedding dim
        
    Returns:
        eigenvalues: (d,) sorted in descending order
        eigenvectors: (d, d)
    """
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    
    N = embeddings.shape[0]
    Sigma = (embeddings_centered.T @ embeddings_centered) / N
    
    eigenvalues, eigenvectors = eigh(Sigma)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def compute_effective_dimension(eigenvalues: np.ndarray) -> float:
    """
    Compute effective dimension (participation ratio).
    
    d_eff = (sum λ_i)² / (sum λ_i²)
    
    Measures "how many dimensions are effectively used"
    """
    eigenvalues = np.maximum(eigenvalues, 0)
    return float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())


def compute_effective_dimension_threshold(eigenvalues: np.ndarray, threshold: float = 0.99) -> int:
    """
    Compute number of dimensions needed to explain threshold of variance.
    """
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var
    return int(np.argmax(cumulative_var >= threshold) + 1)


def analyze_all_embeddings(embedding_dict: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Compute eigenvalue analysis for multiple embedding types.
    
    Args:
        embedding_dict: Dict mapping model_name -> (N, d) embeddings
        
    Returns:
        Dict with eigenvalues, d_eff, and spectrum for each model
    """
    results = {}
    
    for model_name, embeddings in embedding_dict.items():
        eigenvalues, eigenvectors = compute_eigenvalue_spectrum(embeddings)
        d_eff = compute_effective_dimension(eigenvalues)
        d_99 = compute_effective_dimension_threshold(eigenvalues, 0.99)
        
        results[model_name] = {
            'eigenvalues': eigenvalues,
            'd_eff': d_eff,
            'd_99': d_99,
            'top_10_eigenvalues': eigenvalues[:10].tolist(),
            'eigenvalue_sum': float(eigenvalues.sum()),
            'max_eigenvalue': float(eigenvalues.max()),
        }
        
        print(f"{model_name:12s}: d_eff = {d_eff:.1f}, d_99 = {d_99}")
    
    return results


def plot_eigenvalue_spectra(results: Dict[str, dict], save_path: str = None):
    """
    Plot eigenvalue spectra for all models.
    Shows anisotropic vs contrastive separation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    anisotropic_models = ['bert', 'roberta', 'llama3']
    contrastive_models = ['simcse', 'jina', 'llm2vec']
    
    colors = {
        'bert': '#e41a1c',
        'roberta': '#377eb8',
        'llama3': '#4daf4a',
        'simcse': '#984ea3',
        'jina': '#ff7f00',
        'llm2vec': '#a65628'
    }
    
    ax1 = axes[0]
    for model_name in results:
        if model_name in anisotropic_models or model_name in contrastive_models:
            eigs = results[model_name]['eigenvalues']
            style = '--' if model_name in anisotropic_models else '-'
            color = colors.get(model_name, 'gray')
            ax1.plot(np.log(eigs + 1e-10), label=model_name, 
                    linestyle=style, linewidth=2, color=color)
    
    ax1.set_xlabel('Dimension Index')
    ax1.set_ylabel('Log Eigenvalue')
    ax1.set_title('Eigenvalue Spectra: Anisotropic (dashed) vs Contrastive (solid)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 300)
    
    ax2 = axes[1]
    model_names = list(results.keys())
    d_effs = [results[m]['d_eff'] for m in model_names]
    bar_colors = [colors.get(m, 'gray') for m in model_names]
    
    bars = ax2.bar(model_names, d_effs, color=bar_colors)
    ax2.set_ylabel('Effective Dimension (d_eff)')
    ax2.set_title('Effective Dimension Comparison')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Theory threshold')
    
    for bar, val in zip(bars, d_effs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved eigenvalue spectra to {save_path}")
    
    return fig


def generate_demo_eigenvalue_data() -> Dict[str, dict]:
    """Generate demo data for visualization when models aren't available."""
    np.random.seed(42)
    
    results = {}
    
    for model_name in ['bert', 'roberta', 'llama3']:
        eigenvalues = np.exp(-np.arange(768) / 15) + 0.01
        eigenvalues = eigenvalues * (np.random.rand(768) * 0.2 + 0.9)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        results[model_name] = {
            'eigenvalues': eigenvalues,
            'd_eff': compute_effective_dimension(eigenvalues),
            'd_99': compute_effective_dimension_threshold(eigenvalues),
            'top_10_eigenvalues': eigenvalues[:10].tolist(),
        }
    
    for model_name in ['simcse', 'jina', 'llm2vec']:
        eigenvalues = np.exp(-np.arange(768) / 80) + 0.01
        eigenvalues = eigenvalues * (np.random.rand(768) * 0.1 + 0.95)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        results[model_name] = {
            'eigenvalues': eigenvalues,
            'd_eff': compute_effective_dimension(eigenvalues),
            'd_99': compute_effective_dimension_threshold(eigenvalues),
            'top_10_eigenvalues': eigenvalues[:10].tolist(),
        }
    
    return results
