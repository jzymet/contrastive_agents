import numpy as np
import torch
from typing import Dict, Optional


def compute_rkhs_norm(reward_weights: np.ndarray, eigenvalues: np.ndarray) -> float:
    """
    Compute RKHS norm of reward function.
    
    ||R||² = Σ (w_i² / λ_i)
    
    Args:
        reward_weights: (d,) learned weights from reward network
        eigenvalues: (d,) from embedding covariance
        
    Returns:
        rkhs_norm: scalar
    """
    rkhs_norm_squared = np.sum(reward_weights**2 / (eigenvalues + 1e-10))
    return float(np.sqrt(rkhs_norm_squared))


def extract_reward_weights_from_bandit(trained_model) -> np.ndarray:
    """
    Extract final layer weights from neural bandit.
    """
    if hasattr(trained_model, 'network'):
        for layer in reversed(list(trained_model.network.children())):
            if hasattr(layer, 'weight'):
                weights = layer.weight.data.cpu().numpy()
                return weights.squeeze()
    
    if hasattr(trained_model, 'weights'):
        return trained_model.weights
    
    raise ValueError("Could not extract weights from model")


def extract_reward_weights_from_critic(trained_model) -> np.ndarray:
    """
    Extract final layer weights from transformer critic.
    """
    if hasattr(trained_model, 'value_head'):
        for layer in reversed(list(trained_model.value_head.children())):
            if hasattr(layer, 'weight'):
                weights = layer.weight.data.cpu().numpy()
                return weights.squeeze()
    
    if hasattr(trained_model, 'critic'):
        for layer in reversed(list(trained_model.critic.children())):
            if hasattr(layer, 'weight'):
                weights = layer.weight.data.cpu().numpy()
                return weights.squeeze()
    
    raise ValueError("Could not extract weights from model")


def analyze_rkhs_norms(
    trained_models: Dict[str, object],
    eigenvalue_results: Dict[str, dict],
    model_type: str = 'bandit'
) -> Dict[str, float]:
    """
    Compute RKHS norms for all trained models.
    
    Args:
        trained_models: Dict mapping model_name -> trained model
        eigenvalue_results: Dict from eigenvalue analysis
        model_type: 'bandit' or 'critic'
        
    Returns:
        Dict mapping model_name -> RKHS norm
    """
    results = {}
    
    for model_name, model in trained_models.items():
        if model_name not in eigenvalue_results:
            continue
        
        try:
            if model_type == 'bandit':
                weights = extract_reward_weights_from_bandit(model)
            else:
                weights = extract_reward_weights_from_critic(model)
            
            eigenvalues = eigenvalue_results[model_name]['eigenvalues']
            
            weight_dim = len(weights)
            eig_dim = len(eigenvalues)
            
            if weight_dim != eig_dim:
                min_dim = min(weight_dim, eig_dim)
                weights = weights[:min_dim]
                eigenvalues = eigenvalues[:min_dim]
            
            norm = compute_rkhs_norm(weights, eigenvalues)
            results[model_name] = norm
            print(f"{model_name:12s}: ||R|| = {norm:.2f}")
            
        except Exception as e:
            print(f"Error computing RKHS norm for {model_name}: {e}")
            results[model_name] = None
    
    return results


def generate_demo_rkhs_data() -> Dict[str, float]:
    """Generate demo RKHS data for visualization."""
    return {
        'bert': 324.51,
        'roberta': 287.34,
        'llama3': 298.67,
        'simcse': 8.42,
        'jina': 6.78,
        'llm2vec': 7.91
    }
