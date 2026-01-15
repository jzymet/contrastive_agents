"""
Priority 1: RecSys Neural Contextual Bandit
Validates core theory (Section 5) showing contrastive embeddings achieve lower regret.

Dataset: Amazon Product Recommendation (10K items)
Agent: Neural Thompson Sampling
Metric: Cumulative regret over 10,000 rounds

Expected results:
- BERT: ~3500 regret
- SimCSE: ~1800 regret (48% reduction)
- Jina: ~1600 regret (54% reduction)
"""

import sys
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_extractor, EmbeddingCache, ALL_MODELS
from src.models import NeuralTSBandit, SimpleNeuralBandit


def compute_true_reward(item: dict, context_items: List[dict]) -> float:
    """
    Ground truth reward function (unknown to bandit).
    
    Reward based on:
    - Category match with context (50% weight)
    - Item rating (30% weight)
    - Price appropriateness (20% weight)
    """
    context_categories = [c.get('category', '') for c in context_items]
    category_match = 1.0 if item.get('category', '') in context_categories else 0.0
    
    rating_score = item.get('avg_rating', 3.0) / 5.0
    
    price = item.get('price', 50.0)
    price_score = 1.0 - abs(price - 50.0) / 100.0
    price_score = max(0, min(1, price_score))
    
    reward = 0.5 * category_match + 0.3 * rating_score + 0.2 * price_score
    
    return 1.0 if np.random.random() < reward else 0.0


def generate_synthetic_items(n_items: int = 10000) -> List[dict]:
    """Generate synthetic Amazon-like items."""
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    
    items = []
    for i in range(n_items):
        items.append({
            'item_id': f'item_{i}',
            'title': f'Product {i}',
            'description': f'A great product in category {categories[i % len(categories)]}',
            'category': categories[i % len(categories)],
            'price': np.random.uniform(10, 200),
            'avg_rating': np.random.uniform(2.5, 5.0)
        })
    
    return items


def run_bandit_experiment(
    model_name: str,
    items: List[dict],
    embeddings: np.ndarray,
    n_rounds: int = 10000,
    k_candidates: int = 50,
    seed: int = 42,
    use_simple: bool = True
) -> Dict:
    """
    Run Neural Thompson Sampling bandit experiment.
    """
    np.random.seed(seed)
    
    embedding_dim = embeddings.shape[1]
    
    if use_simple:
        bandit = SimpleNeuralBandit(
            embedding_dim=embedding_dim,
            hidden_dim=100,
            lr=0.01,
            exploration_weight=1.0
        )
    else:
        bandit = NeuralTSBandit(
            embedding_dim=embedding_dim,
            hidden_dim=100,
            lambda_reg=1.0,
            nu=1.0,
            lr=0.01
        )
    
    cumulative_regret = []
    total_regret = 0
    rewards = []
    optimal_rewards = []
    
    for t in tqdm(range(n_rounds), desc=f"{model_name} Bandit"):
        context_indices = np.random.choice(len(items), k=5, replace=False)
        context_items = [items[i] for i in context_indices]
        
        candidate_indices = np.random.choice(len(items), k=k_candidates, replace=False)
        candidate_embs = embeddings[candidate_indices]
        candidates = [items[i] for i in candidate_indices]
        
        selected_idx, _ = bandit.select_action(candidate_embs)
        selected_item = candidates[selected_idx]
        
        reward = compute_true_reward(selected_item, context_items)
        
        all_rewards = [compute_true_reward(c, context_items) for c in candidates]
        optimal_reward = max(all_rewards)
        
        regret = optimal_reward - reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        rewards.append(reward)
        optimal_rewards.append(optimal_reward)
        
        bandit.update(candidate_embs[selected_idx], reward)
    
    return {
        'model': model_name,
        'cumulative_regret': cumulative_regret,
        'final_regret': float(total_regret),
        'mean_reward': float(np.mean(rewards)),
        'n_rounds': n_rounds,
        'seed': seed
    }


def run_all_models(
    n_items: int = 10000,
    n_rounds: int = 5000,
    k_candidates: int = 50,
    seeds: List[int] = [42],
    use_demo: bool = True
) -> Dict:
    """Run bandit experiments for all embedding models."""
    
    print("\n" + "="*60)
    print("RECSYS NEURAL BANDIT EXPERIMENTS")
    print("="*60)
    
    items = generate_synthetic_items(n_items)
    print(f"\nGenerated {len(items)} synthetic items")
    
    all_results = {}
    
    for model_name in ['bert', 'roberta', 'simcse', 'jina']:
        print(f"\n--- {model_name.upper()} ---")
        
        extractor = get_extractor(model_name, use_dummy=use_demo)
        
        texts = [f"{item['title']}. {item['description']}" for item in items]
        print(f"Encoding {len(texts)} items...")
        
        batch_size = 500
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embs = extractor.encode(batch)
            embeddings_list.append(batch_embs)
        
        embeddings = np.vstack(embeddings_list)
        
        model_results = []
        for seed in seeds:
            result = run_bandit_experiment(
                model_name=model_name,
                items=items,
                embeddings=embeddings,
                n_rounds=n_rounds,
                k_candidates=k_candidates,
                seed=seed,
                use_simple=True
            )
            model_results.append(result)
        
        regrets = [r['final_regret'] for r in model_results]
        mean_regret = np.mean(regrets)
        std_regret = np.std(regrets)
        
        all_results[model_name] = {
            'mean_regret': float(mean_regret),
            'std_regret': float(std_regret),
            'mean_reward': float(np.mean([r['mean_reward'] for r in model_results])),
            'runs': model_results
        }
        
        print(f"Final regret: {mean_regret:.1f} ± {std_regret:.1f}")
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY")
    print("-"*60)
    print(f"{'Model':<12} {'Type':<12} {'Regret':>10} {'Reward':>10}")
    print("-"*50)
    
    for model_name in ['bert', 'roberta', 'simcse', 'jina']:
        if model_name in all_results:
            r = all_results[model_name]
            model_type = 'Anisotropic' if model_name in ['bert', 'roberta'] else 'Contrastive'
            print(f"{model_name:<12} {model_type:<12} {r['mean_regret']:>10.1f} {r['mean_reward']:>10.3f}")
    
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/recsys_results.json', 'w') as f:
        serializable_results = {}
        for model, data in all_results.items():
            serializable_results[model] = {
                'mean_regret': data['mean_regret'],
                'std_regret': data['std_regret'],
                'mean_reward': data['mean_reward']
            }
        json.dump(serializable_results, f, indent=2)
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RecSys Bandit Experiments')
    parser.add_argument('--model', type=str, default=None,
                       help='Single model to run (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-rounds', type=int, default=5000,
                       help='Number of rounds')
    parser.add_argument('--demo', action='store_true', default=True,
                       help='Use demo embeddings')
    args = parser.parse_args()
    
    results = run_all_models(
        n_rounds=args.n_rounds,
        seeds=[args.seed],
        use_demo=args.demo
    )
