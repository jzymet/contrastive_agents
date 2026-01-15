"""
Run larger-scale bandit experiment with 10K rounds, 10K items, 500 candidates/round.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_item_texts(n_items: int = 10000) -> list:
    """Generate diverse product-like texts for bandit experiment."""
    categories = [
        "Electronics", "Clothing", "Home", "Books", "Sports", 
        "Toys", "Beauty", "Garden", "Automotive", "Office",
        "Food", "Pet Supplies", "Music", "Movies", "Health",
        "Jewelry", "Shoes", "Outdoors", "Baby", "Tools"
    ]
    
    adjectives = [
        "Premium", "Affordable", "High-quality", "Durable", "Compact", 
        "Professional", "Lightweight", "Versatile", "Innovative", "Classic",
        "Modern", "Eco-friendly", "Budget", "Luxury", "Essential",
        "Portable", "Wireless", "Smart", "Ergonomic", "Stylish"
    ]
    
    products = [
        "headphones", "speaker", "cable", "charger", "adapter",
        "shirt", "jacket", "pants", "shoes", "bag",
        "lamp", "chair", "table", "shelf", "container",
        "novel", "guide", "manual", "journal", "album",
        "ball", "racket", "weights", "mat", "gear",
        "toy", "game", "puzzle", "blocks", "doll",
        "cream", "lotion", "serum", "brush", "kit",
        "plant", "pot", "tools", "seeds", "fertilizer",
        "watch", "ring", "necklace", "bracelet", "earrings",
        "sneakers", "boots", "sandals", "slippers", "heels"
    ]
    
    features = [
        "with excellent reviews",
        "bestseller item",
        "top rated",
        "customer favorite",
        "highly recommended",
        "new arrival",
        "trending now",
        "limited edition",
        "exclusive design",
        "award winning",
        "editor's choice",
        "staff pick",
        "most popular",
        "value pack",
        "gift ready"
    ]
    
    items = []
    for i in range(n_items):
        cat = categories[i % len(categories)]
        adj = adjectives[(i * 7) % len(adjectives)]
        prod = products[(i * 3) % len(products)]
        feat = features[(i * 11) % len(features)]
        
        items.append({
            'id': i,
            'text': f"{adj} {prod} for {cat.lower()} enthusiasts. {feat.capitalize()}. Fast shipping available.",
            'category': cat,
            'quality': np.random.uniform(0.2, 1.0),
            'popularity': np.random.uniform(0.3, 1.0)
        })
    
    return items


def compute_reward(item: dict, context_category: str, user_preference: float) -> float:
    """
    Compute reward based on multiple factors.
    More complex reward function to show embedding quality difference.
    """
    category_match = 1.0 if item['category'] == context_category else 0.2
    quality_factor = item['quality']
    popularity_factor = item['popularity']
    preference_match = 1.0 - abs(user_preference - item['quality'])
    
    reward_prob = (
        0.35 * category_match + 
        0.25 * quality_factor + 
        0.20 * popularity_factor +
        0.20 * preference_match
    )
    
    return 1.0 if np.random.random() < reward_prob else 0.0


def run_bandit_experiment():
    """Run bandit experiment with real embeddings at large scale."""
    
    print("\n" + "="*60)
    print("LARGE-SCALE BANDIT EXPERIMENT")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    n_items = 3000
    n_rounds = 10000
    k_candidates = 500
    
    items = get_item_texts(n_items)
    texts = [item['text'] for item in items]
    categories = list(set(item['category'] for item in items))
    
    print(f"\nConfig: {n_items} items, {n_rounds} rounds, {k_candidates} candidates/round")
    print(f"Categories: {len(categories)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_items': n_items,
        'n_rounds': n_rounds,
        'k_candidates': k_candidates,
        'models': {}
    }
    
    from src.models import SimpleNeuralBandit
    
    models_to_test = ['bert', 'simcse']
    all_embeddings = {}
    
    for model_name in models_to_test:
        print(f"\n--- Encoding {n_items} items with {model_name.upper()} ---")
        
        if model_name == 'bert':
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            model = AutoModel.from_pretrained('bert-base-uncased')
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model.eval()
            
            embeddings = []
            batch_size = 128
            
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT encoding"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=64, padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_embs = outputs.last_hidden_state.mean(dim=1).numpy()
                    embeddings.append(batch_embs)
            
            all_embeddings[model_name] = np.vstack(embeddings)
            del model, tokenizer
            gc.collect()
            
        elif model_name == 'simcse':
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
            all_embeddings[model_name] = model.encode(texts, show_progress_bar=True, batch_size=128)
            del model
            gc.collect()
        
        print(f"Embeddings shape: {all_embeddings[model_name].shape}")
    
    for model_name in models_to_test:
        print(f"\n--- Running 10K Bandit with {model_name.upper()} (500 candidates/round) ---")
        
        embeddings = all_embeddings[model_name]
        
        bandit = SimpleNeuralBandit(
            embedding_dim=embeddings.shape[1],
            hidden_dim=128,
            lr=0.005,
            exploration_weight=0.3
        )
        
        np.random.seed(42)
        
        cumulative_regret = []
        total_regret = 0
        rewards = []
        
        for t in tqdm(range(n_rounds), desc=f"{model_name} bandit"):
            context_category = categories[t % len(categories)]
            user_preference = 0.3 + 0.4 * np.sin(t / 500)
            
            candidate_indices = np.random.choice(n_items, k_candidates, replace=False)
            candidate_embs = embeddings[candidate_indices]
            candidate_items = [items[i] for i in candidate_indices]
            
            selected_idx, _ = bandit.select_action(candidate_embs)
            selected_item = candidate_items[selected_idx]
            
            reward = compute_reward(selected_item, context_category, user_preference)
            
            all_potential_rewards = []
            for item in candidate_items:
                all_potential_rewards.append(compute_reward(item, context_category, user_preference))
            optimal_reward = max(all_potential_rewards)
            
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)
            rewards.append(reward)
            
            bandit.update(candidate_embs[selected_idx], reward)
        
        results['models'][model_name] = {
            'final_regret': float(total_regret),
            'mean_reward': float(np.mean(rewards)),
            'reward_last_1000': float(np.mean(rewards[-1000:])),
            'cumulative_regret': cumulative_regret[::100],
        }
        
        print(f"  Final regret: {total_regret:.1f}")
        print(f"  Mean reward: {np.mean(rewards):.3f}")
        print(f"  Mean reward (last 1000): {np.mean(rewards[-1000:]):.3f}")
    
    print("\n" + "-"*60)
    print("10K BANDIT RESULTS (500 candidates/round)")
    print("-"*60)
    
    bert_regret = results['models']['bert']['final_regret']
    simcse_regret = results['models']['simcse']['final_regret']
    
    print(f"\n{'Model':<12} {'Final Regret':>15} {'Mean Reward':>12} {'Last 1K Reward':>15}")
    print("-"*55)
    for model_name in models_to_test:
        r = results['models'][model_name]
        print(f"{model_name:<12} {r['final_regret']:>15.1f} {r['mean_reward']:>12.3f} {r['reward_last_1000']:>15.3f}")
    
    improvement = (bert_regret - simcse_regret) / bert_regret * 100
    print(f"\nRegret reduction with SimCSE: {improvement:.1f}%")
    
    if improvement > 5:
        print("Confirmed: Contrastive embeddings achieve lower regret at scale")
    elif improvement > 0:
        print("Slight advantage for contrastive embeddings")
    else:
        print("Results inconclusive at this scale")
    
    os.makedirs('results/metrics', exist_ok=True)
    output_path = 'results/metrics/real_bandit_10k_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_bandit_experiment()
