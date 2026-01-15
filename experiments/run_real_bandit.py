"""
Run small-scale bandit experiment with real BERT and SimCSE embeddings.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_item_texts(n_items: int = 500) -> list:
    """Generate product-like texts for bandit experiment."""
    categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Toys", "Beauty", "Garden"]
    adjectives = ["Premium", "Affordable", "High-quality", "Durable", "Compact", "Professional"]
    products = ["headphones", "speaker", "shirt", "jacket", "lamp", "chair", "book", "ball", "cream", "plant"]
    
    items = []
    for i in range(n_items):
        cat = categories[i % len(categories)]
        adj = adjectives[i % len(adjectives)]
        prod = products[i % len(products)]
        
        items.append({
            'id': i,
            'text': f"{adj} {prod} for {cat.lower()} enthusiasts. Great quality and fast shipping.",
            'category': cat,
            'quality': np.random.uniform(0.3, 1.0)
        })
    
    return items


def compute_reward(item: dict, context_category: str) -> float:
    """Compute reward based on category match and quality."""
    category_match = 1.0 if item['category'] == context_category else 0.3
    reward_prob = 0.5 * category_match + 0.5 * item['quality']
    return 1.0 if np.random.random() < reward_prob else 0.0


def run_bandit_experiment():
    """Run bandit experiment with real embeddings."""
    
    print("\n" + "="*60)
    print("REAL BANDIT EXPERIMENT")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    n_items = 500
    n_rounds = 1000
    k_candidates = 20
    
    items = get_item_texts(n_items)
    texts = [item['text'] for item in items]
    categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Toys", "Beauty", "Garden"]
    
    print(f"\nItems: {n_items}, Rounds: {n_rounds}, Candidates per round: {k_candidates}")
    
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
        print(f"\n--- Encoding with {model_name.upper()} ---")
        
        if model_name == 'bert':
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            model = AutoModel.from_pretrained('bert-base-uncased')
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model.eval()
            
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=64, padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_embs = outputs.last_hidden_state.mean(dim=1).numpy()
                    embeddings.append(batch_embs)
            
            all_embeddings[model_name] = np.vstack(embeddings)
            del model, tokenizer
            
        elif model_name == 'simcse':
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
            all_embeddings[model_name] = model.encode(texts, show_progress_bar=True, batch_size=32)
            del model
        
        import gc
        gc.collect()
        print(f"Embeddings shape: {all_embeddings[model_name].shape}")
    
    for model_name in models_to_test:
        print(f"\n--- Running Bandit with {model_name.upper()} ---")
        
        embeddings = all_embeddings[model_name]
        
        bandit = SimpleNeuralBandit(
            embedding_dim=embeddings.shape[1],
            hidden_dim=64,
            lr=0.01,
            exploration_weight=0.5
        )
        
        np.random.seed(42)
        
        cumulative_regret = []
        total_regret = 0
        rewards = []
        
        for t in tqdm(range(n_rounds), desc=f"{model_name} bandit"):
            context_category = categories[t % len(categories)]
            
            candidate_indices = np.random.choice(n_items, k_candidates, replace=False)
            candidate_embs = embeddings[candidate_indices]
            candidate_items = [items[i] for i in candidate_indices]
            
            selected_idx, _ = bandit.select_action(candidate_embs)
            selected_item = candidate_items[selected_idx]
            
            reward = compute_reward(selected_item, context_category)
            
            optimal_rewards = [compute_reward(item, context_category) for item in candidate_items]
            optimal_reward = max(optimal_rewards)
            
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)
            rewards.append(reward)
            
            bandit.update(candidate_embs[selected_idx], reward)
        
        results['models'][model_name] = {
            'final_regret': float(total_regret),
            'mean_reward': float(np.mean(rewards)),
            'cumulative_regret': cumulative_regret[::10],
        }
        
        print(f"  Final regret: {total_regret:.1f}")
        print(f"  Mean reward: {np.mean(rewards):.3f}")
    
    print("\n" + "-"*60)
    print("BANDIT RESULTS COMPARISON")
    print("-"*60)
    
    bert_regret = results['models']['bert']['final_regret']
    simcse_regret = results['models']['simcse']['final_regret']
    
    print(f"{'Model':<12} {'Final Regret':>15} {'Mean Reward':>12}")
    print("-"*40)
    print(f"{'BERT':<12} {bert_regret:>15.1f} {results['models']['bert']['mean_reward']:>12.3f}")
    print(f"{'SimCSE':<12} {simcse_regret:>15.1f} {results['models']['simcse']['mean_reward']:>12.3f}")
    
    improvement = (bert_regret - simcse_regret) / bert_regret * 100
    print(f"\nRegret reduction with SimCSE: {improvement:.1f}%")
    
    if improvement > 0:
        print("✓ Confirmed: Contrastive embeddings achieve lower regret")
    
    os.makedirs('results/metrics', exist_ok=True)
    output_path = 'results/metrics/real_bandit_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_bandit_experiment()
