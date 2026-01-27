"""
Ranking-based RecSys experiment for Books dataset with config sweep.

Uses CONTINUOUS REWARDS (actual ratings normalized to [0,1]) for better learning signal.

Sweeps over:
- Lambda values (regularization)
- Dimensionality reduction (UMAP to 10D, 50D, or no reduction)
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import umap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.neural_ts import NeuralContextualBandit


def load_embeddings(cache_dir='../data/embeddings'):
    """Load cached ITEM embeddings only."""
    cache_dir = Path(cache_dir)

    print("Loading item embeddings...")
    with open(cache_dir / 'bert_embeddings_books.pkl', 'rb') as f:
        bert_item_embs = pickle.load(f)
    with open(cache_dir / 'simcse_embeddings_books.pkl', 'rb') as f:
        simcse_item_embs = pickle.load(f)

    print(f"✓ Loaded {len(bert_item_embs)} BERT items")
    print(f"✓ Loaded {len(simcse_item_embs)} SimCSE items")

    return bert_item_embs, simcse_item_embs


def load_dataset(cache_file='../data/amazon_books/Books_processed.pkl'):
    """Load processed dataset."""
    print("Loading dataset...")
    with open(cache_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def compute_user_embedding(user_id, dataset, item_embs, k=10):
    """Compute user embedding on-the-fly from recent K items."""
    if user_id not in dataset['train_histories']:
        embedding_dim = len(next(iter(item_embs.values())))
        return np.zeros(embedding_dim, dtype=np.float32)
    
    history = dataset['train_histories'][user_id]
    recent_items = history[-k:] if len(history) >= k else history
    recent_item_ids = [h['item_id'] for h in recent_items]
    
    valid_embs = [item_embs[iid] for iid in recent_item_ids if iid in item_embs]
    
    if len(valid_embs) == 0:
        embedding_dim = len(next(iter(item_embs.values())))
        return np.zeros(embedding_dim, dtype=np.float32)
    
    return np.mean(valid_embs, axis=0)


def reduce_dimensionality(item_embs_dict, method='umap', n_components=10, seed=42):
    """Reduce embedding dimensionality using UMAP or PCA."""
    print(f"\nReducing dimensionality: {method.upper()} to {n_components}D")
    
    item_ids = list(item_embs_dict.keys())
    embs_array = np.array([item_embs_dict[iid] for iid in item_ids])
    
    print(f"  Input: {embs_array.shape}")
    
    if method == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=seed
        )
        reduced_embs = reducer.fit_transform(embs_array)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=seed)
        reduced_embs = reducer.fit_transform(embs_array)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  Output: {reduced_embs.shape}")
    
    return {item_ids[i]: reduced_embs[i] for i in range(len(item_ids))}


def run_ranking_bandit(dataset, item_embs, config, k_history=10, seed=42):
    """Run ranking bandit experiment with CONTINUOUS REWARDS."""
    np.random.seed(seed)
    
    # Extract config
    lambda_reg = config.get('lambda_reg', 1.0)
    n_rounds = config.get('n_rounds', 3000)
    n_candidates = config.get('n_candidates', 50)

    # Setup
    item_asins = list(item_embs.keys())
    item_array = np.array([item_embs[asin] for asin in item_asins])
    asin_to_idx = {asin: i for i, asin in enumerate(item_asins)}
    embedding_dim = item_array.shape[1]

    # Initialize bandit
    bandit = NeuralContextualBandit(
        embedding_dim=embedding_dim * 2,
        hidden_dim=100,
        lambda_reg=lambda_reg,
        nu=1.0,
        use_diagonal_approx=True
    )

    # Tracking
    true_item_ranks = []
    test_data = dataset['test_interactions']
    n_rounds = min(n_rounds, len(test_data))

    print(f"\n{config['name']}")
    print(f"  Lambda: {lambda_reg}, Dims: {embedding_dim}, Rounds: {n_rounds}")

    for t in tqdm(range(n_rounds), desc=config['name'][:30]):
        user_id, true_item, true_reward = test_data[t]

        if true_item not in asin_to_idx:
            continue

        user_emb = compute_user_embedding(user_id, dataset, item_embs, k=k_history)
        
        if np.allclose(user_emb, 0):
            continue

        true_idx = asin_to_idx[true_item]

        # Sample candidates
        other_indices = np.random.choice(
            [i for i in range(len(item_asins)) if i != true_idx],
            min(n_candidates - 1, len(item_asins) - 1),
            replace=False)
        candidate_indices = [true_idx] + other_indices.tolist()
        np.random.shuffle(candidate_indices)

        true_position = candidate_indices.index(true_idx)

        # Features: [item; user]
        candidate_embs = item_array[candidate_indices]
        user_embs_tiled = np.tile(user_emb, (len(candidate_indices), 1))
        features = np.concatenate([candidate_embs, user_embs_tiled], axis=1)

        # Score and rank
        scores = bandit.predict_scores(features)
        ranking = np.argsort(-scores)
        true_item_rank = np.where(ranking == true_position)[0][0] + 1
        true_item_ranks.append(true_item_rank)

        # ===== UPDATE WITH CONTINUOUS REWARDS =====
        # Update on all candidates with graded rewards
        for idx in range(len(candidate_indices)):
            if candidate_indices[idx] == true_idx:
                # TRUE ITEM: Use actual rating (already normalized to [0,1])
                reward = true_reward
            else:
                # OTHER ITEMS: Zero reward (user didn't interact)
                reward = 0.0
            
            bandit.update(features[idx], reward)

    # Metrics
    true_item_ranks = np.array(true_item_ranks)
    
    return {
        'config': config,
        'true_item_ranks': true_item_ranks,
        'mean_rank': np.mean(true_item_ranks),
        'median_rank': np.median(true_item_ranks),
        'mrr': np.mean(1.0 / true_item_ranks),
        'hit@1': np.mean(true_item_ranks == 1),
        'hit@3': np.mean(true_item_ranks <= 3),
        'hit@5': np.mean(true_item_ranks <= 5),
        'hit@10': np.mean(true_item_ranks <= 10),
        'n_rounds': len(true_item_ranks)
    }


def main():
    """Run complete ranking experiment with config sweep."""

    # Load data
    dataset = load_dataset()
    bert_item_embs, simcse_item_embs = load_embeddings()

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Users: {len(dataset['train_histories'])}")
    print(f"Items: {len(bert_item_embs)}")
    print(f"Test interactions: {len(dataset['test_interactions'])}")

    # Config sweep
    n_rounds = 3000
    configs = [
        {
            'name': 'Neural - UMAP (10) - λ=0.1',
            'reduce': 'umap',
            'dims': 10,
            'lambda_reg': 0.1,
            'n_rounds': n_rounds
        },
        {
            'name': 'Neural - UMAP (10) - λ=0.01',
            'reduce': 'umap',
            'dims': 10,
            'lambda_reg': 0.01,
            'n_rounds': n_rounds
        },
    ]

    # Run experiments
    bert_results = []
    simcse_results = []

    for config in configs:
        print("\n" + "=" * 60)
        print(f"CONFIG: {config['name']}")
        print("=" * 60)
        
        # Apply dimensionality reduction
        if config.get('reduce') == 'umap':
            bert_reduced = reduce_dimensionality(bert_item_embs, 'umap', config['dims'])
            simcse_reduced = reduce_dimensionality(simcse_item_embs, 'umap', config['dims'])
        else:
            bert_reduced = bert_item_embs
            simcse_reduced = simcse_item_embs
        
        # BERT
        print("\n--- BERT ---")
        bert_res = run_ranking_bandit(dataset, bert_reduced, config)
        bert_results.append(bert_res)
        
        # SimCSE
        print("\n--- SimCSE ---")
        simcse_res = run_ranking_bandit(dataset, simcse_reduced, config)
        simcse_results.append(simcse_res)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for i, config in enumerate(configs):
        print(f"\n{config['name']}:")
        print(f"  BERT:   MRR={bert_results[i]['mrr']:.4f}, Hit@5={bert_results[i]['hit@5']*100:.1f}%")
        print(f"  SimCSE: MRR={simcse_results[i]['mrr']:.4f}, Hit@5={simcse_results[i]['hit@5']*100:.1f}%")
        gap = ((simcse_results[i]['mrr'] - bert_results[i]['mrr']) / bert_results[i]['mrr']) * 100
        print(f"  Gap: {gap:+.1f}%")

    # Save results
    results_dir = Path('results/books_ranking')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'bert_results.pkl', 'wb') as f:
        pickle.dump(bert_results, f)
    with open(results_dir / 'simcse_results.pkl', 'wb') as f:
        pickle.dump(simcse_results, f)

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()