"""
Ranking-based RecSys experiment for Books dataset.

Key differences from Beauty:
1. Uses Books data (larger, more items, longer histories)
2. Computes user embeddings ON-THE-FLY from recent K items
3. No pre-computed user embeddings needed
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.neural_ts import NeuralContextualBandit


def load_embeddings(cache_dir='../data/embeddings'):
    """Load cached ITEM embeddings only (user embeddings computed on-the-fly)."""
    cache_dir = Path(cache_dir)

    print("Loading item embeddings...")
    with open(cache_dir / 'bert_embeddings_books.pkl', 'rb') as f:
        bert_item_embs = pickle.load(f)
    with open(cache_dir / 'simcse_embeddings_books.pkl', 'rb') as f:
        simcse_item_embs = pickle.load(f)

    print(f"✓ Loaded {len(bert_item_embs)} BERT items")
    print(f"✓ Loaded {len(simcse_item_embs)} SimCSE items")

    return bert_item_embs, simcse_item_embs


def load_dataset(cache_file='../data/amazon_reviews/Books_processed.pkl'):
    """Load processed dataset."""
    print("Loading dataset...")
    with open(cache_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def compute_user_embedding(user_id, dataset, item_embs, k=10):
    """
    Compute user embedding on-the-fly from recent K items in history.
    
    Args:
        user_id: User ID
        dataset: Dataset dict with 'train_histories'
        item_embs: Dict mapping item_id -> embedding
        k: Number of recent items to use (default 10)
    
    Returns:
        user_emb: Average embedding of recent K items (np.array)
    """
    if user_id not in dataset['train_histories']:
        # Fallback: return zero vector
        embedding_dim = len(next(iter(item_embs.values())))
        return np.zeros(embedding_dim, dtype=np.float32)
    
    history = dataset['train_histories'][user_id]
    
    # Get last K items
    recent_items = history[-k:] if len(history) >= k else history
    recent_item_ids = [h['item_id'] for h in recent_items]
    
    # Get embeddings for items that exist
    valid_embs = []
    for item_id in recent_item_ids:
        if item_id in item_embs:
            valid_embs.append(item_embs[item_id])
    
    if len(valid_embs) == 0:
        # Fallback: return zero vector
        embedding_dim = len(next(iter(item_embs.values())))
        return np.zeros(embedding_dim, dtype=np.float32)
    
    # Average embeddings
    user_emb = np.mean(valid_embs, axis=0)
    return user_emb


def run_ranking_bandit(
        dataset,
        item_embs: dict,
        n_rounds: int = 3000,
        n_candidates: int = 50,
        k_history: int = 10,
        update_mode: str = 'full',
        seed: int = 42):
    """
    Ranking bandit experiment with on-the-fly user embeddings.

    Args:
        dataset: Amazon dataset with train_histories and test_interactions
        item_embs: Dict mapping item_id -> embedding
        n_rounds: Number of rounds to run
        n_candidates: Number of candidates per round (25-50)
        k_history: Number of recent items to use for user embedding
        update_mode: 'full' = update on all candidates, 'selected_only' = update only on selected
        seed: Random seed

    Returns:
        dict with results
    """
    np.random.seed(seed)

    # Setup
    item_asins = list(item_embs.keys())
    item_array = np.array([item_embs[asin] for asin in item_asins])
    asin_to_idx = {asin: i for i, asin in enumerate(item_asins)}

    embedding_dim = item_array.shape[1]

    # Initialize neural bandit with diagonal approximation (avoids OOM)
    bandit = NeuralContextualBandit(
        input_dim=embedding_dim * 2,  # Concatenated [item; user]
        hidden_dim=100,
        lambda_reg=1.0,
        nu=1.0,
        use_diagonal_approx=True  # Critical for large action spaces!
    )

    # Tracking
    true_item_ranks = []
    selected_ranks = []

    test_data = dataset['test_interactions']
    n_rounds = min(n_rounds, len(test_data))

    print(f"\nStarting experiment:")
    print(f"  Rounds: {n_rounds}")
    print(f"  Candidates per round: {n_candidates}")
    print(f"  User history length: {k_history}")
    print(f"  Update mode: {update_mode}")

    for t in tqdm(range(n_rounds), desc="Ranking Bandit"):
        user_id, true_item, true_reward = test_data[t]

        # Skip if missing
        if true_item not in asin_to_idx:
            continue

        # Compute user embedding on-the-fly from recent K items
        user_emb = compute_user_embedding(user_id, dataset, item_embs, k=k_history)
        
        # Skip if user has no history
        if np.allclose(user_emb, 0):
            continue

        true_idx = asin_to_idx[true_item]

        # Sample candidates (always include true item)
        other_indices = np.random.choice(
            [i for i in range(len(item_asins)) if i != true_idx],
            min(n_candidates - 1, len(item_asins) - 1),
            replace=False)
        candidate_indices = [true_idx] + other_indices.tolist()
        np.random.shuffle(candidate_indices)

        # Find position of true item in candidate list
        true_position_in_candidates = candidate_indices.index(true_idx)

        # Compute features: [item; user] concatenation
        candidate_embs = item_array[candidate_indices]
        user_embs_tiled = np.tile(user_emb, (len(candidate_indices), 1))
        features = np.concatenate([candidate_embs, user_embs_tiled], axis=1)

        # Bandit scores all candidates
        scores = bandit.predict_scores(features)

        # Rank candidates by scores (descending)
        ranking = np.argsort(-scores)

        # Find where true item ranked
        true_item_rank = np.where(ranking == true_position_in_candidates)[0][0] + 1
        true_item_ranks.append(true_item_rank)

        # Bandit selects top-ranked item
        selected_position = ranking[0]
        selected_idx = candidate_indices[selected_position]
        selected_rank = 1
        selected_ranks.append(selected_rank)

        # ===== UPDATE =====
        if update_mode == 'full':
            # Update on ALL candidates (full information)
            for idx in range(len(candidate_indices)):
                # Label: 1 if this is the true item, 0 otherwise
                label = 1.0 if candidate_indices[idx] == true_idx else 0.0
                bandit.update(features[idx], label)

        else:  # 'selected_only'
            # Only update on selected item
            label = 1.0 if selected_idx == true_idx else 0.0
            bandit.update(features[selected_position], label)

    # Compute metrics
    true_item_ranks = np.array(true_item_ranks)

    return {
        'true_item_ranks': true_item_ranks,
        'mean_rank': np.mean(true_item_ranks),
        'median_rank': np.median(true_item_ranks),
        'mrr': np.mean(1.0 / true_item_ranks),  # Mean Reciprocal Rank
        'hit@1': np.mean(true_item_ranks == 1),
        'hit@3': np.mean(true_item_ranks <= 3),
        'hit@5': np.mean(true_item_ranks <= 5),
        'hit@10': np.mean(true_item_ranks <= 10),
        'n_rounds': len(true_item_ranks),
        'n_candidates': n_candidates
    }


def plot_results(bert_results,
                 simcse_results,
                 save_path='results/plots/books_ranking_comparison.png'):
    """Plot comparison between BERT and SimCSE bandits."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Rank distribution over time (rolling average)
    ax1 = axes[0]
    window = 100
    bert_rolling = np.convolve(bert_results['true_item_ranks'],
                               np.ones(window) / window,
                               mode='valid')
    simcse_rolling = np.convolve(simcse_results['true_item_ranks'],
                                 np.ones(window) / window,
                                 mode='valid')

    ax1.plot(bert_rolling, label='BERT', color='red', alpha=0.8)
    ax1.plot(simcse_rolling, label='SimCSE', color='blue', alpha=0.8)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Rank (lower = better)')
    ax1.set_title('True Item Rank Over Time (100-round window)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative MRR over time
    ax2 = axes[1]
    bert_cumulative_mrr = np.cumsum(
        1.0 / bert_results['true_item_ranks']) / np.arange(
            1,
            len(bert_results['true_item_ranks']) + 1)
    simcse_cumulative_mrr = np.cumsum(
        1.0 / simcse_results['true_item_ranks']) / np.arange(
            1,
            len(simcse_results['true_item_ranks']) + 1)

    ax2.plot(bert_cumulative_mrr, label='BERT', color='red', alpha=0.8)
    ax2.plot(simcse_cumulative_mrr, label='SimCSE', color='blue', alpha=0.8)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative MRR (higher = better)')
    ax2.set_title('Mean Reciprocal Rank Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Hit@K comparison
    ax3 = axes[2]
    k_values = [1, 3, 5, 10]
    bert_hits = [bert_results[f'hit@{k}'] for k in k_values]
    simcse_hits = [simcse_results[f'hit@{k}'] for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    ax3.bar(x - width / 2,
            bert_hits,
            width,
            label='BERT',
            color='red',
            alpha=0.7)
    ax3.bar(x + width / 2,
            simcse_hits,
            width,
            label='SimCSE',
            color='blue',
            alpha=0.7)
    ax3.set_xlabel('K')
    ax3.set_ylabel('Hit@K (higher = better)')
    ax3.set_title('Hit Rate Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'@{k}' for k in k_values])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    """Run complete ranking experiment on Books."""

    # Load data
    dataset = load_dataset()
    bert_item_embs, simcse_item_embs = load_embeddings()

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total users: {len(dataset['train_histories'])}")
    print(f"Total items: {len(bert_item_embs)}")
    print(f"Test interactions: {len(dataset['test_interactions'])}")
    
    # Compute average history length
    history_lengths = [len(h) for h in dataset['train_histories'].values()]
    print(f"Avg history length: {np.mean(history_lengths):.1f} items")
    print(f"Median history length: {np.median(history_lengths):.0f} items")

    # Experiment parameters
    n_rounds = 3000
    n_candidates = 50
    k_history = 10  # Use last 10 items for user embedding

    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENTS")
    print("=" * 60)
    print(f"Rounds: {n_rounds}")
    print(f"Candidates per round: {n_candidates}")
    print(f"User history items: {k_history}")
    print(f"Update mode: full information")

    # Run BERT bandit
    print("\n--- BERT Bandit ---")
    bert_results = run_ranking_bandit(dataset,
                                      bert_item_embs,
                                      n_rounds=n_rounds,
                                      n_candidates=n_candidates,
                                      k_history=k_history,
                                      update_mode='full',
                                      seed=42)

    # Run SimCSE bandit
    print("\n--- SimCSE Bandit ---")
    simcse_results = run_ranking_bandit(dataset,
                                        simcse_item_embs,
                                        n_rounds=n_rounds,
                                        n_candidates=n_candidates,
                                        k_history=k_history,
                                        update_mode='full',
                                        seed=42)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nBERT Bandit:")
    print(f"  Mean Rank: {bert_results['mean_rank']:.2f}")
    print(f"  Median Rank: {bert_results['median_rank']:.0f}")
    print(f"  MRR: {bert_results['mrr']:.4f}")
    print(f"  Hit@1: {bert_results['hit@1']*100:.2f}%")
    print(f"  Hit@3: {bert_results['hit@3']*100:.2f}%")
    print(f"  Hit@5: {bert_results['hit@5']*100:.2f}%")
    print(f"  Hit@10: {bert_results['hit@10']*100:.2f}%")

    print("\nSimCSE Bandit:")
    print(f"  Mean Rank: {simcse_results['mean_rank']:.2f}")
    print(f"  Median Rank: {simcse_results['median_rank']:.0f}")
    print(f"  MRR: {simcse_results['mrr']:.4f}")
    print(f"  Hit@1: {simcse_results['hit@1']*100:.2f}%")
    print(f"  Hit@3: {simcse_results['hit@3']*100:.2f}%")
    print(f"  Hit@5: {simcse_results['hit@5']*100:.2f}%")
    print(f"  Hit@10: {simcse_results['hit@10']*100:.2f}%")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    rank_improvement = (
        (bert_results['mean_rank'] - simcse_results['mean_rank']) /
        bert_results['mean_rank'] * 100)
    mrr_improvement = ((simcse_results['mrr'] - bert_results['mrr']) /
                       bert_results['mrr'] * 100)
    hit5_improvement = ((simcse_results['hit@5'] - bert_results['hit@5']) /
                        bert_results['hit@5'] * 100)

    print(f"Mean Rank: SimCSE is {rank_improvement:+.1f}% better (lower is better)")
    print(f"MRR: SimCSE is {mrr_improvement:+.1f}% better (higher is better)")
    print(f"Hit@5: SimCSE is {hit5_improvement:+.1f}% better")
    print(f"  SimCSE: {simcse_results['hit@5']*100:.2f}% vs BERT: {bert_results['hit@5']*100:.2f}%")

    # Plot
    plot_results(bert_results, simcse_results)

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
