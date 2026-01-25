"""
Ranking-based RecSys experiment with sparse rewards workaround.

Instead of binary hit/miss, we:
1. Sample 25-50 candidates (including true item)
2. Bandit ranks all candidates
3. Track where true item ranks
4. Metrics: MRR, Hit@5, Hit@10, average rank
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

# Add after existing imports (around line 16)
from src.embeddings.reduce_dim import reduce_embeddings
from src.models.neural_ts import LinearContextualBandit, NeuralContextualBandit  # If not already imported


def load_embeddings(cache_dir='../data/embeddings'):
    """Load cached embeddings."""
    cache_dir = Path(cache_dir)

    print("Loading embeddings...")
    with open(cache_dir / 'bert_embeddings.pkl', 'rb') as f:
        bert_item_embs = pickle.load(f)
    with open(cache_dir / 'simcse_embeddings.pkl', 'rb') as f:
        simcse_item_embs = pickle.load(f)
    with open(cache_dir / 'bert_user_embeddings.pkl', 'rb') as f:
        bert_user_embs = pickle.load(f)
    with open(cache_dir / 'simcse_user_embeddings.pkl', 'rb') as f:
        simcse_user_embs = pickle.load(f)

    return bert_item_embs, simcse_item_embs, bert_user_embs, simcse_user_embs


def load_dataset(cache_file='../data/amazon_reviews/All_Beauty_processed.pkl'):
    """Load processed dataset."""
    print("Loading dataset...")
    with open(cache_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def compute_user_history_concat(train_histories, item_embs):
    """
    User representation: concatenate last 2 items purchased.

    Returns dict mapping user_id -> [item1_emb; item2_emb]
    """
    from tqdm import tqdm

    user_reprs = {}
    embedding_dim = len(next(iter(item_embs.values())))

    for user_id, history in tqdm(train_histories.items(),
                                 desc="User histories (concat)"):
        item_ids = [item_id for item_id, _, _ in history]
        valid_embs = [
            item_embs[item_id] for item_id in item_ids if item_id in item_embs
        ]

        if len(valid_embs) >= 2:
            # Concat last 2 items
            user_reprs[user_id] = np.concatenate(
                [valid_embs[-2], valid_embs[-1]])
        elif len(valid_embs) == 1:
            # Duplicate single item
            user_reprs[user_id] = np.concatenate(
                [valid_embs[0], valid_embs[0]])
        else:
            # Zero vector
            user_reprs[user_id] = np.zeros(embedding_dim * 2)

    return user_reprs


def run_ranking_bandit(
        dataset,
        item_embs: dict,
        user_embs: dict,
        n_rounds: int = 3000,
        n_candidates: int = 50,
        bandit_type: str = 'neural',  # ← ADD: 'linear' or 'neural'
        hidden_dim: int = 100,  # ← ADD: for neural
        lambda_reg: float = 10.0,  # ← CHANGE default to 10.0
        weight_decay: float = 0.01,  # ← ADD: for neural
        use_diagonal: bool = True,  # ← ADD: for neural
        seed: int = 42):
    """
    Ranking bandit experiment.

    Args:
        dataset: Amazon reviews dataset
        item_embs: Dict mapping item_id -> embedding
        user_embs: Dict mapping user_id -> embedding
        n_rounds: Number of rounds to run
        n_candidates: Number of candidates per round (25-50)
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

    # OLD:
    # bandit = LinearContextualBandit(
    #     embedding_dim * 2,
    #     algorithm='ucb',
    #     ucb_alpha=1.0,
    #     lambda_reg=1.0
    # )

    # NEW:
    if bandit_type == 'linear':
        bandit = LinearContextualBandit(
            embedding_dim * 2,  # [item; user]
            algorithm='ucb',
            ucb_alpha=1.0,
            lambda_reg=lambda_reg)
    elif bandit_type == 'neural':
        bandit = NeuralContextualBandit(
            embedding_dim * 2,  # [item; user] 
            hidden_dim=hidden_dim,
            lambda_reg=lambda_reg,
            weight_decay=weight_decay,
            learning_rate=0.01,
            algorithm='ucb',
            ucb_alpha=1.0,
            use_cuda=False,
            use_diagonal_approximation=use_diagonal)
    else:
        raise ValueError(f"Unknown bandit_type: {bandit_type}")

    # Tracking
    true_item_ranks = []  # Where did true item rank?
    selected_ranks = []  # What rank did we select?

    test_data = dataset['test_interactions']
    n_rounds = min(n_rounds, len(test_data))

    for t in tqdm(range(n_rounds), desc="Ranking Bandit"):
        user_id, true_item, true_reward = test_data[t]

        # Skip if missing
        if user_id not in user_embs or true_item not in asin_to_idx:
            continue

        user_emb = user_embs[user_id]
        true_idx = asin_to_idx[true_item]

        # Sample candidates (always include true item)
        other_indices = np.random.choice(
            [i for i in range(len(item_asins)) if i != true_idx],
            min(n_candidates - 1,
                len(item_asins) - 1),
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
        ranking = np.argsort(
            -scores)  # Indices sorted by score (highest first)

        # Find where true item ranked
        true_item_rank = np.where(
            ranking == true_position_in_candidates)[0][0] + 1  # 1-indexed
        true_item_ranks.append(true_item_rank)

        # Bandit selects top-ranked item
        selected_position = ranking[0]
        selected_idx = candidate_indices[selected_position]
        selected_rank = 1  # Always selects rank 1
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
                 save_path='results/plots/ranking_comparison.png'):
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
    """Run ranking experiments with multiple configurations."""

    # Load data
    dataset = load_dataset()
    bert_item_embs, simcse_item_embs, bert_user_embs, simcse_user_embs = load_embeddings(
    )

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total users: {len(dataset['train_histories'])}")
    print(f"Total items: {len(bert_item_embs)}")
    print(f"Test interactions: {len(dataset['test_interactions'])}")
    print(
        f"Original embedding dim: {len(next(iter(bert_item_embs.values())))}")

    # Test configurations
    configs = [
        {
            'name': 'Neural - No Reduction - λ=1.0',
            'reduce': None,
            'bandit_type': 'neural',
            'lambda_reg': 1.0,
            'n_rounds': 1000
        },
        {
            'name': 'Neural - No Reduction - λ=10.0',
            'reduce': None,
            'bandit_type': 'neural',
            'lambda_reg': 10.0,
            'n_rounds': 1000
        },
        {
            'name': 'Neural - PCA (50) - λ=10.0',
            'reduce': 'pca',
            'bandit_type': 'neural',
            'lambda_reg': 10.0,
            'n_rounds': 1000
        },
        {
            'name': 'Neural - UMAP (10) - λ=10.0',
            'reduce': 'umap',
            'bandit_type': 'neural',
            'lambda_reg': 10.0,
            'n_rounds': 1000
        },
    ]

    all_results = {}

    for config in configs:
        print("\n" + "=" * 70)
        print(f"CONFIG: {config['name']}")
        print("=" * 70)

        # Apply dimensionality reduction if specified
        if config['reduce'] == 'pca':
            print("\nApplying PCA to 50 dimensions...")
            bert_reduced, _ = reduce_embeddings(bert_item_embs,
                                                method='pca',
                                                n_components=50)
            simcse_reduced, _ = reduce_embeddings(simcse_item_embs,
                                                  method='pca',
                                                  n_components=50)
        elif config['reduce'] == 'umap':
            print("\nApplying UMAP to 10 dimensions...")
            try:
                bert_reduced, _ = reduce_embeddings(bert_item_embs,
                                                    method='umap',
                                                    n_components=10)
                simcse_reduced, _ = reduce_embeddings(simcse_item_embs,
                                                      method='umap',
                                                      n_components=10)
            except ImportError:
                print("⚠️  UMAP not installed, skipping this config")
                print("   Install with: pip install umap-learn")
                continue
        else:
            bert_reduced = bert_item_embs
            simcse_reduced = simcse_item_embs

        # Compute user representations (concatenate last 2 items)
        print("\nComputing user representations (concat last 2 items)...")
        bert_user_reprs = compute_user_history_concat(
            dataset['train_histories'], bert_reduced)
        simcse_user_reprs = compute_user_history_concat(
            dataset['train_histories'], simcse_reduced)

        print(
            f"User representation dim: {len(next(iter(bert_user_reprs.values())))}"
        )
        print(f"Item dim: {len(next(iter(bert_reduced.values())))}")
        print(
            f"Total feature dim: {len(next(iter(bert_user_reprs.values()))) + len(next(iter(bert_reduced.values())))}"
        )

        # Run BERT bandit
        print("\n--- BERT Bandit ---")
        bert_results = run_ranking_bandit(dataset,
                                          bert_reduced,
                                          bert_user_reprs,
                                          n_rounds=config['n_rounds'],
                                          n_candidates=50,
                                          bandit_type=config['bandit_type'],
                                          hidden_dim=100,
                                          lambda_reg=config['lambda_reg'],
                                          weight_decay=0.01,
                                          use_diagonal=True,
                                          seed=42)

        # Run SimCSE bandit
        print("\n--- SimCSE Bandit ---")
        simcse_results = run_ranking_bandit(dataset,
                                            simcse_reduced,
                                            simcse_user_reprs,
                                            n_rounds=config['n_rounds'],
                                            n_candidates=50,
                                            bandit_type=config['bandit_type'],
                                            hidden_dim=100,
                                            lambda_reg=config['lambda_reg'],
                                            weight_decay=0.01,
                                            use_diagonal=True,
                                            seed=42)

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nBERT:")
        print(f"  Mean Rank: {bert_results['mean_rank']:.2f} / 50")
        print(f"  MRR: {bert_results['mrr']:.4f}")
        print(f"  Hit@1: {bert_results['hit@1']*100:.1f}%")
        print(f"  Hit@5: {bert_results['hit@5']*100:.1f}%")
        print(f"  Hit@10: {bert_results['hit@10']*100:.1f}%")

        print(f"\nSimCSE:")
        print(f"  Mean Rank: {simcse_results['mean_rank']:.2f} / 50")
        print(f"  MRR: {simcse_results['mrr']:.4f}")
        print(f"  Hit@1: {simcse_results['hit@1']*100:.1f}%")
        print(f"  Hit@5: {simcse_results['hit@5']*100:.1f}%")
        print(f"  Hit@10: {simcse_results['hit@10']*100:.1f}%")

        # Check if learning happened
        print("\n" + "-" * 70)
        bert_learning = bert_results['mean_rank'] < 24.5
        simcse_learning = simcse_results['mean_rank'] < 24.5

        print(
            f"{'✓' if bert_learning else '✗'} BERT: {'Learning!' if bert_learning else 'Not learning (random)'}"
        )
        print(
            f"{'✓' if simcse_learning else '✗'} SimCSE: {'Learning!' if simcse_learning else 'Not learning (random)'}"
        )

        if simcse_learning and bert_learning:
            improvement = (
                (bert_results['mean_rank'] - simcse_results['mean_rank']) /
                bert_results['mean_rank'] * 100)
            print(
                f"\n💡 SimCSE mean rank is {improvement:+.1f}% better than BERT"
            )

        # Store results
        all_results[config['name']] = {
            'bert': bert_results,
            'simcse': simcse_results
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: WHICH CONFIGS WORK?")
    print("=" * 70)

    for config_name, results in all_results.items():
        bert_works = results['bert']['mean_rank'] < 24.5
        simcse_works = results['simcse']['mean_rank'] < 24.5

        status = "✓ Both learning" if (bert_works and simcse_works) else \
                 "⚠️  Only one learning" if (bert_works or simcse_works) else \
                 "✗ Neither learning"

        print(f"{status:20s} - {config_name}")


if __name__ == "__main__":
    main()
