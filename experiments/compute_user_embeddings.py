"""
Compute and cache user embeddings (average of purchased item embeddings).
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def compute_user_embeddings_concat(dataset, item_embs):
    """Concatenate first 2 purchased items instead of averaging."""
    user_embs = {}
    embedding_dim = len(next(iter(item_embs.values())))

    for user_id, history in tqdm(dataset['train_histories'].items()):
        item_ids = [item_id for item_id, _, _ in history]
        valid_embs = [
            item_embs[item_id] for item_id in item_ids if item_id in item_embs
        ]

        if len(valid_embs) >= 2:
            user_embs[user_id] = np.concatenate([valid_embs[0], valid_embs[1]])
        elif len(valid_embs) == 1:
            user_embs[user_id] = np.concatenate([valid_embs[0], valid_embs[0]])
        else:
            user_embs[user_id] = np.zeros(embedding_dim * 2)

    return user_embs


def compute_user_embeddings(dataset, item_embs):
    """
    Compute user embeddings as average of their purchased items.

    Args:
        dataset: Dict with 'train_histories' and 'test_interactions'
        item_embs: Dict mapping item_id -> embedding

    Returns:
        user_embs: Dict mapping user_id -> embedding
    """
    user_embs = {}

    for user_id, history in tqdm(dataset['train_histories'].items(),
                                 desc="Computing user embeddings"):
        # Get item IDs from history
        item_ids = [item_id for item_id, reward, timestamp in history]

        # Get embeddings for items that exist
        valid_embs = []
        for item_id in item_ids:
            if item_id in item_embs:
                valid_embs.append(item_embs[item_id])

        if len(valid_embs) > 0:
            # Average embeddings
            user_embs[user_id] = np.mean(valid_embs, axis=0)
        else:
            # Fallback: zero vector
            embedding_dim = len(next(iter(item_embs.values())))
            user_embs[user_id] = np.zeros(embedding_dim)

    return user_embs


def main():
    """Compute and cache user embeddings."""

    # Load dataset
    print("Loading dataset...")
    with open('../data/amazon_reviews/All_Beauty_processed.pkl', 'rb') as f:
        dataset = pickle.load(f)

    # Load item embeddings
    print("Loading item embeddings...")
    embeddings_dir = Path('../data/embeddings')

    with open(embeddings_dir / 'bert_embeddings.pkl', 'rb') as f:
        bert_item_embs = pickle.load(f)

    with open(embeddings_dir / 'simcse_embeddings.pkl', 'rb') as f:
        simcse_item_embs = pickle.load(f)

    # Compute user embeddings
    print("\nComputing BERT user embeddings...")
    bert_user_embs = compute_user_embeddings(dataset, bert_item_embs)

    print("Computing SimCSE user embeddings...")
    simcse_user_embs = compute_user_embeddings(dataset, simcse_item_embs)

    # Save
    print("\nSaving user embeddings...")
    with open(embeddings_dir / 'bert_user_embeddings.pkl', 'wb') as f:
        pickle.dump(bert_user_embs, f)

    with open(embeddings_dir / 'simcse_user_embeddings.pkl', 'wb') as f:
        pickle.dump(simcse_user_embs, f)

    print(f"\n✓ Saved {len(bert_user_embs)} BERT user embeddings")
    print(f"✓ Saved {len(simcse_user_embs)} SimCSE user embeddings")
    print(f"\nFiles saved to: {embeddings_dir}/")


if __name__ == "__main__":
    main()
