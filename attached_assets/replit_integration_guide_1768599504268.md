# Integration Guide: Real User-Item Interactions for Amazon Bandit

## What You Need to Add to Your Codebase

### New Files to Add:

1. **`src/datasets/amazon_reviews.py`** - New dataset class for user-item interactions
2. **`src/models/linear_kernel_bandit.py`** - Linear kernel bandit implementation
3. **`experiments/02_amazon_real_users.ipynb`** - New notebook for real user experiments

### Files to Modify:

1. **`src/datasets/__init__.py`** - Add new dataset import
2. **`src/models/__init__.py`** - Add new bandit import

---

## Step-by-Step Integration

### Step 1: Add New Dataset Class

Create **`src/datasets/amazon_reviews.py`**:

```python
"""
Amazon Reviews 2023 Dataset with User-Item Interactions

Loads real user purchase histories from Amazon Reviews 2023.
Each user has a history of items they purchased/reviewed.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm

class AmazonReviewsDataset:
    """
    Process Amazon Reviews 2023 into user-item interaction format.
    
    Key differences from existing AmazonDataset:
    - Loads user reviews (interactions), not just item metadata
    - Builds user purchase histories
    - Splits into train/test temporally
    """
    
    def __init__(
        self, 
        category='All_Beauty',
        min_interactions_per_user=5,
        cache_dir='data/amazon_reviews'
    ):
        """
        Args:
            category: Amazon category (All_Beauty, Electronics, etc.)
            min_interactions_per_user: Filter users with fewer interactions
            cache_dir: Where to cache data
        """
        from pathlib import Path
        self.category = category
        self.min_interactions = min_interactions_per_user
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load from cache first
        cache_file = self.cache_dir / f"{category}_processed.pkl"
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                self.reviews_df = cached['reviews_df']
                self.items_dict = cached['items_dict']
                self.user_histories = cached['user_histories']
                self.train_histories = cached['train_histories']
                self.test_interactions = cached['test_interactions']
            print(f"Loaded {len(self.user_histories)} users, {len(self.items_dict)} items")
            return
        
        # Otherwise, load and process
        print(f"Loading {category} reviews and metadata...")
        self.reviews_df, self.items_dict = self._load_data()
        
        print("Building user interaction histories...")
        self.user_histories = self._build_user_histories()
        
        print("Splitting train/test...")
        self.train_histories, self.test_interactions = self._temporal_split()
        
        # Cache for next time
        print(f"Caching to {cache_file}...")
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'reviews_df': self.reviews_df,
                'items_dict': self.items_dict,
                'user_histories': self.user_histories,
                'train_histories': self.train_histories,
                'test_interactions': self.test_interactions,
            }, f)
        
        print(f"\nDataset loaded:")
        print(f"  Total users: {len(self.user_histories)}")
        print(f"  Total items: {len(self.items_dict)}")
        print(f"  Total interactions: {len(self.reviews_df)}")
        print(f"  Train users: {len(self.train_histories)}")
        print(f"  Test interactions: {len(self.test_interactions)}")
    
    def _load_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load reviews and item metadata from HuggingFace"""
        
        # Load reviews (user-item interactions)
        print(f"  Downloading reviews...")
        reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{self.category}",
            trust_remote_code=True,
            split="full"
        )
        
        # Convert to DataFrame
        reviews_df = pd.DataFrame(reviews)
        print(f"  Loaded {len(reviews_df)} reviews")
        
        # Filter: only verified purchases with ratings
        reviews_df = reviews_df[
            (reviews_df['verified_purchase'] == True) & 
            (reviews_df['rating'].notna())
        ]
        print(f"  After filtering: {len(reviews_df)} verified reviews")
        
        # Binary reward: rating >= 4 is positive
        reviews_df['reward'] = (reviews_df['rating'] >= 4.0).astype(int)
        
        # Use parent_asin as item ID
        reviews_df['item_id'] = reviews_df['parent_asin']
        
        # Filter users with at least min_interactions
        user_counts = reviews_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.min_interactions].index
        reviews_df = reviews_df[reviews_df['user_id'].isin(active_users)]
        print(f"  Active users (>={self.min_interactions} reviews): {len(active_users)}")
        
        # Sort by timestamp
        reviews_df = reviews_df.sort_values('timestamp').reset_index(drop=True)
        
        # Load item metadata
        print(f"  Downloading item metadata...")
        items = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            trust_remote_code=True,
            split="full"
        )
        
        # Process items
        items_dict = {}
        for item in tqdm(items, desc="  Processing items"):
            asin = item.get('parent_asin', item.get('asin'))
            if not asin:
                continue
            
            # Build description
            desc_parts = []
            if item.get('title'):
                desc_parts.append(item['title'])
            if item.get('features'):
                features = item['features']
                if isinstance(features, list):
                    desc_parts.extend(features[:3])
            if item.get('description'):
                desc = item['description']
                if isinstance(desc, list):
                    desc_parts.extend(desc[:2])
                elif desc:
                    desc_parts.append(str(desc))
            
            description = '. '.join(str(p) for p in desc_parts if p)
            
            # Get image URL
            image_url = None
            if item.get('images'):
                imgs = item['images']
                if isinstance(imgs, dict) and imgs.get('large'):
                    large = imgs['large']
                    image_url = large[0] if isinstance(large, list) else large
            
            items_dict[asin] = {
                'asin': asin,
                'title': item.get('title', ''),
                'description': description,
                'category': item.get('main_category', ''),
                'price': item.get('price', 'N/A'),
                'avg_rating': item.get('average_rating', 0.0),
                'image_url': image_url,
            }
        
        # Keep only items in reviews
        reviewed_items = set(reviews_df['item_id'].unique())
        items_dict = {asin: item for asin, item in items_dict.items() 
                     if asin in reviewed_items}
        
        # Filter reviews to only include items with metadata
        reviews_df = reviews_df[reviews_df['item_id'].isin(items_dict.keys())]
        
        return reviews_df, items_dict
    
    def _build_user_histories(self) -> Dict[str, List[Tuple[str, int, int]]]:
        """Build chronological purchase history for each user"""
        histories = defaultdict(list)
        
        for _, row in self.reviews_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            reward = row['reward']
            timestamp = row['timestamp']
            
            histories[user_id].append((item_id, reward, timestamp))
        
        # Sort by timestamp
        for user_id in histories:
            histories[user_id].sort(key=lambda x: x[2])
        
        return dict(histories)
    
    def _temporal_split(self, test_ratio=0.2) -> Tuple[Dict, List]:
        """Split each user's history: last 20% for testing"""
        train_histories = {}
        test_interactions = []
        
        for user_id, history in self.user_histories.items():
            split_idx = max(1, int(len(history) * (1 - test_ratio)))
            
            # Training history
            train_histories[user_id] = history[:split_idx]
            
            # Test interactions
            for item_id, reward, ts in history[split_idx:]:
                test_interactions.append((user_id, item_id, reward))
        
        return train_histories, test_interactions
    
    def get_item_texts(self) -> Dict[str, str]:
        """Get text descriptions for all items"""
        return {asin: item['description'] for asin, item in self.items_dict.items()}
    
    def get_item_asins(self) -> List[str]:
        """Get list of all item IDs"""
        return list(self.items_dict.keys())
    
    def __len__(self):
        return len(self.items_dict)
```

### Step 2: Add Linear Kernel Bandit

Create **`src/models/linear_kernel_bandit.py`**:

```python
"""
Linear Kernel Contextual Bandit

Uses direct dot product: reward ≈ (user_emb * item_emb)^T theta
This directly tests whether embedding geometry captures preferences.
"""

import numpy as np
from typing import Dict

class LinearKernelBandit:
    """
    Thompson Sampling for linear kernel bandit
    
    Feature: user_emb * item_emb (element-wise product)
    Reward: theta^T feature
    """
    
    def __init__(self, embedding_dim: int, algorithm='ts'):
        self.d = embedding_dim
        self.algorithm = algorithm
        
        if algorithm == 'ts':
            # Thompson Sampling
            self.B = np.eye(self.d)  # Precision
            self.mu = np.zeros(self.d)  # Mean
            self.f = np.zeros(self.d)  # sum(feature * reward)
            self.sigma_sq = 1.0
            
        elif algorithm == 'ucb':
            # LinUCB
            self.A = np.eye(self.d)
            self.b = np.zeros(self.d)
            self.alpha = 1.0
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _compute_feature(self, user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
        """Feature = user * item (element-wise)"""
        return user_emb * item_emb
    
    def select_arm(
        self, 
        user_emb: np.ndarray,
        candidate_items: np.ndarray
    ) -> int:
        """Select best item for user"""
        K = len(candidate_items)
        
        # Compute features
        features = np.array([
            self._compute_feature(user_emb, item)
            for item in candidate_items
        ])
        
        if self.algorithm == 'ts':
            # Sample theta from posterior
            try:
                B_inv = np.linalg.inv(self.B)
                theta = np.random.multivariate_normal(self.mu, self.sigma_sq * B_inv)
            except:
                theta = self.mu
            
            rewards = features @ theta
            return np.argmax(rewards)
        
        elif self.algorithm == 'ucb':
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
            
            ucb_scores = []
            for x in features:
                mean = x @ theta
                std = np.sqrt(x @ A_inv @ x)
                ucb_scores.append(mean + self.alpha * std)
            
            return np.argmax(ucb_scores)
    
    def update(self, user_emb: np.ndarray, item_emb: np.ndarray, reward: float):
        """Update with observation"""
        x = self._compute_feature(user_emb, item_emb)
        
        if self.algorithm == 'ts':
            self.B += np.outer(x, x) / self.sigma_sq
            self.f += x * reward
            try:
                self.mu = np.linalg.solve(self.B, self.f)
            except:
                pass
        
        elif self.algorithm == 'ucb':
            self.A += np.outer(x, x)
            self.b += x * reward


class UserEmbeddingManager:
    """Compute user embeddings from purchase history"""
    
    def __init__(self, item_embeddings: Dict[str, np.ndarray]):
        self.item_embeddings = item_embeddings
        self.embedding_dim = next(iter(item_embeddings.values())).shape[0]
        self._cache = {}
    
    def get_user_embedding(
        self, 
        user_history: list,
        strategy='mean',
        user_id=None
    ) -> np.ndarray:
        """
        Compute user embedding from purchase history
        
        Args:
            user_history: [(item_id, reward, timestamp), ...]
            strategy: 'mean' | 'weighted_mean' | 'recent'
            user_id: For caching
        """
        if user_id and user_id in self._cache:
            return self._cache[user_id]
        
        # Get item IDs
        item_ids = [item_id for item_id, _, _ in user_history]
        
        # Get embeddings
        embs = [self.item_embeddings[item_id] for item_id in item_ids 
                if item_id in self.item_embeddings]
        
        if not embs:
            user_emb = np.zeros(self.embedding_dim)
        elif strategy == 'mean':
            user_emb = np.mean(embs, axis=0)
        elif strategy == 'weighted_mean':
            weights = [reward for _, reward, _ in user_history][:len(embs)]
            user_emb = np.average(embs, axis=0, weights=weights)
        elif strategy == 'recent':
            user_emb = np.mean(embs[-5:], axis=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Normalize
        norm = np.linalg.norm(user_emb)
        if norm > 1e-8:
            user_emb = user_emb / norm
        
        if user_id:
            self._cache[user_id] = user_emb
        
        return user_emb
```

### Step 3: Update __init__.py Files

**`src/datasets/__init__.py`**:
```python
from .amazon import AmazonDataset
from .amazon_reviews import AmazonReviewsDataset  # ADD THIS
from .toolbench import ToolBenchDataset
from .math_data import MathDataset

__all__ = [
    'AmazonDataset',
    'AmazonReviewsDataset',  # ADD THIS
    'ToolBenchDataset',
    'MathDataset'
]
```

**`src/models/__init__.py`**:
```python
from .neural_ts import NeuralThompsonSampling, SimpleNeuralBandit
from .reward_transformer import RewardTransformer
from .linear_kernel_bandit import LinearKernelBandit, UserEmbeddingManager  # ADD THIS

__all__ = [
    'NeuralThompsonSampling',
    'SimpleNeuralBandit',
    'RewardTransformer',
    'LinearKernelBandit',  # ADD THIS
    'UserEmbeddingManager',  # ADD THIS
]
```

### Step 4: Create New Experiment Notebook

Create **`experiments/02_amazon_real_users.ipynb`**:

```python
"""
Amazon Real User-Item Interactions Experiment

Tests BERT vs SimCSE on real user purchase data
"""

# Cell 1: Imports
import sys
sys.path.insert(0, '..')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.datasets import AmazonReviewsDataset
from src.embeddings import get_extractor
from src.models import LinearKernelBandit, UserEmbeddingManager
from src.analysis import compute_effective_dimension, compute_eigenvalue_spectrum

# Cell 2: Load Dataset
dataset = AmazonReviewsDataset(
    category='All_Beauty',
    min_interactions_per_user=5
)

print(f"Users: {len(dataset.user_histories)}")
print(f"Items: {len(dataset.items_dict)}")
print(f"Test interactions: {len(dataset.test_interactions)}")

# Cell 3: Compute Embeddings
item_texts = dataset.get_item_texts()
item_asins = dataset.get_item_asins()
texts = [item_texts[asin] for asin in item_asins]

print("Computing BERT embeddings...")
bert_enc = get_extractor('bert')
bert_embs = bert_enc.encode(texts)
bert_item_embs = {asin: bert_embs[i] for i, asin in enumerate(item_asins)}

print("Computing SimCSE embeddings...")
simcse_enc = get_extractor('simcse')
simcse_embs = simcse_enc.encode(texts)
simcse_item_embs = {asin: simcse_embs[i] for i, asin in enumerate(item_asins)}

# Cell 4: Compute Effective Dimensions
bert_eigs, _ = compute_eigenvalue_spectrum(bert_embs)
simcse_eigs, _ = compute_eigenvalue_spectrum(simcse_embs)

bert_deff = compute_effective_dimension(bert_eigs)
simcse_deff = compute_effective_dimension(simcse_eigs)

print(f"BERT d_eff: {bert_deff:.1f}")
print(f"SimCSE d_eff: {simcse_deff:.1f}")

# Cell 5: Compute User Embeddings
bert_user_mgr = UserEmbeddingManager(bert_item_embs)
simcse_user_mgr = UserEmbeddingManager(simcse_item_embs)

print("Computing user embeddings...")
bert_user_embs = {
    user_id: bert_user_mgr.get_user_embedding(history, user_id=user_id)
    for user_id, history in tqdm(dataset.train_histories.items())
}

simcse_user_embs = {
    user_id: simcse_user_mgr.get_user_embedding(history, user_id=user_id)
    for user_id, history in tqdm(dataset.train_histories.items())
}

# Cell 6: Run Bandit Experiments
def run_bandit(dataset, item_embs, user_embs, n_rounds=3000):
    """Run linear kernel bandit"""
    # ... (copy from linear_kernel_bandit.py)
    pass

print("Running BERT bandit...")
bert_results = run_bandit(dataset, bert_item_embs, bert_user_embs)

print("Running SimCSE bandit...")
simcse_results = run_bandit(dataset, simcse_item_embs, simcse_user_embs)

# Cell 7: Results
print(f"\nFinal Cumulative Regret:")
print(f"  BERT: {bert_results['final_regret']:.1f}")
print(f"  SimCSE: {simcse_results['final_regret']:.1f}")

improvement = 100 * (bert_results['final_regret'] - simcse_results['final_regret']) / bert_results['final_regret']
print(f"  Improvement: {improvement:.1f}%")
```

---

## Summary: What to Give Replit

### Files to Add:
1. `src/datasets/amazon_reviews.py` (new dataset class)
2. `src/models/linear_kernel_bandit.py` (new bandit)
3. `experiments/02_amazon_real_users.ipynb` (new notebook)

### Files to Modify:
1. `src/datasets/__init__.py` (add import)
2. `src/models/__init__.py` (add import)

### Replit Prompt:
"Integrate the new Amazon Reviews dataset and linear kernel bandit into the existing codebase. The new dataset loads user-item interactions from Amazon Reviews 2023, computes user embeddings from purchase history, and runs a linear kernel bandit that tests whether embedding geometry captures real user preferences. Make sure all imports work and the new notebook runs end-to-end."

---

## Expected Runtime

On All_Beauty dataset:
- Loading data: ~5-10 minutes (first time, then cached)
- Computing embeddings: ~10-15 minutes
- Running bandit (3000 rounds): ~5-10 minutes

**Total: ~30 minutes for full experiment**
