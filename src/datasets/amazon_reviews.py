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
        
        print(f"Loading {category} reviews and metadata...")
        self.reviews_df, self.items_dict = self._load_data()
        
        print("Building user interaction histories...")
        self.user_histories = self._build_user_histories()
        
        print("Splitting train/test...")
        self.train_histories, self.test_interactions = self._temporal_split()
        
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
        
        print(f"  Downloading reviews...")
        reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{self.category}",
            trust_remote_code=True,
            split="full"
        )
        
        reviews_df = pd.DataFrame(reviews)
        print(f"  Loaded {len(reviews_df)} reviews")
        
        reviews_df = reviews_df[
            (reviews_df['verified_purchase'] == True) & 
            (reviews_df['rating'].notna())
        ]
        print(f"  After filtering: {len(reviews_df)} verified reviews")
        
        reviews_df['reward'] = (reviews_df['rating'] >= 4.0).astype(int)
        reviews_df['item_id'] = reviews_df['parent_asin']
        
        user_counts = reviews_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.min_interactions].index
        reviews_df = reviews_df[reviews_df['user_id'].isin(active_users)]
        print(f"  Active users (>={self.min_interactions} reviews): {len(active_users)}")
        
        reviews_df = reviews_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Downloading item metadata...")
        items = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            trust_remote_code=True,
            split="full"
        )
        
        items_dict = {}
        for item in tqdm(items, desc="  Processing items"):
            asin = item.get('parent_asin', item.get('asin'))
            if not asin:
                continue
            
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
        
        reviewed_items = set(reviews_df['item_id'].unique())
        items_dict = {asin: item for asin, item in items_dict.items() 
                     if asin in reviewed_items}
        
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
        
        for user_id in histories:
            histories[user_id].sort(key=lambda x: x[2])
        
        return dict(histories)
    
    def _temporal_split(self, test_ratio=0.2) -> Tuple[Dict, List]:
        """Split each user's history: last 20% for testing"""
        train_histories = {}
        test_interactions = []
        
        for user_id, history in self.user_histories.items():
            split_idx = max(1, int(len(history) * (1 - test_ratio)))
            
            train_histories[user_id] = history[:split_idx]
            
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
