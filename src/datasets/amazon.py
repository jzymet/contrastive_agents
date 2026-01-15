import gzip
import json
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

class AmazonDataset:
    """
    Real Amazon product dataset with streaming download.

    Uses Amazon Reviews 2023 dataset from McAuley lab:
    https://amazon-reviews-2023.github.io/
    """

    def __init__(
        self,
        categories: List[str] = None,
        n_items_per_category: int = 3333,
        cache_dir: str = 'data/amazon',
        seed: int = 42
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Default categories
        if categories is None:
            categories = ['Electronics', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen']

        self.categories = categories
        self.n_items_per_category = n_items_per_category

        # Load or download items
        self.items = self._load_or_download_items()

    def _download_category_streaming(self, category: str, n_items: int) -> List[Dict]:
        """
        Stream-download products from one category.
        Stops at n_items (doesn't download full dataset).
        """
        url = f"https://amazon-reviews-2023.github.io/data/{category}_metadata.jsonl.gz"

        print(f"Downloading {n_items} items from {category}...")
        items = []

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with gzip.open(response.raw, 'rt', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(tqdm(f, total=n_items, desc=category)):
                    if len(items) >= n_items:
                        break  # STOP EARLY

                    try:
                        item = json.loads(line)

                        # Filter to required fields only (save memory)
                        filtered_item = {
                            'item_id': item.get('asin', f'{category}_{line_num}'),
                            'title': item.get('title', ''),
                            'description': self._extract_description(item),
                            'category': item.get('main_category', category),
                            'price': self._extract_price(item),
                            'avg_rating': item.get('average_rating', 0.0),
                            'images': item.get('images', [])[:1]  # Keep first image only
                        }

                        # Skip if missing critical fields
                        if not filtered_item['title'] or not filtered_item['description']:
                            continue

                        items.append(filtered_item)

                    except (json.JSONDecodeError, KeyError) as e:
                        continue

        except Exception as e:
            print(f"Error downloading {category}: {e}")
            return []

        print(f"{category}: Downloaded {len(items)} items")
        return items

    def _extract_description(self, item: Dict) -> str:
        """Extract description from various possible fields."""
        # Try multiple fields
        if 'description' in item:
            desc = item['description']
            if isinstance(desc, list):
                return ' '.join(desc[:3])  # First 3 sentences
            return str(desc)

        if 'details' in item:
            return str(item['details'])

        if 'features' in item:
            features = item['features']
            if isinstance(features, list):
                return ' '.join(features[:5])
            return str(features)

        # Fallback: use title
        return item.get('title', '')

    def _extract_price(self, item: Dict) -> float:
        """Extract price, handling various formats."""
        if 'price' in item:
            try:
                price = item['price']
                if isinstance(price, str):
                    # Remove $ and commas
                    price = price.replace('$', '').replace(',', '')
                return float(price)
            except:
                pass

        # Default price
        return 50.0

    def _load_or_download_items(self) -> List[Dict]:
        """Load from cache or download."""
        cache_file = self.cache_dir / 'amazon_items.json'

        # Check cache
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                items = json.load(f)
            print(f"Loaded {len(items)} items from cache")
            return items

        # Download
        all_items = []
        for category in self.categories:
            items = self._download_category_streaming(category, self.n_items_per_category)
            all_items.extend(items)

            # Clear memory
            import gc
            gc.collect()

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(all_items, f, indent=2)
        print(f"Saved {len(all_items)} items to {cache_file}")

        return all_items

    def get_item_texts(self) -> List[str]:
        """Get text for embedding (title + description)."""
        return [f"{item['title']}. {item['description']}" for item in self.items]

    def get_item_image_urls(self) -> List[str]:
        """Get image URLs for multimodal embeddings."""
        urls = []
        for item in self.items:
            if item['images']:
                urls.append(item['images'][0]['large'])
            else:
                urls.append(None)
        return urls

    def compute_true_reward(self, item_idx: int, context_items: List[int]) -> float:
        """
        Ground truth reward function (unknown to bandit).

        Based on:
        - Category match with context (50% weight)
        - Item rating (30% weight)  
        - Price appropriateness (20% weight)
        """
        item = self.items[item_idx]

        # Category match
        if context_items:
            context_categories = [self.items[i]['category'] for i in context_items]
            category_match = 1.0 if item['category'] in context_categories else 0.0
        else:
            category_match = 0.5  # Neutral if no context

        # Rating (normalize to [0, 1])
        rating_score = item['avg_rating'] / 5.0

        # Price (prefer mid-range $20-100)
        price = item['price']
        if 20 <= price <= 100:
            price_score = 1.0
        elif price < 20:
            price_score = 0.5
        else:
            price_score = max(0.0, 1.0 - (price - 100) / 200)

        # Weighted combination
        reward_prob = (
            0.5 * category_match +
            0.3 * rating_score +
            0.2 * price_score
        )

        # Binarize with probability = reward_prob
        return 1.0 if np.random.random() < reward_prob else 0.0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]