import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

class AmazonDataset:
    """
    Amazon product dataset using HuggingFace datasets (reliable streaming).
    
    Uses McAuley Amazon Reviews dataset hosted on HuggingFace.
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
        np.random.seed(seed)

        # Default categories
        if categories is None:
            categories = ['Electronics']

        self.categories = categories
        self.n_items_per_category = n_items_per_category

        # Load or download items
        self.items = self._load_or_download_items()

    def _download_category_hf(self, category: str, n_items: int) -> List[Dict]:
        """
        Download products from HuggingFace datasets (streaming mode).
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("HuggingFace datasets not installed. Using synthetic fallback.")
            return self._generate_synthetic_items(category, n_items)

        print(f"Downloading {n_items} items from {category} via HuggingFace...")
        items = []

        try:
            # Map category names to HuggingFace dataset names
            hf_category_map = {
                'Electronics': 'Electronics',
                'Clothing_Shoes_and_Jewelry': 'Clothing_Shoes_and_Jewelry', 
                'Home_and_Kitchen': 'Home_and_Kitchen',
                'Books': 'Books',
                'Sports_and_Outdoors': 'Sports_and_Outdoors'
            }
            hf_cat = hf_category_map.get(category, category)
            
            # Load in streaming mode
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_meta_{hf_cat}",
                split="full",
                streaming=True,
                trust_remote_code=True
            )

            for i, item in enumerate(tqdm(ds, total=n_items, desc=category)):
                if len(items) >= n_items:
                    break

                # Extract fields
                filtered_item = {
                    'item_id': item.get('parent_asin', f'{category}_{i}'),
                    'title': item.get('title', ''),
                    'description': self._extract_description(item),
                    'category': item.get('main_category', category),
                    'price': self._extract_price(item),
                    'avg_rating': item.get('average_rating', 3.5),
                    'images': item.get('images', {}).get('large', [])[:1]
                }

                # Skip if missing critical fields
                if not filtered_item['title']:
                    continue

                items.append(filtered_item)

            print(f"{category}: Downloaded {len(items)} items")
            return items

        except Exception as e:
            print(f"Error downloading {category}: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_items(category, n_items)

    def _generate_synthetic_items(self, category: str, n_items: int) -> List[Dict]:
        """Generate synthetic product data as fallback."""
        print(f"Generating {n_items} synthetic items for {category}...")
        
        product_types = {
            'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Camera', 'Speaker', 'Monitor', 'Keyboard'],
            'Clothing_Shoes_and_Jewelry': ['Shirt', 'Pants', 'Dress', 'Jacket', 'Shoes', 'Watch', 'Necklace'],
            'Home_and_Kitchen': ['Blender', 'Toaster', 'Pan', 'Knife Set', 'Vacuum', 'Lamp', 'Rug'],
            'Books': ['Novel', 'Textbook', 'Biography', 'Cookbook', 'Guide', 'Manual'],
            'Sports_and_Outdoors': ['Bike', 'Tent', 'Weights', 'Yoga Mat', 'Running Shoes', 'Backpack']
        }
        
        brands = ['ProTech', 'HomeMax', 'ValuePlus', 'PremiumGear', 'EcoSmart', 'UltraLite', 'MaxPower']
        adjectives = ['Premium', 'Professional', 'Compact', 'Wireless', 'Smart', 'Ultra', 'Classic', 'Modern']
        
        types = product_types.get(category, ['Product'])
        items = []
        
        for i in range(n_items):
            product_type = np.random.choice(types)
            brand = np.random.choice(brands)
            adj = np.random.choice(adjectives)
            
            title = f"{brand} {adj} {product_type} - Model {np.random.randint(100, 999)}"
            
            features = [
                f"High quality {product_type.lower()} for everyday use",
                f"Features {adj.lower()} design and construction", 
                f"Perfect for home or office",
                f"Easy to use and maintain",
                f"Comes with {np.random.randint(1, 3)} year warranty"
            ]
            
            items.append({
                'item_id': f'{category}_{i}',
                'title': title,
                'description': ' '.join(features[:3]),
                'category': category,
                'price': round(np.random.uniform(15, 500), 2),
                'avg_rating': round(np.random.uniform(3.0, 5.0), 1),
                'images': []
            })
        
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

        # Check cache (only use if non-empty)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    items = json.load(f)
                if len(items) > 0:
                    print(f"Loading from cache: {cache_file}")
                    print(f"Loaded {len(items)} items from cache")
                    return items
                else:
                    print("Cache is empty, re-downloading...")
            except json.JSONDecodeError:
                print("Cache corrupted, re-downloading...")

        # Download using HuggingFace
        all_items = []
        for category in self.categories:
            items = self._download_category_hf(category, self.n_items_per_category)
            all_items.extend(items)

            # Clear memory
            import gc
            gc.collect()

        # Save to cache
        if all_items:
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

    def compute_reward_prob(self, item_idx: int, user_embedding: np.ndarray, item_embeddings: np.ndarray) -> float:
        """
        Compute expected reward probability based on embedding similarity.
        
        This directly tests the embedding geometry hypothesis:
        - Good embeddings (high d_eff) = better similarity estimates = better recommendations
        
        Args:
            item_idx: Index of item in dataset
            user_embedding: User preference embedding (normalized)
            item_embeddings: All item embeddings matrix
            
        Returns:
            reward_prob: Expected reward in [0, 1]
        """
        item = self.items[item_idx]
        
        # Embedding similarity (primary signal - 60% weight)
        item_emb = item_embeddings[item_idx]
        item_emb = item_emb / (np.linalg.norm(item_emb) + 1e-8)
        similarity = np.dot(user_embedding, item_emb)
        similarity_score = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]
        
        # Item quality (rating - 25% weight)
        rating_score = item['avg_rating'] / 5.0
        
        # Price appropriateness (15% weight)
        price = item['price']
        if 20 <= price <= 100:
            price_score = 1.0
        elif price < 20:
            price_score = 0.6
        else:
            price_score = max(0.2, 1.0 - (price - 100) / 300)
        
        # Weighted combination
        reward_prob = (
            0.60 * similarity_score +
            0.25 * rating_score +
            0.15 * price_score
        )
        
        return float(np.clip(reward_prob, 0, 1))
    
    def sample_reward(self, item_idx: int, user_embedding: np.ndarray, item_embeddings: np.ndarray) -> float:
        """Sample binary reward from reward probability."""
        prob = self.compute_reward_prob(item_idx, user_embedding, item_embeddings)
        return 1.0 if np.random.random() < prob else 0.0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]