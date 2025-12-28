import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class AmazonElectronicsDataset:
    def __init__(self, n_items: int = 10000, n_users: int = 1000, seed: int = 42):
        np.random.seed(seed)
        self.n_items = n_items
        self.n_users = n_users
        
        self.items = self._generate_synthetic_items()
        self.users = self._generate_synthetic_users()
        self.interactions = self._generate_interactions()
    
    def _generate_synthetic_items(self) -> List[Dict]:
        categories = [
            "Smartphones", "Laptops", "Tablets", "Headphones", "Speakers",
            "Cameras", "TVs", "Gaming", "Wearables", "Accessories",
            "Smart Home", "Audio Equipment", "Computer Parts", "Networking", "Storage"
        ]
        
        brands = [
            "TechPro", "ElectroniX", "SmartGear", "DigitalEdge", "PowerTech",
            "InnovateCo", "FutureTech", "PrimeElec", "UltraGadget", "MegaTech"
        ]
        
        adjectives = ["Premium", "Ultra", "Pro", "Elite", "Essential", "Advanced", "Basic", "Smart", "Wireless", "Portable"]
        nouns = ["Device", "System", "Unit", "Kit", "Set", "Package", "Bundle", "Edition", "Series", "Collection"]
        
        items = []
        for i in range(self.n_items):
            category = categories[i % len(categories)]
            brand = brands[i % len(brands)]
            adj = adjectives[np.random.randint(len(adjectives))]
            noun = nouns[np.random.randint(len(nouns))]
            
            title = f"{brand} {adj} {category} {noun} - Model {i}"
            description = f"High-quality {category.lower()} from {brand}. Features include advanced technology, " \
                         f"durable construction, and excellent performance. Perfect for {category.lower()} enthusiasts."
            
            price = round(np.random.uniform(20, 2000), 2)
            rating = round(np.random.uniform(3.0, 5.0), 1)
            
            items.append({
                "item_id": i,
                "title": title,
                "description": description,
                "category": category,
                "brand": brand,
                "price": price,
                "rating": rating
            })
        
        return items
    
    def _generate_synthetic_users(self) -> List[Dict]:
        users = []
        for i in range(self.n_users):
            preferred_categories = np.random.choice(
                ["Smartphones", "Laptops", "Tablets", "Headphones", "Speakers",
                 "Cameras", "TVs", "Gaming", "Wearables", "Accessories"],
                size=np.random.randint(1, 4),
                replace=False
            ).tolist()
            
            price_sensitivity = np.random.uniform(0.2, 1.0)
            brand_loyalty = np.random.uniform(0.0, 1.0)
            
            users.append({
                "user_id": i,
                "preferred_categories": preferred_categories,
                "price_sensitivity": price_sensitivity,
                "brand_loyalty": brand_loyalty
            })
        
        return users
    
    def _generate_interactions(self) -> Dict[int, List[Tuple[int, float]]]:
        interactions = {u["user_id"]: [] for u in self.users}
        
        for user in self.users:
            n_interactions = np.random.randint(5, 50)
            
            for _ in range(n_interactions):
                item_idx = np.random.randint(self.n_items)
                item = self.items[item_idx]
                
                base_prob = 0.5
                if item["category"] in user["preferred_categories"]:
                    base_prob += 0.3
                
                price_factor = 1.0 - (item["price"] / 2000) * user["price_sensitivity"]
                rating_factor = (item["rating"] - 3.0) / 2.0
                
                reward_prob = min(1.0, max(0.0, base_prob + price_factor * 0.1 + rating_factor * 0.1))
                reward = 1.0 if np.random.random() < reward_prob else 0.0
                
                interactions[user["user_id"]].append((item_idx, reward))
        
        return interactions
    
    def get_item_texts(self) -> List[str]:
        return [f"{item['title']}. {item['description']}" for item in self.items]
    
    def get_user_context(self, user_id: int) -> Dict:
        return self.users[user_id]
    
    def get_reward(self, user_id: int, item_id: int) -> float:
        user = self.users[user_id]
        item = self.items[item_id]
        
        base_prob = 0.3
        if item["category"] in user["preferred_categories"]:
            base_prob += 0.4
        
        price_factor = 1.0 - (item["price"] / 2000) * user["price_sensitivity"]
        rating_factor = (item["rating"] - 3.0) / 2.0
        
        reward_prob = min(1.0, max(0.0, base_prob + price_factor * 0.1 + rating_factor * 0.1))
        return 1.0 if np.random.random() < reward_prob else 0.0
    
    def simulate_session(self, user_id: int, n_steps: int = 10) -> List[Tuple[int, float]]:
        session = []
        for _ in range(n_steps):
            item_id = np.random.randint(self.n_items)
            reward = self.get_reward(user_id, item_id)
            session.append((item_id, reward))
        return session
    
    def __len__(self) -> int:
        return self.n_items
