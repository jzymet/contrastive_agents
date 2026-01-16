"""
Linear Kernel Contextual Bandit

Uses direct dot product: reward ≈ (user_emb * item_emb)^T theta
This directly tests whether embedding geometry captures preferences.
"""

import numpy as np
from typing import Dict, List, Tuple

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
            self.B = np.eye(self.d)
            self.mu = np.zeros(self.d)
            self.f = np.zeros(self.d)
            self.sigma_sq = 1.0
            
        elif algorithm == 'ucb':
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
        
        features = np.array([
            self._compute_feature(user_emb, item)
            for item in candidate_items
        ])
        
        if self.algorithm == 'ts':
            try:
                B_inv = np.linalg.inv(self.B)
                theta = np.random.multivariate_normal(self.mu, self.sigma_sq * B_inv)
            except:
                theta = self.mu
            
            rewards = features @ theta
            return int(np.argmax(rewards))
        
        elif self.algorithm == 'ucb':
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
            
            ucb_scores = []
            for x in features:
                mean = x @ theta
                std = np.sqrt(x @ A_inv @ x)
                ucb_scores.append(mean + self.alpha * std)
            
            return int(np.argmax(ucb_scores))
    
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
        
        item_ids = [item_id for item_id, _, _ in user_history]
        
        embs = [self.item_embeddings[item_id] for item_id in item_ids 
                if item_id in self.item_embeddings]
        
        if not embs:
            user_emb = np.zeros(self.embedding_dim)
        elif strategy == 'mean':
            user_emb = np.mean(embs, axis=0)
        elif strategy == 'weighted_mean':
            weights = [reward for _, reward, _ in user_history][:len(embs)]
            if sum(weights) == 0:
                user_emb = np.mean(embs, axis=0)
            else:
                user_emb = np.average(embs, axis=0, weights=weights)
        elif strategy == 'recent':
            user_emb = np.mean(embs[-5:], axis=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        norm = np.linalg.norm(user_emb)
        if norm > 1e-8:
            user_emb = user_emb / norm
        
        if user_id:
            self._cache[user_id] = user_emb
        
        return user_emb
    
    def clear_cache(self):
        """Clear the user embedding cache"""
        self._cache = {}
