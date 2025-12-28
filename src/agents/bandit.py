import numpy as np
from typing import List, Tuple, Optional, Dict


class ThompsonSamplingBandit:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.total_pulls = 0
        self.arm_pulls = np.zeros(n_arms)
        self.rewards_history = []
        self.cumulative_reward = 0.0
    
    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float):
        self.alpha[arm] += reward
        self.beta[arm] += (1.0 - reward)
        
        self.arm_pulls[arm] += 1
        self.total_pulls += 1
        self.rewards_history.append(reward)
        self.cumulative_reward += reward
    
    def reset(self):
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.total_pulls = 0
        self.arm_pulls = np.zeros(self.n_arms)
        self.rewards_history = []
        self.cumulative_reward = 0.0
    
    def get_expected_rewards(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)
    
    def get_ucb_scores(self, c: float = 2.0) -> np.ndarray:
        means = self.get_expected_rewards()
        exploration = c * np.sqrt(np.log(self.total_pulls + 1) / (self.arm_pulls + 1))
        return means + exploration


class UCBBandit:
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
    
    def select_arm(self) -> int:
        if self.total_pulls < self.n_arms:
            return self.total_pulls
        
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_pulls) / (self.counts + 1e-5)
        )
        return int(np.argmax(ucb_values))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n
        self.total_pulls += 1
        self.rewards_history.append(reward)
        self.cumulative_reward += reward
    
    def reset(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_pulls = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0


class PersistentContextualBandit:
    def __init__(self, n_total_items: int, k: int = 500):
        self.n_total_items = n_total_items
        self.k = k
        
        self.item_alpha = np.ones(n_total_items)
        self.item_beta = np.ones(n_total_items)
        self.item_pulls = np.zeros(n_total_items)
        
        self.total_pulls = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
        
        self.current_candidates = None
        self.candidate_to_item_map = None
    
    def set_candidates(self, candidate_indices: np.ndarray):
        self.current_candidates = candidate_indices
        self.candidate_to_item_map = {i: idx for i, idx in enumerate(candidate_indices)}
    
    def select_arm(self) -> int:
        if self.current_candidates is None:
            raise ValueError("Call set_candidates first")
        
        candidate_samples = np.random.beta(
            self.item_alpha[self.current_candidates],
            self.item_beta[self.current_candidates]
        )
        return int(np.argmax(candidate_samples))
    
    def update(self, arm: int, reward: float):
        if self.current_candidates is None:
            raise ValueError("Call set_candidates first")
        
        item_idx = self.current_candidates[arm]
        
        self.item_alpha[item_idx] += reward
        self.item_beta[item_idx] += (1.0 - reward)
        self.item_pulls[item_idx] += 1
        
        self.total_pulls += 1
        self.rewards_history.append(reward)
        self.cumulative_reward += reward
    
    def get_item_stats(self, item_idx: int) -> Tuple[float, float, int]:
        expected = self.item_alpha[item_idx] / (self.item_alpha[item_idx] + self.item_beta[item_idx])
        variance = (self.item_alpha[item_idx] * self.item_beta[item_idx]) / \
                   ((self.item_alpha[item_idx] + self.item_beta[item_idx]) ** 2 * 
                    (self.item_alpha[item_idx] + self.item_beta[item_idx] + 1))
        return expected, variance, int(self.item_pulls[item_idx])
    
    def reset(self):
        self.item_alpha = np.ones(self.n_total_items)
        self.item_beta = np.ones(self.n_total_items)
        self.item_pulls = np.zeros(self.n_total_items)
        self.total_pulls = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
        self.current_candidates = None
        self.candidate_to_item_map = None


class ContextualBandit:
    def __init__(self, embedding_dim: int, n_candidate_arms: int = 500):
        self.embedding_dim = embedding_dim
        self.n_candidate_arms = n_candidate_arms
        self.bandit = None
    
    def initialize_episode(self, candidate_embeddings: np.ndarray):
        self.candidate_embeddings = candidate_embeddings
        self.bandit = ThompsonSamplingBandit(len(candidate_embeddings))
    
    def select_arm(self, context_embedding: Optional[np.ndarray] = None) -> int:
        if self.bandit is None:
            raise ValueError("Call initialize_episode first")
        
        if context_embedding is not None:
            similarities = self.candidate_embeddings @ context_embedding
            priors = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
            self.bandit.alpha = 1 + priors
        
        return self.bandit.select_arm()
    
    def update(self, arm: int, reward: float):
        if self.bandit is None:
            raise ValueError("Call initialize_episode first")
        self.bandit.update(arm, reward)
