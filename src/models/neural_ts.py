import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


class NeuralTSBandit(nn.Module):
    """
    Neural Thompson Sampling with diagonal approximation.
    
    Based on: https://github.com/ZeroWeight/NeuralTS/
    
    Uses neural network to estimate reward function with uncertainty
    quantification via neural tangent kernel approximation.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 768, 
        hidden_dim: int = 100,
        lambda_reg: float = 1.0,
        nu: float = 1.0,
        lr: float = 0.01
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.lr = lr
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.num_params = sum(p.numel() for p in self.network.parameters())
        self.U = torch.eye(self.num_params)
        
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.U = self.U.to(self.device)
        
        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
    
    def forward(
        self, 
        embedding: torch.Tensor, 
        compute_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute mean and variance for Thompson Sampling.
        
        Args:
            embedding: (embedding_dim,) or (batch, embedding_dim)
            compute_variance: Whether to compute posterior variance
            
        Returns:
            mu: Mean reward prediction
            sigma: Standard deviation (if compute_variance=True)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.to(self.device)
        
        embedding.requires_grad_(True)
        mu = self.network(embedding).squeeze(-1)
        
        if not compute_variance:
            return mu, None
        
        try:
            grads = torch.autograd.grad(
                mu.sum(), 
                self.network.parameters(),
                create_graph=False,
                retain_graph=True
            )
            grad_vec = torch.cat([g.view(-1) for g in grads])
            
            U_inv = torch.inverse(self.U + torch.eye(self.num_params, device=self.device) * 1e-4)
            sigma_sq = (grad_vec @ U_inv @ grad_vec) / self.hidden_dim
            sigma = torch.sqrt(torch.clamp(self.lambda_reg * self.nu * sigma_sq, min=1e-6))
        except Exception:
            sigma = torch.tensor(0.1, device=self.device)
        
        return mu, sigma
    
    def select_action(
        self, 
        candidate_embeddings: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, np.ndarray]:
        """
        Thompson Sampling action selection.
        
        Args:
            candidate_embeddings: (K, embedding_dim) K candidate items
            deterministic: If True, select argmax of means
            
        Returns:
            selected_idx: Index of selected item
            sampled_rewards: (K,) Sampled/predicted rewards
        """
        candidates_t = torch.tensor(candidate_embeddings, dtype=torch.float32, device=self.device)
        
        sampled_rewards = []
        
        with torch.no_grad():
            for emb in candidates_t:
                mu, sigma = self.forward(emb, compute_variance=not deterministic)
                
                if deterministic or sigma is None:
                    sampled_rewards.append(mu.item())
                else:
                    reward_sample = torch.normal(mu, sigma)
                    sampled_rewards.append(reward_sample.item())
        
        sampled_rewards = np.array(sampled_rewards)
        selected_idx = int(np.argmax(sampled_rewards))
        
        return selected_idx, sampled_rewards
    
    def update(self, embedding: np.ndarray, reward: float):
        """
        Update network parameters via SGD and update U matrix.
        
        Args:
            embedding: Selected item embedding
            reward: Observed reward (0 to 1)
        """
        self.total_rounds += 1
        self.rewards_history.append(reward)
        self.cumulative_reward += reward
        
        embedding_t = torch.tensor(embedding, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        embedding_t.requires_grad_(True)
        pred = self.network(embedding_t.unsqueeze(0)).squeeze()
        
        reg_loss = sum(p.norm()**2 for p in self.network.parameters())
        loss = (reward_t - pred)**2 + (self.lambda_reg / max(self.total_rounds, 1)) * reg_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        try:
            with torch.no_grad():
                pred_new = self.network(embedding_t.unsqueeze(0)).squeeze()
            
            grads = torch.autograd.grad(
                pred_new, 
                self.network.parameters(),
                create_graph=False
            )
            grad_vec = torch.cat([g.view(-1) for g in grads])
            
            self.U = self.U + torch.outer(grad_vec, grad_vec)
        except Exception:
            pass
    
    def get_cumulative_regret(self, optimal_rewards: Optional[List[float]] = None) -> np.ndarray:
        """Compute cumulative regret."""
        if optimal_rewards is None:
            optimal_rewards = [1.0] * len(self.rewards_history)
        
        regrets = np.array(optimal_rewards) - np.array(self.rewards_history)
        return np.cumsum(regrets)
    
    def reset(self):
        """Reset bandit state."""
        self.U = torch.eye(self.num_params, device=self.device)
        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
        
        for layer in self.network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SimpleNeuralBandit:
    """
    Simplified neural bandit for faster experimentation.
    Uses numpy for computation without GPU requirements.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 768, 
        hidden_dim: int = 100,
        lr: float = 0.01,
        exploration_weight: float = 1.0
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.exploration_weight = exploration_weight
        
        scale = np.sqrt(2.0 / embedding_dim)
        self.W1 = np.random.randn(embedding_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale
        self.b2 = np.zeros(1)
        
        self.feature_count = np.ones(hidden_dim)
        
        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
    
    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass returning prediction and hidden features."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        return y.squeeze(), h
    
    def select_action(
        self, 
        candidate_embeddings: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, np.ndarray]:
        """Select action using UCB-style exploration."""
        scores = []
        
        for emb in candidate_embeddings:
            pred, h = self._forward(emb)
            
            if deterministic:
                score = pred
            else:
                uncertainty = self.exploration_weight * np.sqrt(
                    np.sum(h**2 / (self.feature_count + 1e-6))
                )
                score = pred + uncertainty * np.random.randn()
            
            scores.append(score)
        
        scores = np.array(scores)
        selected_idx = int(np.argmax(scores))
        
        return selected_idx, scores
    
    def update(self, embedding: np.ndarray, reward: float):
        """Update network via simple gradient descent."""
        self.total_rounds += 1
        self.rewards_history.append(reward)
        self.cumulative_reward += reward
        
        pred, h = self._forward(embedding)
        error = reward - pred
        
        grad_W2 = np.outer(h, error)
        grad_b2 = error
        
        grad_h = error * self.W2.squeeze()
        grad_h = grad_h * (h > 0)
        grad_W1 = np.outer(embedding, grad_h)
        grad_b1 = grad_h
        
        self.W2 += self.lr * grad_W2
        self.b2 += self.lr * grad_b2
        self.W1 += self.lr * grad_W1
        self.b1 += self.lr * grad_b1
        
        self.feature_count += h**2
    
    def reset(self):
        """Reset bandit state."""
        scale = np.sqrt(2.0 / self.embedding_dim)
        self.W1 = np.random.randn(self.embedding_dim, self.hidden_dim) * scale
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, 1) * scale
        self.b2 = np.zeros(1)
        self.feature_count = np.ones(self.hidden_dim)
        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0
