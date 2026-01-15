import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List, Literal


class NeuralContextualBandit:
    """
    Neural bandit with Thompson Sampling OR UCB exploration.

    Adapted from NeuralTS: https://github.com/ZeroWeight/NeuralTS
    """

    def __init__(
            self,
            embedding_dim: int = 768,
            hidden_dim: int = 100,
            lambda_reg: float = 1.0,
            nu: float = 1.0,
            learning_rate: float = 0.01,
            algorithm: Literal['ts', 'ucb'] = 'ts',
            ucb_alpha: float = 1.0,
            use_cuda: bool = False):
        self.d = embedding_dim
        self.m = hidden_dim
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.lr = learning_rate
        self.algorithm = algorithm
        self.ucb_alpha = ucb_alpha

        self.device = torch.device(
            'cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)

        self.num_params = sum(p.numel() for p in self.network.parameters())
        self.U = torch.eye(self.num_params).to(self.device)

        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0

    def forward(self, embedding: torch.Tensor, compute_variance: bool = True):
        """Forward pass with optional uncertainty estimation."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        mu = self.network(embedding).squeeze(-1)

        if not compute_variance:
            return mu, None

        self.network.zero_grad()
        mu_single = mu[0] if mu.dim() > 0 else mu
        mu_single.backward(retain_graph=True)

        grads = []
        for p in self.network.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        grad_vec = torch.cat(grads)

        U_inv = torch.inverse(self.U)
        sigma_sq = (grad_vec @ U_inv @ grad_vec) / self.m
        sigma = torch.sqrt(self.lambda_reg * self.nu * sigma_sq)

        return mu, sigma

    def predict_ts(self, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """Thompson Sampling: sample from posterior."""
        sampled_rewards = []

        with torch.no_grad():
            for emb in candidate_embeddings:
                mu, sigma = self.forward(emb, compute_variance=True)

                if sigma is not None and sigma > 0:
                    reward_sample = torch.normal(mu, sigma)
                else:
                    reward_sample = mu
                sampled_rewards.append(reward_sample.item())

        return torch.tensor(sampled_rewards)

    def predict_ucb(self, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """UCB: mean + alpha * uncertainty."""
        ucb_scores = []

        with torch.no_grad():
            for emb in candidate_embeddings:
                mu, sigma = self.forward(emb, compute_variance=True)

                if sigma is not None:
                    ucb_score = mu + self.ucb_alpha * sigma
                else:
                    ucb_score = mu
                ucb_scores.append(ucb_score.item())

        return torch.tensor(ucb_scores)

    def select_arm(self, candidate_embeddings: np.ndarray) -> int:
        """
        Select arm via Thompson Sampling or UCB.

        Args:
            candidate_embeddings: (K, d) numpy array

        Returns:
            selected_idx: Index in [0, K-1]
        """
        embeddings_tensor = torch.tensor(candidate_embeddings,
                                         dtype=torch.float32).to(self.device)

        if self.algorithm == 'ts':
            scores = self.predict_ts(embeddings_tensor)
        elif self.algorithm == 'ucb':
            scores = self.predict_ucb(embeddings_tensor)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        selected_idx = int(torch.argmax(scores))

        return selected_idx

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

        embedding_t = torch.tensor(embedding,
                                   dtype=torch.float32,
                                   device=self.device)
        reward_t = torch.tensor(reward,
                                dtype=torch.float32,
                                device=self.device)

        embedding_t.requires_grad_(True)
        pred = self.network(embedding_t.unsqueeze(0)).squeeze()

        reg_loss = sum(p.norm()**2 for p in self.network.parameters())
        loss = (reward_t - pred)**2 + (self.lambda_reg /
                                       max(self.total_rounds, 1)) * reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        try:
            with torch.no_grad():
                pred_new = self.network(embedding_t.unsqueeze(0)).squeeze()

            grads = torch.autograd.grad(pred_new,
                                        list(self.network.parameters()),
                                        create_graph=False)
            grad_vec = torch.cat([g.view(-1) for g in grads])

            self.U = self.U + torch.outer(grad_vec, grad_vec)
        except Exception:
            pass

    def get_cumulative_regret(self,
                              optimal_rewards: Optional[List[float]] = None
                              ) -> np.ndarray:
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


class LinearContextualBandit:
    """
    Linear contextual bandit with Thompson Sampling or UCB.
    Uses closed-form posterior updates (no neural network).
    """

    def __init__(
            self,
            embedding_dim: int = 768,
            algorithm: Literal['ts', 'ucb'] = 'ts',
            lambda_reg: float = 1.0,
            ucb_alpha: float = 1.0):
        self.d = embedding_dim
        self.algorithm = algorithm
        self.lambda_reg = lambda_reg
        self.ucb_alpha = ucb_alpha

        self.B = np.eye(embedding_dim) * lambda_reg
        self.f = np.zeros(embedding_dim)
        self.B_inv = np.eye(embedding_dim) / lambda_reg

        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0

    def _get_theta(self) -> np.ndarray:
        """Get current estimate of theta."""
        return self.B_inv @ self.f

    def select_arm(self, candidate_embeddings: np.ndarray) -> int:
        """
        Select arm via Thompson Sampling or UCB.

        Args:
            candidate_embeddings: (K, d) numpy array

        Returns:
            selected_idx: Index in [0, K-1]
        """
        theta = self._get_theta()

        if self.algorithm == 'ts':
            try:
                theta_sample = np.random.multivariate_normal(theta, self.B_inv)
            except np.linalg.LinAlgError:
                theta_sample = theta + np.random.randn(self.d) * 0.1

            scores = candidate_embeddings @ theta_sample

        elif self.algorithm == 'ucb':
            means = candidate_embeddings @ theta

            stds = np.sqrt(np.sum(
                (candidate_embeddings @ self.B_inv) * candidate_embeddings,
                axis=1
            ))

            scores = means + self.ucb_alpha * stds

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return int(np.argmax(scores))

    def update(self, embedding: np.ndarray, reward: float):
        """
        Update posterior with new observation.

        Args:
            embedding: Selected item embedding
            reward: Observed reward
        """
        self.total_rounds += 1
        self.rewards_history.append(reward)
        self.cumulative_reward += reward

        self.B += np.outer(embedding, embedding)
        self.f += reward * embedding

        u = self.B_inv @ embedding
        self.B_inv -= np.outer(u, u) / (1 + embedding @ u)

    def get_cumulative_regret(self,
                              optimal_rewards: Optional[List[float]] = None
                              ) -> np.ndarray:
        """Compute cumulative regret."""
        if optimal_rewards is None:
            optimal_rewards = [1.0] * len(self.rewards_history)

        regrets = np.array(optimal_rewards) - np.array(self.rewards_history)
        return np.cumsum(regrets)

    def reset(self):
        """Reset bandit state."""
        self.B = np.eye(self.d) * self.lambda_reg
        self.f = np.zeros(self.d)
        self.B_inv = np.eye(self.d) / self.lambda_reg
        self.total_rounds = 0
        self.rewards_history = []
        self.cumulative_reward = 0.0


class SimpleNeuralBandit:
    """
    Simplified neural bandit for faster experimentation.
    Uses numpy for computation without GPU requirements.
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 100,
                 lr: float = 0.01,
                 exploration_weight: float = 1.0):
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

    def select_action(self,
                      candidate_embeddings: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """Select action using UCB-style exploration."""
        scores = []

        for emb in candidate_embeddings:
            pred, h = self._forward(emb)

            if deterministic:
                score = pred
            else:
                uncertainty = self.exploration_weight * np.sqrt(
                    np.sum(h**2 / (self.feature_count + 1e-6)))
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
