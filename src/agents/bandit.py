import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal


class LinearContextualBandit:
    """
    Linear contextual bandit with UCB or Thompson Sampling.

    Assumes reward function: r(x) = w^T x + noise
    where x is the embedding (768-dim)

    This is the baseline to compare against NeuralTS.
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 algorithm: Literal['ucb', 'ts'] = 'ts',
                 lambda_reg: float = 1.0,
                 delta: float = 0.1):
        self.d = embedding_dim
        self.algorithm = algorithm
        self.lambda_reg = lambda_reg
        self.delta = delta

        # Sufficient statistics
        self.A = lambda_reg * np.eye(self.d)  # (d, d) precision matrix
        self.b = np.zeros(self.d)  # (d,) weighted rewards

        # Weights (posterior mean)
        self.w = np.zeros(self.d)

        # For TS: posterior covariance
        self.A_inv = np.linalg.inv(self.A)

        # Tracking
        self.t = 0
        self.rewards_history = []
        self.cumulative_regret = []

    def update_posterior(self):
        """Update posterior mean and covariance."""
        self.A_inv = np.linalg.inv(self.A)
        self.w = self.A_inv @ self.b

    def predict_ucb(self,
                    embeddings: np.ndarray,
                    alpha: float = 1.0) -> np.ndarray:
        """
        UCB scores for all candidate embeddings.

        Args:
            embeddings: (K, d) candidate embeddings
            alpha: Exploration parameter

        Returns:
            ucb_scores: (K,) upper confidence bounds
        """
        # Mean prediction
        mean = embeddings @ self.w  # (K,)

        # Uncertainty (confidence width)
        # σ²(x) = x^T A^{-1} x
        uncertainty = np.sqrt(
            np.sum((embeddings @ self.A_inv) * embeddings, axis=1))  # (K,)

        # UCB = mean + α * uncertainty
        ucb_scores = mean + alpha * uncertainty

        return ucb_scores

    def predict_ts(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Thompson Sampling: sample weights from posterior, score candidates.

        Args:
            embeddings: (K, d) candidate embeddings

        Returns:
            sampled_scores: (K,) scores under sampled weights
        """
        # Sample from posterior: w ~ N(w_hat, A^{-1})
        w_sample = np.random.multivariate_normal(self.w, self.A_inv)

        # Score candidates with sampled weights
        scores = embeddings @ w_sample  # (K,)

        return scores

    def select_arm(self, candidate_embeddings: np.ndarray) -> int:
        """
        Select arm (item) from candidates.

        Args:
            candidate_embeddings: (K, d) embeddings for K candidates

        Returns:
            selected_idx: Index of selected candidate (0 to K-1)
        """
        if self.algorithm == 'ucb':
            scores = self.predict_ucb(candidate_embeddings)
        elif self.algorithm == 'ts':
            scores = self.predict_ts(candidate_embeddings)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return int(np.argmax(scores))

    def update(self, embedding: np.ndarray, reward: float):
        """
        Update posterior after observing reward.

        Args:
            embedding: (d,) selected item's embedding
            reward: Observed reward (0 or 1)
        """
        # Update sufficient statistics
        self.A += np.outer(embedding, embedding)
        self.b += reward * embedding

        # Update posterior
        self.update_posterior()

        # Track
        self.t += 1
        self.rewards_history.append(reward)

    def get_weights(self) -> np.ndarray:
        """Return current weight vector (for RKHS analysis)."""
        return self.w.copy()

    def reset(self):
        """Reset bandit to initial state."""
        self.A = self.lambda_reg * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.w = np.zeros(self.d)
        self.A_inv = np.linalg.inv(self.A)
        self.t = 0
        self.rewards_history = []
        self.cumulative_regret = []


class NeuralContextualBandit:
    """
    Neural Thompson Sampling bandit (adapted from NeuralTS).

    Paper: "Neural Contextual Bandits with UCB-based Exploration"
    Code: https://github.com/ZeroWeight/NeuralTS

    Key differences from linear:
    - Non-linear reward: r(x) = f_θ(x)
    - Neural tangent kernel for uncertainty
    - Diagonal approximation for efficiency
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 100,
                 lambda_reg: float = 1.0,
                 nu: float = 1.0,
                 learning_rate: float = 0.01,
                 use_cuda: bool = False):
        self.d = embedding_dim
        self.m = hidden_dim
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.lr = learning_rate

        # Device
        self.device = torch.device(
            'cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        # Neural network: 2-layer MLP
        self.network = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                     nn.ReLU(), nn.Linear(hidden_dim,
                                                          1)).to(self.device)

        # Optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)

        # Diagonal approximation of precision matrix
        # U_t ≈ diag of gradient outer products
        n_params = sum(p.numel() for p in self.network.parameters())
        self.U = torch.eye(n_params).to(self.device)  # Diagonal approx

        # Tracking
        self.t = 0
        self.rewards_history = []
        self.training_losses = []

    def forward(self, embedding: torch.Tensor, compute_variance: bool = True):
        """
        Forward pass with optional uncertainty estimation.

        Args:
            embedding: (d,) or (batch, d)
            compute_variance: Whether to compute posterior variance

        Returns:
            mu: Mean prediction
            sigma: Standard deviation (if compute_variance=True)
        """
        # Ensure 2D
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Mean prediction
        mu = self.network(embedding).squeeze(-1)  # (batch,)

        if not compute_variance:
            return mu, None

        # Compute gradient (Jacobian) for uncertainty
        # g_t = ∇_θ f_θ(x_t)
        self.network.zero_grad()
        mu_single = mu[0] if mu.dim() > 0 else mu
        mu_single.backward(retain_graph=True)

        # Collect gradients into vector
        grads = []
        for p in self.network.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        grad_vec = torch.cat(grads)  # (n_params,)

        # Variance via neural tangent kernel
        # σ²(x) = g_t^T U_t^{-1} g_t / m
        U_inv = torch.inverse(self.U)
        sigma_sq = (grad_vec @ U_inv @ grad_vec) / self.m
        sigma = torch.sqrt(self.lambda_reg * self.nu * sigma_sq)

        return mu, sigma

    def predict_ts(self, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Thompson Sampling: sample from posterior for each candidate.

        Args:
            candidate_embeddings: (K, d)

        Returns:
            sampled_rewards: (K,)
        """
        sampled_rewards = []

        with torch.no_grad():
            for emb in candidate_embeddings:
                mu, sigma = self.forward(emb, compute_variance=True)

                # Sample from N(mu, sigma²)
                reward_sample = torch.normal(mu, sigma)
                sampled_rewards.append(reward_sample.item())

        return torch.tensor(sampled_rewards)

    def select_arm(self, candidate_embeddings: np.ndarray) -> int:
        """
        Select arm via Thompson Sampling.

        Args:
            candidate_embeddings: (K, d) numpy array

        Returns:
            selected_idx: Index in [0, K-1]
        """
        # Convert to tensor
        embeddings_tensor = torch.tensor(candidate_embeddings,
                                         dtype=torch.float32).to(self.device)

        # Thompson Sampling
        sampled_rewards = self.predict_ts(embeddings_tensor)

        # Select highest sampled reward
        selected_idx = int(torch.argmax(sampled_rewards))

        return selected_idx

    def update(self, embedding: np.ndarray, reward: float):
        """
        Update network via SGD with time-decaying regularization.

        Args:
            embedding: (d,) selected item embedding
            reward: Observed reward
        """
        # Convert to tensor
        emb_tensor = torch.tensor(embedding,
                                  dtype=torch.float32).to(self.device)

        # Forward pass
        pred, _ = self.forward(emb_tensor, compute_variance=False)

        # Loss: MSE + time-decaying L2 regularization
        mse_loss = (reward - pred)**2

        # Regularization: (λ / t) * ||θ||²
        reg_loss = (self.lambda_reg /
                    (self.t + 1)) * sum(p.norm()**2
                                        for p in self.network.parameters())

        loss = mse_loss + reg_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        # Update U matrix (diagonal approximation)
        # U_t = U_{t-1} + g_t g_t^T
        with torch.no_grad():
            grads = []
            for p in self.network.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            grad_vec = torch.cat(grads)

            # Diagonal update: U_ii += g_i²
            self.U += torch.outer(grad_vec, grad_vec)

        # Track
        self.t += 1
        self.rewards_history.append(reward)
        self.training_losses.append(loss.item())

    def get_weights(self) -> np.ndarray:
        """
        Extract first layer weights for RKHS analysis.

        Returns:
            weights: (d,) averaged across hidden units
        """
        first_layer = list(self.network.children())[0]
        weights = first_layer.weight.data.cpu().numpy()  # (hidden_dim, d)

        # Average across hidden units
        return weights.mean(axis=0)  # (d,)

    def reset(self):
        """Reset bandit."""
        # Re-initialize network
        for layer in self.network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Reset U matrix
        n_params = sum(p.numel() for p in self.network.parameters())
        self.U = torch.eye(n_params).to(self.device)

        # Reset tracking
        self.t = 0
        self.rewards_history = []
        self.training_losses = []
