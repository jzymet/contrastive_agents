import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class RewardTransformer(nn.Module):
    """
    Transformer critic for scoring tool/reasoning sequences.

    Architecture (from paper):
    - Query projection: Projects task embedding
    - Sequence encoder: Self-attention over action history
    - Value head: Predicts cumulative reward for this sequence

    This is the CRITIC in A2C, not an actor!
    Policy is external (BM25 for tools, LLM for math).
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()

        self.d_model = d_model

        # Query projection (for task instruction embedding)
        self.query_proj = nn.Linear(d_model, d_model)

        # Sequence encoder (self-attention over action history)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            self.sequence_encoder.gradient_checkpointing_enable()

        # Value head (predicts cumulative reward)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(
        self,
        query_emb: torch.Tensor,      # (batch, d_model) - task embedding
        sequence_embs: torch.Tensor    # (batch, seq_len, d_model) - action sequence
    ) -> torch.Tensor:
        """
        Args:
            query_emb: (batch, d_model) Task instruction embedding
            sequence_embs: (batch, seq_len, d_model) Tool/step sequence embeddings

        Returns:
            value: (batch, 1) Predicted cumulative reward
        """
        batch_size = query_emb.shape[0]

        # Project query context
        query_ctx = self.query_proj(query_emb)  # (batch, d_model)

        # Encode sequence (self-attention over tools/steps)
        if sequence_embs.shape[1] > 0:
            seq_encoding = self.sequence_encoder(sequence_embs)  # (batch, seq_len, d_model)

            # Pool to final state (last tool/step encoding)
            final_state = seq_encoding[:, -1, :]  # (batch, d_model)
        else:
            # Empty sequence (start of episode)
            final_state = torch.zeros_like(query_ctx)

        # Combine query + final state
        combined = torch.cat([query_ctx, final_state], dim=-1)  # (batch, d_model*2)

        # Predict value
        value = self.value_head(combined)  # (batch, 1)

        return value

    def get_critic_weights(self) -> np.ndarray:
        """
        Extract weights for RKHS analysis.
        Returns query projection weights (operate on embeddings).
        """
        weights = self.query_proj.weight.data.cpu().numpy()  # (d_model, d_model)

        # Average across output dimensions (more stable)
        return weights.mean(axis=0)  # (d_model,)


class A2CTrainer:
    """
    A2C training loop for RewardTransformer.

    Key insight: Policy is EXTERNAL (BM25 for tools, LLM for math).
    We only train the CRITIC (RewardTransformer).

    Training loop:
    1. Policy proposes K candidates
    2. Critic scores each candidate sequence
    3. Select action via softmax(critic_scores / temperature)
    4. Execute action, observe reward
    5. Compute returns (Monte Carlo or TD)
    6. Update critic to predict returns better
    """

    def __init__(
        self,
        critic: RewardTransformer,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True
    ):
        self.critic = critic
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=learning_rate
        )

        # Mixed precision training
        self.use_amp = use_mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Trajectory buffer
        self.trajectory_buffer = []

    def select_action(
        self,
        task_emb: np.ndarray,           # (d_model,) - task embedding
        history_embs: List[np.ndarray], # List of (d_model,) - action history
        candidate_embs: List[np.ndarray], # List of (d_model,) - K candidates
        temperature: float = 0.1,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using critic to score candidates.

        Args:
            task_emb: Task instruction embedding
            history_embs: List of embeddings for actions taken so far
            candidate_embs: List of embeddings for K candidate next actions
            temperature: Softmax temperature (lower = less random)
            deterministic: If True, select argmax instead of sampling

        Returns:
            action_idx: Index of selected candidate (0 to K-1)
            log_prob: Log probability of selected action
            value: Critic's value estimate for selected sequence
        """
        with torch.no_grad():
            # Convert to tensors
            task_tensor = torch.tensor(
                task_emb, dtype=torch.float32
            ).unsqueeze(0).to(self.critic.device)

            # Score each candidate
            scores = []
            for candidate_emb in candidate_embs:
                # Build sequence: history + candidate
                if len(history_embs) > 0:
                    sequence = history_embs + [candidate_emb]
                else:
                    sequence = [candidate_emb]

                seq_tensor = torch.tensor(
                    np.stack(sequence), dtype=torch.float32
                ).unsqueeze(0).to(self.critic.device)

                # Critic scores this sequence
                value = self.critic(task_tensor, seq_tensor)
                scores.append(value.item())

            scores = np.array(scores)

            # Softmax with temperature
            scores_scaled = scores / temperature
            scores_scaled = scores_scaled - scores_scaled.max()  # Numerical stability
            probs = np.exp(scores_scaled)
            probs = probs / (probs.sum() + 1e-10)

            # Select action
            if deterministic:
                action_idx = int(np.argmax(probs))
            else:
                action_idx = int(np.random.choice(len(probs), p=probs))

            log_prob = float(np.log(probs[action_idx] + 1e-10))

            # Get value for selected sequence
            selected_value = scores[action_idx]

        return action_idx, log_prob, selected_value

    def store_transition(
        self,
        task_emb: np.ndarray,
        history_embs: List[np.ndarray],
        action_emb: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ):
        """Store transition for later training."""
        self.trajectory_buffer.append({
            'task_emb': task_emb,
            'history_embs': history_embs.copy() if history_embs else [],
            'action_emb': action_emb,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        })

    def compute_returns(self) -> np.ndarray:
        """
        Compute Monte Carlo returns (discounted cumulative rewards).

        Returns:
            returns: Array of shape (T,) where T = trajectory length
        """
        returns = []
        G = 0.0

        for t in reversed(range(len(self.trajectory_buffer))):
            transition = self.trajectory_buffer[t]

            if transition['done']:
                G = 0.0

            G = transition['reward'] + self.gamma * G
            returns.insert(0, G)

        return np.array(returns)

    def update(self) -> Dict[str, float]:
        """
        Update critic using trajectory data.

        A2C update:
        1. Compute returns (discounted cumulative rewards)
        2. Compute advantages (returns - values)
        3. Critic loss: MSE(value, return)
        4. Optional: Add policy gradient term (but policy is external!)

        Returns:
            Dict with loss metrics
        """
        if len(self.trajectory_buffer) == 0:
            return {'critic_loss': 0.0}

        # Compute returns
        returns = self.compute_returns()
        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.critic.device
        )

        # Compute advantages
        values = np.array([t['value'] for t in self.trajectory_buffer])
        advantages = returns - values

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=self.critic.device
        )

        # Update critic
        total_critic_loss = 0.0

        for t, transition in enumerate(self.trajectory_buffer):
            # Prepare inputs
            task_tensor = torch.tensor(
                transition['task_emb'], dtype=torch.float32
            ).unsqueeze(0).to(self.critic.device)

            # Build sequence up to this point
            sequence = transition['history_embs'] + [transition['action_emb']]
            seq_tensor = torch.tensor(
                np.stack(sequence), dtype=torch.float32
            ).unsqueeze(0).to(self.critic.device)

            # Forward pass (with mixed precision if enabled)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    value_pred = self.critic(task_tensor, seq_tensor)
                    critic_loss = F.mse_loss(value_pred.squeeze(), returns_tensor[t])
            else:
                value_pred = self.critic(task_tensor, seq_tensor)
                critic_loss = F.mse_loss(value_pred.squeeze(), returns_tensor[t])

            total_critic_loss += critic_loss

        # Average loss
        loss = self.value_loss_coef * total_critic_loss / len(self.trajectory_buffer)

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Clear buffer
        self.trajectory_buffer.clear()

        return {
            'critic_loss': loss.item(),
            'mean_return': returns.mean(),
            'mean_advantage': advantages.mean()
        }

    def reset_trajectory(self):
        """Clear trajectory buffer."""
        self.trajectory_buffer.clear()