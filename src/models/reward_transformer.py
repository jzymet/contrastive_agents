import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List


class RewardTransformer(nn.Module):
    """
    Lightweight transformer critic for scoring tool/step sequences.
    
    Architecture: ~36M parameters
    - Input: Query embedding + tool/step sequence embeddings
    - Output: Scalar value (quality of this sequence)
    
    Used as the critic in A2C for tool selection and math reasoning.
    """
    
    def __init__(
        self, 
        d_model: int = 768, 
        nhead: int = 8, 
        num_layers: int = 2,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        use_checkpointing: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing
        
        self.query_proj = nn.Linear(d_model, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.to(self.device)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"RewardTransformer initialized with {total_params / 1e6:.1f}M parameters")
    
    def forward(
        self, 
        query_emb: torch.Tensor, 
        sequence_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute value for a query + sequence.
        
        Args:
            query_emb: (batch_size, d_model) Task/query embedding
            sequence_embs: (batch_size, seq_len, d_model) Tool/step sequence
            mask: Optional attention mask
            
        Returns:
            value: (batch_size, 1) Predicted cumulative reward
        """
        query_emb = query_emb.to(self.device)
        sequence_embs = sequence_embs.to(self.device)
        
        query_ctx = self.query_proj(query_emb)
        
        if self.use_checkpointing and self.training:
            seq_encoding = torch.utils.checkpoint.checkpoint(
                self.sequence_encoder, sequence_embs, mask
            )
        else:
            seq_encoding = self.sequence_encoder(sequence_embs, src_key_padding_mask=mask)
        
        final_state = seq_encoding[:, -1, :]
        
        combined = torch.cat([query_ctx, final_state], dim=-1)
        
        value = self.value_head(combined)
        
        return value
    
    def score_candidates(
        self,
        query_emb: np.ndarray,
        current_sequence: List[np.ndarray],
        candidate_embs: np.ndarray,
        return_all_scores: bool = False
    ) -> Tuple[int, float]:
        """
        Score all candidates and return best one.
        
        Args:
            query_emb: (d_model,) Query embedding
            current_sequence: List of (d_model,) embeddings in sequence so far
            candidate_embs: (n_candidates, d_model) Candidate embeddings
            return_all_scores: Whether to return all scores
            
        Returns:
            best_idx: Index of highest-scoring candidate
            best_value: Value of best candidate
            (optional) all_scores: All candidate scores
        """
        query_t = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        scores = []
        
        with torch.no_grad():
            for c_emb in candidate_embs:
                if len(current_sequence) > 0:
                    seq = current_sequence + [c_emb]
                else:
                    seq = [c_emb]
                
                seq_t = torch.tensor(np.stack(seq), dtype=torch.float32).unsqueeze(0).to(self.device)
                
                value = self.forward(query_t, seq_t)
                scores.append(value.item())
        
        scores = np.array(scores)
        best_idx = int(np.argmax(scores))
        
        if return_all_scores:
            return best_idx, scores[best_idx], scores
        
        return best_idx, scores[best_idx]


class A2CTransformerAgent(nn.Module):
    """
    Full A2C agent with transformer-based actor and critic.
    
    Actor: Proposes actions via softmax over critic scores
    Critic: RewardTransformer for value estimation
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        hidden_dim: int = 768,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ):
        super().__init__()
        
        self.d_model = d_model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.critic = RewardTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.device = self.critic.device
        
        self.trajectory_buffer = []
    
    def select_action(
        self,
        query_emb: np.ndarray,
        current_sequence: List[np.ndarray],
        candidate_embs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using softmax over critic scores.
        
        Returns:
            action_idx: Selected candidate index
            log_prob: Log probability of action
            value: Value estimate
        """
        query_t = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        scores = []
        with torch.no_grad():
            for c_emb in candidate_embs:
                if len(current_sequence) > 0:
                    seq = current_sequence + [c_emb]
                else:
                    seq = [c_emb]
                
                seq_t = torch.tensor(np.stack(seq), dtype=torch.float32).unsqueeze(0).to(self.device)
                value = self.critic(query_t, seq_t)
                scores.append(value.item())
        
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        probs = F.softmax(scores_t / self.temperature.abs(), dim=0)
        
        if deterministic:
            action_idx = int(torch.argmax(probs).item())
        else:
            action_idx = int(torch.multinomial(probs, 1).item())
        
        log_prob = float(torch.log(probs[action_idx] + 1e-10).item())
        value = scores[action_idx]
        
        return action_idx, log_prob, value
    
    def store_transition(
        self,
        query_emb: np.ndarray,
        sequence: List[np.ndarray],
        candidate_embs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ):
        """Store transition for later update."""
        self.trajectory_buffer.append({
            'query': query_emb.copy(),
            'sequence': [s.copy() for s in sequence],
            'candidates': candidate_embs.copy(),
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        })
    
    def update(self) -> Dict[str, float]:
        """Update agent using collected trajectories."""
        if len(self.trajectory_buffer) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        returns = []
        G = 0
        for t in reversed(range(len(self.trajectory_buffer))):
            transition = self.trajectory_buffer[t]
            if transition['done']:
                G = 0
            G = transition['reward'] + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.tensor([t['value'] for t in self.trajectory_buffer], 
                             dtype=torch.float32, device=self.device)
        log_probs = torch.tensor([t['log_prob'] for t in self.trajectory_buffer],
                                dtype=torch.float32, device=self.device)
        
        advantages = returns - values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        
        entropy = -(torch.exp(log_probs) * log_probs).mean()
        
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        self.trajectory_buffer = []
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }
    
    def reset(self):
        """Reset trajectory buffer."""
        self.trajectory_buffer = []
