import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 768, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.position_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim) * 0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :]
        return self.transformer(x, src_key_padding_mask=mask)


class CrossAttention(nn.Module):
    def __init__(self, embedding_dim: int = 768, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(query, key_value, key_value)
        return attn_output


class A2CAgent(nn.Module):
    def __init__(self, embedding_dim: int = 768, n_heads: int = 8, n_layers: int = 2, 
                 dropout: float = 0.1, lr: float = 3e-4, gamma: float = 0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        
        self.history_encoder = TransformerEncoder(embedding_dim, n_heads, n_layers, dropout)
        self.cross_attention = CrossAttention(embedding_dim, n_heads)
        
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        
        self.trajectory_buffer = []
    
    def forward(self, history_embeddings: torch.Tensor, 
                candidate_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = candidate_embeddings.size(0)
        n_candidates = candidate_embeddings.size(1)
        
        if history_embeddings.size(1) > 0:
            history_encoded = self.history_encoder(history_embeddings)
            history_context = history_encoded.mean(dim=1, keepdim=True)
            history_context = history_context.expand(-1, n_candidates, -1)
            
            attended = self.cross_attention(candidate_embeddings, history_context)
            features = candidate_embeddings + attended
        else:
            features = candidate_embeddings
        
        action_logits = self.actor(features).squeeze(-1)
        value = self.critic(features.mean(dim=1))
        
        return action_logits, value
    
    def select_action(self, history_embeddings: np.ndarray, 
                      candidate_embeddings: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        history_t = torch.tensor(history_embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        candidates_t = torch.tensor(candidate_embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.forward(history_t, candidates_t)
        
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            action = torch.multinomial(probs, 1).squeeze().item()
        
        log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
        
        return action, log_prob, value.item()
    
    def store_transition(self, history: np.ndarray, candidates: np.ndarray, 
                         action: int, log_prob: float, value: float, reward: float, done: bool):
        self.trajectory_buffer.append({
            "history": history.copy() if len(history) > 0 else np.zeros((0, self.embedding_dim)),
            "candidates": candidates.copy(),
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "done": done
        })
    
    def update(self) -> Dict[str, float]:
        if len(self.trajectory_buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        
        returns = []
        advantages = []
        
        G = 0
        for t in reversed(range(len(self.trajectory_buffer))):
            transition = self.trajectory_buffer[t]
            if transition["done"]:
                G = 0
            G = transition["reward"] + self.gamma * G
            returns.insert(0, G)
            advantages.insert(0, G - transition["value"])
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for t, transition in enumerate(self.trajectory_buffer):
            history_t = torch.tensor(transition["history"], dtype=torch.float32).unsqueeze(0).to(self.device)
            candidates_t = torch.tensor(transition["candidates"], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            logits, value = self.forward(history_t, candidates_t)
            
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            action = transition["action"]
            
            actor_loss = -log_probs[0, action] * advantages[t]
            critic_loss = F.mse_loss(value.squeeze(), returns[t])
            
            entropy = -(probs * log_probs).sum()
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy += entropy
        
        n = len(self.trajectory_buffer)
        loss = total_actor_loss / n + 0.5 * total_critic_loss / n - 0.01 * total_entropy / n
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        self.trajectory_buffer = []
        
        return {
            "actor_loss": (total_actor_loss / n).item(),
            "critic_loss": (total_critic_loss / n).item(),
            "entropy": (total_entropy / n).item()
        }


class SimpleA2CAgent:
    def __init__(self, embedding_dim: int = 768, lr: float = 0.01, gamma: float = 0.99):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.gamma = gamma
        
        self.weights = np.random.randn(embedding_dim) * 0.01
        self.value_weights = np.random.randn(embedding_dim) * 0.01
        
        self.trajectory_buffer = []
    
    def select_action(self, history_embeddings: np.ndarray, 
                      candidate_embeddings: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        if len(history_embeddings) > 0:
            context = np.mean(history_embeddings, axis=0)
            combined_weights = self.weights + 0.1 * context
        else:
            combined_weights = self.weights
        
        scores = candidate_embeddings @ combined_weights
        
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs = probs / (probs.sum() + 1e-10)
        
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        
        log_prob = float(np.log(probs[action] + 1e-10))
        
        if len(history_embeddings) > 0:
            state_repr = np.mean(history_embeddings, axis=0)
        else:
            state_repr = np.zeros(self.embedding_dim)
        value = float(state_repr @ self.value_weights)
        
        return action, log_prob, value
    
    def store_transition(self, history: np.ndarray, candidates: np.ndarray, 
                         action: int, log_prob: float, value: float, reward: float, done: bool):
        self.trajectory_buffer.append({
            "history": history.copy() if len(history) > 0 else np.zeros((0, self.embedding_dim)),
            "candidates": candidates.copy(),
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "done": done
        })
    
    def update(self) -> Dict[str, float]:
        if len(self.trajectory_buffer) == 0:
            return {"loss": 0.0}
        
        returns = []
        G = 0
        for t in reversed(range(len(self.trajectory_buffer))):
            transition = self.trajectory_buffer[t]
            if transition["done"]:
                G = 0
            G = transition["reward"] + self.gamma * G
            returns.insert(0, G)
        
        total_actor_grad = np.zeros_like(self.weights)
        total_value_grad = np.zeros_like(self.value_weights)
        
        for t, transition in enumerate(self.trajectory_buffer):
            advantage = returns[t] - transition["value"]
            
            candidate_embedding = transition["candidates"][transition["action"]]
            total_actor_grad += advantage * candidate_embedding
            
            if len(transition["history"]) > 0:
                state_repr = np.mean(transition["history"], axis=0)
            else:
                state_repr = np.zeros(self.embedding_dim)
            
            value_error = returns[t] - transition["value"]
            total_value_grad += value_error * state_repr
        
        n = len(self.trajectory_buffer)
        self.weights += self.lr * total_actor_grad / n
        self.value_weights += self.lr * 0.5 * total_value_grad / n
        
        self.trajectory_buffer = []
        
        return {"loss": float(np.abs(total_actor_grad).mean())}
