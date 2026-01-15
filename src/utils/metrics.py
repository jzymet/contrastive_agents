import numpy as np
from typing import List, Dict, Any
from collections import Counter


def compute_regret(rewards: List[float], optimal_reward: float = 1.0) -> np.ndarray:
    regrets = optimal_reward - np.array(rewards)
    return np.cumsum(regrets)


def compute_hit_rate(predictions: List[int], ground_truth: List[int], k: int = 10) -> float:
    hits = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred in gt[:k] if isinstance(gt, list) else pred == gt:
            hits += 1
    return hits / len(predictions) if predictions else 0.0


def compute_strategy_entropy(strategies: List[str]) -> float:
    if not strategies:
        return 0.0
    
    counts = Counter(strategies)
    total = len(strategies)
    probs = np.array([c / total for c in counts.values()])
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def compute_solve_rate(solved: List[bool], window: int = 100) -> List[float]:
    rates = []
    for i in range(len(solved)):
        start = max(0, i - window + 1)
        window_solved = solved[start:i + 1]
        rates.append(sum(window_solved) / len(window_solved))
    return rates


def compute_rollouts_to_threshold(solve_rates: List[float], threshold: float = 0.7) -> int:
    for i, rate in enumerate(solve_rates):
        if rate >= threshold:
            return i + 1
    return len(solve_rates)


def extract_strategy_type(step_text: str) -> str:
    step_lower = step_text.lower()
    
    if any(op in step_text for op in ['+', '-', '*', '/', '×', '÷']):
        if 'let' in step_lower or 'solve' in step_lower:
            return 'algebraic_manipulation'
        return 'direct_calculation'
    
    if '%' in step_text or 'percent' in step_lower:
        return 'percentage_calculation'
    
    if any(w in step_lower for w in ['convert', 'unit', 'miles', 'km', 'meters']):
        return 'unit_conversion'
    
    if any(w in step_lower for w in ['area', 'volume', 'angle', 'triangle', 'circle']):
        return 'geometric_reasoning'
    
    if any(w in step_lower for w in ['if', 'then', 'therefore', 'because']):
        return 'logical_deduction'
    
    return 'other'


class ExperimentMetrics:
    def __init__(self):
        self.rewards = []
        self.predictions = []
        self.strategies = []
        self.solved = []
        self.rollout_count = 0
    
    def log_reward(self, reward: float):
        self.rewards.append(reward)
    
    def log_prediction(self, pred: int, ground_truth: Any):
        self.predictions.append((pred, ground_truth))
    
    def log_strategy(self, strategy: str):
        self.strategies.append(strategy)
    
    def log_solve(self, solved: bool):
        self.solved.append(solved)
        self.rollout_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        solve_rates = compute_solve_rate(self.solved)
        
        return {
            "cumulative_regret": compute_regret(self.rewards)[-1] if self.rewards else 0.0,
            "mean_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "strategy_entropy": compute_strategy_entropy(self.strategies),
            "unique_strategies": len(set(self.strategies)),
            "solve_rate": float(np.mean(self.solved)) if self.solved else 0.0,
            "rollouts_to_70": compute_rollouts_to_threshold(solve_rates, 0.7),
            "total_rollouts": self.rollout_count,
        }
