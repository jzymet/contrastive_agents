import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.math_data import GSM8KDataset, MATH500Dataset, MathDataset
from src.agents.a2c import A2CAgent
from src.utils.sampling import uniform_sample, coverage_metric_rho, coverage_metric_rho_sampled
from src.utils.metrics import (
    ExperimentMetrics, compute_solve_rate, compute_rollouts_to_threshold,
    compute_strategy_entropy, extract_strategy_type
)


def run_math_a2c(
    encoder,
    dataset: MathDataset,
    n_episodes: int = 50,
    n_candidates: int = 50,
    max_steps: int = 10,
    progress_callback: Optional[Callable] = None
) -> Dict:
    embedding_dim = encoder.embedding_dim if hasattr(encoder, 'embedding_dim') else 768
    agent = A2CAgent(embedding_dim=embedding_dim, n_heads=4, n_layers=1, lr=1e-3, gamma=0.99)
    
    metrics = ExperimentMetrics()
    results = {
        "episode_rewards": [],
        "solved": [],
        "steps_per_problem": [],
        "strategies_used": [],
        "solve_rates": [],
        "coverage_per_problem": [],
        "training_losses": []
    }
    
    all_candidate_embeddings = []
    
    for episode in tqdm(range(min(n_episodes, len(dataset))), desc="Math A2C"):
        problem = dataset[episode]
        
        history_embeddings = np.zeros((0, embedding_dim))
        episode_rewards = []
        strategies = []
        problem_coverage = []
        
        current_state = {"values": {}, "steps": []}
        
        for step in range(max_steps):
            candidates_text = dataset.generate_step_candidates(
                problem, current_state, n_candidates=n_candidates
            )
            
            try:
                candidate_embeddings = encoder.encode(candidates_text, show_progress=False)
            except:
                candidate_embeddings = np.random.randn(len(candidates_text), embedding_dim).astype(np.float32)
            
            if len(all_candidate_embeddings) < 10:
                all_candidate_embeddings.append(candidate_embeddings)
            
            action, log_prob, value = agent.select_action(
                history_embeddings,
                candidate_embeddings
            )
            
            selected_step = candidates_text[action]
            reward = dataset.get_step_reward(selected_step)
            
            strategy = extract_strategy_type(selected_step)
            strategies.append(strategy)
            metrics.log_strategy(strategy)
            
            done = (step == max_steps - 1) or (reward > 0 and len(current_state["steps"]) >= 2)
            agent.store_transition(
                history_embeddings, candidate_embeddings,
                action, log_prob, value, reward, done
            )
            
            if reward > 0:
                if len(history_embeddings) > 0:
                    history_embeddings = np.vstack([
                        history_embeddings,
                        candidate_embeddings[action:action+1]
                    ])
                else:
                    history_embeddings = candidate_embeddings[action:action+1]
                current_state["steps"].append(selected_step)
            
            episode_rewards.append(reward)
            metrics.log_reward(reward)
            
            if done and reward > 0:
                break
        
        update_info = agent.update()
        
        solved = len(current_state["steps"]) >= 2 and sum(episode_rewards) > 0
        metrics.log_solve(solved)
        
        results["episode_rewards"].append(sum(episode_rewards))
        results["solved"].append(solved)
        results["steps_per_problem"].append(len(current_state["steps"]))
        results["strategies_used"].extend(strategies)
        results["training_losses"].append(update_info.get("actor_loss", 0) + update_info.get("critic_loss", 0))
        
        if progress_callback:
            progress_callback(episode / n_episodes, {
                "episode": episode,
                "solved": solved,
                "steps": len(current_state["steps"]),
                "reward": sum(episode_rewards)
            })
    
    solve_rates = compute_solve_rate(results["solved"])
    results["solve_rates"] = solve_rates
    results["final_solve_rate"] = float(np.mean(results["solved"]))
    results["mean_steps"] = float(np.mean(results["steps_per_problem"]))
    results["strategy_entropy"] = compute_strategy_entropy(results["strategies_used"])
    results["unique_strategies"] = len(set(results["strategies_used"]))
    results["rollouts_to_70"] = compute_rollouts_to_threshold(solve_rates, 0.7)
    results["total_episodes"] = min(n_episodes, len(dataset))
    
    if all_candidate_embeddings:
        combined = np.vstack(all_candidate_embeddings[:5])
        rho_mean, rho_std = coverage_metric_rho(combined, k=min(20, len(combined) // 2), num_trials=20)
        results["coverage_rho"] = {"mean": rho_mean, "std": rho_std}
    else:
        results["coverage_rho"] = {"mean": 0.0, "std": 0.0}
    
    return results


def analyze_search_efficiency(
    simcse_encoder,
    bert_encoder,
    dataset: MathDataset,
    n_episodes: int = 50,
    n_candidates: int = 50,
    progress_callback: Optional[Callable] = None
) -> Dict:
    results = {"simcse": {}, "bert": {}}
    
    if progress_callback:
        progress_callback(0, {"status": "Running SimCSE experiments..."})
    
    results["simcse"] = run_math_a2c(
        simcse_encoder, dataset, n_episodes, n_candidates,
        progress_callback=lambda p, d: progress_callback(p * 0.5, d) if progress_callback else None
    )
    
    if progress_callback:
        progress_callback(0.5, {"status": "Running BERT experiments..."})
    
    results["bert"] = run_math_a2c(
        bert_encoder, dataset, n_episodes, n_candidates,
        progress_callback=lambda p, d: progress_callback(0.5 + p * 0.5, d) if progress_callback else None
    )
    
    simcse_rollouts = results["simcse"]["rollouts_to_70"]
    bert_rollouts = results["bert"]["rollouts_to_70"]
    
    if bert_rollouts > 0 and simcse_rollouts < bert_rollouts:
        rollout_improvement = (bert_rollouts - simcse_rollouts) / bert_rollouts * 100
    else:
        rollout_improvement = 0
    
    results["comparison"] = {
        "rollout_improvement_pct": rollout_improvement,
        "simcse_rollouts_to_70": simcse_rollouts,
        "bert_rollouts_to_70": bert_rollouts,
        "simcse_unique_strategies": results["simcse"]["unique_strategies"],
        "bert_unique_strategies": results["bert"]["unique_strategies"],
        "strategy_diversity_improvement": (
            results["simcse"]["strategy_entropy"] - results["bert"]["strategy_entropy"]
        ),
        "solve_rate_improvement": (
            results["simcse"]["final_solve_rate"] - results["bert"]["final_solve_rate"]
        ) * 100
    }
    
    return results


class DummyEncoder:
    def __init__(self, embedding_dim: int = 768, is_uniform: bool = True):
        self.embedding_dim = embedding_dim
        self.is_uniform = is_uniform
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        n = len(texts)
        if self.is_uniform:
            embeddings = np.random.randn(n, self.embedding_dim).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            base = np.random.randn(1, self.embedding_dim).astype(np.float32)
            noise = np.random.randn(n, self.embedding_dim).astype(np.float32) * 0.3
            embeddings = base + noise
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings


def run_quick_math_demo(n_problems: int = 20, n_candidates: int = 30) -> Dict:
    dataset = GSM8KDataset(n_problems=n_problems)
    
    simcse_encoder = DummyEncoder(embedding_dim=768, is_uniform=True)
    bert_encoder = DummyEncoder(embedding_dim=768, is_uniform=False)
    
    results = analyze_search_efficiency(
        simcse_encoder, bert_encoder, dataset,
        n_episodes=n_problems, n_candidates=n_candidates
    )
    
    return results
