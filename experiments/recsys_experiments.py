import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.amazon import AmazonElectronicsDataset
from src.agents.bandit import PersistentContextualBandit
from src.agents.a2c import A2CAgent
from src.utils.sampling import uniform_sample, coverage_metric_rho, coverage_metric_rho_sampled
from src.utils.metrics import ExperimentMetrics, compute_regret


def run_recsys_bandit(
    embeddings: np.ndarray,
    dataset: AmazonElectronicsDataset,
    n_episodes: int = 100,
    k: int = 500,
    steps_per_episode: int = 10,
    progress_callback: Optional[Callable] = None
) -> Dict:
    n_items = len(embeddings)
    bandit = PersistentContextualBandit(n_total_items=n_items, k=k)
    
    metrics = ExperimentMetrics()
    results = {
        "rewards": [],
        "episode_rewards": [],
        "cumulative_regret": [],
        "coverage_per_episode": [],
        "items_explored": set()
    }
    
    for episode in tqdm(range(n_episodes), desc="RecSys Bandit"):
        user_id = np.random.randint(len(dataset.users))
        episode_rewards = []
        episode_coverage = []
        
        for step in range(steps_per_episode):
            candidate_indices = uniform_sample(n_items, k=k)
            bandit.set_candidates(candidate_indices)
            
            coverage = coverage_metric_rho_sampled(embeddings, candidate_indices)
            episode_coverage.append(coverage)
            
            arm = bandit.select_arm()
            item_idx = candidate_indices[arm]
            
            reward = dataset.get_reward(user_id, item_idx)
            bandit.update(arm, reward)
            
            episode_rewards.append(reward)
            results["rewards"].append(reward)
            results["items_explored"].add(int(item_idx))
            metrics.log_reward(reward)
        
        results["episode_rewards"].append(sum(episode_rewards))
        results["coverage_per_episode"].append(np.mean(episode_coverage))
        
        if progress_callback:
            progress_callback(episode / n_episodes, {
                "episode": episode,
                "mean_reward": np.mean(episode_rewards),
                "coverage": np.mean(episode_coverage)
            })
    
    results["cumulative_regret"] = compute_regret(results["rewards"]).tolist()
    results["mean_reward"] = float(np.mean(results["rewards"]))
    results["mean_episode_reward"] = float(np.mean(results["episode_rewards"]))
    results["total_episodes"] = n_episodes
    results["unique_items_explored"] = len(results["items_explored"])
    results["items_explored"] = list(results["items_explored"])[:100]
    
    rho_mean, rho_std = coverage_metric_rho(embeddings, k=k, num_trials=50)
    results["coverage_rho"] = {"mean": rho_mean, "std": rho_std}
    results["mean_sampled_coverage"] = float(np.mean(results["coverage_per_episode"]))
    
    return results


def run_recsys_a2c(
    embeddings: np.ndarray,
    dataset: AmazonElectronicsDataset,
    n_episodes: int = 50,
    k: int = 500,
    session_length: int = 10,
    progress_callback: Optional[Callable] = None
) -> Dict:
    embedding_dim = embeddings.shape[1]
    agent = A2CAgent(embedding_dim=embedding_dim, n_heads=4, n_layers=1, lr=1e-3, gamma=0.99)
    
    metrics = ExperimentMetrics()
    results = {
        "episode_rewards": [],
        "cumulative_regret": [],
        "training_losses": [],
        "coverage_per_episode": []
    }
    
    all_rewards = []
    
    for episode in tqdm(range(n_episodes), desc="RecSys A2C"):
        user_id = np.random.randint(len(dataset.users))
        
        history_embeddings = np.zeros((0, embedding_dim))
        episode_rewards = []
        episode_coverage = []
        
        for step in range(session_length):
            candidate_indices = uniform_sample(len(embeddings), k=k)
            candidate_embeddings = embeddings[candidate_indices]
            
            coverage = coverage_metric_rho_sampled(embeddings, candidate_indices)
            episode_coverage.append(coverage)
            
            action, log_prob, value = agent.select_action(
                history_embeddings, 
                candidate_embeddings
            )
            
            item_idx = candidate_indices[action]
            reward = dataset.get_reward(user_id, item_idx)
            
            done = (step == session_length - 1)
            agent.store_transition(
                history_embeddings, candidate_embeddings,
                action, log_prob, value, reward, done
            )
            
            if reward > 0.5:
                if len(history_embeddings) > 0:
                    history_embeddings = np.vstack([
                        history_embeddings, 
                        candidate_embeddings[action:action+1]
                    ])
                else:
                    history_embeddings = candidate_embeddings[action:action+1]
            
            episode_rewards.append(reward)
            all_rewards.append(reward)
            metrics.log_reward(reward)
        
        update_info = agent.update()
        
        results["episode_rewards"].append(sum(episode_rewards))
        results["training_losses"].append(update_info.get("actor_loss", 0) + update_info.get("critic_loss", 0))
        results["coverage_per_episode"].append(np.mean(episode_coverage))
        
        if progress_callback:
            progress_callback(episode / n_episodes, {
                "episode": episode,
                "episode_reward": sum(episode_rewards),
                "mean_reward": np.mean(episode_rewards),
                "loss": update_info.get("actor_loss", 0)
            })
    
    results["cumulative_regret"] = compute_regret(all_rewards).tolist()
    results["mean_episode_reward"] = float(np.mean(results["episode_rewards"]))
    results["total_episodes"] = n_episodes
    results["mean_sampled_coverage"] = float(np.mean(results["coverage_per_episode"]))
    
    rho_mean, rho_std = coverage_metric_rho(embeddings, k=k, num_trials=50)
    results["coverage_rho"] = {"mean": rho_mean, "std": rho_std}
    
    return results


def compare_embedding_types(
    simcse_embeddings: np.ndarray,
    bert_embeddings: np.ndarray,
    dataset: AmazonElectronicsDataset,
    n_episodes: int = 50,
    k: int = 500,
    experiment_type: str = "bandit"
) -> Dict:
    results = {}
    
    if experiment_type == "bandit":
        results["simcse"] = run_recsys_bandit(simcse_embeddings, dataset, n_episodes, k)
        results["bert"] = run_recsys_bandit(bert_embeddings, dataset, n_episodes, k)
    else:
        results["simcse"] = run_recsys_a2c(simcse_embeddings, dataset, n_episodes, k)
        results["bert"] = run_recsys_a2c(bert_embeddings, dataset, n_episodes, k)
    
    simcse_rho = results["simcse"]["coverage_rho"]["mean"]
    bert_rho = results["bert"]["coverage_rho"]["mean"]
    
    results["comparison"] = {
        "coverage_improvement": (bert_rho - simcse_rho) / bert_rho * 100 if bert_rho > 0 else 0,
        "simcse_better_coverage": simcse_rho < bert_rho,
        "reward_improvement": (
            results["simcse"].get("mean_reward", results["simcse"].get("mean_episode_reward", 0)) -
            results["bert"].get("mean_reward", results["bert"].get("mean_episode_reward", 0))
        ),
        "simcse_unique_items": results["simcse"].get("unique_items_explored", 0),
        "bert_unique_items": results["bert"].get("unique_items_explored", 0)
    }
    
    return results
