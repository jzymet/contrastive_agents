import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from src.datasets.toolbench import ToolBenchDataset
from src.agents.bandit import PersistentContextualBandit
from src.agents.a2c import A2CAgent
from src.utils.sampling import uniform_sample, coverage_metric_rho, coverage_metric_rho_sampled
from src.utils.metrics import ExperimentMetrics, compute_regret


def run_tool_bandit(
    embeddings: np.ndarray,
    dataset: ToolBenchDataset,
    task_type: str = "I1",
    n_episodes: int = 100,
    k: int = 500,
    progress_callback: Optional[Callable] = None
) -> Dict:
    queries = dataset.get_queries(task_type)
    if not queries:
        return {"error": f"No queries found for task type {task_type}"}
    
    n_tools = len(embeddings)
    bandit = PersistentContextualBandit(n_total_items=n_tools, k=k)
    
    metrics = ExperimentMetrics()
    results = {
        "rewards": [],
        "success_rate": [],
        "cumulative_regret": [],
        "coverage_per_query": []
    }
    
    for episode in tqdm(range(min(n_episodes, len(queries))), desc=f"Tool Bandit ({task_type})"):
        query = queries[episode % len(queries)]
        ground_truth = set(query["ground_truth"])
        
        candidate_indices = uniform_sample(n_tools, k=k)
        bandit.set_candidates(candidate_indices)
        
        coverage = coverage_metric_rho_sampled(embeddings, candidate_indices)
        results["coverage_per_query"].append(coverage)
        
        arm = bandit.select_arm()
        tool_idx = candidate_indices[arm]
        
        result, success = dataset.execute_tool(tool_idx)
        is_correct = tool_idx in ground_truth
        
        reward = 1.0 if is_correct else (0.3 if success else 0.0)
        bandit.update(arm, reward)
        
        metrics.log_reward(reward)
        results["rewards"].append(reward)
        results["success_rate"].append(1.0 if is_correct else 0.0)
        
        if progress_callback:
            progress_callback(episode / n_episodes, {
                "episode": episode,
                "success": is_correct,
                "reward": reward,
                "coverage": coverage
            })
    
    results["cumulative_regret"] = compute_regret(results["rewards"]).tolist()
    results["mean_reward"] = float(np.mean(results["rewards"]))
    results["overall_success_rate"] = float(np.mean(results["success_rate"]))
    results["total_episodes"] = min(n_episodes, len(queries))
    results["task_type"] = task_type
    results["mean_sampled_coverage"] = float(np.mean(results["coverage_per_query"]))
    
    rho_mean, rho_std = coverage_metric_rho(embeddings, k=k, num_trials=50)
    results["coverage_rho"] = {"mean": rho_mean, "std": rho_std}
    
    return results


def run_tool_a2c(
    embeddings: np.ndarray,
    dataset: ToolBenchDataset,
    task_type: str = "I2",
    n_episodes: int = 50,
    k: int = 500,
    max_steps: int = 10,
    progress_callback: Optional[Callable] = None
) -> Dict:
    queries = dataset.get_queries(task_type)
    if not queries:
        return {"error": f"No queries found for task type {task_type}"}
    
    embedding_dim = embeddings.shape[1]
    agent = A2CAgent(embedding_dim=embedding_dim, n_heads=4, n_layers=1, lr=1e-3, gamma=0.99)
    
    metrics = ExperimentMetrics()
    results = {
        "episode_rewards": [],
        "success_rate": [],
        "steps_per_episode": [],
        "tool_chains": [],
        "coverage_per_episode": []
    }
    
    all_rewards = []
    
    for episode in tqdm(range(min(n_episodes, len(queries))), desc=f"Tool A2C ({task_type})"):
        query = queries[episode % len(queries)]
        ground_truth = set(query["ground_truth"])
        
        history_embeddings = np.zeros((0, embedding_dim))
        episode_rewards = []
        tool_chain = []
        found_tools = set()
        episode_coverage = []
        
        for step in range(max_steps):
            candidate_indices = uniform_sample(len(embeddings), k=k)
            candidate_embeddings = embeddings[candidate_indices]
            
            coverage = coverage_metric_rho_sampled(embeddings, candidate_indices)
            episode_coverage.append(coverage)
            
            action, log_prob, value = agent.select_action(
                history_embeddings,
                candidate_embeddings
            )
            
            tool_idx = candidate_indices[action]
            result, success = dataset.execute_tool(tool_idx)
            
            is_correct = tool_idx in ground_truth
            if is_correct:
                found_tools.add(tool_idx)
            
            reward = 1.0 if is_correct else (0.2 if success else -0.1)
            
            done = (step == max_steps - 1) or (found_tools == ground_truth)
            agent.store_transition(
                history_embeddings, candidate_embeddings,
                action, log_prob, value, reward, done
            )
            
            if len(history_embeddings) > 0:
                history_embeddings = np.vstack([
                    history_embeddings,
                    candidate_embeddings[action:action+1]
                ])
            else:
                history_embeddings = candidate_embeddings[action:action+1]
            
            episode_rewards.append(reward)
            all_rewards.append(reward)
            tool_chain.append(int(tool_idx))
            metrics.log_reward(reward)
            
            if found_tools == ground_truth:
                bonus = 5.0
                episode_rewards.append(bonus)
                all_rewards.append(bonus)
                break
        
        update_info = agent.update()
        
        task_success = len(found_tools.intersection(ground_truth)) >= max(1, len(ground_truth) // 2)
        
        results["episode_rewards"].append(sum(episode_rewards))
        results["success_rate"].append(1.0 if task_success else 0.0)
        results["steps_per_episode"].append(len(tool_chain))
        results["tool_chains"].append(tool_chain)
        results["coverage_per_episode"].append(np.mean(episode_coverage))
        
        if progress_callback:
            progress_callback(episode / n_episodes, {
                "episode": episode,
                "success": task_success,
                "steps": len(tool_chain),
                "found_ratio": len(found_tools.intersection(ground_truth)) / max(1, len(ground_truth))
            })
    
    results["cumulative_regret"] = compute_regret(all_rewards).tolist()
    results["mean_episode_reward"] = float(np.mean(results["episode_rewards"]))
    results["overall_success_rate"] = float(np.mean(results["success_rate"]))
    results["mean_steps"] = float(np.mean(results["steps_per_episode"]))
    results["total_episodes"] = min(n_episodes, len(queries))
    results["task_type"] = task_type
    results["mean_sampled_coverage"] = float(np.mean(results["coverage_per_episode"]))
    
    rho_mean, rho_std = coverage_metric_rho(embeddings, k=k, num_trials=50)
    results["coverage_rho"] = {"mean": rho_mean, "std": rho_std}
    
    return results


def compare_tool_embedding_types(
    simcse_embeddings: np.ndarray,
    bert_embeddings: np.ndarray,
    dataset: ToolBenchDataset,
    task_type: str = "I3",
    n_episodes: int = 50,
    experiment_type: str = "a2c"
) -> Dict:
    results = {}
    
    if experiment_type == "bandit":
        results["simcse"] = run_tool_bandit(simcse_embeddings, dataset, task_type, n_episodes)
        results["bert"] = run_tool_bandit(bert_embeddings, dataset, task_type, n_episodes)
    else:
        results["simcse"] = run_tool_a2c(simcse_embeddings, dataset, task_type, n_episodes)
        results["bert"] = run_tool_a2c(bert_embeddings, dataset, task_type, n_episodes)
    
    simcse_rho = results["simcse"]["coverage_rho"]["mean"]
    bert_rho = results["bert"]["coverage_rho"]["mean"]
    
    results["comparison"] = {
        "coverage_improvement": (bert_rho - simcse_rho) / bert_rho * 100 if bert_rho > 0 else 0,
        "simcse_better_coverage": simcse_rho < bert_rho,
        "success_rate_improvement": (
            results["simcse"]["overall_success_rate"] - results["bert"]["overall_success_rate"]
        ) * 100,
        "task_type": task_type
    }
    
    return results
