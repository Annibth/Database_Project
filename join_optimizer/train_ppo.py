#!/usr/bin/env python3
"""
Training script for PPO-based join order optimization with stable learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import math

from env.join_env import JoinOrderEnv
from agent.ppo import PPOAgent, PPOBuffer
from data_splitter import QueryDataSplitter


class JoinOrderTrainer:
    """
    Trainer for join order optimization using PPO with stable learning.
    """
    
    def __init__(self, query_data_dir: str, 
                 lr: float = 0.0003,  # Higher learning rate for faster learning
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.1,  # Tighter clipping for stability
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.05,  # Higher entropy for exploration
                 max_grad_norm: float = 0.5,
                 target_kl: float = 0.01,
                 buffer_size: int = 512,   # Smaller buffer for more frequent updates
                 batch_size: int = 32,     # Smaller batch size for more updates
                 episodes_per_query: int = 20,  # More episodes per query
                 updates_per_query: int = 4,    # More updates per query
                 eval_freq: int = 100,     # More frequent evaluation
                 save_freq: int = 500,
                 max_episode_length: int = None, # Will be set dynamically
                 relations_nr: int = 0,
                 type: str = "all",
                 train_ratio: float = 0.9):  
        
        self.query_data_dir = Path(query_data_dir)
        self.train_ratio = train_ratio
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episodes_per_query = episodes_per_query
        self.updates_per_query = updates_per_query
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.max_episode_length = max_episode_length
        
        # NEW
        self.relations = set()
        
        # Calculate total episodes
        self.train_queries, self.test_queries = self._load_splits(relations_nr, type)
        self.num_episodes = len(self.train_queries) * self.episodes_per_query
        
        print(f"Loaded {len(self.train_queries)} training queries and {len(self.test_queries)} test queries")
        print(f"Training structure: {self.episodes_per_query} episodes per query, {self.updates_per_query} updates per query")
        print(f"Total episodes: {self.num_episodes}")
        
        # Calculate max dimensions from sample queries
        max_state_dim = 0
        max_action_dim = 0
        
        for query in self.train_queries[:100]:  # Sample first 100 queries
            env = JoinOrderEnv(query, max_relations=20)
            state = env.reset()
            max_state_dim = max(max_state_dim, len(state))
            max_action_dim = max(max_action_dim, len(env.relation_names))
        
        # New state dimension: 20 + 1 + 80 + 3 + 20 + 20 = 144 features
        # joined_vector(20) + cardinality(1) + table_features(80) + progress(3) + history(20) + optimal(20)
        max_state_dim = max(max_state_dim, 144)
        max_action_dim = 20  # Use max_relations for action dimension
        
        print(f"Using max state dim: {max_state_dim}, max action dim: {max_action_dim}")
        
        # Initialize agent with calculated dimensions
        self.agent = PPOAgent(
            state_dim=max_state_dim,
            action_dim=max_action_dim,
            lr=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_ratio=self.clip_ratio,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl
        )
        
        # Initialize training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_costs': [],
            'eval_rewards': [],
            'eval_costs': [],
            'eval_gt_accuracy': [],  # Ground truth accuracy during evaluation
            'eval_cost_ratio': [],   # Mean cost ratio (agent/GT) during evaluation
            'eval_completion_rate': [],  # Fraction of completed join orders
            'eval_baseline_cost_ratio': [],  # Mean cost ratio of greedy baseline
            'eval_episodes': [],     # Episode numbers when evaluation occurred
            'update_stats': [],
            'best_eval_reward': float('-inf'),
            'best_eval_accuracy': 0.0,
            'best_episode': 0,
            'query_performance': {}  # Track performance per query
        }
        
        # Create output directories (per run)
        if type == "all":
            self.run_output_path = Path(f"outputs/{type}")
        else:
            self.run_output_path = Path(f"outputs/{type}/{relations_nr}")
        self.run_output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
    
    def _load_splits(self, relations_nr, type) -> Tuple[List, List]:
        """Load train-test splits with curriculum ordering"""
        splitter = QueryDataSplitter(self.query_data_dir, relations_nr, type, train_ratio=self.train_ratio)
        train_queries, test_queries = splitter.load_splits()
        self.relations = splitter.relations
        
        # Sort training queries by complexity for curriculum learning
        # Complexity = number of relations + average cardinality
        def query_complexity(query):
            num_relations = len(query.relations)
            avg_cardinality = sum(r["cardinality"] for r in query.relations) / num_relations
            return num_relations + math.log(avg_cardinality + 1)
        
        train_queries.sort(key=query_complexity)
        print(f"Sorted {len(train_queries)} training queries by complexity for curriculum learning")
        
        return train_queries, test_queries
    
    def _evaluate_agent(self) -> Dict[str, float]:
        """Evaluate the agent on test queries with comprehensive metrics"""
        eval_rewards = []
        eval_costs = []
        eval_lengths = []
        eval_join_orders = []
        eval_ground_truth_accuracy = []
        eval_cost_ratios = []
        eval_completions = []
        baseline_cost_ratios = []
        
        for query in np.random.choice(self.test_queries, 
                                      size=min(50, len(self.test_queries)),
                                     replace=False):
            try:
                env = JoinOrderEnv(query, max_relations=20)
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                join_order = []
                
                # Run until episode is complete - same as training
                while not env.done:
                    action_mask = env.get_action_mask()
                    action, _, _ = self.agent.actor_critic.get_action(state, 
                                                                     action_mask, deterministic=True)
                    state, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    # Track join order
                    if 'joined_relation' in info:
                        join_order.append(info['joined_relation'])
                
                # Compute costs and accuracy when possible
                gt_order = []
                if hasattr(query, 'left_deep_tree_min_order') and query.left_deep_tree_min_order:
                    gt_order = self._parse_ground_truth_order(query)
                    accuracy = self._calculate_join_order_accuracy(gt_order, join_order)
                    eval_ground_truth_accuracy.append(accuracy)

                    # Compute costs
                    gt_cost = self._calculate_join_order_cost(query, gt_order)
                    agent_cost = self._calculate_join_order_cost(query, join_order)
                    if gt_cost > 0 and np.isfinite(agent_cost):
                        eval_cost_ratios.append(agent_cost / gt_cost)

                # Baseline greedy cost ratio (if GT available)
                if gt_order:
                    greedy_order = self._get_greedy_join_order(query)
                    if greedy_order:
                        greedy_cost = self._calculate_join_order_cost(query, greedy_order)
                        if gt_cost > 0 and np.isfinite(greedy_cost):
                            baseline_cost_ratios.append(greedy_cost / gt_cost)

                # Completion flag
                eval_completions.append(1.0 if len(join_order) == len(query.relations) else 0.0)

                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                eval_join_orders.append(join_order)
                
                
            except Exception as e:
                print(f"Error evaluating query {query.name}: {e}")
                continue
        
        # Calculate additional metrics
        mean_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        std_reward = np.std(eval_rewards) if eval_rewards else 0.0
        mean_gt_accuracy = np.mean(eval_ground_truth_accuracy) if eval_ground_truth_accuracy else 0.0
        mean_cost_ratio = np.mean(eval_cost_ratios) if eval_cost_ratios else float('inf')
        completion_rate = np.mean(eval_completions) if eval_completions else 0.0
        mean_baseline_cost_ratio = np.mean(baseline_cost_ratios) if baseline_cost_ratios else float('inf')
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": np.mean(eval_lengths) if eval_lengths else 0.0,
            "mean_gt_accuracy": mean_gt_accuracy,
            "num_queries_evaluated": len(eval_rewards),
            "num_with_ground_truth": len(eval_ground_truth_accuracy),
            "mean_cost_ratio": mean_cost_ratio,
            "completion_rate": completion_rate,
            "mean_baseline_cost_ratio": mean_baseline_cost_ratio
        }

    def _calculate_join_order_cost(self, query, join_order: List[str]) -> float:
        """Calculate cost for a join order using query.sizes (order-insensitive stepwise)."""
        if not join_order:
            return float('inf')
        # Direct match for full order if present
        for size_info in query.sizes:
            if size_info.get("relations") == join_order:
                return size_info.get("cardinality", float('inf'))
        # Stepwise accumulate last known cardinality
        total_cost = 0
        current = []
        for rel in join_order:
            current.append(rel)
            joined_sorted = sorted(current)
            for size_info in query.sizes:
                if sorted(size_info.get("relations", [])) == joined_sorted:
                    total_cost = size_info.get("cardinality", total_cost)
                    break
        # If we couldn't find any intermediate cardinality, return inf to indicate unknown/poor
        return total_cost if total_cost > 0 else float('inf')
    
    def _parse_ground_truth_order(self, query) -> List[str]:
        """Parse ground truth join order from string format"""
        if not hasattr(query, 'left_deep_tree_min_order') or not query.left_deep_tree_min_order:
            return []
        
        order_str = query.left_deep_tree_min_order
        # Remove parentheses and 'join' keywords
        cleaned = order_str.replace('(', '').replace(')', '').replace(' join ', ' ')
        table_names = cleaned.split()
        
        # Extract unique tables in order (handle duplicates)
        seen = set()
        unique_tables = []
        for table in table_names:
            if table not in seen:
                seen.add(table)
                unique_tables.append(table)
        
        return unique_tables

    def _get_greedy_join_order(self, query) -> List[str]:
        """Compute a simple greedy join order: smallest-table-first with feasibility.
        Uses the environment's legal-action mask to ensure connectivity.
        """
        try:
            env = JoinOrderEnv(query, max_relations=20)
            env.reset()
            # Precompute per-relation base cardinalities
            base_card = {}
            for rel_info in query.relations:
                base_card[rel_info["name"]] = rel_info["cardinality"]
            order = []
            while not env.done:
                legal_idxs = env.get_legal_actions()
                if not legal_idxs:
                    break
                # pick legal relation with smallest base cardinality
                best_idx = min(legal_idxs, key=lambda i: base_card.get(env.relation_names[i], float('inf')))
                state, reward, done, info = env.step(best_idx)
                if 'joined_relation' in info:
                    order.append(info['joined_relation'])
            return order
        except Exception:
            return []
    
    def _calculate_join_order_accuracy(self, ground_truth: List[str], agent_order: List[str]) -> float:
        """Calculate accuracy of join order prediction"""
        if not ground_truth or not agent_order:
            return 0.0
        
        # Calculate position-based accuracy
        correct_positions = 0
        min_length = min(len(ground_truth), len(agent_order))
        
        for i in range(min_length):
            if i < len(ground_truth) and i < len(agent_order):
                if ground_truth[i] == agent_order[i]:
                    correct_positions += 1
        
        return correct_positions / len(ground_truth) if ground_truth else 0.0
    
    def train(self):
        """Train the agent with stable learning"""
        print("Starting training with stable learning approach...")
        print(f"Buffer size: {self.buffer_size}, Episodes per query: {self.episodes_per_query}")
        print(f"Updates per query: {self.updates_per_query}")
        
        update_count = 0
        total_episode_count = 0
        
        # Train on each query
        for query_idx, query in enumerate(tqdm(self.train_queries, desc="Training on queries")):
            query_buffer = PPOBuffer(self.buffer_size)  # Buffer for this query
            query_rewards = []  # Track rewards for this query
            
            # Run episodes for this query
            for ep_idx in range(self.episodes_per_query):
                env = JoinOrderEnv(query, max_relations=20)
                
                # Run episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_cost = 0
                
                # Run until episode is complete or timeout
                while not env.done:
                    action_mask = env.get_action_mask()
                    action, log_prob, value = self.agent.actor_critic.get_action(state, action_mask)
                    try:
                        next_state, reward, done, info = env.step(action)
                    except Exception as e:
                        # Skip problematic episode
                        break
                    query_buffer.add(state, action, reward, value, log_prob, action_mask, done)
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    episode_cost = info.get('current_cardinality', episode_cost)
                
                # Store episode statistics
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['episode_lengths'].append(episode_length)
                self.training_stats['episode_costs'].append(episode_cost)
                query_rewards.append(episode_reward)
                total_episode_count += 1
                
                # Update policy whenever enough samples are accumulated
                if len(query_buffer) >= self.batch_size:
                    update_stats = self.agent.update(query_buffer)
                    self.training_stats['update_stats'].append(update_stats)
                    update_count += 1
                    query_buffer.reset()
                
                # Evaluation
                if total_episode_count % self.eval_freq == 0:
                    eval_stats = self._evaluate_agent()
                    self.training_stats['eval_rewards'].append(eval_stats['mean_reward'])
                    self.training_stats['eval_costs'].append(eval_stats['mean_length'])
                    self.training_stats['eval_gt_accuracy'].append(eval_stats['mean_gt_accuracy'])
                    # New metrics appended (no change to existing plots)
                    self.training_stats['eval_cost_ratio'].append(eval_stats.get('mean_cost_ratio', float('inf')))
                    self.training_stats['eval_completion_rate'].append(eval_stats.get('completion_rate', 0.0))
                    self.training_stats['eval_baseline_cost_ratio'].append(eval_stats.get('mean_baseline_cost_ratio', float('inf')))
                    self.training_stats['eval_episodes'].append(total_episode_count)
                    
                    # Save best model based on ground truth accuracy if available, otherwise reward
                    if eval_stats['num_with_ground_truth'] > 0:
                        if eval_stats['mean_gt_accuracy'] > self.training_stats['best_eval_accuracy']:
                            self.training_stats['best_eval_accuracy'] = eval_stats['mean_gt_accuracy']
                            self.training_stats['best_eval_reward'] = eval_stats['mean_reward']
                            self.training_stats['best_episode'] = total_episode_count
                            self.agent.save(f"models/best_model.pt")
                    else:
                        if eval_stats['mean_reward'] > self.training_stats['best_eval_reward']:
                            self.training_stats['best_eval_reward'] = eval_stats['mean_reward']
                            self.training_stats['best_episode'] = total_episode_count
                            self.agent.save(f"models/best_model.pt")
                    
                    print(f"Episode {total_episode_count}: Query {query_idx}, "
                          f"Reward={episode_reward:.2f}, "
                          f"Eval Reward={eval_stats['mean_reward']:.2f}, "
                          f"GT Accuracy={eval_stats['mean_gt_accuracy']:.3f}, "
                          f"Updates={update_count}")
                
                # Save checkpoint
                if total_episode_count % self.save_freq == 0 and total_episode_count > 0:
                    self.agent.save(f"models/checkpoint_episode_{total_episode_count}.pt")
            
            # Store query performance
            self.training_stats['query_performance'][query_idx] = {
                'mean_reward': np.mean(query_rewards),
                'std_reward': np.std(query_rewards),
                'min_reward': np.min(query_rewards),
                'max_reward': np.max(query_rewards),
                'improvement': np.mean(query_rewards[-3:]) - np.mean(query_rewards[:3]) if len(query_rewards) >= 6 else 0
            }
        
        # Save final model
        self.agent.save("models/final_model.pt")
        # Save training statistics
        try:
            with open(self.run_output_path / "training_stats.json", "w") as f:
                json.dump(self.training_stats, f, indent=2, default=str)
        except Exception:
            pass
        
        print(f"\nTraining completed! Total episodes: {total_episode_count}")
        print(f"Total updates: {update_count}")
        print(f"Average updates per episode: {update_count / total_episode_count:.2f}")
        print(f"Best evaluation reward: {self.training_stats['best_eval_reward']:.2f} "
              f"at episode {self.training_stats['best_episode']}")
        print("Individual plots saved to outputs/")
    
    def create_individual_plots(self, rel_nr, type):
        """Create individual plots for each metric"""
        if type == "all":
            directory = Path(f"outputs/{type}")
            directory.mkdir(exist_ok = True)
            path = f"outputs/{type}"
        else:
            directory = Path(f"outputs/{type}")
            directory.mkdir(exist_ok = True)
            Path(f"outputs/{type}/{str(rel_nr)}").mkdir(exist_ok= True)
            path = f"outputs/{type}/{str(rel_nr)}"
        
        # 1. Episode Rewards
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['episode_rewards'])
        plt.title('Episode Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path}/episode_rewards.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Moving Average of Rewards
        window = 50
        rewards = self.training_stats['episode_rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.figure(figsize=(12, 8))
            plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
            plt.plot(rewards, alpha=0.3, label='Raw rewards')
            plt.title('Reward Moving Average')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{path}/reward_moving_average.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2.5. Learning Progress Analysis
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Episode rewards with trend
        plt.subplot(2, 2, 1)
        plt.plot(rewards, alpha=0.3, color='lightblue', label='Raw rewards')
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, 'b-', linewidth=2, label=f'Moving Avg (window={window})')
        
        # Add trend line
        if len(rewards) > 10:
            z = np.polyfit(range(len(rewards)), rewards, 1)
            p = np.poly1d(z)
            plt.plot(range(len(rewards)), p(range(len(rewards))), "r--", alpha=0.8, label='Trend')
        
        plt.title('Episode Rewards with Learning Trend')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Evaluation performance
        plt.subplot(2, 2, 2)
        if self.training_stats['eval_rewards']:
            eval_episodes = self.training_stats['eval_episodes']
            plt.plot(eval_episodes, self.training_stats['eval_rewards'], 'g-o', linewidth=2, markersize=4, label='Eval Reward')
            
            if self.training_stats['eval_gt_accuracy'] and any(acc > 0 for acc in self.training_stats['eval_gt_accuracy']):
                ax2 = plt.gca().twinx()
                ax2.plot(eval_episodes, self.training_stats['eval_gt_accuracy'], 'r-s', linewidth=2, markersize=4, label='GT Accuracy')
                ax2.set_ylabel('Ground Truth Accuracy', color='r')
                ax2.set_ylim(0, 1)
                ax2.legend(loc='upper right')
        
        plt.title('Evaluation Performance Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Learning stability (rolling std)
        plt.subplot(2, 2, 3)
        if len(rewards) >= window:
            rolling_std = []
            for i in range(window, len(rewards) + 1):
                rolling_std.append(np.std(rewards[i-window:i]))
            plt.plot(range(window, len(rewards) + 1), rolling_std, 'purple', linewidth=2, label='Rolling Std Dev')
            plt.title('Learning Stability (Lower = More Stable)')
            plt.xlabel('Episode')
            plt.ylabel('Standard Deviation')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Subplot 4: Policy improvement indicators
        plt.subplot(2, 2, 4)
        if self.training_stats['query_performance']:
            query_improvements = [perf['improvement'] for perf in self.training_stats['query_performance'].values()]
            plt.plot(query_improvements, 'o-', linewidth=2, markersize=4, label='Query Improvement')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='No Improvement')
            plt.title('Learning Improvement per Query')
            plt.xlabel('Query Index')
            plt.ylabel('Improvement (Final - Initial)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{path}/learning_progress_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Episode Costs
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['episode_costs'])
        plt.title('Episode Costs Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path}/episode_costs.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Episode Lengths
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['episode_lengths'])
        plt.axhline(y=17, color='r', linestyle='--', alpha=0.7, label='Expected Length (17)')
        plt.title('Episode Lengths Over Time (Should be 17 for all episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{path}/episode_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Evaluation Rewards
        if self.training_stats['eval_rewards']:
            eval_episodes = self.training_stats['eval_episodes']
            plt.figure(figsize=(12, 8))
            plt.plot(eval_episodes, self.training_stats['eval_rewards'], 'b-', linewidth=2, label='Evaluation Reward')
            plt.title('Evaluation Rewards Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{path}/evaluation_rewards.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Ground Truth Accuracy (if available)
        if self.training_stats['eval_gt_accuracy'] and any(acc > 0 for acc in self.training_stats['eval_gt_accuracy']):
            eval_episodes = self.training_stats['eval_episodes']
            plt.figure(figsize=(12, 8))
            plt.plot(eval_episodes, self.training_stats['eval_gt_accuracy'], 'g-', linewidth=2, label='Ground Truth Accuracy')
            plt.title('Ground Truth Join Order Accuracy Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{path}/ground_truth_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. Combined Evaluation Plot
        if self.training_stats['eval_rewards']:
            eval_episodes = self.training_stats['eval_episodes']
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Rewards
            ax1.plot(eval_episodes, self.training_stats['eval_rewards'], 'b-', linewidth=2, label='Evaluation Reward')
            ax1.set_title('Evaluation Performance Over Time')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Ground Truth Accuracy
            if self.training_stats['eval_gt_accuracy'] and any(acc > 0 for acc in self.training_stats['eval_gt_accuracy']):
                ax2.plot(eval_episodes, self.training_stats['eval_gt_accuracy'], 'g-', linewidth=2, label='Ground Truth Accuracy')
                ax2.set_ylabel('Accuracy')
                ax2.set_ylim(0, 1)
            else:
                ax2.text(0.5, 0.5, 'No Ground Truth Data Available', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_ylabel('Accuracy')
            
            ax2.set_xlabel('Episode')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{path}/combined_evaluation.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7.1 Evaluation Cost Ratio over Time (new)
        if self.training_stats['eval_cost_ratio']:
            eval_episodes = np.array(self.training_stats['eval_episodes'])
            ratios = np.array(self.training_stats['eval_cost_ratio'], dtype=float)
            base_ratios = np.array(self.training_stats['eval_baseline_cost_ratio'], dtype=float) if self.training_stats['eval_baseline_cost_ratio'] else None
            mask = np.isfinite(ratios)
            plt.figure(figsize=(12, 8))
            if mask.any():
                plt.plot(eval_episodes[mask], ratios[mask], 'b-o', linewidth=2, markersize=4, label='Agent Cost Ratio')
            if base_ratios is not None:
                bmask = np.isfinite(base_ratios)
                if bmask.any():
                    plt.plot(eval_episodes[bmask], base_ratios[bmask], 'r--s', linewidth=2, markersize=4, label='Greedy Baseline Cost Ratio')
            plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Optimal (1.0)')
            plt.title('Evaluation Cost Ratio Over Time (Agent vs Baseline)')
            plt.xlabel('Episode')
            plt.ylabel('Cost Ratio (Agent/GT)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{path}/evaluation_cost_ratio.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 7.2 Evaluation Completion Rate over Time (new)
        if self.training_stats['eval_completion_rate']:
            eval_episodes = self.training_stats['eval_episodes']
            plt.figure(figsize=(12, 8))
            plt.plot(eval_episodes, self.training_stats['eval_completion_rate'], 'm-^', linewidth=2, markersize=4, label='Completion Rate')
            plt.title('Evaluation Completion Rate Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Completion Rate')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{path}/evaluation_completion_rate.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Query Performance Analysis
        if self.training_stats['query_performance']:
            query_improvements = [perf['improvement'] for perf in self.training_stats['query_performance'].values()]
            plt.figure(figsize=(12, 8))
            plt.plot(query_improvements)
            plt.title('Learning Improvement per Query')
            plt.xlabel('Query Index')
            plt.ylabel('Improvement (Final - Initial Reward)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{path}/query_improvements.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot update statistics if available
        if self.training_stats['update_stats']:
            update_episodes = list(range(len(self.training_stats['update_stats'])))
            policy_losses = [stat['policy_loss'] for stat in self.training_stats['update_stats']]
            value_losses = [stat['value_loss'] for stat in self.training_stats['update_stats']]
            kl_divs = [stat['kl_div'] for stat in self.training_stats['update_stats']]
            
            # Policy Loss
            plt.figure(figsize=(12, 8))
            plt.plot(update_episodes, policy_losses, 'b-', alpha=0.7)
            plt.title('Policy Loss Over Updates')
            plt.xlabel('Update Number')
            plt.ylabel('Policy Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{path}/policy_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Value Loss
            plt.figure(figsize=(12, 8))
            plt.plot(update_episodes, value_losses, 'r-', alpha=0.7)
            plt.title('Value Loss Over Updates')
            plt.xlabel('Update Number')
            plt.ylabel('Value Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{path}/value_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # KL Divergence
            plt.figure(figsize=(12, 8))
            plt.plot(update_episodes, kl_divs, 'g-', alpha=0.7)
            plt.title('KL Divergence Over Updates')
            plt.xlabel('Update Number')
            plt.ylabel('KL Divergence')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{path}/kl_divergence.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 8. Final Learning Summary
        self._create_learning_summary_plot(path)
        
        print(f"Individual plots saved to {self.run_output_path}/")
    
    def _create_learning_summary_plot(self, path):
        """Create a comprehensive learning summary plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Join Order Optimization Learning Summary', fontsize=16, fontweight='bold')
        
        # 1. Reward progression
        rewards = self.training_stats['episode_rewards']
        ax1.plot(rewards, alpha=0.3, color='lightblue', label='Raw rewards')
        
        # Moving average
        window = min(50, len(rewards) // 4)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), moving_avg, 'b-', linewidth=2, label=f'Moving Avg (window={window})')
        
        # Trend analysis
        if len(rewards) > 10:
            z = np.polyfit(range(len(rewards)), rewards, 1)
            p = np.poly1d(z)
            ax1.plot(range(len(rewards)), p(range(len(rewards))), "r--", alpha=0.8, label='Trend')
            
            # Calculate improvement
            initial_avg = np.mean(rewards[:len(rewards)//4])
            final_avg = np.mean(rewards[-len(rewards)//4:])
            improvement = final_avg - initial_avg
            
            ax1.text(0.02, 0.98, f'Improvement: {improvement:.2f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_title('Episode Rewards Progression')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Evaluation performance
        if self.training_stats['eval_rewards']:
            eval_episodes = self.training_stats['eval_episodes']
            ax2.plot(eval_episodes, self.training_stats['eval_rewards'], 'g-o', linewidth=2, markersize=4, label='Eval Reward')
            
            if self.training_stats['eval_gt_accuracy'] and any(acc > 0 for acc in self.training_stats['eval_gt_accuracy']):
                ax2_twin = ax2.twinx()
                ax2_twin.plot(eval_episodes, self.training_stats['eval_gt_accuracy'], 'r-s', linewidth=2, markersize=4, label='GT Accuracy')
                ax2_twin.set_ylabel('Ground Truth Accuracy', color='r')
                ax2_twin.set_ylim(0, 1)
                ax2_twin.legend(loc='upper right')
        
        ax2.set_title('Evaluation Performance')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning stability
        if len(rewards) >= window:
            rolling_std = []
            for i in range(window, len(rewards) + 1):
                rolling_std.append(np.std(rewards[i-window:i]))
            ax3.plot(range(window, len(rewards) + 1), rolling_std, 'purple', linewidth=2, label='Rolling Std Dev')
            ax3.set_title('Learning Stability')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Standard Deviation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Policy improvement per query
        if self.training_stats['query_performance']:
            query_improvements = [perf['improvement'] for perf in self.training_stats['query_performance'].values()]
            ax4.plot(query_improvements, 'o-', linewidth=2, markersize=4, label='Query Improvement')
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='No Improvement')
            
            # Calculate improvement statistics
            positive_improvements = [imp for imp in query_improvements if imp > 0]
            improvement_rate = len(positive_improvements) / len(query_improvements) if query_improvements else 0
            
            ax4.text(0.02, 0.98, f'Improvement Rate: {improvement_rate:.1%}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            ax4.set_title('Learning per Query')
            ax4.set_xlabel('Query Index')
            ax4.set_ylabel('Improvement (Final - Initial)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{path}/learning_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_performance(self):
        """Analyze training performance"""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        episode_rewards = self.training_stats['episode_rewards']
        eval_rewards = self.training_stats['eval_rewards']
        
        # Calculate statistics
        initial_reward = np.mean(episode_rewards[:50])
        final_reward = np.mean(episode_rewards[-50:])
        improvement = final_reward - initial_reward
        
        print(f"Training Statistics:")
        print(f"  Total episodes: {len(episode_rewards)}")
        print(f"  Initial average reward: {initial_reward:.2f}")
        print(f"  Final average reward: {final_reward:.2f}")
        print(f"  Improvement: {improvement:.2f}")
        print(f"  Best episode reward: {max(episode_rewards):.2f}")
        print(f"  Worst episode reward: {min(episode_rewards):.2f}")
        
        if self.training_stats['update_stats']:
            total_updates = len(self.training_stats['update_stats'])
            print(f"  Total policy updates: {total_updates}")
            print(f"  Updates per episode: {total_updates/len(episode_rewards):.2f}")
        
        if eval_rewards:
            print(f"\nEvaluation Statistics:")
            print(f"  Final evaluation reward: {eval_rewards[-1]:.2f}")
            print(f"  Best evaluation reward: {max(eval_rewards):.2f}")
            print(f"  Evaluation improvement: {eval_rewards[-1] - eval_rewards[0]:.2f}")
            print(f"  Best model saved at episode: {self.training_stats['best_episode']}")
            
            # Ground truth accuracy statistics
            if self.training_stats['eval_gt_accuracy'] and any(acc > 0 for acc in self.training_stats['eval_gt_accuracy']):
                gt_accuracies = [acc for acc in self.training_stats['eval_gt_accuracy'] if acc > 0]
                print(f"\nGround Truth Accuracy Statistics:")
                print(f"  Final GT accuracy: {gt_accuracies[-1]:.3f}")
                print(f"  Best GT accuracy: {max(gt_accuracies):.3f}")
                print(f"  GT accuracy improvement: {gt_accuracies[-1] - gt_accuracies[0]:.3f}")
                print(f"  Queries with GT data: {self.training_stats['eval_gt_accuracy'].count(0)}/{len(self.training_stats['eval_gt_accuracy'])}")
            else:
                print(f"\nGround Truth Accuracy: No ground truth data available")
        
        # Analyze query performance
        if self.training_stats['query_performance']:
            improvements = [perf['improvement'] for perf in self.training_stats['query_performance'].values()]
            positive_improvements = [imp for imp in improvements if imp > 0]
            print(f"\nQuery Learning Analysis:")
            print(f"  Queries with positive improvement: {len(positive_improvements)}/{len(improvements)}")
            print(f"  Average improvement per query: {np.mean(improvements):.2f}")
            print(f"  Best query improvement: {max(improvements):.2f}")
            print(f"  Worst query improvement: {min(improvements):.2f}")
        
        # Check for real improvements
        if improvement > 1.0:  # Lower threshold for more realistic improvement
            print(f"\n✅ REAL IMPROVEMENT DETECTED!")
            print(f"   Improvement: {improvement:.2f} (threshold: 1.0)")
        else:
            print(f"\n⚠️  Limited improvement detected")
            print(f"   Improvement: {improvement:.2f} (threshold: 1.0)")
            print(f"   Consider adjusting hyperparameters for better learning")

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for join order optimization")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "series-exact", "series-leq"], help="Run mode: single or series runs")
    parser.add_argument("--relations", type=int, default=0, help="For single mode: relation filter (ignored for type=all)")
    parser.add_argument("--type", type=str, default="all", choices=["exact","leq","meq","all"], help="For single mode: query selection type")
    parser.add_argument("--min_relations", type=int, default=4, help="Series mode: min relation count (inclusive)")
    parser.add_argument("--max_relations", type=int, default=12, help="Series mode: max relation count (inclusive)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/test split ratio")
    parser.add_argument("--episodes_per_query", type=int, default=10, help="Episodes per query")
    parser.add_argument("--updates_per_query", type=int, default=2, help="Policy updates per query")
    parser.add_argument("--buffer_size", type=int, default=256, help="PPO buffer size")
    parser.add_argument("--batch_size", type=int, default=16, help="PPO batch size")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency in episodes")
    parser.add_argument("--save_freq", type=int, default=1000, help="Checkpoint save frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seeds
    PPOAgent.set_seed(args.seed)
    
    # Decide run mode
    if args.mode == "single":
        trainer = JoinOrderTrainer(
            query_data_dir="query_data",
            episodes_per_query=args.episodes_per_query,
            updates_per_query=args.updates_per_query,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            max_episode_length=17,
            relations_nr=args.relations,
            type=args.type,
            train_ratio=args.train_ratio
        )

        if len(trainer.train_queries) == 0:
            print("No training queries found for the given filter. Skipping.")
            return
        trainer.train()
        trainer.create_individual_plots(args.relations, args.type)
        trainer.analyze_performance()
    elif args.mode == "series-exact":
        for k in range(args.min_relations, args.max_relations + 1):
            print(f"\n===== Running EXACT series for k={k} =====")
            trainer = JoinOrderTrainer(
                query_data_dir="query_data",
                episodes_per_query=args.episodes_per_query,
                updates_per_query=args.updates_per_query,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                eval_freq=args.eval_freq,
                save_freq=args.save_freq,
                max_episode_length=17,
                relations_nr=k,
                type="exact",
                train_ratio=args.train_ratio
            )
            if len(trainer.train_queries) == 0:
                print(f"No training queries found for exact k={k}. Skipping.")
                continue
            trainer.train()
            trainer.create_individual_plots(k, "exact")
            trainer.analyze_performance()
    elif args.mode == "series-leq":
        # Accumulation runs should start at least from 5 (<=5, <=6, ...)
        start_k = max(5, args.min_relations)
        for k in range(start_k, args.max_relations + 1):
            print(f"\n===== Running LEQ series for k={k} (<= {k}) =====")
            trainer = JoinOrderTrainer(
                query_data_dir="query_data",
                episodes_per_query=args.episodes_per_query,
                updates_per_query=args.updates_per_query,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                eval_freq=args.eval_freq,
                save_freq=args.save_freq,
                max_episode_length=17,
                relations_nr=k,
                type="leq",
                train_ratio=args.train_ratio
            )
            if len(trainer.train_queries) == 0:
                print(f"No training queries found for leq k={k}. Skipping.")
                continue
            trainer.train()
            trainer.create_individual_plots(k, "leq")
            trainer.analyze_performance()

if __name__ == "__main__":
    main()