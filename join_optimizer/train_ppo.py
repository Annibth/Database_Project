#!/usr/bin/env python3
"""
Training script for PPO-based join order optimization with stable learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
                 lr: float = 3e-4,  # Standard PPO learning rate
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,  # Standard PPO clipping
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,  # Standard entropy
                 max_grad_norm: float = 0.5,  # Standard gradient clipping
                 target_kl: float = 0.01,  # Standard KL target
                 buffer_size: int = 1024,   # Standard buffer size
                 batch_size: int = 64,      # Standard batch size
                 episodes_per_query: int = 10,  # 10 episodes per query
                 updates_per_query: int = 2,    # 2 updates per query (after 5 episodes)
                 eval_freq: int = 200,      # Evaluation frequency
                 save_freq: int = 1000,
                 max_episode_length: int = 17):  # Exactly 17 steps
        
        self.query_data_dir = Path(query_data_dir)
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
        
        # Calculate total episodes
        self.train_queries, self.test_queries = self._load_splits()
        self.num_episodes = len(self.train_queries) * self.episodes_per_query
        
        print(f"Loaded {len(self.train_queries)} training queries and {len(self.test_queries)} test queries")
        print(f"Training structure: {self.episodes_per_query} episodes per query, {self.updates_per_query} updates per query")
        print(f"Total episodes: {self.num_episodes}")
        
        # Calculate max dimensions from sample queries
        max_state_dim = 0
        max_action_dim = 0
        
        for query in self.train_queries[:100]:  # Sample first 100 queries
            env = JoinOrderEnv(query, max_relations=20, 
                              max_episode_length=self.max_episode_length)
            state = env._get_state()
            max_state_dim = max(max_state_dim, len(state))
            max_action_dim = max(max_action_dim, len(env.relation_names))
        
        # Ensure we have reasonable bounds
        max_state_dim = max(max_state_dim, 20 + 1 + 20*2 + 1)  # joined + cardinality + table_features + progress
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
            'update_stats': [],
            'best_eval_reward': float('-inf'),
            'best_episode': 0,
            'query_performance': {}  # Track performance per query
        }
        
        # Create output directories
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
    
    def _load_splits(self) -> Tuple[List, List]:
        """Load train-test splits with curriculum ordering"""
        splitter = QueryDataSplitter(self.query_data_dir)
        train_queries, test_queries = splitter.load_splits()
        
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
        """Evaluate the agent on test queries"""
        eval_rewards = []
        eval_costs = []
        eval_lengths = []
        
        for query in np.random.choice(self.test_queries, 
                                     size=min(50, len(self.test_queries)), 
                                     replace=False):
            try:
                env = JoinOrderEnv(query, max_relations=20, 
                                  max_episode_length=self.max_episode_length)
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Run for exactly max_episode_length steps (17) - same as training
                for step in range(self.max_episode_length):
                    action_mask = env.get_action_mask()
                    action, _, _ = self.agent.actor_critic.get_action(state, 
                                                                     action_mask)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                
            except Exception as e:
                print(f"Error evaluating query {query.name}: {e}")
                continue
        
        return {
            "mean_reward": np.mean(eval_rewards) if eval_rewards else 0.0,
            "std_reward": np.std(eval_rewards) if eval_rewards else 0.0,
            "mean_length": np.mean(eval_lengths) if eval_lengths else 0.0
        }
    
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
                env = JoinOrderEnv(query, max_relations=20, 
                                  max_episode_length=self.max_episode_length)
                
                # Run episode
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_cost = 0
                
                # Run for exactly max_episode_length steps (17)
                for step in range(self.max_episode_length):
                    action_mask = env.get_action_mask()
                    action, log_prob, value = self.agent.actor_critic.get_action(state, action_mask)
                    
                    next_state, reward, done, info = env.step(action)
                    
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
                
                # Update policy after every 5 episodes (episodes_per_query // updates_per_query)
                if (ep_idx + 1) % (self.episodes_per_query // self.updates_per_query) == 0:
                    if len(query_buffer) >= self.batch_size:
                        update_stats = self.agent.update(query_buffer)
                        self.training_stats['update_stats'].append(update_stats)
                        update_count += 1
                        query_buffer.reset()
                
                # Evaluation
                if total_episode_count % self.eval_freq == 0:
                    eval_stats = self._evaluate_agent()
                    self.training_stats['eval_rewards'].append(eval_stats['mean_reward'])
                    self.training_stats['eval_costs'].append(eval_stats['mean_reward'])
                    
                    # Save best model
                    if eval_stats['mean_reward'] > self.training_stats['best_eval_reward']:
                        self.training_stats['best_eval_reward'] = eval_stats['mean_reward']
                        self.training_stats['best_episode'] = total_episode_count
                        self.agent.save(f"models/best_model.pt")
                    
                    print(f"Episode {total_episode_count}: Query {query_idx}, "
                          f"Reward={episode_reward:.2f}, "
                          f"Eval Reward={eval_stats['mean_reward']:.2f}, "
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
        
        print(f"\nTraining completed! Total episodes: {total_episode_count}")
        print(f"Total updates: {update_count}")
        print(f"Average updates per episode: {update_count / total_episode_count:.2f}")
        print(f"Best evaluation reward: {self.training_stats['best_eval_reward']:.2f} "
              f"at episode {self.training_stats['best_episode']}")
        print("Individual plots saved to outputs/")
    
    def create_individual_plots(self):
        """Create individual plots for each metric"""
        # 1. Episode Rewards
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['episode_rewards'])
        plt.title('Episode Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "episode_rewards.png", dpi=300, bbox_inches='tight')
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
            plt.savefig(self.output_dir / "reward_moving_average.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Episode Costs
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_stats['episode_costs'])
        plt.title('Episode Costs Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "episode_costs.png", dpi=300, bbox_inches='tight')
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
        plt.savefig(self.output_dir / "episode_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Evaluation Rewards
        if self.training_stats['eval_rewards']:
            # Fix the array length mismatch
            eval_episodes = np.arange(0, len(self.training_stats['eval_rewards']) * self.eval_freq, self.eval_freq)
            plt.figure(figsize=(12, 8))
            plt.plot(eval_episodes, self.training_stats['eval_rewards'])
            plt.title('Evaluation Rewards Over Time')
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Reward')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "evaluation_rewards.png", dpi=300, bbox_inches='tight')
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
            plt.savefig(self.output_dir / "query_improvements.png", dpi=300, bbox_inches='tight')
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
            plt.savefig('outputs/policy_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Value Loss
            plt.figure(figsize=(12, 8))
            plt.plot(update_episodes, value_losses, 'r-', alpha=0.7)
            plt.title('Value Loss Over Updates')
            plt.xlabel('Update Number')
            plt.ylabel('Value Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/value_loss.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # KL Divergence
            plt.figure(figsize=(12, 8))
            plt.plot(update_episodes, kl_divs, 'g-', alpha=0.7)
            plt.title('KL Divergence Over Updates')
            plt.xlabel('Update Number')
            plt.ylabel('KL Divergence')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/kl_divergence.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual plots saved to {self.output_dir}/")
    
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
    """Main training function"""
    trainer = JoinOrderTrainer(
        query_data_dir="query_data",
        episodes_per_query=10,    # 10 episodes per query
        updates_per_query=2,      # 2 updates per query (after 5 episodes)
        buffer_size=1024,         # Standard buffer size
        batch_size=64,            # Standard batch size
        eval_freq=200,            # Evaluation frequency
        save_freq=1000,
        max_episode_length=17
    )
    
    trainer.train()
    trainer.create_individual_plots()
    trainer.analyze_performance()


if __name__ == "__main__":
    main() 