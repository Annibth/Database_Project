import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import random


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with improved architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        
        # Simplified but effective network architecture
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy) - simplified
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Critic head (value function) - simplified
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights with better initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use orthogonal initialization for better gradient flow
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Forward pass through the network
        
        Args:
            state: Input state tensor
            action_mask: Binary mask for valid actions
            
        Returns:
            action_probs: Action probabilities
            value: State value
        """
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Actor: compute action logits
        action_logits = self.actor(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Mask invalid actions by setting their logits to -inf
            masked_logits = action_logits.masked_fill(action_mask == 0, -1e9)
            action_probs = F.softmax(masked_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: compute state value
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray = None, 
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action from current state.
        
        Args:
            state: Current state
            action_mask: Action mask for valid actions
            deterministic: Whether to use deterministic action selection
        
        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: State value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
        else:
            action_mask_tensor = None
        
        with torch.no_grad():
            action_probs, value = self.forward(state_tensor, action_mask_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
            
            log_prob = torch.log(action_probs[0, action] + 1e-8).item()
            value = value.item()
        
        return action, log_prob, value


class PPOBuffer:
    """
    Buffer for storing PPO training data.
    """
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        """Reset the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.action_masks = []
        self.dones = []
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            value: float, log_prob: float, action_mask: np.ndarray, 
            done: bool):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask)
        self.dones.append(done)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors"""
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'log_probs': torch.FloatTensor(self.log_probs),
            'action_masks': torch.FloatTensor(np.array(self.action_masks)),
            'dones': torch.BoolTensor(self.dones)
        }
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO agent for join order optimization with improved hyperparameters.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 3e-4,  # Standard PPO learning rate
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.95, 
                 clip_ratio: float = 0.2,  # Standard PPO clipping
                 value_loss_coef: float = 0.5, 
                 entropy_coef: float = 0.01,  # Lower entropy for more stable learning
                 max_grad_norm: float = 0.5,  # Standard gradient clipping
                 target_kl: float = 0.01,  # Standard KL target
                 update_epochs: int = 4):  # Standard PPO epochs
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(state_dim, action_dim)
        
        # Initialize optimizer with better settings
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=lr,
            eps=1e-5  # Better epsilon for Adam
        )
        
        # Training statistics
        self.update_count = 0
        
        # Networks
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_div': [],
            'clip_fraction': []
        }
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool]) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
        
        Returns:
            advantages: Computed advantages
        """
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        Update the policy using PPO with standard hyperparameters.
        
        Args:
            buffer: Buffer containing training data
        
        Returns:
            stats: Training statistics
        """
        batch = buffer.get_batch()
        
        # Compute advantages
        advantages = self.compute_gae(
            batch['rewards'].numpy().tolist(),
            batch['values'].numpy().tolist(),
            batch['dones'].numpy().tolist()
        )
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + batch['values']
        
        # PPO update with standard parameters
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fractions = []
        
        for epoch in range(self.update_epochs):
            # Forward pass
            action_probs, values = self.actor_critic(
                batch['states'], batch['action_masks']
            )
            
            # Get log probs for taken actions
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(batch['actions'])
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (MSE)
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy loss (for exploration)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                         self.max_grad_norm)
            self.optimizer.step()
            
            # Compute KL divergence for early stopping
            kl_div = (batch['log_probs'] - new_log_probs).mean().item()
            
            # Compute clip fraction
            clip_fraction = (abs(ratio - 1) > self.clip_ratio).float().mean().item()
            
            # Store losses
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            kl_divs.append(kl_div)
            clip_fractions.append(clip_fraction)
            
            # Early stopping if KL divergence is too high
            if kl_div > self.target_kl:
                break
        
        self.update_count += 1
        
        return {
            'episode': self.update_count,
            'update_count': self.update_count,
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_div': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
