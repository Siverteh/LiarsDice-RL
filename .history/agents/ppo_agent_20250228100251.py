"""
Proximal Policy Optimization (PPO) agent for Liar's Dice.

This module implements a PPO agent, which is a policy gradient method
that uses a clipped surrogate objective function to ensure stable updates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Any, Optional, Tuple

from .base_agent import BaseAgent
from environment.state import ObservationEncoder


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for PPO.
    
    This network outputs both a policy distribution (actor) and
    a state value estimate (critic).
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize the actor-critic network.
        
        Args:
            input_dim: Dimension of the input (observation)
            output_dim: Dimension of the output (action space)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        shared_features = self.shared(x)
        
        # Actor: convert to action probabilities with softmax
        action_logits = self.actor(shared_features)
        
        # Critic: estimate state value
        state_value = self.critic(shared_features)
        
        return action_logits, state_value


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent for Liar's Dice.
    
    This agent uses PPO, a policy gradient method with a clipped surrogate
    objective function for stable updates.
    
    Attributes:
        player_id (int): ID of the player this agent controls
        observation_encoder (ObservationEncoder): Encoder for observations
        action_dim (int): Dimension of the action space
        device (torch.device): Device to run the model on
        network (ActorCriticNetwork): Actor-critic network
        optimizer (torch.optim.Optimizer): Optimizer for the network
        clip_param (float): Clipping parameter for PPO
        value_coef (float): Coefficient for value loss
        entropy_coef (float): Coefficient for entropy bonus
        max_grad_norm (float): Maximum gradient norm for clipping
        ppo_epochs (int): Number of PPO epochs per update
        mini_batch_size (int): Size of mini-batches for PPO updates
        gamma (float): Discount factor for future rewards
        gae_lambda (float): GAE lambda parameter
        experiences (list): List to store experiences for a PPO update
    """
    
    def __init__(
        self,
        player_id: int,
        observation_encoder: ObservationEncoder,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 32,
        gae_lambda: float = 0.95,
        hidden_dim: int = 128,
        device: str = None
    ):
        """
        Initialize the PPO agent.
        
        Args:
            player_id: ID of the player this agent controls
            observation_encoder: Encoder for observations
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            clip_param: Clipping parameter for PPO
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Size of mini-batches for PPO updates
            gae_lambda: GAE lambda parameter
            hidden_dim: Dimension of hidden layers
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__(player_id, name=f"PPO-{player_id}")
        
        self.observation_encoder = observation_encoder
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gae_lambda = gae_lambda
        
        # Set device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actor-critic network
        input_dim = observation_encoder.get_observation_shape()[0]
        self.network = ActorCriticNetwork(input_dim, action_dim, hidden_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Storage for experiences to update with PPO
        self.experiences = []
        
        # Keep mapping from action indices to actual actions during an episode
        self.action_maps = {}
    
    def act(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action using the current policy.
        
        Args:
            observation: Current observation
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        if not valid_actions:
            return None
        
        # Create mapping from action indices to actions
        action_map = {i: action for i, action in enumerate(valid_actions)}
        
        # Store the action map for this state
        obs_id = id(observation)  # Use object ID as a key
        self.action_maps[obs_id] = action_map
        
        # Get observation tensor
        obs_encoded = self.observation_encoder.encode(observation)
        obs_tensor = torch.FloatTensor(obs_encoded).unsqueeze(0).to(self.device)
        
        # Get action logits and value estimate
        with torch.no_grad():
            action_logits, state_value = self.network(obs_tensor)
        
        # Mask invalid actions with a large negative number
        mask = torch.full((1, self.action_dim), float('-inf'), device=self.device)
        
        valid_indices = list(action_map.keys())
        mask[0, valid_indices] = 0
        
        masked_logits = action_logits + mask
        
        # Create probability distribution
        probs = F.softmax(masked_logits, dim=1)
        m = Categorical(probs)
        
        # Sample action
        action_idx = m.sample().item()
        
        # Store experience for training
        log_prob = m.log_prob(torch.tensor([action_idx], device=self.device))
        
        self.experiences.append({
            'state': obs_encoded,
            'action': action_idx,
            'action_log_prob': log_prob.item(),
            'value': state_value.item(),
            'mask': mask.squeeze(0).cpu().numpy(),
            'obs_id': obs_id,
            'reward': 0.0,  # Will be filled in update()
            'done': False   # Will be filled in update()
        })
        
        return action_map[action_idx]
    
    def update(self, 
             observation: Dict[str, Any], 
             action: Dict[str, Any], 
             reward: float, 
             next_observation: Dict[str, Any], 
             done: bool, 
             valid_actions: List[Dict[str, Any]]) -> None:
        """
        Update the most recent experience with reward and done signal.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            valid_actions: Valid actions for the current state
        """
        if not self.experiences:
            return
        
        # Update the most recent experience
        self.experiences[-1]['reward'] = reward
        self.experiences[-1]['done'] = done
        
        # If done, we need the value of the final state for GAE
        if done:
            # Get the next state value estimate
            next_obs_encoded = self.observation_encoder.encode(next_observation)
            next_obs_tensor = torch.FloatTensor(next_obs_encoded).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, next_value = self.network(next_obs_tensor)
            
            # Add the final state value to experiences
            self.experiences[-1]['next_value'] = next_value.item()
    
    def train(self) -> Optional[float]:
        """
        Train the agent using PPO.
        
        This is called to indicate that we should perform a PPO update
        with the current batch of experiences.
        
        Returns:
            Average loss value if training occurred, None otherwise
        """
        # Need enough experiences to train
        if len(self.experiences) < self.mini_batch_size:
            return None
        
        # Check if we have all the required information
        if any('reward' not in exp for exp in self.experiences):
            return None
        
        # Compute advantages and returns
        self._compute_returns_and_advantages()
        
        # Convert experiences to tensors
        states = torch.FloatTensor([exp['state'] for exp in self.experiences]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in self.experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['action_log_prob'] for exp in self.experiences]).to(self.device)
        advantages = torch.FloatTensor([exp['advantage'] for exp in self.experiences]).to(self.device)
        returns = torch.FloatTensor([exp['return'] for exp in self.experiences]).to(self.device)
        masks = torch.FloatTensor([exp['mask'] for exp in self.experiences]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Generate random mini-batch indices
            batch_size = len(self.experiences)
            indices = np.random.permutation(batch_size)
            
            # Perform mini-batch updates
            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_masks = masks[mb_indices]
                
                # Forward pass
                action_logits, values = self.network(mb_states)
                
                # Apply action masks
                masked_logits = action_logits + mb_masks
                
                # Get new action probabilities
                probs = F.softmax(masked_logits, dim=1)
                dist = Categorical(probs)
                
                # Get new log probabilities
                new_log_probs = dist.log_prob(mb_actions)
                
                # Calculate ratios
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                values = values.squeeze(1)
                value_loss = F.mse_loss(values, mb_returns)
                
                # Calculate entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Perform backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Clear experiences after update
        self.experiences = []
        self.action_maps = {}
        
        # Return average loss
        num_updates = self.ppo_epochs * (len(indices) // self.mini_batch_size + 1)
        return total_loss / num_updates if num_updates > 0 else 0.0
    
    def _compute_returns_and_advantages(self) -> None:
        """
        Compute returns and advantages for all experiences using GAE.
        
        This method adds 'return' and 'advantage' fields to each experience.
        """
        # Get the last value if the episode isn't done
        if not self.experiences[-1]['done'] and 'next_value' not in self.experiences[-1]:
            # We need to estimate the value of the next state
            last_exp = self.experiences[-1]
            obs_id = last_exp['obs_id']
            
            # Create a placeholder next observation
            next_obs_tensor = torch.FloatTensor(last_exp['state']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, next_value = self.network(next_obs_tensor)
            
            self.experiences[-1]['next_value'] = next_value.item()
        
        # Initialize gae
        gae = 0
        
        # Loop through experiences in reverse order
        for i in reversed(range(len(self.experiences))):
            # Get current experience
            exp = self.experiences[i]
            
            # Get next value
            if i == len(self.experiences) - 1:
                next_value = exp['next_value'] if 'next_value' in exp else 0.0
            else:
                next_value = self.experiences[i + 1]['value']
            
            # Get current value and reward
            value = exp['value']
            reward = exp['reward']
            done = exp['done']
            
            # Calculate delta
            delta = reward + self.gamma * next_value * (1 - done) - value
            
            # Calculate GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            
            # Add advantage and return to experience
            self.experiences[i]['advantage'] = gae
            self.experiences[i]['return'] = gae + value
    
    def save(self, path: str) -> None:
        """
        Save the agent's network.
        
        Args:
            path: Path to save the model to
        """
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the agent's network.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])