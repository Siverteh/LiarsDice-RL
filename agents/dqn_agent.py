"""
Enhanced DQN agent implementation for Liar's Dice.

This module implements an improved DQN agent with Prioritized Experience 
Replay, Double DQN, and Dueling architecture.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import namedtuple, deque

from agents.base_agent import RLAgent

# Define transition tuple for experience replay
Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done', 'weight'))


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network for better state-action value estimation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for dim in hidden_dims[:-1]:
            self.feature_layers.append(nn.Linear(prev_dim, dim))
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process input through feature layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Get value and advantage estimates
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine value and advantages using the dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform sampling, 1 = full prioritization)
            beta_start: Initial importance sampling correction (0 = no correction, 1 = full correction)
            beta_frames: Number of frames over which beta increases to 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences based on their priorities."""
        if self.size == 0:
            return [], [], [], [], [], []
        
        # Update beta value
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get batch elements
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones), 
            weights,
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            # Add a small constant to avoid zero priority
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return self.size


class DQNAgent(RLAgent):
    """
    Enhanced DQN agent for Liar's Dice with:
    - Prioritized Experience Replay
    - Double DQN for more stable learning
    - Dueling network architecture
    - Linear epsilon annealing
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        target_update_freq: int = 1000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        device: str = 'auto',
        double_dqn: bool = True
    ):
        """
        Initialize the enhanced DQN agent.
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Number of steps to decay epsilon
            target_update_freq: Frequency of target network updates
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            hidden_dims: Dimensions of hidden layers
            per_alpha: Alpha parameter for prioritized replay
            per_beta_start: Initial beta for importance sampling correction
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            double_dqn: Whether to use Double DQN algorithm
        """
        super(DQNAgent, self).__init__()
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.double_dqn = double_dqn
        
        # Initialize Q-networks with dueling architecture
        self.q_network = DuelingQNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DuelingQNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=epsilon_decay_steps
        )
        
        # Action mapping from indices to game actions
        self.action_to_game_action = None
        
        # Tracking variables
        self.update_counter = 0
        self.total_steps = 0
        self.losses = []  # Track recent losses for logging
    
    def select_action(self, obs: np.ndarray, valid_actions: List[Dict[str, Any]], training: bool = True) -> Dict[str, Any]:
        """
        Select an action using epsilon-greedy policy with linear decay.
        
        Args:
            obs: Current observation
            valid_actions: List of valid actions in game format
            training: Whether to use epsilon-greedy (True) or greedy (False)
            
        Returns:
            Selected action in game format
        """
        if len(valid_actions) == 0:
            raise ValueError("No valid actions provided")
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection with linear decay
        if training and random.random() < self.epsilon:
            # Random action
            return random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                # Get Q-values for all actions
                q_values = self.q_network(obs_tensor).cpu().numpy()[0]
                
                # Get indices of valid actions
                valid_indices = [self._get_action_index(action) for action in valid_actions]
                
                # Find action with highest Q-value among valid actions
                valid_q_values = [q_values[idx] for idx in valid_indices]
                max_idx = valid_indices[np.argmax(valid_q_values)]
                
                # Return the game action
                return self.action_to_game_action[max_idx]
    
    def _get_action_index(self, action: Dict[str, Any]) -> int:
        """
        Get the index of a game action in the action mapping.
        
        Args:
            action: Game action
            
        Returns:
            Index of the action
        """
        for idx, game_action in enumerate(self.action_to_game_action):
            if self._actions_equal(action, game_action):
                return idx
        
        raise ValueError(f"Action {action} not found in action mapping")
    
    def add_experience(self, obs: np.ndarray, action: Dict[str, Any], 
                     reward: float, next_obs: np.ndarray, done: bool):
        """
        Add experience to prioritized replay buffer.
        
        Args:
            obs: Current observation
            action: Action taken in game format
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        # Convert action to index
        action_idx = self._get_action_index(action)
        
        # Add to replay buffer
        self.replay_buffer.add(obs, action_idx, reward, next_obs, done)
        
        # Update epsilon with linear decay
        if self.total_steps < (self.epsilon_start - self.epsilon_end) / self.epsilon_decay:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        self.total_steps += 1
    
    def update(self) -> float:
        """
        Update the Q-network using prioritized experience replay and Double DQN.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        
        # Compute weighted Huber loss for more stable learning
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Update Q-network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Track loss for logging
        loss_value = weighted_loss.item()
        self.losses.append(loss_value)
        if len(self.losses) > 100:
            self.losses.pop(0)
        
        return loss_value
    
    def save(self, path: str):
        """
        Save the agent to the specified path.
        
        Args:
            path: Directory to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters
        torch.save(self.q_network.state_dict(), os.path.join(path, 'q_network.pth'))
        torch.save(self.target_network.state_dict(), os.path.join(path, 'target_network.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pth'))
        
        # Save other parameters
        torch.save({
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'total_steps': self.total_steps,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'target_update_freq': self.target_update_freq,
            'batch_size': self.batch_size,
            'per_alpha': self.per_alpha,
            'per_beta_start': self.per_beta_start,
            'double_dqn': self.double_dqn
        }, os.path.join(path, 'parameters.pth'))
        
        # If action mapping exists, save it
        if self.action_to_game_action:
            with open(os.path.join(path, 'action_mapping.txt'), 'w') as f:
                for action in self.action_to_game_action:
                    f.write(str(action) + '\n')
    
    def load(self, path: str):
        """
        Load the agent from the specified path.
        
        Args:
            path: Directory to load the agent from
        """
        # Load model parameters with weights_only=True for security
        self.q_network.load_state_dict(torch.load(
            os.path.join(path, 'q_network.pth'),
            map_location=self.device,
            weights_only=True
        ))
        self.target_network.load_state_dict(torch.load(
            os.path.join(path, 'target_network.pth'),
            map_location=self.device,
            weights_only=True
        ))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(
            os.path.join(path, 'optimizer.pth'),
            map_location=self.device,
            weights_only=True
        ))
        
        # Load other parameters - can't use weights_only here since this is a dictionary
        params = torch.load(
            os.path.join(path, 'parameters.pth'),
            map_location=self.device
        )
        self.epsilon = params['epsilon']
        self.update_counter = params['update_counter']
        self.total_steps = params.get('total_steps', 0)
        if 'double_dqn' in params:
            self.double_dqn = params['double_dqn']
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent for logging.
        
        Returns:
            Dictionary of agent statistics
        """
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'update_counter': self.update_counter,
            'total_steps': self.total_steps,
            'learning_rate': self.learning_rate,
            'network_size': self.hidden_dims,
            'device': str(self.device),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'double_dqn': self.double_dqn
        }