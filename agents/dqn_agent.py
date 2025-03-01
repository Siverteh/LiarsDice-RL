"""
DQN Agent implementation for Liar's Dice.

This module implements a Deep Q-Network (DQN) agent for learning
to play Liar's Dice through reinforcement learning.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Tuple, Any, Optional


class QNetwork(nn.Module):
    """
    Neural network architecture for approximating the Q-function.
    
    The network takes an observation as input and outputs Q-values
    for each possible action.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize the Q-Network.
        
        Args:
            input_dim: Dimension of the input observation
            output_dim: Dimension of the output (number of actions)
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Add dropout for regularization
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better training dynamics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, output_dim) with Q-values
        """
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores transitions (observation, action, reward, next_observation, done)
    and provides random sampling for training.
    """
    
    def __init__(self, capacity: int = 50000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
            as PyTorch tensors
        """
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        obs, actions, rewards, next_obs, dones = zip(*[self.buffer[i] for i in indices])
        
        # Convert to PyTorch tensors
        obs = torch.FloatTensor(np.array(obs))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_obs = torch.FloatTensor(np.array(next_obs))
        dones = torch.FloatTensor(np.array(dones))
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of transitions in the buffer
        """
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for Liar's Dice.
    
    This agent uses DQN to learn a policy for playing Liar's Dice,
    with experience replay and a target network for stable learning.
    
    Attributes:
        obs_dim: Dimension of the observation space
        action_dim: Dimension of the action space
        device: Device to run the model on (CPU or GPU)
        q_network: Main Q-network for action selection
        target_network: Target Q-network for stable learning
        optimizer: Optimizer for updating the Q-network
        epsilon: Exploration rate for epsilon-greedy policy
        gamma: Discount factor for future rewards
        target_update_freq: Frequency of target network updates
        replay_buffer: Buffer for experience replay
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        buffer_size: int = 50000,
        batch_size: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        device: str = 'auto'
    ):
        """
        Initialize the DQN agent.
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            target_update_freq: Frequency of target network updates (in steps)
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            hidden_dims: Hidden dimensions of the Q-network
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        # IMPORTANT: Print the observation dimension to verify it matches what the environment produces
        print(f"Initializing DQN agent with obs_dim={obs_dim}, action_dim={action_dim}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Determine device (CPU or GPU)
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize Q-networks
        self.q_network = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training step counter
        self.training_steps = 0
        
        # Action mapping for Liar's Dice
        self.action_to_game_action = None
    
    def set_action_mapping(self, action_mapping: List[Dict[str, Any]]):
        """
        Set the mapping from action indices to game actions.
        
        Args:
            action_mapping: List of valid game actions where the index
                corresponds to the action index in the Q-network output
        """
        self.action_to_game_action = action_mapping
    
    def select_action(self, observation: np.ndarray, valid_actions: List[Dict[str, Any]], training: bool = True) -> Dict[str, Any]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            valid_actions: List of valid actions in the current state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action as a dictionary compatible with the environment
        """
        # Map valid actions to indices
        valid_indices = []
        for action in valid_actions:
            if self.action_to_game_action is not None:
                # Find the index of the action in the action mapping
                for i, mapped_action in enumerate(self.action_to_game_action):
                    if self._actions_equal(action, mapped_action):
                        valid_indices.append(i)
                        break
            else:
                # Without a mapping, we can't properly select actions
                raise ValueError("Action mapping not set. Call set_action_mapping first.")
        
        # If there are no valid actions, return None
        if not valid_indices:
            return None
        
        if training and random.random() < self.epsilon:
            # Exploration: pick a random valid action
            action_idx = random.choice(valid_indices)
        else:
            # Exploitation: pick the best valid action according to the Q-network
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor).squeeze(0).cpu().numpy()
            
            # Filter q_values to only include valid actions
            valid_q_values = [(idx, q_values[idx]) for idx in valid_indices]
            action_idx = max(valid_q_values, key=lambda x: x[1])[0]
        
        # Convert action index to game action
        return self.action_to_game_action[action_idx].copy()
    
    def update(self) -> float:
        """
        Update the Q-network using a batch of experiences.
        
        Returns:
            Loss value from the update
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch of transitions
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = (
            obs.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_obs.to(self.device),
            dones.to(self.device)
        )
        
        # Compute current Q-values
        q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values using Double DQN approach
        with torch.no_grad():
            # Get actions from the main network
            next_actions = self.q_network(next_obs).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for those actions
            next_q_values = self.target_network(next_obs).gather(1, next_actions).squeeze(1)
            # Compute target
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(q_values, targets)
        
        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Step the learning rate scheduler
        if self.training_steps % 1000 == 0:
            self.scheduler.step()
        
        return loss.item()
    
    def _actions_equal(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """
        Check if two actions are equal (used for action mapping).
        
        Args:
            action1: First action dictionary
            action2: Second action dictionary
            
        Returns:
            True if the actions are equal, False otherwise
        """
        if action1['type'] != action2['type']:
            return False
        
        if action1['type'] == 'challenge':
            return True  # Challenges are always equal
        
        # For bids, compare quantity and value
        return (action1['quantity'] == action2['quantity'] and 
                action1['value'] == action2['value'])
    
    def add_experience(self, obs: np.ndarray, action: Dict[str, Any], reward: float, next_obs: np.ndarray, done: bool):
        """
        Add an experience to the replay buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        # Convert action to index
        action_idx = None
        for i, mapped_action in enumerate(self.action_to_game_action):
            if self._actions_equal(action, mapped_action):
                action_idx = i
                break
        
        if action_idx is None:
            raise ValueError(f"Action {action} not found in action mapping")
        
        # Add to buffer
        self.replay_buffer.add(obs, action_idx, reward, next_obs, done)
    
    def save(self, path: str):
        """
        Save the agent's models and parameters.
        
        Args:
            path: Directory path to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        # Save models
        torch.save(self.q_network.state_dict(), os.path.join(path, 'q_network.pth'))
        torch.save(self.target_network.state_dict(), os.path.join(path, 'target_network.pth'))
        
        # Save parameters
        params = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'target_update_freq': self.target_update_freq,
            'batch_size': self.batch_size,
            'training_steps': self.training_steps,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        torch.save(params, os.path.join(path, 'params.pth'))
        
        # Save action mapping if available
        if self.action_to_game_action is not None:
            np.save(os.path.join(path, 'action_mapping.npy'), self.action_to_game_action)
    
    def load(self, path: str):
        """
        Load the agent's models and parameters.
        
        Args:
            path: Directory path to load the agent from
        """
        # Load models
        self.q_network.load_state_dict(torch.load(os.path.join(path, 'q_network.pth'), map_location=self.device))
        self.target_network.load_state_dict(torch.load(os.path.join(path, 'target_network.pth'), map_location=self.device))
        
        # Load parameters
        params = torch.load(os.path.join(path, 'params.pth'), map_location=self.device)
        self.obs_dim = params['obs_dim']
        self.action_dim = params['action_dim']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        self.target_update_freq = params['target_update_freq']
        self.batch_size = params['batch_size']
        self.training_steps = params['training_steps']
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = params['learning_rate']
        
        # Load action mapping if available
        action_mapping_path = os.path.join(path, 'action_mapping.npy')
        if os.path.exists(action_mapping_path):
            self.action_to_game_action = np.load(action_mapping_path, allow_pickle=True).tolist()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics for monitoring training progress.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_steps': self.training_steps
        }