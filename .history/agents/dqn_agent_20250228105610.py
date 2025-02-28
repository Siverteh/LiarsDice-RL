"""
Deep Q-Network (DQN) agent for Liar's Dice.

This module implements a DQN agent with various enhancements for playing Liar's Dice.
Features include experience replay, target networks, and dueling architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

from .base_agent import BaseAgent
from environment.state import ObservationEncoder


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    
    This network architecture separates state value and action advantage,
    which can improve learning stability and performance.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize the Dueling DQN network.
        
        Args:
            input_dim: Dimension of the input (observation)
            output_dim: Dimension of the output (action space)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions and samples them randomly for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Tuple) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done, valid_actions)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for Liar's Dice.
    
    This agent uses a dueling DQN architecture with experience replay
    for learning to play Liar's Dice.
    
    Attributes:
        player_id (int): ID of the player this agent controls
        observation_encoder (ObservationEncoder): Encoder for observations
        action_dim (int): Dimension of the action space
        device (torch.device): Device to run the model on
        policy_net (DuelingDQN): Policy network
        target_net (DuelingDQN): Target network
        memory (ReplayBuffer): Experience replay buffer
        optimizer (torch.optim.Optimizer): Optimizer for the policy network
        epsilon (float): Current exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (float): Decay rate for exploration
        gamma (float): Discount factor for future rewards
        batch_size (int): Batch size for training
        target_update (int): Frequency of target network updates
        training_steps (int): Number of training steps performed
    """
    
    def __init__(
        self,
        player_id: int,
        observation_encoder: ObservationEncoder,
        action_dim: int,
        input_dim: int = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        memory_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 100,
        hidden_dim: int = 128,
        device: str = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            player_id: ID of the player this agent controls
            observation_encoder: Encoder for observations
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update: Frequency of target network updates
            hidden_dim: Dimension of hidden layers
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__(player_id, name=f"DQN-{player_id}")
        
        self.observation_encoder = observation_encoder
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Set device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        input_dim = observation_encoder.get_observation_shape()[0]
        self.policy_net = DuelingDQN(input_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(input_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        if input_dim is None:
        input_dim = observation_encoder.get_observation_shape()[0]
    self.policy_net = DuelingDQN(input_dim, action_dim, hidden_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Tracking variables
        self.training_steps = 0
    
    def act(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        if not valid_actions:
            return None
        
        # Use epsilon-greedy policy
        if random.random() < self.epsilon:
            # Random action
            return random.choice(valid_actions)
        
        # Get observation tensor
        obs_encoded = self.observation_encoder.encode(observation)
        obs_tensor = torch.FloatTensor(obs_encoded).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        
        # Create a mapping from action index to action
        action_map = {i: action for i, action in enumerate(valid_actions)}
        
        # Select best valid action
        valid_indices = list(action_map.keys())
        q_valid = q_values[0, valid_indices]
        best_valid_idx = valid_indices[q_valid.argmax().item()]
        
        return action_map[best_valid_idx]
    
    def update(self, 
             observation: Dict[str, Any], 
             action: Dict[str, Any], 
             reward: float, 
             next_observation: Dict[str, Any], 
             done: bool, 
             valid_actions: List[Dict[str, Any]]) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            valid_actions: Valid actions for the current state
        """
        # Encode observations
        obs_encoded = self.observation_encoder.encode(observation)
        next_obs_encoded = self.observation_encoder.encode(next_observation)
        
        # Get action index and valid action indices
        action_idx = valid_actions.index(action)
        valid_indices = [i for i in range(len(valid_actions))]
        
        # Store in memory
        self.memory.add((
            obs_encoded,
            action_idx,
            reward,
            next_obs_encoded,
            done,
            valid_indices
        ))
    
    def train(self) -> Optional[float]:
        """
        Train the agent using experience replay.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if we have enough samples
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample a batch
        batch = self.memory.sample(self.batch_size)
        states, action_indices, rewards, next_states, dones, valid_action_indices = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        action_indices = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, action_indices).squeeze(1)
        
        # Compute next Q values (using target network)
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            if not dones[i]:
                # Get Q-values for next state
                next_q = self.target_net(next_states[i].unsqueeze(0)).squeeze(0)
                
                # Only consider valid actions
                valid_indices = valid_action_indices[i]
                if valid_indices:  # Check if there are valid actions
                    valid_q = next_q[valid_indices]
                    next_q_values[i] = valid_q.max()
        
        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """
        Save the agent's policy network.
        
        Args:
            path: Path to save the model to
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the agent's policy network.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']