"""
Advantage Actor-Critic (A2C) agent for Liar's Dice.

This module implements an A2C agent, which is a policy gradient method
that uses a critic to estimate the advantage function.
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


class RecurrentActorCritic(nn.Module):
    """
    Recurrent Actor-Critic network for A2C.
    
    This network uses an LSTM layer to maintain a history of observations,
    which is useful for partially observable environments like Liar's Dice.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 128,
        lstm_layers: int = 1
    ):
        """
        Initialize the recurrent actor-critic network.
        
        Args:
            input_dim: Dimension of the input (observation)
            output_dim: Dimension of the output (action space)
            hidden_dim: Dimension of hidden layers
            lstm_layers: Number of LSTM layers
        """
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
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
        
        # Initialize hidden state
        self.hidden = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state for LSTM
            
        Returns:
            Tuple of (action_logits, state_value, hidden_state)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Ensure batch dimension for LSTM
        if features.dim() == 2:
            # Add sequence dimension (batch_size, seq_len=1, hidden_dim)
            features = features.unsqueeze(1)
        
        # Use provided hidden state or internal state
        if hidden is not None:
            lstm_out, hidden = self.lstm(features, hidden)
        else:
            if self.hidden is None:
                # Initialize hidden state if needed
                batch_size = features.size(0)
                h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=features.device)
                c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=features.device)
                self.hidden = (h0, c0)
            
            lstm_out, self.hidden = self.lstm(features, self.hidden)
        
        # Extract last timestep output
        lstm_out = lstm_out[:, -1]
        
        # Actor: convert to action logits
        action_logits = self.actor(lstm_out)
        
        # Critic: estimate state value
        state_value = self.critic(lstm_out)
        
        return action_logits, state_value, self.hidden if hidden is None else hidden
    
    def reset_hidden(self) -> None:
        """Reset the hidden state of the LSTM."""
        self.hidden = None


class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic agent for Liar's Dice.
    
    This agent uses A2C with a recurrent network to handle the
    partially observable nature of Liar's Dice.
    
    Attributes:
        player_id (int): ID of the player this agent controls
        observation_encoder (ObservationEncoder): Encoder for observations
        action_dim (int): Dimension of the action space
        device (torch.device): Device to run the model on
        network (RecurrentActorCritic): Actor-critic network
        optimizer (torch.optim.Optimizer): Optimizer for the network
        gamma (float): Discount factor for future rewards
        entropy_coef (float): Coefficient for entropy bonus
        value_coef (float): Coefficient for value loss
        max_grad_norm (float): Maximum gradient norm for clipping
        experiences (list): List to store experiences before update
        action_maps (dict): Maps observations to valid action maps
    """
    
    def __init__(
        self,
        player_id: int,
        observation_encoder: ObservationEncoder,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        device: str = None
    ):
        """
        Initialize the A2C agent.
        
        Args:
            player_id: ID of the player this agent controls
            observation_encoder: Encoder for observations
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            entropy_coef: Coefficient for entropy bonus
            value_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
            hidden_dim: Dimension of hidden layers
            lstm_layers: Number of LSTM layers
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__(player_id, name=f"A2C-{player_id}")
        
        self.observation_encoder = observation_encoder
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Set device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create network
        input_dim = observation_encoder.get_observation_shape()[0]
        self.network = RecurrentActorCritic(
            input_dim, action_dim, hidden_dim, lstm_layers
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Storage for experiences
        self.experiences = []
        
        # Keep mapping from observation IDs to action maps
        self.action_maps = {}
    
    def reset(self) -> None:
        """Reset the agent's state for a new episode."""
        self.network.reset_hidden()
        self.experiences = []
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
            action_logits, state_value, hidden = self.network(obs_tensor)
        
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
    
    def train(self) -> Optional[float]:
        """
        Train the agent using A2C.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.experiences:
            return None
        
        # Check if all experiences have rewards
        if any('reward' not in exp for exp in self.experiences):
            return None
        
        # Calculate returns for each experience
        returns = []
        
        # Initialize with bootstrap value for non-terminal states
        R = 0
        if not self.experiences[-1]['done']:
            # Estimate value of the next state
            last_state = self.experiences[-1]['state']
            state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, value, _ = self.network(state_tensor)
            
            R = value.item()
        
        # Calculate returns in reverse order
        for i in reversed(range(len(self.experiences))):
            R = self.experiences[i]['reward'] + self.gamma * R * (1 - self.experiences[i]['done'])
            returns.insert(0, R)
        
        # Convert experiences to tensors
        states = torch.FloatTensor([exp['state'] for exp in self.experiences]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in self.experiences]).to(self.device)
        masks = torch.FloatTensor([exp['mask'] for exp in self.experiences]).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Reset LSTM hidden state
        self.network.reset_hidden()
        
        # Forward pass to get new action probabilities and values
        action_logits, values, _ = self.network(states)
        values = values.squeeze(-1)
        
        # Apply action masks
        masked_logits = action_logits + masks
        
        # Calculate new probabilities
        probs = F.softmax(masked_logits, dim=1)
        dist = Categorical(probs)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Calculate action log probabilities
        selected_log_probs = dist.log_prob(actions)
        
        # Calculate losses
        actor_loss = -(selected_log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -dist.entropy().mean()
        
        # Calculate total loss
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear experiences
        self.experiences = []
        self.action_maps = {}
        
        # Reset LSTM hidden state
        self.network.reset_hidden()
        
        return loss.item()
    
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