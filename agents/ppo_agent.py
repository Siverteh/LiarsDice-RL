"""
Enhanced PPO agent implementation for Liar's Dice.

This module implements an improved PPO agent with adaptive entropy coefficient,
learning rate scheduling, and better action masking.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Any, Optional, Tuple, Union

from agents.base_agent import RLAgent


class ActorCritic(nn.Module):
    """Enhanced neural network for PPO actor-critic architecture."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor with layer normalization instead of batch normalization
        self.feature_network = nn.Sequential()
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims[:-1]):
            self.feature_network.add_module(f"fc{i}", nn.Linear(prev_dim, dim))
            self.feature_network.add_module(f"ln{i}", nn.LayerNorm(dim))  # Use LayerNorm instead of BatchNorm
            self.feature_network.add_module(f"relu{i}", nn.ReLU())
            prev_dim = dim
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        # Log softmax layer
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor-critic network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension for single sample
            
        features = self.feature_network(x)
        logits = self.actor(features)
        log_probs = self.log_softmax(logits)
        action_probs = torch.exp(log_probs)
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def evaluate(self, x: torch.Tensor, action: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions and compute values for PPO updates.
        
        Args:
            x: Input tensor
            action: Actions tensor
            mask: Optional mask for valid actions
            
        Returns:
            Tuple of (action_log_probs, state_values, entropy)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension for single sample
            
        features = self.feature_network(x)
        logits = self.actor(features)
        
        # Apply action mask if provided
        if mask is not None:
            # Set logits for invalid actions to large negative value
            masked_logits = logits.clone()
            masked_logits[mask == 0] = -1e20
            log_probs = self.log_softmax(masked_logits)
        else:
            log_probs = self.log_softmax(logits)
        
        # Get log probabilities of actions
        action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Compute state values
        state_value = self.critic(features).squeeze(-1)
        
        # Compute entropy
        if mask is not None:
            # Only consider valid actions for entropy
            probs = torch.exp(log_probs)
            # Renormalize probabilities
            normalized_probs = probs.clone()
            normalized_probs[mask == 0] = 0.0
            row_sums = normalized_probs.sum(dim=1, keepdim=True)
            normalized_probs = normalized_probs / (row_sums + 1e-8)
            
            # Compute entropy manually for masked probabilities
            log_probs = torch.log(normalized_probs + 1e-8)
            entropy = -(normalized_probs * log_probs).sum(dim=1)
        else:
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=1)
        
        return action_log_probs, state_value, entropy.mean()


class PPOMemory:
    """Improved memory buffer for PPO algorithm with better handling of masks."""
    
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.action_indices = []  # Store action indices for training
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size
        self.masks = []  # For masking invalid actions
        self.advantages = []  # Store pre-computed advantages
        
    def add(self, state, action, action_idx, reward, done, value, log_prob, mask=None):
        """Add experience to memory."""
        self.states.append(state)
        self.actions.append(action)
        self.action_indices.append(action_idx)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        if mask is not None:
            self.masks.append(mask)
    
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float = 0.0):
        """
        Pre-compute GAE advantages before update.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value of final state (0 for terminal, estimated for non-terminal)
        """
        self.advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
    
    def clear(self):
        """Clear memory after update."""
        self.states.clear()
        self.actions.clear()
        self.action_indices.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
        self.advantages = []
    
    def generate_batches(self):
        """Generate mini-batches for PPO updates."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches
    
    def get_tensors(self, device):
        """Get tensors for PPO updates."""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.tensor(self.action_indices, dtype=torch.int64).to(device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32).to(device)
        
        if self.masks:
            masks = torch.FloatTensor(np.array(self.masks)).to(device)
        else:
            masks = None
        
        return states, actions, old_log_probs, values, rewards, dones, advantages, masks
    
    def __len__(self):
        return len(self.states)


class LearningRateScheduler:
    """Learning rate scheduler with cosine annealing."""
    
    def __init__(self, optimizer, initial_lr: float, min_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update the learning rate based on current step."""
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        
        # Cosine annealing schedule
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EntropyScheduler:
    """Scheduler for entropy coefficient in PPO."""
    
    def __init__(self, initial_value: float, min_value: float, decay_steps: int):
        self.initial_value = initial_value
        self.min_value = min_value
        self.decay_rate = (initial_value - min_value) / decay_steps
        self.current_value = initial_value
        self.steps = 0
    
    def get_value(self):
        """Get current entropy coefficient value."""
        return self.current_value
    
    def step(self):
        """Update entropy coefficient."""
        self.steps += 1
        self.current_value = max(self.min_value, self.current_value - self.decay_rate)
        return self.current_value


class PPOAgent(RLAgent):
    """
    Enhanced PPO agent for Liar's Dice.
    
    This implements an improved Proximal Policy Optimization algorithm with:
    - Learning rate scheduling
    - Adaptive entropy coefficient
    - Improved action masking
    - Better advantage estimation
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        min_learning_rate: float = 3e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        entropy_min: float = 0.002,
        entropy_decay_steps: int = 100000,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        update_frequency: int = 2048,
        max_grad_norm: float = 0.5,
        device: str = 'auto',
        total_training_steps: int = 1000000
    ):
        """
        Initialize the enhanced PPO agent.
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            learning_rate: Initial learning rate for optimizer
            min_learning_rate: Minimum learning rate for scheduler
            gamma: Discount factor
            gae_lambda: Lambda parameter for GAE
            policy_clip: Clipping parameter for PPO objective
            value_coef: Coefficient for value loss
            entropy_coef: Initial coefficient for entropy bonus
            entropy_min: Minimum entropy coefficient
            entropy_decay_steps: Steps over which entropy coefficient decays
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            hidden_dims: Dimensions of hidden layers
            update_frequency: Number of steps between updates
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            total_training_steps: Total expected training steps for schedulers
        """
        super(PPOAgent, self).__init__()
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.initial_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.hidden_dims = hidden_dims
        self.update_frequency = update_frequency
        self.max_grad_norm = max_grad_norm
        self.total_training_steps = total_training_steps
        
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(obs_dim, action_dim, hidden_dims).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Initialize learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer, 
            initial_lr=learning_rate, 
            min_lr=min_learning_rate,
            total_steps=total_training_steps // update_frequency
        )
        
        # Initialize entropy scheduler
        self.entropy_scheduler = EntropyScheduler(
            initial_value=entropy_coef,
            min_value=entropy_min,
            decay_steps=entropy_decay_steps // update_frequency
        )
        
        # Initialize memory
        self.memory = PPOMemory(batch_size)
        
        # Action mapping from indices to game actions
        self.action_to_game_action = None
        
        # Tracking variables
        self.step_counter = 0
        self.update_counter = 0
        self.episode_reward = 0
        self.current_lr = learning_rate
        self.current_entropy_coef = entropy_coef
        
        # Statistics tracking
        self.loss_stats = {'total': [], 'policy': [], 'value': [], 'entropy': []}
        self.reward_history = []
        self.last_state = None  # For better advantage estimation
    
    
    def select_action(self, obs: np.ndarray, valid_actions: List[Dict[str, Any]], training: bool = True) -> Dict[str, Any]:
        """
        Select an action using the policy network with improved masking.
        
        Args:
            obs: Current observation
            valid_actions: List of valid actions in game format
            training: Whether actions are for training
            
        Returns:
            Selected action in game format
        """
        if len(valid_actions) == 0:
            raise ValueError("No valid actions provided")
        
        # Save last observed state for better advantage estimation
        self.last_state = obs
        
        # Special case: only one valid action
        if len(valid_actions) == 1:
            action = valid_actions[0]
            if training:
                # Need to store data even for singleton choices
                with torch.no_grad():
                    # Set network to eval mode temporarily to avoid BatchNorm issues
                    was_training = self.actor_critic.training
                    self.actor_critic.eval()
                    
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    action_probs, value = self.actor_critic(obs_tensor)
                    # Create mask for valid actions
                    valid_indices = [self._get_action_index(a) for a in valid_actions]
                    mask = torch.zeros(self.action_dim, device=self.device)
                    mask[valid_indices] = 1.0
                    # Since only one action is valid, log probability is 0 (100% chance)
                    self.memory.add(
                        obs, action, valid_indices[0], self.episode_reward, False, 
                        value.item(), 0, mask.cpu().numpy()
                    )
                    
                    # Restore previous training mode
                    if was_training:
                        self.actor_critic.train()
            return action
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # Get valid action indices
        valid_indices = [self._get_action_index(a) for a in valid_actions]
        
        # Create mask for valid actions
        mask = torch.zeros(self.action_dim, device=self.device)
        mask[valid_indices] = 1.0
        
        with torch.no_grad():
            # Set network to eval mode temporarily to avoid BatchNorm issues
            was_training = self.actor_critic.training
            self.actor_critic.eval()
            
            # Get action probabilities and value with improved masking
            action_probs, value = self.actor_critic(obs_tensor)
            
            # Mask out invalid actions
            masked_probs = action_probs * mask
            # Normalize to ensure sum to 1
            masked_probs = masked_probs / (masked_probs.sum() + 1e-10)
            
            # Sample from distribution
            dist = Categorical(masked_probs)
            action_idx = dist.sample().item()
            
            # Get log probability
            log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
            
            # Restore previous training mode
            if was_training:
                self.actor_critic.train()
        
        # Get game action
        game_action = self.action_to_game_action[action_idx]
        
        # Store experience if training
        if training:
            self.memory.add(
                obs, game_action, action_idx, self.episode_reward, False, 
                value.item(), log_prob, mask.cpu().numpy()
            )
            self.step_counter += 1
        
        return game_action
    
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
    
    def add_experience(self, obs: np.ndarray, action: Dict[str, Any], reward: float, next_obs: np.ndarray, done: bool):
        """
        Add reward and done flag to the last experience in memory.
        
        Args:
            obs: Current observation (not used, already stored)
            action: Action taken (not used, already stored)
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        # Update episode reward
        self.episode_reward = reward
        
        # Update done flag for the last experience
        if len(self.memory.dones) > 0:
            self.memory.dones[-1] = done
            self.memory.rewards[-1] = reward
        
        # Track episode rewards
        if done:
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            self.episode_reward = 0
    
    def update(self) -> float:
        """
        Update the policy and value networks with enhanced techniques.
        
        Returns:
            Loss value
        """
        # Check if enough steps have been collected
        if self.step_counter < self.update_frequency or len(self.memory.states) < self.batch_size:
            return 0.0
        
        # Reset step counter
        self.step_counter = 0
        self.update_counter += 1
        
        # Get final state value for better advantage estimation
        if self.last_state is not None:
            with torch.no_grad():
                # Set to eval mode temporarily to avoid BatchNorm issues with batch size 1
                was_training = self.actor_critic.training
                self.actor_critic.eval()
                
                last_state_tensor = torch.FloatTensor(self.last_state).to(self.device)
                _, last_value = self.actor_critic(last_state_tensor)
                last_value = last_value.item()
                
                # Restore previous training mode
                if was_training:
                    self.actor_critic.train()
        else:
            last_value = 0.0
        
        # Pre-compute advantages using GAE
        self.memory.compute_advantages(self.gamma, self.gae_lambda, last_value)
        
        # Convert experiences to tensors
        states, actions, old_log_probs, values, rewards, dones, advantages, masks = self.memory.get_tensors(self.device)
        
        # Normalize advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate returns
        returns = advantages + values
        
        # Update learning rate and entropy coefficient
        self.current_lr = self.lr_scheduler.step()
        self.current_entropy_coef = self.entropy_scheduler.step()
        
        # PPO update loop
        total_loss = 0
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_total = 0
        
        for _ in range(self.ppo_epochs):
            # Generate mini-batches
            for batch_indices in self.memory.generate_batches():
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get batch masks if available
                if masks is not None:
                    batch_masks = masks[batch_indices]
                else:
                    batch_masks = None
                
                # Evaluate current policy with improved masking
                new_log_probs, new_values, entropy = self.actor_critic.evaluate(
                    batch_states, batch_actions, batch_masks
                )
                
                # Compute policy ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate objectives with clipping
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                
                # Compute actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute critic loss with clipping for stability
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # Compute total loss with adaptive entropy coefficient
                loss = actor_loss + self.value_coef * value_loss - self.current_entropy_coef * entropy
                
                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_loss += loss.item()
                actor_loss_total += actor_loss.item()
                critic_loss_total += value_loss.item()
                entropy_total += entropy.item()
        
        # Clear memory after update
        self.memory.clear()
        
        # Compute average losses
        num_batches = self.ppo_epochs * len(self.memory.generate_batches())
        if num_batches > 0:
            avg_total_loss = total_loss / num_batches
            avg_actor_loss = actor_loss_total / num_batches
            avg_critic_loss = critic_loss_total / num_batches
            avg_entropy = entropy_total / num_batches
        else:
            avg_total_loss = 0.0
            avg_actor_loss = 0.0
            avg_critic_loss = 0.0
            avg_entropy = 0.0
        
        # Store statistics
        self.loss_stats['total'].append(avg_total_loss)
        self.loss_stats['policy'].append(avg_actor_loss)
        self.loss_stats['value'].append(avg_critic_loss)
        self.loss_stats['entropy'].append(avg_entropy)
        
        # Keep only the most recent stats
        max_stats = 100
        for k in self.loss_stats:
            if len(self.loss_stats[k]) > max_stats:
                self.loss_stats[k] = self.loss_stats[k][-max_stats:]
        
        return avg_total_loss
    
    def save(self, path: str):
        """
        Save the agent to the specified path.
        
        Args:
            path: Directory to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters
        torch.save(self.actor_critic.state_dict(), os.path.join(path, 'actor_critic.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pth'))
        
        # Save other parameters
        torch.save({
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'policy_clip': self.policy_clip,
            'value_coef': self.value_coef,
            'current_entropy_coef': self.current_entropy_coef,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'initial_learning_rate': self.initial_learning_rate,
            'current_lr': self.current_lr,
            'min_learning_rate': self.min_learning_rate,
            'update_frequency': self.update_frequency,
            'update_counter': self.update_counter,
            'step_counter': self.step_counter,
            'max_grad_norm': self.max_grad_norm,
            'lr_scheduler_step': self.lr_scheduler.current_step,
            'entropy_scheduler_step': self.entropy_scheduler.steps
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
        # Load model parameters
        self.actor_critic.load_state_dict(torch.load(
            os.path.join(path, 'actor_critic.pth'),
            map_location=self.device
        ))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(
            os.path.join(path, 'optimizer.pth'),
            map_location=self.device
        ))
        
        # Load other parameters
        params = torch.load(
            os.path.join(path, 'parameters.pth'),
            map_location=self.device
        )
        
        # Update counters and current values
        self.update_counter = params['update_counter']
        self.step_counter = params['step_counter']
        self.current_lr = params['current_lr']
        self.current_entropy_coef = params['current_entropy_coef']
        
        # Recreate schedulers
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer, 
            initial_lr=params['initial_learning_rate'], 
            min_lr=params['min_learning_rate'],
            total_steps=self.total_training_steps // params['update_frequency']
        )
        self.lr_scheduler.current_step = params.get('lr_scheduler_step', 0)
        
        self.entropy_scheduler = EntropyScheduler(
            initial_value=params['current_entropy_coef'],
            min_value=0.002,  # Default value if not saved
            decay_steps=100000 // params['update_frequency']
        )
        self.entropy_scheduler.steps = params.get('entropy_scheduler_step', 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent for logging.
        
        Returns:
            Dictionary of agent statistics
        """
        return {
            'update_counter': self.update_counter,
            'step_counter': self.step_counter,
            'current_lr': self.current_lr,
            'current_entropy': self.current_entropy_coef,
            'network_size': self.hidden_dims,
            'device': str(self.device),
            'memory_size': len(self.memory),
            'avg_policy_loss': np.mean(self.loss_stats['policy']) if self.loss_stats['policy'] else 0.0,
            'avg_value_loss': np.mean(self.loss_stats['value']) if self.loss_stats['value'] else 0.0,
            'avg_entropy': np.mean(self.loss_stats['entropy']) if self.loss_stats['entropy'] else 0.0,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }