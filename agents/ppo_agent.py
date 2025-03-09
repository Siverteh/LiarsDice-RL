import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Any, Optional, Tuple, Union

class ContextualMemoryModule(nn.Module):
    """Enhanced memory module that can handle contextual information and provide memory resets."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_size: int, 
        num_layers: int = 1,
        use_gru: bool = False,
        context_size: int = 8,
        attention_heads: int = 4,
    ):
        super(ContextualMemoryModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.context_size = context_size
        
        # Context embedding to encode game-specific information
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        # Recurrent layer - either GRU or LSTM
        if use_gru:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            
        # Self-attention mechanism for better memory integration
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True
        )
        
        # Memory modulation to control influence of past states
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
    def init_hidden(
        self, 
        batch_size: int = 1, 
        device: torch.device = None,
        context: Optional[torch.Tensor] = None
    ):
        """Initialize hidden states with optional context embedding."""
        device = device or torch.device("cpu")
        
        # Create base hidden state
        if self.use_gru:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            hidden = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            )
        
        # If context is provided, use it to initialize the hidden state
        if context is not None:
            context_embed = self.context_encoder(context)
            
            if self.use_gru:
                hidden[0] = context_embed.unsqueeze(0)
            else:
                hidden[0][0] = context_embed.unsqueeze(0)
        
        return hidden
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
        memory_influence: float = 1.0
    ):
        """
        Forward pass through memory module with controllable memory influence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            hidden: Hidden state from previous step
            memory_influence: Value between 0-1 controlling how much previous memory impacts current output
        
        Returns:
            output: Output tensor of shape [batch_size, seq_len, hidden_size]
            new_hidden: Updated hidden state
        """
        # Process through recurrent layer
        output, new_hidden = self.rnn(x, hidden)
        
        # Apply self-attention mechanism for better temporal dependencies
        attn_output, _ = self.attention(output, output, output)
        
        # Control memory influence using a gating mechanism
        if memory_influence < 1.0:
            # Create a context-only representation by passing zeros through RNN
            zero_input = torch.zeros_like(x)
            context_output, _ = self.rnn(zero_input, hidden)
            
            # Compute memory gate
            gate_input = torch.cat([output, context_output], dim=-1)
            mem_gate = self.memory_gate(gate_input)
            
            # Apply memory influence control
            effective_gate = mem_gate * memory_influence
            output = context_output + effective_gate * (output - context_output)
        
        return attn_output, new_hidden


class ImprovedRecurrentActorCritic(nn.Module):
    """Enhanced actor-critic architecture with better memory management."""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        memory_size: int = 128,
        memory_layers: int = 1,
        use_gru: bool = False,
        context_size: int = 8,
        attention_heads: int = 2,
        opponent_embedding_dim: int = 8
    ):
        super(ImprovedRecurrentActorCritic, self).__init__()
        
        self.memory_size = memory_size
        self.memory_layers = memory_layers
        self.use_gru = use_gru
        self.context_size = context_size
        
        # Feature extraction
        self.feature_network = nn.Sequential()
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims[:-1]):
            self.feature_network.add_module(f"fc{i}", nn.Linear(prev_dim, dim))
            self.feature_network.add_module(f"ln{i}", nn.LayerNorm(dim))
            self.feature_network.add_module(f"relu{i}", nn.ReLU())
            prev_dim = dim
        
        # Position-specific processing layer (assuming first 5 features relate to position)
        self.position_embedding = nn.Sequential(
            nn.Linear(5, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Opponent embedding (for encoding opponent identity)
        self.opponent_embedding = nn.Embedding(32, opponent_embedding_dim)  # Support up to 32 distinct opponents
        
        # Contextual memory module
        self.memory = ContextualMemoryModule(
            input_dim=prev_dim + 32,  # Feature dim + position embedding
            hidden_size=memory_size,
            num_layers=memory_layers,
            use_gru=use_gru,
            context_size=context_size,
            attention_heads=attention_heads
        )
        
        # Policy head with residual connection
        self.actor = nn.Sequential(
            nn.Linear(memory_size, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        # Value head with residual connection
        self.critic = nn.Sequential(
            nn.Linear(memory_size, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Adaptive memory control
        self.memory_controller = nn.Sequential(
            nn.Linear(memory_size, 1),
            nn.Sigmoid()
        )
    
    def init_hidden(
        self, 
        batch_size: int = 1, 
        device: torch.device = None,
        opponent_id: Optional[int] = None,
        game_context: Optional[torch.Tensor] = None
    ):
        """Initialize hidden states with game context and opponent information."""
        device = device or torch.device("cpu")
        
        # Create game context tensor if provided
        context = None
        if game_context is not None:
            context = game_context
        elif opponent_id is not None:
            # Create a simple context based on opponent ID
            context = torch.zeros(batch_size, self.context_size, device=device)
            context[:, 0] = opponent_id / 32.0  # Normalize to 0-1 range
        
        return self.memory.init_hidden(batch_size, device, context)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        opponent_id: Optional[int] = None,
        memory_influence: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the actor-critic network."""
        batch_size = x.size(0) if x.dim() > 1 else 1
        
        # Reshape if needed
        if x.dim() == 2:  # [batch_size, input_dim]
            x = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, input_dim]
        elif x.dim() == 1:  # [input_dim]
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, x.device, opponent_id)
        
        # Extract position features (first 5 elements)
        position_features = x[:, :, :5]
        batch_size, seq_len = position_features.size(0), position_features.size(1)
        
        # Process position features
        position_features = position_features.reshape(-1, 5)
        position_embedding = self.position_embedding(position_features)
        position_embedding = position_embedding.reshape(batch_size, seq_len, -1)
        
        # Process features
        features = self.feature_network(x.reshape(-1, x.size(-1)))
        features = features.reshape(batch_size, seq_len, -1)
        
        # Combine features and position embedding
        combined_features = torch.cat([features, position_embedding], dim=2)
        
        # Determine memory influence factor
        if memory_influence is None:
            # Adaptively compute memory influence from the last hidden state
            if self.use_gru:
                last_hidden = hidden_state[-1]  # Last layer's hidden state
            else:
                last_hidden = hidden_state[0][-1]  # LSTM hidden state (not cell state)
                
            # Compute adaptive memory influence between 0.2 and 1.0
            memory_influence = 0.2 + 0.8 * self.memory_controller(last_hidden).mean()
        
        # Pass through memory module
        memory_out, new_hidden = self.memory(
            combined_features, 
            hidden_state, 
            memory_influence=memory_influence
        )
        
        # Extract last output for actor and critic
        last_memory = memory_out[:, -1]
        
        # Pass through actor and critic heads
        logits = self.actor(last_memory)
        action_probs = nn.functional.softmax(logits, dim=-1)
        state_value = self.critic(last_memory)
        
        return action_probs, state_value, new_hidden


class SequentialMemoryBuffer:
    """Enhanced memory buffer that preserves episode sequences for recurrent networks."""
    
    def __init__(self, batch_size: int, seq_length: int = 8, overlap: int = 4):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.overlap = overlap  # Controls how much sequences overlap
        
        # Store sequences
        self.states = []  # List of episode lists of states
        self.actions = []  # List of episode lists of actions
        self.action_indices = []  # List of episode lists of action indices
        self.rewards = []  # List of episode lists of rewards
        self.dones = []  # List of episode lists of dones
        self.values = []  # List of episode lists of values
        self.log_probs = []  # List of episode lists of log_probs
        self.masks = []  # List of episode lists of masks
        
        # Current episode buffers
        self.current_states = []
        self.current_actions = []
        self.current_action_indices = []
        self.current_rewards = []
        self.current_dones = []
        self.current_values = []
        self.current_log_probs = []
        self.current_masks = []
        
        # Meta information
        self.episode_opponent_ids = []  # Track opponent for each episode
        self.current_opponent_id = None
        
        # Advantages
        self.advantages = []  # List of episode lists of advantages
        self.returns = []  # List of episode lists of returns
    
    def start_new_episode(self, opponent_id: Optional[int] = None):
        """Start a new episode."""
        # Store the previous episode if it exists
        if self.current_states:
            self.states.append(self.current_states)
            self.actions.append(self.current_actions)
            self.action_indices.append(self.current_action_indices)
            self.rewards.append(self.current_rewards)
            self.dones.append(self.current_dones)
            self.values.append(self.current_values)
            self.log_probs.append(self.current_log_probs)
            
            if self.current_masks:
                self.masks.append(self.current_masks)
            
            self.episode_opponent_ids.append(self.current_opponent_id)
        
        # Reset current episode buffers
        self.current_states = []
        self.current_actions = []
        self.current_action_indices = []
        self.current_rewards = []
        self.current_dones = []
        self.current_values = []
        self.current_log_probs = []
        self.current_masks = []
        
        # Set opponent ID
        self.current_opponent_id = opponent_id
    
    def add(self, state, action, action_idx, reward, done, value, log_prob, mask=None):
        """Add experience to current episode."""
        self.current_states.append(state)
        self.current_actions.append(action)
        self.current_action_indices.append(action_idx)
        self.current_rewards.append(reward)
        self.current_dones.append(done)
        self.current_values.append(value)
        self.current_log_probs.append(log_prob)
        
        if mask is not None:
            self.current_masks.append(mask)
    
    def compute_advantages(self, gamma: float, gae_lambda: float, final_value: float = 0.0):
        """Compute advantages for each episode using GAE."""
        # Make sure current episode is stored
        self.start_new_episode()
        
        self.advantages = []
        self.returns = []
        
        # Compute for each episode
        for ep_idx in range(len(self.states)):
            ep_rewards = self.rewards[ep_idx]
            ep_values = self.values[ep_idx]
            ep_dones = self.dones[ep_idx]
            
            if not ep_rewards:  # Skip empty episodes
                continue
            
            # Append final value
            values = ep_values + [final_value if ep_idx == len(self.states) - 1 else 0.0]
            
            # Compute GAE advantages
            advantages = np.zeros(len(ep_rewards), dtype=np.float32)
            gae = 0
            
            for t in reversed(range(len(ep_rewards))):
                delta = ep_rewards[t] + gamma * values[t+1] * (1 - ep_dones[t]) - values[t]
                gae = delta + gamma * gae_lambda * (1 - ep_dones[t]) * gae
                advantages[t] = gae
            
            self.advantages.append(advantages)
            self.returns.append(advantages + np.array(ep_values))
    
    def generate_batches(self):
        """Generate mini-batches for PPO updates while respecting episode boundaries."""
        # Prepare sequences for training
        all_sequences = self._prepare_sequences()
        if not all_sequences:
            return []
        
        # Shuffle sequences
        indices = np.arange(len(all_sequences))
        np.random.shuffle(indices)
        
        # Generate batches
        batch_start = np.arange(0, len(indices), self.batch_size)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches, all_sequences
    
    def _prepare_sequences(self):
        """Prepare sequences from episodes for recurrent training."""
        all_sequences = []
        
        # For each episode
        for ep_idx in range(len(self.states)):
            ep_states = self.states[ep_idx]
            ep_action_indices = self.action_indices[ep_idx]
            ep_log_probs = self.log_probs[ep_idx]
            ep_values = self.values[ep_idx]
            
            # Check if ep_idx is in range for returns and advantages lists
            if ep_idx >= len(self.returns) or ep_idx >= len(self.advantages):
                continue
                
            ep_returns = self.returns[ep_idx]
            ep_advantages = self.advantages[ep_idx]
            
            # Check if lists/arrays are empty - use len() which works for both lists and numpy arrays
            if len(ep_states) == 0 or len(ep_returns) == 0:
                continue
            
            # Get masks for this episode if available
            ep_masks = self.masks[ep_idx] if ep_idx < len(self.masks) else None
            
            # Get opponent ID for this episode
            opponent_id = self.episode_opponent_ids[ep_idx] if ep_idx < len(self.episode_opponent_ids) else None
            
            # Create sequences with specified overlap
            step = self.seq_length - self.overlap
            if step <= 0:
                step = 1  # Ensure we make progress
                
            # Create sequences from the episode
            for i in range(0, len(ep_states), step):
                end_idx = min(i + self.seq_length, len(ep_states))
                
                # Skip sequences that are too short
                if end_idx - i < 2:
                    continue
                
                seq = {
                    'states': ep_states[i:end_idx],
                    'action_indices': ep_action_indices[i:end_idx],
                    'log_probs': ep_log_probs[i:end_idx],
                    'values': ep_values[i:end_idx],
                    'returns': ep_returns[i:end_idx],
                    'advantages': ep_advantages[i:end_idx],
                    'opponent_id': opponent_id,
                    'seq_length': end_idx - i
                }
                
                if ep_masks is not None:
                    seq['masks'] = ep_masks[i:end_idx]
                
                all_sequences.append(seq)
        
        return all_sequences
    
    def get_batch_tensors(self, batch_indices, all_sequences, device):
        """Convert a batch of sequence indices to tensors."""
        # Get maximum sequence length in this batch
        max_seq_len = max(all_sequences[idx]['seq_length'] for idx in batch_indices)
        
        # Initialize tensors
        batch_size = len(batch_indices)
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        opponent_ids = []
        seq_lengths = []
        masks = []
        
        # Collect sequences
        for i, idx in enumerate(batch_indices):
            seq = all_sequences[idx]
            
            # Pad sequences to max_seq_len
            seq_len = seq['seq_length']
            seq_lengths.append(seq_len)
            
            # Get opponent ID
            opponent_ids.append(seq['opponent_id'] if seq['opponent_id'] is not None else 0)
            
            # Properly handle padding for numpy arrays and lists
            # Convert to lists first for consistent handling
            seq_states = list(seq['states'])
            seq_actions = list(seq['action_indices'])
            seq_log_probs = list(seq['log_probs'])
            seq_returns = list(seq['returns'])
            seq_advantages = list(seq['advantages'])
            
            # Pad using Python list concatenation
            padded_states = seq_states + [np.zeros_like(seq_states[0])] * (max_seq_len - seq_len)
            padded_actions = seq_actions + [0] * (max_seq_len - seq_len)
            padded_log_probs = seq_log_probs + [0.0] * (max_seq_len - seq_len)
            padded_returns = seq_returns + [0.0] * (max_seq_len - seq_len)
            padded_advantages = seq_advantages + [0.0] * (max_seq_len - seq_len)
            
            states.append(padded_states)
            actions.append(padded_actions)
            old_log_probs.append(padded_log_probs)
            returns.append(padded_returns)
            advantages.append(padded_advantages)
            
            # Pad masks if available
            if 'masks' in seq:
                seq_masks = list(seq['masks'])
                padded_masks = seq_masks + [np.zeros_like(seq_masks[0])] * (max_seq_len - seq_len)
                masks.append(padded_masks)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(device)
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs)).to(device)
        returns_tensor = torch.FloatTensor(np.array(returns)).to(device)
        advantages_tensor = torch.FloatTensor(np.array(advantages)).to(device)
        opponent_ids_tensor = torch.LongTensor(np.array(opponent_ids)).to(device)
        seq_lengths_tensor = torch.LongTensor(np.array(seq_lengths)).to(device)
        
        masks_tensor = None
        if masks:
            masks_tensor = torch.FloatTensor(np.array(masks)).to(device)
        
        return (
            states_tensor, 
            actions_tensor, 
            old_log_probs_tensor, 
            returns_tensor, 
            advantages_tensor, 
            opponent_ids_tensor, 
            seq_lengths_tensor, 
            masks_tensor
        )
    
    def clear(self):
        """Clear all stored episodes."""
        self.states = []
        self.actions = []
        self.action_indices = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.advantages = []
        self.returns = []
        self.episode_opponent_ids = []
        
        self.current_states = []
        self.current_actions = []
        self.current_action_indices = []
        self.current_rewards = []
        self.current_dones = []
        self.current_values = []
        self.current_log_probs = []
        self.current_masks = []
        self.current_opponent_id = None
    
    def __len__(self):
        """Return total number of steps across all episodes."""
        return sum(len(ep_states) for ep_states in self.states) + len(self.current_states)


class LearningRateScheduler:
    """Learning rate scheduler with cosine annealing."""
    
    def __init__(self, optimizer, initial_lr: float, min_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
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
        self.decay_rate = (initial_value - min_value) / decay_steps if decay_steps > 0 else 0
        self.current_value = initial_value
        self.steps = 0
    
    def step(self):
        """Update entropy coefficient."""
        self.steps += 1
        self.current_value = max(self.min_value, self.current_value - self.decay_rate)
        return self.current_value


class PPOAgent:
    """
    Enhanced Recurrent PPO Agent for Liar's Dice with better memory management.
    
    Key improvements:
    1. Better memory state management between opponents
    2. Contextual memory initialization
    3. Sequence-based batch training
    4. Adaptive memory influence based on context
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        memory_size: int = 128,
        memory_layers: int = 1,
        use_gru: bool = True,
        seq_length: int = 8,
        memory_overlap: int = 2,
        reset_memory_on_done: bool = True,
        reset_memory_on_opponent_change: bool = True,
        learning_rate: float = 0.0003,
        min_learning_rate: float = 3e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.08,
        entropy_min: float = 0.01,
        entropy_decay_steps: int = 200000,
        ppo_epochs: int = 4,
        batch_size: int = 8,  # Smaller batch size for sequences
        update_frequency: int = 2048,
        max_grad_norm: float = 0.5,
        device: str = 'auto',
        total_training_steps: int = 1000000
    ):
        """Initialize the agent with enhanced recurrent networks."""
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
        
        # Recurrent parameters
        self.memory_size = memory_size
        self.memory_layers = memory_layers
        self.use_gru = use_gru
        self.seq_length = seq_length
        self.memory_overlap = memory_overlap
        self.reset_memory_on_done = reset_memory_on_done
        self.reset_memory_on_opponent_change = reset_memory_on_opponent_change
        
        # Initialize improved actor-critic network
        self.actor_critic = ImprovedRecurrentActorCritic(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            memory_size=memory_size,
            memory_layers=memory_layers,
            use_gru=use_gru
        ).to(self.device)
        
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
        
        # Initialize improved memory buffer
        self.memory = SequentialMemoryBuffer(
            batch_size=batch_size,
            seq_length=seq_length,
            overlap=memory_overlap
        )
        
        # Action mapping from indices to game actions
        self.action_to_game_action = None
        
        # Tracking variables
        self.step_counter = 0
        self.update_counter = 0
        self.episode_reward = 0
        self.current_lr = learning_rate
        self.entropy_coef = entropy_coef
        
        # Hidden state management
        self.hidden_state = None
        self.current_opponent_id = None
        
        # Statistics tracking
        self.loss_stats = {'total': [], 'policy': [], 'value': [], 'entropy': []}
        self.reward_history = []
        self.last_state = None  # For better advantage estimation
    
    def init_hidden(self, opponent_id: Optional[int] = None):
        """Initialize the hidden state with optional opponent context."""
        self.hidden_state = self.actor_critic.init_hidden(
            batch_size=1, 
            device=self.device,
            opponent_id=opponent_id
        )
        self.current_opponent_id = opponent_id
    
    def reset_hidden(self):
        """Reset the hidden state completely."""
        self.hidden_state = None
        self.current_opponent_id = None
    
    def set_opponent(self, opponent_id: int):
        """Set current opponent ID and optionally reset memory."""
        if opponent_id != self.current_opponent_id and self.reset_memory_on_opponent_change:
            self.init_hidden(opponent_id)
        self.current_opponent_id = opponent_id
    
    def select_action(
        self, 
        obs: np.ndarray, 
        valid_actions: List[Dict[str, Any]], 
        training: bool = True,
        opponent_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Select an action with improved memory management."""
        if len(valid_actions) == 0:
            raise ValueError("No valid actions provided")
        
        # Handle opponent change
        if opponent_id is not None and opponent_id != self.current_opponent_id:
            if self.reset_memory_on_opponent_change:
                self.init_hidden(opponent_id)
            else:
                self.current_opponent_id = opponent_id
        
        # Initialize hidden state if None
        if self.hidden_state is None:
            self.init_hidden(opponent_id)
        
        # Save last observed state for advantage estimation
        self.last_state = obs
        
        # Special case: only one valid action
        if len(valid_actions) == 1:
            action = valid_actions[0]
            if training:
                with torch.no_grad():
                    was_training = self.actor_critic.training
                    self.actor_critic.eval()
                    
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    action_probs, value, new_hidden = self.actor_critic(
                        obs_tensor, 
                        self.hidden_state,
                        opponent_id=self.current_opponent_id
                    )
                    
                    # Create mask for valid actions
                    valid_indices = [self._get_action_index(a) for a in valid_actions]
                    mask = torch.zeros(self.action_dim, device=self.device)
                    mask[valid_indices] = 1.0
                    
                    # Store experience
                    self.memory.add(
                        obs, action, valid_indices[0], self.episode_reward, False, 
                        value.item(), 0, mask.cpu().numpy()
                    )
                    
                    # Update hidden state
                    self.hidden_state = new_hidden
                    
                    # Restore training mode
                    if was_training:
                        self.actor_critic.train()
            return action
        
        # Process observation
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        # Get valid action indices
        valid_indices = [self._get_action_index(a) for a in valid_actions]
        
        # Create mask for valid actions
        mask = torch.zeros(self.action_dim, device=self.device)
        mask[valid_indices] = 1.0
        
        with torch.no_grad():
            # Set network to eval mode temporarily
            was_training = self.actor_critic.training
            self.actor_critic.eval()
            
            # Get action probabilities and value with adaptive memory influence
            action_probs, value, new_hidden = self.actor_critic(
                obs_tensor, 
                self.hidden_state,
                opponent_id=self.current_opponent_id
            )
            
            # Mask out invalid actions
            masked_probs = action_probs * mask
            # Normalize to ensure sum to 1
            masked_probs = masked_probs / (masked_probs.sum() + 1e-10)
            
            # Sample from distribution
            dist = Categorical(masked_probs)
            action_idx = dist.sample().item()
            
            # Get log probability
            log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
            
            # Restore training mode
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
            
            # Update hidden state
            self.hidden_state = new_hidden
            self.step_counter += 1
        
        return game_action
    
    def _get_action_index(self, action: Dict[str, Any]) -> int:
        """Get the index of a game action in the action mapping."""
        for idx, game_action in enumerate(self.action_to_game_action):
            if self._actions_equal(action, game_action):
                return idx
        
        raise ValueError(f"Action {action} not found in action mapping")
    
    def _actions_equal(self, a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
        """Check if two game actions are equal."""
        if a1.keys() != a2.keys():
            return False
        
        for key in a1:
            if a1[key] != a2[key]:
                return False
        
        return True
    
    def add_experience(
        self, 
        obs: np.ndarray, 
        action: Dict[str, Any], 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool,
        opponent_id: Optional[int] = None
    ):
        """Add reward and done flag to the last experience in memory."""
        # Update episode reward
        self.episode_reward = reward
        
        # Update done flag for the last experience
        if self.memory.current_states:
            self.memory.current_dones[-1] = done
            self.memory.current_rewards[-1] = reward
        
        # Track episode rewards
        if done:
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            self.episode_reward = 0
            
            # Reset hidden state if episode is done
            if self.reset_memory_on_done:
                self.reset_hidden()
                
                # Start a new episode in memory
                self.memory.start_new_episode(opponent_id)
    
    def update(self) -> float:
        """Update policy and value networks with improved sequence handling."""
        # Check if enough steps have been collected
        if self.step_counter < self.update_frequency or len(self.memory) < self.batch_size * self.seq_length:
            return 0.0
        
        # Reset step counter
        self.step_counter = 0
        self.update_counter += 1
        
        # Get final state value for better advantage estimation
        if self.last_state is not None:
            with torch.no_grad():
                was_training = self.actor_critic.training
                self.actor_critic.eval()
                
                last_state_tensor = torch.FloatTensor(self.last_state).to(self.device)
                _, last_value, _ = self.actor_critic(last_state_tensor, None)
                last_value = last_value.item()
                
                if was_training:
                    self.actor_critic.train()
        else:
            last_value = 0.0
        
        # Pre-compute advantages using GAE
        self.memory.compute_advantages(self.gamma, self.gae_lambda, last_value)
        
        # Generate sequences and batches
        batches, all_sequences = self.memory.generate_batches()
        if not batches:
            return 0.0
        
        # Update learning rate and entropy coefficient
        self.current_lr = self.lr_scheduler.step()
        self.entropy_coef = self.entropy_scheduler.step()
        
        # PPO update loop
        total_loss = 0
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_total = 0
        
        for _ in range(self.ppo_epochs):
            for batch_indices in batches:
                # Get batch tensors
                (
                    states, 
                    actions, 
                    old_log_probs, 
                    returns, 
                    advantages, 
                    opponent_ids, 
                    seq_lengths, 
                    masks
                ) = self.memory.get_batch_tensors(batch_indices, all_sequences, self.device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Initialize hidden states for each sequence in batch
                batch_size = states.size(0)
                hidden_states = [
                    self.actor_critic.init_hidden(1, self.device, opponent_ids[i].item())
                    for i in range(batch_size)
                ]
                
                # Process each sequence separately to maintain proper hidden state
                batch_actor_loss = 0
                batch_value_loss = 0
                batch_entropy = 0
                
                for i in range(batch_size):
                    # Get sequence data
                    seq_state = states[i, :seq_lengths[i]].unsqueeze(0)  # Add batch dim
                    seq_action = actions[i, :seq_lengths[i]]
                    seq_old_log_prob = old_log_probs[i, :seq_lengths[i]]
                    seq_advantage = advantages[i, :seq_lengths[i]]
                    seq_return = returns[i, :seq_lengths[i]]
                    
                    # Get sequence mask if available
                    seq_mask = None
                    if masks is not None:
                        seq_mask = masks[i, :seq_lengths[i]]
                    
                    # Forward pass with proper hidden state init
                    action_probs, values, _ = self.actor_critic(
                        seq_state, 
                        hidden_states[i], 
                        opponent_id=opponent_ids[i].item()
                    )
                    
                    # Make sure shapes are correct
                    # action_probs should be [1, seq_len, action_dim]
                    # values should be [1, seq_len, 1]
                    seq_len = seq_lengths[i].item()
                    
                    # Ensure action_probs and values have correct sequence dimension
                    if action_probs.dim() == 2:  # [batch, action_dim] without seq dim
                        action_probs = action_probs.unsqueeze(1)  # [batch, 1, action_dim]
                    
                    if values.dim() == 2:  # [batch, 1] without seq dim
                        values = values.unsqueeze(1)  # [batch, 1, 1]
                    
                    # Reshape for sequence operations
                    action_probs = action_probs.squeeze(0)  # [seq_len, action_dim]
                    values = values.squeeze(0).squeeze(-1)  # [seq_len]
                    
                    # Make sure we have the right shape for our sequence
                    if action_probs.size(0) != seq_len:
                        action_probs = action_probs.expand(seq_len, -1)
                    
                    if values.size(0) != seq_len:
                        # If values don't have the right sequence length, repeat the single value
                        values = values[-1].repeat(seq_len)
                    
                    # Calculate log probs for the actions that were taken
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(seq_action)
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratios and surrogates for PPO
                    ratios = torch.exp(new_log_probs - seq_old_log_prob)
                    surr1 = ratios * seq_advantage
                    surr2 = torch.clamp(ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * seq_advantage
                    
                    # Calculate actor, critic, and entropy losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Now values and seq_return should have the same shape
                    value_loss = nn.functional.mse_loss(values, seq_return)
                    
                    # Accumulate losses for batch
                    batch_actor_loss += actor_loss
                    batch_value_loss += value_loss
                    batch_entropy += entropy
                
                # Average losses over batch
                batch_actor_loss /= batch_size
                batch_value_loss /= batch_size
                batch_entropy /= batch_size
                
                # Combine losses
                loss = batch_actor_loss + self.value_coef * batch_value_loss - self.entropy_coef * batch_entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                actor_loss_total += batch_actor_loss.item()
                critic_loss_total += batch_value_loss.item()
                entropy_total += batch_entropy.item()
                total_loss += loss.item()
        
        # Clear memory after update
        self.memory.clear()
        
        # Calculate average losses
        num_batches = self.ppo_epochs * len(batches)
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
        
        # Keep only recent stats
        max_stats = 100
        for k in self.loss_stats:
            if len(self.loss_stats[k]) > max_stats:
                self.loss_stats[k] = self.loss_stats[k][-max_stats:]
        
        return avg_total_loss
    
    def save(self, path: str):
        """Save the agent to the specified path."""
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
            'memory_size': self.memory_size,
            'memory_layers': self.memory_layers,
            'use_gru': self.use_gru,
            'seq_length': self.seq_length,
            'memory_overlap': self.memory_overlap,
            'reset_memory_on_done': self.reset_memory_on_done,
            'reset_memory_on_opponent_change': self.reset_memory_on_opponent_change,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'policy_clip': self.policy_clip,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
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
        
        # Save action mapping if exists
        if self.action_to_game_action:
            with open(os.path.join(path, 'action_mapping.txt'), 'w') as f:
                for action in self.action_to_game_action:
                    f.write(str(action) + '\n')
    
    def load(self, path: str):
        """Load the agent from the specified path."""
        # Load model parameters
        self.actor_critic.load_state_dict(torch.load(
            os.path.join(path, 'actor_critic.pth'),
            map_location=self.device
        ))
        
        # Load optimizer state
        try:
            self.optimizer.load_state_dict(torch.load(
                os.path.join(path, 'optimizer.pth'),
                map_location=self.device
            ))
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
        
        # Load other parameters
        try:
            params = torch.load(
                os.path.join(path, 'parameters.pth'),
                map_location=self.device
            )
            
            # Update agent parameters
            self.memory_size = params.get('memory_size', self.memory_size)
            self.memory_layers = params.get('memory_layers', self.memory_layers)
            self.use_gru = params.get('use_gru', self.use_gru)
            self.seq_length = params.get('seq_length', self.seq_length)
            self.memory_overlap = params.get('memory_overlap', self.memory_overlap)
            self.reset_memory_on_done = params.get('reset_memory_on_done', self.reset_memory_on_done)
            self.reset_memory_on_opponent_change = params.get('reset_memory_on_opponent_change', self.reset_memory_on_opponent_change)
            
            # Update counters and current values
            self.update_counter = params['update_counter']
            self.step_counter = params['step_counter']
            self.current_lr = params['current_lr']
            self.entropy_coef = params['entropy_coef']
        except Exception as e:
            print(f"Warning: Could not load all parameters: {e}")
        
        # Reset hidden state
        self.reset_hidden()
        
        # Recreate schedulers
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer, 
            initial_lr=self.initial_learning_rate, 
            min_lr=self.min_learning_rate,
            total_steps=self.total_training_steps // self.update_frequency
        )
        self.lr_scheduler.current_step = getattr(params, 'lr_scheduler_step', 0)
        
        self.entropy_scheduler = EntropyScheduler(
            initial_value=self.entropy_coef,
            min_value=0.01,  # Default value if not saved
            decay_steps=200000 // self.update_frequency
        )
        self.entropy_scheduler.steps = getattr(params, 'entropy_scheduler_step', 0)
    
    def get_entropy(self) -> float:
        """Get the current entropy coefficient."""
        return self.entropy_coef
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get complete statistics about the agent for logging."""
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'memory_size': self.memory_size,
            'memory_layers': self.memory_layers,
            'use_gru': self.use_gru,
            'seq_length': self.seq_length,
            'memory_overlap': self.memory_overlap,
            'reset_memory_on_done': self.reset_memory_on_done,
            'reset_memory_on_opponent_change': self.reset_memory_on_opponent_change,
            
            'learning_rate': self.initial_learning_rate,
            'min_learning_rate': self.min_learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'policy_clip': self.policy_clip,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'batch_size': self.batch_size,
            'update_frequency': self.update_frequency,
            'max_grad_norm': self.max_grad_norm,
            
            'update_counter': self.update_counter,
            'step_counter': self.step_counter,
            'total_training_steps': self.total_training_steps,
            
            'current_lr': self.current_lr,
            'device': str(self.device),
            
            'memory_size': len(self.memory),
            'avg_policy_loss': np.mean(self.loss_stats['policy']) if self.loss_stats['policy'] else 0.0,
            'avg_value_loss': np.mean(self.loss_stats['value']) if self.loss_stats['value'] else 0.0,
            'avg_entropy': np.mean(self.loss_stats['entropy']) if self.loss_stats['entropy'] else 0.0,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }
    
    def set_action_mapping(self, action_mapping):
        """Set the mapping from action indices to game actions."""
        self.action_to_game_action = action_mapping