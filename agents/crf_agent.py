"""
Conditional Random Field agent implementation for Liar's Dice.

This module implements a CRF-based agent that uses structured prediction
to model optimal actions in the game state space.
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from agents.base_agent import RLAgent

# Requires: pip install sklearn-crfsuite
import sklearn_crfsuite
from sklearn.preprocessing import StandardScaler


class CRFAgent(RLAgent):
    """
    Conditional Random Field agent for Liar's Dice.
    
    This agent uses a CRF to model the conditional probability of actions
    given the current game state. It extracts features from the observation
    and uses them to predict the optimal action.
    
    Attributes:
        obs_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        feature_engineering (str): Method for feature extraction ('basic', 'advanced')
        model: CRF model for action prediction
        buffer_size (int): Maximum size of experience buffer
        update_frequency (int): How often to update the model
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_engineering: str = 'advanced',
        c1: float = 0.1,  # L1 regularization
        c2: float = 0.1,  # L2 regularization
        max_iterations: int = 100,
        buffer_size: int = 10000,
        update_frequency: int = 500,
        initial_exploration: float = 1.0,
        final_exploration: float = 0.1,
        exploration_decay: int = 100000,
        device: str = 'auto'
    ):
        """
        Initialize the CRF agent.
        
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            feature_engineering: Method for feature extraction
            c1: L1 regularization parameter
            c2: L2 regularization parameter
            max_iterations: Maximum iterations for CRF training
            buffer_size: Maximum size of experience buffer
            update_frequency: How often to update the model (in steps)
            initial_exploration: Initial exploration rate
            final_exploration: Final exploration rate
            exploration_decay: Steps to decay exploration
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        super(CRFAgent, self).__init__()
        
        # Basic parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_engineering = feature_engineering
        
        # Device setting
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize CRF model
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=False
        )
        
        # Experience buffer parameters
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.experience_buffer = []
        
        # Feature normalization
        self.scaler = StandardScaler()
        
        # Exploration parameters
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.exploration_decay = exploration_decay
        self.epsilon = initial_exploration
        
        # Training statistics
        self.total_steps = 0
        self.update_counter = 0
        self.training_history = []
        self.fitted = False
        
        # For calculating success statistics
        self.action_success = defaultdict(lambda: [0, 0])  # [success_count, total_count]
        
        # Action mapping
        self.action_to_game_action = None
    
    def select_action(self, obs: np.ndarray, valid_actions: List[Dict[str, Any]], training: bool = True) -> Dict[str, Any]:
        """
        Select an action using the CRF model.
        
        Args:
            obs: Current observation
            valid_actions: List of valid actions in game format
            training: Whether the agent is in training mode
            
        Returns:
            Selected action in game format
        """
        if len(valid_actions) == 0:
            raise ValueError("No valid actions provided")
        
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Explore with epsilon probability
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # If model not yet fitted, use random action
        if not self.fitted:
            return np.random.choice(valid_actions)
        
        try:
            # Extract features for prediction
            features = self._extract_features(obs, valid_actions)
            
            # Get indices of valid actions
            valid_indices = [self._get_action_index(action) for action in valid_actions]
            
            # Prepare for CRF prediction
            # CRF expects sequence data, but we have a single-step decision
            X = [features]  # List of feature dictionaries
            
            # Get action probabilities
            action_probs = self.model.predict_marginals(X)[0]
            
            # Filter to only valid actions
            valid_probs = {}
            for i in valid_indices:
                if str(i) in action_probs:
                    valid_probs[i] = action_probs[str(i)]
                else:
                    # Default probability if not seen during training
                    valid_probs[i] = 0.01
            
            # If no valid actions found, use random
            if not valid_probs:
                return np.random.choice(valid_actions)
            
            # Select action with highest probability
            best_action_idx = max(valid_probs, key=valid_probs.get)
            return self.action_to_game_action[best_action_idx]
            
        except Exception as e:
            # Fallback to random if prediction fails
            print(f"Error in CRF prediction: {e}")
            return np.random.choice(valid_actions)
    
    def _extract_features(self, obs: np.ndarray, valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from observation for CRF.
        
        Args:
            obs: Observation array
            valid_actions: List of valid actions
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic feature extraction - direct mapping of observation values
        if self.feature_engineering == 'basic':
            # Normalize observation
            if not self.scaler.n_samples_seen_:
                self.scaler.partial_fit([obs])
            
            normalized_obs = self.scaler.transform([obs])[0]
            
            # Add each observation dimension as a feature
            for i in range(min(50, len(normalized_obs))):  # Limit to 50 features for efficiency
                features[f'obs_{i}'] = normalized_obs[i]
            
            # Add valid action types as features
            for action in valid_actions:
                action_type = action['type']
                features[f'valid_{action_type}'] = 1
            
            return features
        
        # Advanced feature engineering - game-specific features
        elif self.feature_engineering == 'advanced':
            # Dice features - assuming first part of obs is dice information
            dice_end = min(self.obs_dim // 2, 30)  # Reasonable limit to avoid out-of-bounds
            
            # Try to extract dice values
            dice_values = []
            for i in range(0, dice_end, 6):  # Assuming 6 faces per die
                if i + 6 <= len(obs):
                    die_encoding = obs[i:i+6]
                    if np.max(die_encoding) > 0:
                        die_value = np.argmax(die_encoding) + 1
                        dice_values.append(die_value)
            
            # Create features for dice counts
            for value in range(1, 7):
                count = dice_values.count(value)
                features[f'dice_count_{value}'] = count
                features[f'has_dice_{value}'] = 1 if count > 0 else 0
            
            features['total_dice'] = len(dice_values)
            
            # Try to extract player position information
            # Assuming this is somewhere in the middle of the observation
            position_start = dice_end
            position_end = min(position_start + 20, self.obs_dim)
            
            if position_start < len(obs):
                position_features = obs[position_start:position_end]
                
                # Normalize these features
                if position_features.size > 0:
                    # Create a scaler just for these features
                    pos_scaler = StandardScaler()
                    pos_scaler.fit([position_features])
                    normalized_pos = pos_scaler.transform([position_features])[0]
                    
                    for i, value in enumerate(normalized_pos):
                        features[f'position_{i}'] = value
            
            # Check for current bid info
            # Typically in the later part of observation
            bid_start = position_end
            bid_end = min(bid_start + 10, self.obs_dim)
            
            if bid_start < len(obs) and np.sum(obs[bid_start:bid_end]) > 0:
                bid_features = obs[bid_start:bid_end]
                
                # Normalize bid features
                if bid_features.size > 0:
                    bid_scaler = StandardScaler()
                    bid_scaler.fit([bid_features])
                    normalized_bid = bid_scaler.transform([bid_features])[0]
                    
                    for i, value in enumerate(normalized_bid):
                        features[f'bid_{i}'] = value
            
            # Valid action features
            features['can_challenge'] = 0
            features['min_bid_quantity'] = float('inf')
            features['max_bid_quantity'] = 0
            features['min_bid_value'] = float('inf')
            features['max_bid_value'] = 0
            
            # Process valid actions
            for action in valid_actions:
                if action['type'] == 'challenge':
                    features['can_challenge'] = 1
                elif action['type'] == 'bid':
                    features['min_bid_quantity'] = min(features['min_bid_quantity'], action['quantity'])
                    features['max_bid_quantity'] = max(features['max_bid_quantity'], action['quantity'])
                    features['min_bid_value'] = min(features['min_bid_value'], action['value'])
                    features['max_bid_value'] = max(features['max_bid_value'], action['value'])
            
            # Fix infinity values
            if features['min_bid_quantity'] == float('inf'):
                features['min_bid_quantity'] = 0
            if features['min_bid_value'] == float('inf'):
                features['min_bid_value'] = 0
            
            # Convert all values to strings (requirement for sklearn-crfsuite)
            string_features = {k: str(v) for k, v in features.items()}
            
            return string_features
        
        else:
            raise ValueError(f"Unknown feature engineering method: {self.feature_engineering}")
    
    def add_experience(self, obs: np.ndarray, action: Dict[str, Any], 
                     reward: float, next_obs: np.ndarray, done: bool):
        """
        Add experience to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken in game format
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        # Get valid actions at the time of observation (approximation)
        valid_actions = self.action_to_game_action
        
        # Convert action to index
        action_idx = self._get_action_index(action)
        
        # Extract features
        features = self._extract_features(obs, valid_actions)
        
        # Add to buffer with additional information
        self.experience_buffer.append({
            'features': features,
            'action_idx': action_idx,
            'reward': reward,
            'done': done
        })
        
        # Track success of actions
        if done:
            # Action led to episode end
            if reward > 0:  # Action led to win
                self.action_success[action_idx][0] += 1  # Success count
            self.action_success[action_idx][1] += 1  # Total count
        
        # Keep buffer size limited
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
        
        # Update total steps and exploration rate
        self.total_steps += 1
        self._update_exploration()
        
        # Check if update is needed
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self.update()
            self.update_counter = 0
    
    def _update_exploration(self):
        """Update exploration rate based on linear decay."""
        if self.total_steps < self.exploration_decay:
            decay_step = (self.initial_exploration - self.final_exploration) / self.exploration_decay
            self.epsilon = max(self.final_exploration, self.epsilon - decay_step)
    
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
    
    def update(self, *args, **kwargs) -> float:
        """
        Update the CRF model using collected experiences.
        
        Returns:
            Loss value (approximate)
        """
        # Need enough experiences to train
        if len(self.experience_buffer) < 100:
            return 0.0
        
        try:
            # Prepare training data
            X = []  # List of feature dictionaries for sequences
            y = []  # List of action indices as labels for sequences
            
            # Group experiences by their feature dictionaries
            # This simplifies the data for CRF training
            feature_groups = {}
            
            for exp in self.experience_buffer:
                # Convert feature dictionary to tuple for hashing
                feature_key = tuple(sorted(exp['features'].items()))
                
                if feature_key not in feature_groups:
                    feature_groups[feature_key] = []
                
                feature_groups[feature_key].append((exp['action_idx'], exp['reward']))
            
            # For each unique feature set, select the action with highest reward
            for feature_key, actions in feature_groups.items():
                # Sort by reward (descending)
                sorted_actions = sorted(actions, key=lambda x: x[1], reverse=True)
                
                # Take action with highest reward
                best_action_idx = sorted_actions[0][0]
                
                # Convert back to dictionary
                features_dict = dict(feature_key)
                
                # Add to training data
                X.append(features_dict)
                y.append(str(best_action_idx))  # CRF expects string labels
            
            # Train CRF model if we have enough data
            if len(X) >= 20 and len(set(y)) >= 2:  # Need at least 20 samples and 2 different actions
                # CRF expects sequence data, so we wrap our data
                X_seq = [X]
                y_seq = [y]
                
                # Fit the model
                self.model.fit(X_seq, y_seq)
                self.fitted = True
                
                # Record training history
                self.training_history.append({
                    'step': self.total_steps,
                    'samples': len(X),
                    'unique_actions': len(set(y))
                })
                
                # Return arbitrary loss (CRF doesn't provide loss values)
                return 1.0
            
        except Exception as e:
            print(f"Error updating CRF model: {e}")
        
        return 0.0
    
    def save(self, path: str):
        """
        Save the agent to the specified path.
        
        Args:
            path: Directory to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        try:
            # Save CRF model
            with open(os.path.join(path, 'crf_model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save other parameters
            params = {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'feature_engineering': self.feature_engineering,
                'epsilon': self.epsilon,
                'total_steps': self.total_steps,
                'update_counter': self.update_counter,
                'fitted': self.fitted,
                'training_history': self.training_history,
                'action_success': dict(self.action_success)
            }
            
            with open(os.path.join(path, 'params.pkl'), 'wb') as f:
                pickle.dump(params, f)
            
            # Save action mapping if exists
            if self.action_to_game_action:
                with open(os.path.join(path, 'action_mapping.txt'), 'w') as f:
                    for action in self.action_to_game_action:
                        f.write(str(action) + '\n')
        except Exception as e:
            print(f"Error saving CRF agent: {e}")
    
    def load(self, path: str):
        """
        Load the agent from the specified path.
        
        Args:
            path: Directory to load the agent from
        """
        try:
            # Load CRF model
            with open(os.path.join(path, 'crf_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            if os.path.exists(os.path.join(path, 'scaler.pkl')):
                with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load other parameters
            with open(os.path.join(path, 'params.pkl'), 'rb') as f:
                params = pickle.load(f)
                
                self.obs_dim = params['obs_dim']
                self.action_dim = params['action_dim']
                self.feature_engineering = params['feature_engineering']
                self.epsilon = params['epsilon']
                self.total_steps = params['total_steps']
                self.update_counter = params['update_counter']
                self.fitted = params['fitted']
                self.training_history = params['training_history']
                
                # Load action success statistics if available
                if 'action_success' in params:
                    self.action_success = defaultdict(lambda: [0, 0])
                    for k, v in params['action_success'].items():
                        self.action_success[k] = v
        except Exception as e:
            print(f"Error loading CRF agent: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent for logging.
        
        Returns:
            Dictionary of agent statistics
        """
        # Calculate success rates for different actions
        action_success_rates = {}
        for action_idx, (success_count, total_count) in self.action_success.items():
            if total_count > 0:
                action_success_rates[str(action_idx)] = success_count / total_count
        
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'feature_engineering': self.feature_engineering,
            'epsilon': self.epsilon,
            'buffer_size': len(self.experience_buffer),
            'total_steps': self.total_steps,
            'fitted': self.fitted,
            'device': self.device,
            'action_success_rates': action_success_rates,
            'unique_features': len(set(tuple(sorted(exp['features'].items())) 
                                    for exp in self.experience_buffer)),
            'training_updates': len(self.training_history)
        }