"""
Environment wrapper for Liar's Dice game.

This module provides a wrapper around the Liar's Dice environment
to make it compatible with the DQN agent and facilitate training.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from agents.rule_agent import RuleAgent, create_agent


class LiarsDiceEnvWrapper:
    """
    Wrapper for the Liar's Dice environment to standardize
    the interface for reinforcement learning training.
    
    This wrapper provides:
    - Action space conversion between DQN outputs and game actions
    - Observation preprocessing
    - Environment reset and step methods
    - Support for playing against rule-based agents
    
    Attributes:
        game: The underlying Liar's Dice game
        observation_encoder: Encoder for observations
        num_players: Number of players in the game
        action_mapping: Mapping from action indices to game actions
        rule_agents: Dict of rule-based opponents, if any
        active_player_id: Index of the learning agent in the game
    """
    
    def __init__(
        self,
        num_players: int = 2,
        num_dice: int = 2,
        dice_faces: int = 6,
        seed: Optional[int] = None,
        rule_agent_types: Optional[List[str]] = None
    ):
        """
        Initialize the environment wrapper.
        
        Args:
            num_players: Number of players in the game
            num_dice: Number of dice per player
            dice_faces: Number of faces on each die
            seed: Random seed for reproducibility
            rule_agent_types: List of rule agent types for opponents
                (if None, environment expects actions for all players)
        """
        self.game = LiarsDiceGame(
            num_players=num_players,
            num_dice=num_dice,
            dice_faces=dice_faces,
            seed=seed
        )
        
        # Create observation encoder with exact same parameters
        self.observation_encoder = ObservationEncoder(
            num_players=num_players, 
            num_dice=num_dice, 
            dice_faces=dice_faces
        )
        
        # Get the actual observation shape directly from the encoder
        test_obs = self.game.reset()
        encoded_test = self.observation_encoder.encode(test_obs[0])
        self.obs_dim = encoded_test.shape[0]
        
        self.num_players = num_players
        self.num_dice = num_dice
        self.dice_faces = dice_faces
        
        # Generate all possible actions
        self.action_mapping = self._generate_action_mapping()
        self.action_dim = len(self.action_mapping)
        
        # Set up rule-based agents if specified
        self.rule_agents = {}
        self.active_player_id = 0  # Default: first player is the learning agent
        
        if rule_agent_types is not None:
            if len(rule_agent_types) != num_players - 1:
                raise ValueError("Number of rule agent types must be num_players - 1")
            
            agent_idx = 1  # Start from player 1 (player 0 is the learning agent)
            for agent_type in rule_agent_types:
                agent = create_agent(agent_type, dice_faces=dice_faces)
                agent.set_player_id(agent_idx, num_players)
                self.rule_agents[agent_idx] = agent
                agent_idx += 1
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            Encoded observation for the active player
        """
        observations = self.game.reset(seed=seed)
        encoded_obs = self.observation_encoder.encode(observations[self.active_player_id])
        return encoded_obs
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment using an action index.
        
        Args:
            action_idx: Index of the action to take (from action_mapping)
            
        Returns:
            Tuple of (next_observation, reward, done, info)
        """
        if action_idx < 0 or action_idx >= len(self.action_mapping):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        # Convert action index to game action
        game_action = self.action_mapping[action_idx].copy()
        
        # Execute the action in the game
        observations, rewards, done, info = self.game.step(game_action)
        
        # If rule agents are specified and game not done, let them take their turns
        current_player = self.game.current_player
        while not done and current_player != self.active_player_id and current_player in self.rule_agents:
            agent = self.rule_agents[current_player]
            valid_actions = self.game.get_valid_actions(current_player)
            agent_action = agent.select_action(observations[current_player], valid_actions)
            observations, rewards, done, info = self.game.step(agent_action)
            current_player = self.game.current_player
        
        # If the game is still not done and it's not the active player's turn,
        # something went wrong
        if not done and current_player != self.active_player_id:
            raise RuntimeError(f"Game is waiting for player {current_player}, not the active player {self.active_player_id}")
        
        # Return the observation, reward, done flag, and info for the active player
        encoded_obs = self.observation_encoder.encode(observations[self.active_player_id])
        reward = rewards[self.active_player_id]
        
        return encoded_obs, reward, done, info
    
    def get_valid_actions(self) -> List[int]:
        """
        Get valid action indices for the current state.
        
        Returns:
            List of valid action indices
        """
        # Get valid game actions
        valid_game_actions = self.game.get_valid_actions(self.active_player_id)
        
        # Convert to action indices
        valid_indices = []
        for game_action in valid_game_actions:
            for idx, action in enumerate(self.action_mapping):
                if self._actions_equal(game_action, action):
                    valid_indices.append(idx)
                    break
        
        return valid_indices
    
    def _generate_action_mapping(self) -> List[Dict[str, Any]]:
        """
        Generate a complete mapping of action indices to game actions.
        
        This creates all possible actions in the game, including:
        - Challenge action
        - Bid actions for all valid quantity-value combinations
        
        Returns:
            List of game actions where the index corresponds to the action index
        """
        action_mapping = []
        
        # Add challenge action
        action_mapping.append({'type': 'challenge'})
        
        # Add all possible bid actions
        # In Liar's Dice, a bid consists of a quantity and a value
        max_quantity = self.num_players * self.num_dice
        for quantity in range(1, max_quantity + 1):
            for value in range(1, self.dice_faces + 1):
                action_mapping.append({
                    'type': 'bid',
                    'quantity': quantity,
                    'value': value
                })
        
        return action_mapping
    
    def _actions_equal(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """
        Check if two actions are equal.
        
        Args:
            action1: First action dictionary
            action2: Second action dictionary
            
        Returns:
            True if the actions are equal, False otherwise
        """
        if action1['type'] != action2['type']:
            return False
        
        if action1['type'] == 'challenge':
            return True  # Challenge actions are always equal
        
        # For bid actions, compare quantity and value
        return (action1['quantity'] == action2['quantity'] and 
                action1['value'] == action2['value'])
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the encoded observations.
        
        Returns:
            Shape of observations as a tuple
        """
        # Return our actual observed dimension, not what the encoder claims
        return (self.obs_dim,)
    
    def get_action_dim(self) -> int:
        """
        Get the dimension of the action space.
        
        Returns:
            Number of possible actions
        """
        return self.action_dim
    
    def set_active_player(self, player_id: int):
        """
        Set the active player (the learning agent).
        
        Args:
            player_id: ID of the player to set as active
        """
        if player_id < 0 or player_id >= self.num_players:
            raise ValueError(f"Invalid player ID: {player_id}")
        
        self.active_player_id = player_id
    
    def get_action_mapping(self) -> List[Dict[str, Any]]:
        """
        Get the action mapping.
        
        Returns:
            List of game actions where the index corresponds to the action index
        """
        return self.action_mapping
    
    def render(self):
        """Render the game state."""
        self.game.render()

    def create_progressive_env(
        num_players: int,
        num_dice: int, 
        dice_faces: int,
        seed: Optional[int],
        opponent_type: str,
        episodes_for_level: int
    ):
        """
        Create an environment with opponents that get progressively harder.
        
        As training progresses, the opponent becomes more challenging.
        
        Args:
            num_players: Number of players in the game
            num_dice: Number of dice per player
            dice_faces: Number of faces on each die
            seed: Random seed for reproducibility
            opponent_type: Type of opponent rule agent
            episodes_for_level: Total episodes planned for this level
            
        Returns:
            A wrapped environment with progressive difficulty
        """
        # Create base opponent agent
        rule_agent = create_agent(opponent_type, dice_faces=dice_faces)
        
        # Create a wrapper env with adaptive difficulty
        class ProgressiveEnvWrapper(LiarsDiceEnvWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.episode_counter = 0
                self.total_episodes = episodes_for_level
                self.base_agent_type = opponent_type
                self.base_agent = rule_agent
                
            def step(self, action_idx):
                # Update opponent difficulty based on training progress
                self._update_opponent_difficulty()
                return super().step(action_idx)
                
            def reset(self, **kwargs):
                self.episode_counter += 1
                return super().reset(**kwargs)
                
            def _update_opponent_difficulty(self):
                """Increase opponent difficulty as training progresses."""
                if self.episode_counter % 100 == 0:  # Update every 100 episodes
                    progress = min(self.episode_counter / self.total_episodes, 1.0)
                    
                    # Adjust opponent parameters based on type
                    if self.base_agent_type == 'conservative' and hasattr(self.rule_agents[1], 'challenge_threshold'):
                        # For conservative agents, lower the threshold 
                        # (they'll challenge more aggressively as training progresses)
                        base = 0.3
                        self.rule_agents[1].challenge_threshold = max(base - progress * 0.15, 0.05)
                    
                    elif self.base_agent_type == 'aggressive' and hasattr(self.rule_agents[1], 'bluff_frequency'):
                        # Increase bluffing as training progresses
                        base = 0.3
                        self.rule_agents[1].bluff_frequency = min(base + progress * 0.3, 0.7)
                    
                    elif self.base_agent_type == 'strategic' and hasattr(self.rule_agents[1], 'strategies'):
                        # Make strategic agent more aggressive over time
                        for phase in self.rule_agents[1].strategies:
                            # Gradually lower challenge threshold
                            self.rule_agents[1].strategies[phase]['challenge_threshold'] = max(
                                0.3 - progress * 0.15, 0.15
                            )
                            # Gradually increase bluffing
                            self.rule_agents[1].strategies[phase]['bluff_frequency'] = min(
                                0.2 + progress * 0.3, 0.5
                            )
                    
                    elif self.base_agent_type == 'adaptive' and hasattr(self.rule_agents[1], 'opponent_model'):
                        # Make adaptive agent more aggressive
                        self.rule_agents[1].bluff_frequency = min(0.3 + progress * 0.4, 0.7)
        
        # Create and return the progressive environment
        env = ProgressiveEnvWrapper(
            num_players=num_players,
            num_dice=num_dice,
            dice_faces=dice_faces,
            seed=seed,
            rule_agent_types=[opponent_type]
        )
        
        return env