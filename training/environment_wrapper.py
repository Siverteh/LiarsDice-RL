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
from typing import Union


class LiarsDiceEnvWrapper:
    """
    Wrapper for the Liar's Dice environment to standardize
    the interface for reinforcement learning training.
    """
    
    def __init__(
        self,
        num_players: int = 2,
        num_dice: int = 2,
        dice_faces: int = 6,
        seed: Optional[int] = None,
        rule_agent_types: Optional[List[str]] = None,
        rl_agent_as_opponent: Optional[Any] = None,
        randomize_positions: Union[bool, float] = False  # Changed to accept probability
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
            rl_agent_as_opponent: RL agent to use as opponent (for self-play)
            randomize_positions: Bool for full randomization or float for probability of randomization
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
        
        # Set up position randomization with probability
        if isinstance(randomize_positions, bool):
            self.randomize_positions = randomize_positions
            self.randomize_prob = 1.0 if randomize_positions else 0.0
        else:
            self.randomize_positions = randomize_positions > 0.0
            self.randomize_prob = float(randomize_positions)
            
        if self.randomize_positions:
            self.position_rng = np.random.RandomState(seed)
        
        # Default: first player is the learning agent
        self.active_player_id = 0  
        
        # Set up rule-based agents if specified
        self.rule_agents = {}
        self.rl_opponent = None
        
        # Set up RL agent as opponent if specified (for self-play)
        if rl_agent_as_opponent is not None:
            self.rl_opponent = rl_agent_as_opponent
            # Set opponent as player 1 (player 0 is the learning agent)
            if hasattr(self.rl_opponent, 'set_player_id'):
                self.rl_opponent.set_player_id(1, num_players)
        # Set up rule-based agents if specified
        elif rule_agent_types is not None:
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
        # Handle seed update if provided
        if seed is not None and self.randomize_positions:
            self.position_rng = np.random.RandomState(seed)
        
        # Apply randomization with probability
        if self.randomize_positions and (self.position_rng.random() < self.randomize_prob):
            new_active = self.position_rng.randint(0, self.num_players)
            self._randomize_player_positions(new_active)
        else:
            # Ensure we reset to position 0 if not randomizing this time
            if self.active_player_id != 0:
                self._randomize_player_positions(0)
        
        # Reset the game
        observations = self.game.reset(seed=seed)
        
        # If it's not the active player's turn initially, we need to take actions
        # for opponent agents until it's the active player's turn
        current_player = self.game.current_player
        done = False
        
        # Take actions for opponents until it's the active player's turn or game is done
        while not done and current_player != self.active_player_id:
            if current_player in self.rule_agents:
                agent = self.rule_agents[current_player]
                valid_actions = self.game.get_valid_actions(current_player)
                agent_action = agent.select_action(observations[current_player], valid_actions)
                observations, _, done, _ = self.game.step(agent_action)
            elif self.rl_opponent is not None:
                # Get observation for RL opponent
                opponent_obs = self.observation_encoder.encode(observations[current_player])
                
                # Get action from RL opponent
                valid_actions = self.get_valid_actions_for_player(current_player)
                
                opponent_action_idx = self.rl_opponent.select_action(
                    opponent_obs, 
                    [self.action_mapping[idx] for idx in valid_actions], 
                    training=False
                )
                
                # Find the action in the valid actions list
                opponent_action = None
                for idx, valid_action in enumerate([self.action_mapping[i] for i in valid_actions]):
                    if self._actions_equal(opponent_action_idx, valid_action):
                        opponent_action = valid_action
                        break
                
                if opponent_action is None:
                    raise ValueError(f"Selected action {opponent_action_idx} not found in valid actions")
                
                # Execute opponent action
                observations, _, done, _ = self.game.step(opponent_action)
            else:
                raise RuntimeError(f"No agent defined for player {current_player}")
            
            # Update current player
            current_player = self.game.current_player
        
        # Return encoded observation for the active player
        encoded_obs = self.observation_encoder.encode(observations[self.active_player_id])
        return encoded_obs
        
    def _randomize_player_positions(self, new_active_id: int):
        """
        Randomize player positions by setting a new active player ID and
        reassigning rule agents to other positions.
        
        Args:
            new_active_id: The new player ID for the active (learning) agent
        """
        if new_active_id == self.active_player_id:
            return  # Nothing to change
            
        # Store the current rule agent types before reassigning
        agent_types = []
        if self.rule_agents:
            for player_id in sorted(self.rule_agents.keys()):
                agent = self.rule_agents[player_id]
                if hasattr(agent, 'agent_type'):
                    agent_types.append(agent.agent_type)
                else:
                    # Fallback if agent_type not available
                    agent_types.append(agent.__class__.__name__)
        
        # Update active player ID
        self.active_player_id = new_active_id
        
        # Reassign rule agents to all positions except the new active player
        if agent_types:
            self.rule_agents = {}  # Clear existing agents
            agent_idx = 0
            
            for player_id in range(self.num_players):
                if player_id != self.active_player_id:
                    if agent_idx < len(agent_types):
                        agent_type = agent_types[agent_idx]
                        agent = create_agent(agent_type, dice_faces=self.dice_faces)
                        agent.set_player_id(player_id, self.num_players)
                        self.rule_agents[player_id] = agent
                        agent_idx += 1
        
        # Handle RL opponent if used for self-play
        if self.rl_opponent is not None and hasattr(self.rl_opponent, 'set_player_id'):
            # Find a position for the RL opponent that's not the active player
            for player_id in range(self.num_players):
                if player_id != self.active_player_id:
                    self.rl_opponent.set_player_id(player_id, self.num_players)
                    break
    
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
        
        # Add position and game information to info dict for reward shaping
        info['active_player_id'] = self.active_player_id
        info['num_players'] = self.num_players
        info['num_dice'] = self.num_dice
        info['dice_faces'] = self.dice_faces
        
        # Handle opponent turns (rule agents or RL agent)
        current_player = self.game.current_player
        
        # Keep taking opponent actions until it's the active player's turn or game is done
        while not done and current_player != self.active_player_id:
            if current_player in self.rule_agents:
                agent = self.rule_agents[current_player]
                valid_actions = self.game.get_valid_actions(current_player)
                agent_action = agent.select_action(observations[current_player], valid_actions)
                observations, rewards, done, info = self.game.step(agent_action)
            elif self.rl_opponent is not None:
                # Handle RL opponent action
                opponent_obs = self.observation_encoder.encode(observations[current_player])
                valid_actions = self.get_valid_actions_for_player(current_player)
                opponent_action_idx = self.rl_opponent.select_action(
                    opponent_obs, 
                    [self.action_mapping[idx] for idx in valid_actions], 
                    training=False
                )
                
                # Find the action in the valid actions list
                opponent_action = None
                for idx, valid_action in enumerate([self.action_mapping[i] for i in valid_actions]):
                    if self._actions_equal(opponent_action_idx, valid_action):
                        opponent_action = valid_action
                        break
                
                if opponent_action is None:
                    raise ValueError(f"Selected action {opponent_action_idx} not found in valid actions")
                
                # Execute opponent action
                observations, rewards, done, info = self.game.step(opponent_action)
            else:
                raise RuntimeError(f"No agent defined for player {current_player}")
            
            # Update current player
            current_player = self.game.current_player
            
            # Add position info again after opponent actions
            info['active_player_id'] = self.active_player_id
            info['num_players'] = self.num_players
            info['num_dice'] = self.num_dice
            info['dice_faces'] = self.dice_faces
        
        # Return the observation, reward, done flag, and info for the active player
        encoded_obs = self.observation_encoder.encode(observations[self.active_player_id])
        reward = rewards[self.active_player_id]
        
        return encoded_obs, reward, done, info
    
    def get_valid_actions(self) -> List[int]:
        """
        Get valid action indices for the current state for the active player.
        
        Returns:
            List of valid action indices
        """
        return self.get_valid_actions_for_player(self.active_player_id)
    
    def get_valid_actions_for_player(self, player_id: int) -> List[int]:
        """
        Get valid action indices for the specified player.
        
        Args:
            player_id: ID of the player to get valid actions for
            
        Returns:
            List of valid action indices
        """
        # Get valid game actions
        valid_game_actions = self.game.get_valid_actions(player_id)
        
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

    def update_rl_opponent(self, new_opponent):
        """
        Update the RL opponent with a new agent.
        
        Args:
            new_opponent: New RL agent to use as opponent
        """
        self.rl_opponent = new_opponent
        # Set opponent as player 1 (player 0 is the learning agent)
        if hasattr(self.rl_opponent, 'set_player_id'):
            self.rl_opponent.set_player_id(1, self.num_players)

    @staticmethod
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