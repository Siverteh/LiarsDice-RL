"""
State encoding for Liar's Dice.

This module handles the conversion of game states to formats suitable for
reinforcement learning algorithms, providing both observation encoding
for agents and full state encoding for training.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ObservationEncoder:
    """
    Encodes player observations into formats suitable for RL agents.
    
    This class handles conversion of game observations into formats that
    can be processed by neural networks or other ML models, including:
    - One-hot encoding of dice and game state
    - Feature extraction for history and bidding patterns
    - Normalization of numerical features
    
    Attributes:
        num_players (int): Number of players in the game
        num_dice (int): Maximum number of dice per player
        dice_faces (int): Number of faces on each die
        observation_shape (tuple): Shape of the encoded observation
    """
    
    def __init__(self, num_players: int, num_dice: int, dice_faces: int):
        """
        Initialize the observation encoder.
        
        Args:
            num_players: Number of players in the game
            num_dice: Maximum number of dice per player
            dice_faces: Number of faces on each die
        """
        self.num_players = num_players
        self.num_dice = num_dice
        self.dice_faces = dice_faces
        
        # Calculate observation shape for a single agent
        # Own dice (one-hot encoded)
        dice_shape = num_dice * dice_faces
        
        # Dice counts for all players
        dice_counts_shape = num_players
        
        # Current bid (quantity, value) - both normalized
        bid_shape = 2
        
        # Player position indicators
        position_shape = 3  # current player, previous player, own player ID
        
        # Round information
        round_shape = 1  # round number (normalized)
        
        # Recent history (last 3 bids encoded as quantity and value)
        history_shape = 3 * 2
        
        # Total observation shape
        self.observation_shape = (
            dice_shape + dice_counts_shape + bid_shape + 
            position_shape + round_shape + history_shape,
        )
    
    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Encode a player's observation into a neural network compatible format.
        
        Args:
            observation: Raw observation dictionary from the environment
            
        Returns:
            Encoded observation as a numpy array
        """
        # Extract elements from the observation
        player_id = observation['player_id']
        dice = observation['dice'][player_id]
        dice_counts = observation['dice_counts']
        current_player = observation['current_player']
        previous_player = observation['previous_player'] if observation['previous_player'] is not None else -1
        current_bid = observation['current_bid']
        history = observation['history']
        round_num = observation['round_num']
        
        # Initialize the encoded observation
        encoded = []
        
        # Encode own dice (one-hot)
        dice_encoding = np.zeros(self.num_dice * self.dice_faces)
        for i, value in enumerate(dice[:dice_counts[player_id]]):
            if value > 0:  # Only encode non-zero dice
                idx = i * self.dice_faces + (value - 1)
                dice_encoding[idx] = 1
        encoded.extend(dice_encoding)
        
        # Encode dice counts (normalized)
        encoded.extend(dice_counts / self.num_dice)
        
        # Encode current bid
        if current_bid is not None:
            quantity, value = current_bid
            # Normalize quantity and value
            encoded.append(quantity / sum(dice_counts))
            encoded.append(value / self.dice_faces)
        else:
            encoded.extend([0, 0])  # No current bid
        
        # Encode player positions (one-hot)
        current_player_encoding = np.zeros(self.num_players)
        if current_player >= 0:
            current_player_encoding[current_player] = 1
        encoded.extend(current_player_encoding)
        
        previous_player_encoding = np.zeros(self.num_players)
        if previous_player >= 0:
            previous_player_encoding[previous_player] = 1
        encoded.extend(previous_player_encoding)
        
        # Encode own player ID (one-hot)
        player_encoding = np.zeros(self.num_players)
        player_encoding[player_id] = 1
        encoded.extend(player_encoding)
        
        # Encode round number (normalized)
        encoded.append(round_num / 20)  # Assuming games rarely go beyond 20 rounds
        
        # Encode recent history (last 3 bids)
        bid_history = []
        for entry in reversed(history):
            if entry['action']['type'] == 'bid':
                bid_history.append((
                    entry['action']['quantity'],
                    entry['action']['value']
                ))
            
            if len(bid_history) >= 3:
                break
        
        # Pad history if needed
        while len(bid_history) < 3:
            bid_history.append((0, 0))
        
        # Add history to encoding (normalized)
        for quantity, value in bid_history:
            encoded.append(quantity / sum(dice_counts) if quantity > 0 else 0)
            encoded.append(value / self.dice_faces if value > 0 else 0)
        
        return np.array(encoded, dtype=np.float32)
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of encoded observations.
        
        Returns:
            Tuple describing the observation shape
        """
        return self.observation_shape


class StateEncoder:
    """
    Encodes full game states for centralized training.
    
    This class provides methods to encode the complete game state including
    all players' dice, which can be used for centralized critics or for
    value function approximation during training.
    
    Attributes:
        num_players (int): Number of players in the game
        num_dice (int): Maximum number of dice per player
        dice_faces (int): Number of faces on each die
        state_shape (tuple): Shape of the encoded state
    """
    
    def __init__(self, num_players: int, num_dice: int, dice_faces: int):
        """
        Initialize the state encoder.
        
        Args:
            num_players: Number of players in the game
            num_dice: Maximum number of dice per player
            dice_faces: Number of faces on each die
        """
        self.num_players = num_players
        self.num_dice = num_dice
        self.dice_faces = dice_faces
        
        # Calculate state shape
        # All dice (one-hot encoded)
        dice_shape = num_players * num_dice * dice_faces
        
        # Dice counts for all players
        dice_counts_shape = num_players
        
        # Current bid (quantity, value)
        bid_shape = 2
        
        # Player position indicators
        position_shape = 2  # current player, previous player
        
        # Round information
        round_shape = 1  # round number
        
        # Total state shape
        self.state_shape = (
            dice_shape + dice_counts_shape + bid_shape + 
            position_shape + round_shape,
        )
    
    def encode(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Encode the full game state for centralized training.
        
        Args:
            game_state: Complete game state including all dice
            
        Returns:
            Encoded state as a numpy array
        """
        # Extract elements from the game state
        dice = game_state['dice']
        dice_counts = game_state['dice_counts']
        current_player = game_state['current_player']
        previous_player = game_state['previous_player'] if game_state['previous_player'] is not None else -1
        current_bid = game_state['current_bid']
        round_num = game_state['round_num']
        
        # Initialize the encoded state
        encoded = []
        
        # Encode all dice (one-hot)
        for player in range(self.num_players):
            for i, value in enumerate(dice[player, :dice_counts[player]]):
                dice_encoding = np.zeros(self.dice_faces)
                if value > 0:  # Only encode non-zero dice
                    dice_encoding[value - 1] = 1
                encoded.extend(dice_encoding)
            
            # Pad with zeros for lost dice
            for _ in range(self.num_dice - dice_counts[player]):
                encoded.extend(np.zeros(self.dice_faces))
        
        # Encode dice counts (normalized)
        encoded.extend(dice_counts / self.num_dice)
        
        # Encode current bid
        if current_bid is not None:
            quantity, value = current_bid
            # Normalize quantity and value
            encoded.append(quantity / sum(dice_counts))
            encoded.append(value / self.dice_faces)
        else:
            encoded.extend([0, 0])  # No current bid
        
        # Encode player positions (one-hot)
        current_player_encoding = np.zeros(self.num_players)
        if current_player >= 0:
            current_player_encoding[current_player] = 1
        encoded.extend(current_player_encoding)
        
        previous_player_encoding = np.zeros(self.num_players)
        if previous_player >= 0:
            previous_player_encoding[previous_player] = 1
        encoded.extend(previous_player_encoding)
        
        # Encode round number (normalized)
        encoded.append(round_num / 20)  # Assuming games rarely go beyond 20 rounds
        
        return np.array(encoded, dtype=np.float32)
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of encoded states.
        
        Returns:
            Tuple describing the state shape
        """
        return self.state_shape