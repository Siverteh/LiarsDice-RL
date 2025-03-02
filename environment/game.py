"""
Core game logic for Liar's Dice.

This module implements the rules and mechanics of Liar's Dice,
handling game state, dice rolling, bidding, and game progression.
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any


class GameState(Enum):
    """Enum representing the possible states of the game."""
    ONGOING = 0
    ROUND_END = 1
    GAME_OVER = 2


class LiarsDiceGame:
    """
    Implementation of Liar's Dice game for reinforcement learning.
    
    This class handles the core game logic including:
    - Dice rolling and management
    - Player turns and action validation
    - Bid tracking and resolution
    - Game state transitions
    
    Attributes:
        num_players (int): Number of players in the game
        num_dice (int): Number of dice each player starts with
        dice_faces (int): Number of faces on each die (typically 6)
        current_player (int): Index of the player whose turn it is
        dice (np.ndarray): Array containing all players' dice
        dice_counts (np.ndarray): Count of remaining dice for each player
        current_bid (Tuple[int, int]): Current bid as (quantity, value)
        previous_player (int): Index of the player who made the last bid
        history (List): History of actions taken in the current round
        round_num (int): Current round number
        game_state (GameState): Current state of the game
    """
    
    def __init__(
        self, 
        num_players: int = 4,
        num_dice: int = 5,
        dice_faces: int = 6,
        seed: Optional[int] = None
    ):
        """
        Initialize a new Liar's Dice game.
        
        Args:
            num_players: Number of players (default 4)
            num_dice: Initial number of dice per player (default 5)
            dice_faces: Number of faces on each die (default 6)
            seed: Random seed for reproducibility
        """
        self.num_players = num_players
        self.num_dice = num_dice
        self.dice_faces = dice_faces
        self.rng = np.random.RandomState(seed)
        
        # Initialize game state
        self.current_player = 0
        self.dice = np.zeros((num_players, num_dice), dtype=int)
        self.dice_counts = np.full(num_players, num_dice, dtype=int)
        self.current_bid = None
        self.previous_player = None
        self.history = []
        self.round_num = 1
        self.game_state = GameState.ONGOING
        
        # Roll initial dice
        self._roll_all_dice()
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the game to its initial state.
        
        Args:
            seed: New random seed (optional)
            
        Returns:
            Dict containing initial observations for all players
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        self.current_player = 0
        self.dice = np.zeros((self.num_players, self.num_dice), dtype=int)
        self.dice_counts = np.full(self.num_players, self.num_dice, dtype=int)
        self.current_bid = None
        self.previous_player = None
        self.history = []
        self.round_num = 1
        self.game_state = GameState.ONGOING
        
        # Roll initial dice
        self._roll_all_dice()
        
        return self._get_observations()
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute a game step based on the current player's action.
        
        Args:
            action: Dictionary containing action details:
                'type': 'bid' or 'challenge'
                'quantity': Number of dice (for bid)
                'value': Die face value (for bid)
        
        Returns:
            Tuple of (observations, rewards, done, info):
                observations: Dict of observations for each player
                rewards: Dict of rewards for each player
                done: Boolean indicating if the game is over
                info: Additional information for debugging
        """
        if self.game_state != GameState.ONGOING:
            raise ValueError("Game is not in an ongoing state")
        
        rewards = {i: 0.0 for i in range(self.num_players)}
        
        # Record action in history
        self.history.append({
            'player': self.current_player,
            'action': action,
            'round': self.round_num
        })
        
        # Process the action
        if action['type'] == 'bid':
            self._handle_bid(action)
        elif action['type'] == 'challenge':
            rewards = self._handle_challenge()
        else:
            raise ValueError(f"Unknown action type: {action['type']}")
        
        # Get observations for the next state
        observations = self._get_observations()
        
        # Check if the game is over
        done = self.game_state == GameState.GAME_OVER
        
        # Compile additional info
        info = {
            'round_num': self.round_num,
            'dice_counts': self.dice_counts.copy(),
            'history': self.history.copy(),
            'state': self.game_state.name,
            'last_bid': self.current_bid,  # Include current bid for reward shaping
        }
        
        # Add dice values for debugging and reward shaping
        for player in range(self.num_players):
            player_dice = [int(d) for d in self.dice[player, :self.dice_counts[player]]]
            info[f'player_{player}_dice'] = player_dice
        
        return observations, rewards, done, info
    
    def _roll_all_dice(self) -> None:
        """Roll all dice for all players who still have dice."""
        for player in range(self.num_players):
            dice_count = self.dice_counts[player]
            if dice_count > 0:
                self.dice[player, :dice_count] = self.rng.randint(1, self.dice_faces + 1, dice_count)
                self.dice[player, dice_count:] = 0  # Zero out dice that have been lost
    
    def _handle_bid(self, action: Dict[str, Any]) -> None:
        """
        Process a bid action.
        
        Args:
            action: Dictionary with 'quantity' and 'value' keys
        """
        quantity = action['quantity']
        value = action['value']
        
        # Validate the bid
        if self.current_bid is not None:
            curr_quantity, curr_value = self.current_bid
            valid = (
                (quantity > curr_quantity) or
                (quantity == curr_quantity and value > curr_value)
            )
            if not valid:
                raise ValueError(f"Invalid bid: {quantity} {value}s must be higher than {curr_quantity} {curr_value}s")
        
        # Update game state
        self.current_bid = (quantity, value)
        self.previous_player = self.current_player
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Skip players who have no dice
        while self.dice_counts[self.current_player] == 0:
            self.current_player = (self.current_player + 1) % self.num_players
    
    def _handle_challenge(self) -> Dict[str, float]:
        """
        Process a challenge action and determine the outcome.
        
        Returns:
            Dict of rewards for each player
        """
        challenger = self.current_player
        bidder = self.previous_player
        
        # Store the current bid before it gets cleared
        last_bid = self.current_bid
        
        # Count the total number of dice matching the bid
        quantity, value = self.current_bid
        total_matching = 0
        
        for player in range(self.num_players):
            player_dice = self.dice[player, :self.dice_counts[player]]
            matching_dice = np.sum(player_dice == value)
            total_matching += int(matching_dice)  # Convert to standard int
        
        # Determine if the challenge was successful
        # A challenge is successful if the actual count is LESS THAN the bid
        bid_valid = total_matching >= quantity
        
        # Set loser: either the challenger or the previous bidder
        if bid_valid:
            # Bid was valid, so challenger loses
            loser = challenger
        else:
            # Bid was invalid, so bidder loses
            loser = bidder
        
        # Implement consequences - loser loses a die
        self.dice_counts[loser] -= 1
        
        # Prepare rewards
        rewards = {i: 0.0 for i in range(self.num_players)}
        rewards[loser] = -1.0
        
        # Check if the game is over or a new round should begin
        active_players = np.sum(self.dice_counts > 0)
        
        if active_players == 1:
            # Game over - find the winner
            winner = np.argmax(self.dice_counts)
            rewards[winner] += 5.0  # Bonus for winning the game
            self.game_state = GameState.GAME_OVER
        else:
            # Start a new round
            self.game_state = GameState.ROUND_END
            self.round_num += 1
            self.current_bid = None
            self._roll_all_dice()
            
            # Find the starting player for the next round
            # According to standard rules, the player who lost a die starts the next round
            start_player = loser
            
            # Skip players who have no dice
            while self.dice_counts[start_player] == 0:
                start_player = (start_player + 1) % self.num_players
                
            self.current_player = start_player
            self.previous_player = None
            self.game_state = GameState.ONGOING
        
        return rewards
    
    def _get_observations(self) -> Dict[str, Any]:
        """
        Get observations for all players.
        
        Returns:
            Dict mapping player indices to their observations
        """
        observations = {}
        
        for player in range(self.num_players):
            # Each player can see their own dice but not others'
            visible_dice = np.zeros_like(self.dice)
            visible_dice[player] = self.dice[player]
            
            observations[player] = {
                'dice': visible_dice.copy(),
                'dice_counts': self.dice_counts.copy(),
                'current_player': self.current_player,
                'current_bid': self.current_bid,
                'previous_player': self.previous_player,
                'history': self.history.copy(),
                'player_id': player,
                'round_num': self.round_num
            }
        
        return observations
    
    def get_valid_actions(self, player_idx: int = None) -> List[Dict[str, Any]]:
        """
        Get list of valid actions for the specified player.
        
        Args:
            player_idx: Index of the player
            
        Returns:
            List of valid action dictionaries
        """
        if player_idx is None:
            player_idx = self.current_player
            
        if player_idx != self.current_player:
            return []
        
        valid_actions = []
        
        # Can always challenge if there's a current bid
        if self.current_bid is not None:
            valid_actions.append({'type': 'challenge'})
        
        # Calculate valid bids
        if self.current_bid is None:
            # First bid of the round - can bid anything
            for quantity in range(1, sum(self.dice_counts) + 1):
                for value in range(1, self.dice_faces + 1):
                    valid_actions.append({
                        'type': 'bid',
                        'quantity': quantity,
                        'value': value
                    })
        else:
            curr_quantity, curr_value = self.current_bid
            
            # Can bid higher quantity
            for quantity in range(curr_quantity + 1, sum(self.dice_counts) + 1):
                for value in range(1, self.dice_faces + 1):
                    valid_actions.append({
                        'type': 'bid',
                        'quantity': quantity,
                        'value': value
                    })
            
            # Can bid same quantity but higher value
            for value in range(curr_value + 1, self.dice_faces + 1):
                valid_actions.append({
                    'type': 'bid',
                    'quantity': curr_quantity,
                    'value': value
                })
        
        return valid_actions
    
    def render(self) -> None:
        """Print the current game state to the console."""
        print(f"=== Round {self.round_num} ===")
        print(f"Current player: {self.current_player}")
        
        if self.current_bid is not None:
            quantity, value = self.current_bid
            print(f"Current bid: {quantity} {value}s")
        else:
            print("No current bid")
        
        print("\nDice counts:", self.dice_counts)
        
        print("\nDice:")
        for player in range(self.num_players):
            dice_str = ", ".join(str(d) for d in self.dice[player, :self.dice_counts[player]])
            print(f"Player {player}: [{dice_str}]")
        
        print("\nValid actions for current player:")
        valid_actions = self.get_valid_actions(self.current_player)
        for i, action in enumerate(valid_actions[:10]):  # Show first 10 actions
            print(f"  {i}: {action}")
        
        if len(valid_actions) > 10:
            print(f"  ... and {len(valid_actions) - 10} more actions")
        
        print("=" * 40)