"""
Reward calculation for Liar's Dice.

This module handles different reward schemes for the Liar's Dice
environment, allowing for experimentation with different incentive
structures during training.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class RewardCalculator:
    """
    Calculates rewards for Liar's Dice players.
    
    This class provides different reward schemes that can be used during
    training, including sparse rewards, shaped rewards, and various
    incentive structures to encourage exploration and strategic play.
    
    Attributes:
        num_players (int): Number of players in the game
        reward_scheme (str): Name of the reward scheme to use
        win_reward (float): Reward for winning the game
        lose_dice_penalty (float): Penalty for losing a die
        successful_bluff_reward (float): Reward for successful bluffing
        successful_challenge_reward (float): Reward for successful challenging
    """
    
    def __init__(
        self,
        num_players: int,
        reward_scheme: str = 'standard',
        win_reward: float = 10.0,
        lose_dice_penalty: float = -1.0,
        successful_bluff_reward: float = 0.5,
        successful_challenge_reward: float = 0.5
    ):
        """
        Initialize the reward calculator.
        
        Args:
            num_players: Number of players in the game
            reward_scheme: Name of the reward scheme to use
            win_reward: Reward for winning the game
            lose_dice_penalty: Penalty for losing a die
            successful_bluff_reward: Reward for successful bluffing
            successful_challenge_reward: Reward for successful challenging
        """
        self.num_players = num_players
        self.reward_scheme = reward_scheme
        self.win_reward = win_reward
        self.lose_dice_penalty = lose_dice_penalty
        self.successful_bluff_reward = successful_bluff_reward
        self.successful_challenge_reward = successful_challenge_reward
    
    def calculate_rewards(
        self, 
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate rewards for all players based on the state transition.
        
        Args:
            prev_state: State before the action
            action: Action taken
            next_state: State after the action
            info: Additional information about the transition
            
        Returns:
            Dict mapping player indices to their rewards
        """
        if self.reward_scheme == 'standard':
            return self._standard_rewards(prev_state, action, next_state, info)
        elif self.reward_scheme == 'shaped':
            return self._shaped_rewards(prev_state, action, next_state, info)
        elif self.reward_scheme == 'sparse':
            return self._sparse_rewards(prev_state, action, next_state, info)
        elif self.reward_scheme == 'strategic':
            return self._strategic_rewards(prev_state, action, next_state, info)
        else:
            raise ValueError(f"Unknown reward scheme: {self.reward_scheme}")
    
    def _standard_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate standard rewards based on game outcomes.
        
        This reward scheme gives:
        - Positive reward for winning the game
        - Negative reward for losing a die
        - Small time penalty to encourage efficient play
        
        Args:
            prev_state: State before the action
            action: Action taken
            next_state: State after the action
            info: Additional information about the transition
            
        Returns:
            Dict mapping player indices to their rewards
        """
        rewards = {i: 0.0 for i in range(self.num_players)}
        
        # Small time penalty to encourage finishing games
        for player in range(self.num_players):
            rewards[player] -= 0.01
        
        # Check if any player lost a die
        prev_dice_counts = prev_state['dice_counts']
        next_dice_counts = next_state['dice_counts']
        
        for player in range(self.num_players):
            if next_dice_counts[player] < prev_dice_counts[player]:
                rewards[player] += self.lose_dice_penalty
        
        # Check if the game is over
        if info.get('state') == 'GAME_OVER':
            # Reward the winner
            winner = np.argmax(next_dice_counts)
            rewards[winner] += self.win_reward
        
        return rewards
    
    def _shaped_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate shaped rewards to encourage strategic play.
        
        This reward scheme adds:
        - Rewards for successful bluffing
        - Rewards for successful challenging
        - Rewards for making bids proportional to risk
        
        Args:
            prev_state: State before the action
            action: Action taken
            next_state: State after the action
            info: Additional information about the transition
            
        Returns:
            Dict mapping player indices to their rewards
        """
        # Start with standard rewards
        rewards = self._standard_rewards(prev_state, action, next_state, info)
        
        # Add shaping rewards
        current_player = prev_state['current_player']
        
        if action['type'] == 'bid':
            # Reward for making a risky but valid bid
            if prev_state['current_bid'] is not None:
                prev_quantity, prev_value = prev_state['current_bid']
                new_quantity, new_value = action['quantity'], action['value']
                
                # Calculate the "riskiness" of the bid
                if new_quantity > prev_quantity:
                    risk_factor = (new_quantity - prev_quantity) / sum(prev_state['dice_counts'])
                    rewards[current_player] += 0.1 * risk_factor
                elif new_value > prev_value:
                    risk_factor = (new_value - prev_value) / 6
                    rewards[current_player] += 0.1 * risk_factor
        
        elif action['type'] == 'challenge':
            # Reward for successful challenging
            prev_dice_counts = prev_state['dice_counts']
            next_dice_counts = next_state['dice_counts']
            
            # Identify the player who lost a die
            for player in range(self.num_players):
                if next_dice_counts[player] < prev_dice_counts[player]:
                    if player != current_player:
                        # Successful challenge
                        rewards[current_player] += self.successful_challenge_reward
                    else:
                        # Failed challenge, probably means someone was bluffing
                        previous_player = prev_state['previous_player']
                        if previous_player is not None:
                            rewards[previous_player] += self.successful_bluff_reward
        
        return rewards
    
    def _sparse_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate sparse rewards (only at game end).
        
        This reward scheme only gives rewards at the end of the game:
        - Positive reward for winning
        - Zero reward for losing
        
        Args:
            prev_state: State before the action
            action: Action taken
            next_state: State after the action
            info: Additional information about the transition
            
        Returns:
            Dict mapping player indices to their rewards
        """
        rewards = {i: 0.0 for i in range(self.num_players)}
        
        # Only give rewards at the end of the game
        if info.get('state') == 'GAME_OVER':
            winner = np.argmax(next_state['dice_counts'])
            rewards[winner] = self.win_reward
        
        return rewards
    
    def _strategic_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate rewards focused on encouraging strategic depth.
        
        This reward scheme encourages:
        - Accurately assessing probabilities
        - Bluffing in advantageous situations
        - Making optimal challenges
        
        Args:
            prev_state: State before the action
            action: Action taken
            next_state: State after the action
            info: Additional information about the transition
            
        Returns:
            Dict mapping player indices to their rewards
        """
        # Start with standard rewards
        rewards = self._standard_rewards(prev_state, action, next_state, info)
        
        current_player = prev_state['current_player']
        
        if action['type'] == 'bid':
            # Calculate the probability of the bid being true based on player's knowledge
            if 'dice' in prev_state:
                player_dice = prev_state['dice'][current_player]
                player_count = sum(player_dice == action['value'])
                total_dice = sum(prev_state['dice_counts'])
                
                # Estimated count from other players (assuming uniform distribution)
                other_dice = total_dice - prev_state['dice_counts'][current_player]
                expected_other = other_dice * (1 / 6)  # Assuming fair dice
                
                expected_total = player_count + expected_other
                
                if action['quantity'] > expected_total * 1.2:
                    # Reward for bold bluffing
                    rewards[current_player] += 0.2
                elif action['quantity'] < expected_total * 0.8:
                    # Small penalty for overly conservative bids
                    rewards[current_player] -= 0.1
        
        elif action['type'] == 'challenge':
            # Analyze the challenge decision
            if prev_state['current_bid'] is not None:
                bid_quantity, bid_value = prev_state['current_bid']
                
                # Calculate the probability of the bid being true
                player_dice = prev_state['dice'][current_player]
                player_count = sum(player_dice == bid_value)
                total_dice = sum(prev_state['dice_counts'])
                
                # Expected number from probabilistic reasoning
                other_dice = total_dice - prev_state['dice_counts'][current_player]
                expected_other = other_dice * (1 / 6)  # Assuming fair dice
                
                expected_total = player_count + expected_other
                
                # Reward for challenging when probability is low
                if bid_quantity > expected_total * 1.3:
                    rewards[current_player] += 0.3
                # Penalty for challenging when probability is high
                elif bid_quantity < expected_total * 0.7:
                    rewards[current_player] -= 0.2
        
        return rewards