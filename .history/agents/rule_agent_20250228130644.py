"""
Rule-based agents for Liar's Dice.

This module contains rule-based (non-learning) agents for Liar's Dice,
which follow predetermined strategies of varying sophistication.
These agents serve as baselines and opponents for reinforcement learning agents.
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects actions uniformly at random from valid actions.
    
    This agent serves as the simplest possible baseline, with no strategic intelligence.
    """
    
    def __init__(self, player_id: int, seed: Optional[int] = None):
        """
        Initialize the random agent.
        
        Args:
            player_id: ID of the player this agent controls
            seed: Optional random seed for reproducibility
        """
        super().__init__(player_id, name=f"Random-{player_id}")
        self.rng = random.Random(seed)
    
    def act(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select a random valid action.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            A randomly selected valid action
        """
        if not valid_actions:
            return None
        
        return self.rng.choice(valid_actions)


class ConservativeAgent(BaseAgent):
    """
    Agent that plays conservatively, based only on what it can see.
    
    This agent:
    - Bids based on actual dice it has
    - Challenges when a bid seems statistically unlikely
    - Does not bluff or make risky bids
    """
    
    def __init__(self, player_id: int, challenge_threshold: float = 1.5):
        """
        Initialize the conservative agent.
        
        Args:
            player_id: ID of the player this agent controls
            challenge_threshold: Threshold for challenging bids (higher = more conservative)
        """
        super().__init__(player_id, name=f"Conservative-{player_id}")
        self.challenge_threshold = challenge_threshold
    
    def act(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action using conservative strategy.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            The selected action
        """
        if not valid_actions:
            return None
        
        # Get own dice
        own_dice = observation['dice'][self.player_id]
        
        # Count values in own dice
        value_counts = {}
        for i in range(1, 7):
            value_counts[i] = np.sum(own_dice == i)
        
        # If no current bid, make a bid based on what we can see
        if observation['current_bid'] is None:
            # Find most common value in our hand
            max_count = 0
            max_value = 1
            for value, count in value_counts.items():
                if count > max_count:
                    max_count = count
                    max_value = value
            
            # Bid the actual count
            for action in valid_actions:
                if action['type'] == 'bid' and action['quantity'] == max_count and action['value'] == max_value:
                    return action
            
            # If no exact match, find closest bid
            best_action = None
            min_diff = float('inf')
            for action in valid_actions:
                if action['type'] == 'bid' and action['value'] == max_value:
                    diff = abs(action['quantity'] - max_count)
                    if diff < min_diff:
                        min_diff = diff
                        best_action = action
            
            if best_action:
                return best_action
            
            # If no suitable bid found, take first bid action
            for action in valid_actions:
                if action['type'] == 'bid':
                    return action
        
        # If there is a current bid, decide whether to challenge or bid
        else:
            curr_quantity, curr_value = observation['current_bid']
            
            # Count total dice in game
            total_dice = sum(observation['dice_counts'])
            
            # Count how many of the bid value we have
            own_count = value_counts.get(curr_value, 0)
            
            # Estimate probability of bid being true
            # Assume other dice show the value with 1/6 probability
            other_dice = total_dice - observation['dice_counts'][self.player_id]
            expected_others = other_dice / 6  # Expected value assuming fair distribution
            expected_total = own_count + expected_others
            
            # Challenge if bid seems unlikely
            if curr_quantity > expected_total * self.challenge_threshold:
                for action in valid_actions:
                    if action['type'] == 'challenge':
                        return action
            
            # Otherwise, make a new bid
            # Find our most common value
            max_count = 0
            max_value = 1
            for value, count in value_counts.items():
                if count > max_count:
                    max_count = count
                    max_value = value
            
            # Try to bid our most common value
            best_action = None
            for action in valid_actions:
                if action['type'] == 'bid' and action['value'] == max_value:
                    if best_action is None or action['quantity'] < best_action['quantity']:
                        best_action = action
            
            if best_action:
                return best_action
            
            # If no suitable bid found, take any valid bid action
            for action in valid_actions:
                if action['type'] == 'bid':
                    return action
            
            # If no bid is possible, challenge
            for action in valid_actions:
                if action['type'] == 'challenge':
                    return action
        
        # Fallback: random action
        return random.choice(valid_actions)

class VeryMildConservativeAgent(ConservativeAgent):
    """An extremely easy conservative agent with minimal challenging."""
    def __init__(self, player_id: int):
        super().__init__(player_id, challenge_threshold=4.0)  # Very reluctant to challenge
        self.name = f"VeryMildConservative-{player_id}"

class MildConservativeAgent(ConservativeAgent):
    """A less aggressive version of ConservativeAgent that's easier to beat."""
    
    def __init__(self, player_id: int, challenge_threshold: float = 2.5):
        """
        Initialize with a higher challenge threshold.
        Higher threshold = less likely to challenge = easier opponent
        """
        super().__init__(player_id, challenge_threshold)
        self.name = f"MildConservative-{player_id}"

class StrategicAgent(BaseAgent):
    """
    Agent that uses more sophisticated strategies, including bluffing.
    
    This agent:
    - Makes strategic bids based on game state and dice counts
    - Uses bluffing in appropriate situations
    - Makes more informed challenge decisions
    - Adapts strategy based on die counts and round number
    """
    
    def __init__(
        self, 
        player_id: int, 
        aggression: float = 0.5,
        bluff_probability: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize the strategic agent.
        
        Args:
            player_id: ID of the player this agent controls
            aggression: How aggressively the agent bids (0-1)
            bluff_probability: Probability of making a bluff
            seed: Random seed for reproducibility
        """
        super().__init__(player_id, name=f"Strategic-{player_id}")
        self.aggression = aggression
        self.bluff_probability = bluff_probability
        self.rng = random.Random(seed)
        
        # Track game history
        self.opponent_bids = {}  # Maps player_id to their bidding patterns
        self.round_num = 0
        self.last_observation = None
    
    def reset(self) -> None:
        """Reset the agent's memory for a new game."""
        self.opponent_bids = {}
        self.round_num = 0
        self.last_observation = None
    
    def act(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action using strategic decision-making.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            The selected action
        """
        if not valid_actions:
            return None
        
        # Track round number
        if self.last_observation is None or observation['round_num'] > self.last_observation['round_num']:
            self.round_num = observation['round_num']
        
        # Store observation for future reference
        self.last_observation = observation
        
        # Update opponent bid history
        self._update_opponent_history(observation)
        
        # Get own dice
        own_dice = observation['dice'][self.player_id]
        dice_counts = observation['dice_counts']
        
        # Count values in own dice
        value_counts = {}
        for i in range(1, 7):
            value_counts[i] = np.sum(own_dice == i)
        
        # If no current bid, make initial bid
        if observation['current_bid'] is None:
            return self._make_initial_bid(value_counts, observation, valid_actions)
        
        # Determine whether to challenge or bid
        curr_quantity, curr_value = observation['current_bid']
        previous_player = observation['previous_player']
        
        # Analyze bid probability
        challenge_score = self._evaluate_challenge(curr_quantity, curr_value, value_counts, observation)
        
        # Decide whether to challenge based on score and game state
        if challenge_score > 0.7:
            for action in valid_actions:
                if action['type'] == 'challenge':
                    return action
        
        # If not challenging, make a new bid
        return self._make_strategic_bid(value_counts, observation, valid_actions)
    
    def _update_opponent_history(self, observation: Dict[str, Any]) -> None:
        """Update history of opponent bidding patterns."""
        # Extract last few actions from history
        history = observation['history']
        
        for entry in history:
            player = entry['player']
            if player != self.player_id and entry['action']['type'] == 'bid':
                if player not in self.opponent_bids:
                    self.opponent_bids[player] = []
                
                self.opponent_bids[player].append(entry['action'])
    
    def _make_initial_bid(
        self, 
        value_counts: Dict[int, int],
        observation: Dict[str, Any], 
        valid_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make an initial bid to start the round."""
        # Find most common value
        max_count = 0
        max_value = 1
        for value, count in value_counts.items():
            if count > max_count:
                max_count = count
                max_value = value
        
        # Decide whether to bluff
        bluffing = self.rng.random() < self.bluff_probability
        
        if bluffing:
            # Bluff by bidding higher than what we have
            bluff_quantity = max_count + 1
            
            # Find a valid bluff action
            for action in valid_actions:
                if action['type'] == 'bid' and action['value'] == max_value:
                    if action['quantity'] >= bluff_quantity:
                        return action
        
        # Not bluffing or no valid bluff found
        # Bid conservatively based on what we have
        target_quantity = max_count
        
        # Find a valid bid action
        for action in valid_actions:
            if action['type'] == 'bid' and action['value'] == max_value:
                if action['quantity'] <= target_quantity:
                    return action
        
        # If no suitable bid found, just take first bid action
        for action in valid_actions:
            if action['type'] == 'bid':
                return action
        
        # Fallback to random action
        return random.choice(valid_actions)
    
    def _evaluate_challenge(
        self,
        curr_quantity: int,
        curr_value: int,
        value_counts: Dict[int, int],
        observation: Dict[str, Any]
    ) -> float:
        """
        Evaluate whether to challenge the current bid.
        
        Returns:
            Score between 0 and 1, higher means more likely to challenge
        """
        # Get total dice in play
        total_dice = sum(observation['dice_counts'])
        
        # Our dice matching the bid value
        own_count = value_counts.get(curr_value, 0)
        
        # Expected total based on probability
        other_dice = total_dice - observation['dice_counts'][self.player_id]
        expected_others = other_dice / 6
        expected_total = own_count + expected_others
        
        # Base challenge score on how unlikely the bid is
        if curr_quantity <= expected_total:
            # Bid is likely true - low challenge score
            challenge_score = 0.0
        else:
            # Bid exceeds expected - challenge score based on difference
            difference = curr_quantity - expected_total
            challenge_score = min(1.0, difference / (total_dice / 3))
        
        # Adjust based on opponent behavior
        previous_player = observation['previous_player']
        if previous_player in self.opponent_bids and len(self.opponent_bids[previous_player]) >= 3:
            # Check if opponent tends to bluff
            bluff_count = 0
            for bid in self.opponent_bids[previous_player][-3:]:
                # Simple heuristic - high bids might be bluffs
                if bid['quantity'] > total_dice / 3:
                    bluff_count += 1
            
            # Increase challenge score if opponent seems to bluff a lot
            if bluff_count >= 2:
                challenge_score += 0.2
        
        # Adjust based on game state
        # More aggressive in later rounds or when losing
        if self.round_num > 5 or observation['dice_counts'][self.player_id] < max(observation['dice_counts']):
            challenge_score += 0.1
        
        return min(1.0, challenge_score)
    
    def _make_strategic_bid(
        self, 
        value_counts: Dict[int, int],
        observation: Dict[str, Any], 
        valid_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make a strategic bid based on game state."""
        curr_quantity, curr_value = observation['current_bid']
        
        # Determine whether to continue with current value or switch
        own_count_curr = value_counts.get(curr_value, 0)
        
        # Find our most common value
        max_count = 0
        max_value = 1
        for value, count in value_counts.items():
            if count > max_count:
                max_count = count
                max_value = value
        
        # Decide whether to bluff
        bluffing = self.rng.random() < self.bluff_probability
        
        # Find valid actions that continue with current value
        valid_continue = [a for a in valid_actions 
                        if a['type'] == 'bid' and a['value'] == curr_value]
        
        # Find valid actions that use our most common value
        valid_switch = [a for a in valid_actions 
                      if a['type'] == 'bid' and a['value'] == max_value]
        
        # If we have the current value, prioritize continuing with it
        if own_count_curr > 0 and valid_continue:
            target_quantity = curr_quantity + 1
            
            best_action = None
            for action in valid_continue:
                if action['quantity'] >= target_quantity:
                    if best_action is None or action['quantity'] < best_action['quantity']:
                        best_action = action
            
            if best_action:
                return best_action
        
        # Otherwise, try to switch to our most common value
        if valid_switch:
            if bluffing:
                target_quantity = max_count + int(self.aggression * 2) + 1
            else:
                target_quantity = max_count
            
            best_action = None
            for action in valid_switch:
                if best_action is None or abs(action['quantity'] - target_quantity) < abs(best_action['quantity'] - target_quantity):
                    best_action = action
            
            if best_action:
                return best_action
        
        # If no good options for continuing or switching, just take any valid bid
        for action in valid_actions:
            if action['type'] == 'bid':
                return action
        
        # If no bid is possible, challenge
        for action in valid_actions:
            if action['type'] == 'challenge':
                return action
        
        # Fallback: random action
        return random.choice(valid_actions)