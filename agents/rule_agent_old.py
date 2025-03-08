"""
Human-like agents for Liar's Dice.

This module implements various agents with different difficulty levels and strategies,
designed to better mimic human play patterns for more effective learning.
"""

import random
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from agents.base_agent import RLAgent
from math import comb


class RuleAgent(RLAgent):
    """
    Base class for human-like Liar's Dice agents.
    
    This class defines the interface for all agents and
    provides utility methods for analyzing game states.
    
    Attributes:
        agent_type (str): Type of the agent
        player_id (int): ID of the player in the game
        num_players (int): Total number of players
        dice_faces (int): Number of faces on each die
    """
    
    def __init__(self, agent_type: str = 'base', dice_faces: int = 6):
        """
        Initialize the rule agent.
        
        Args:
            agent_type: Type identifier for the agent
            dice_faces: Number of faces on each die
        """
        super(RuleAgent, self).__init__()
        self.agent_type = agent_type
        self.player_id = None
        self.num_players = None
        self.dice_faces = dice_faces
    
    def set_player_id(self, player_id: int, num_players: int):
        """
        Set the player ID and total number of players.
        
        Args:
            player_id: ID of the player
            num_players: Total number of players
        """
        self.player_id = player_id
        self.num_players = num_players
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]], training: bool = True) -> Dict[str, Any]:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            training: Whether the agent is training (ignored for rule agents)
            
        Returns:
            Selected action as a dictionary
        """
        # Default implementation: random action
        return random.choice(valid_actions)
    
    def analyze_game_state(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current game state to extract useful information.
        
        Args:
            observation: Current game observation
            
        Returns:
            Dictionary containing analyzed information
        """
        dice = observation['dice']
        dice_counts = observation['dice_counts']
        current_bid = observation['current_bid']
        
        # Count own dice by value
        own_dice = dice[self.player_id]
        own_dice_values = [d for d in own_dice if d > 0]
        own_dice_count = len(own_dice_values)
        
        # Count occurrences of each value in own dice
        own_value_counts = {i: own_dice_values.count(i) for i in range(1, self.dice_faces + 1)}
        
        # Calculate total dice in the game
        total_dice = sum(dice_counts)
        
        # Calculate probabilities for each value
        probabilities = {}
        for value in range(1, self.dice_faces + 1):
            # Known dice of this value (from our hand)
            known_count = own_value_counts.get(value, 0)
            
            # Unknown dice (other players)
            unknown_dice = total_dice - own_dice_count
            
            # Expected additional dice with this value (assuming uniform distribution)
            expected_additional = unknown_dice * (1 / self.dice_faces)
            
            # Total expected
            expected_total = known_count + expected_additional
            
            # Probability of exceeding current bid
            if current_bid is not None:
                bid_quantity, bid_value = current_bid
                if value == bid_value:
                    probabilities[value] = self._calculate_probability(
                        total_dice, bid_quantity, known_count, self.dice_faces
                    )
            else:
                probabilities[value] = 1.0  # No current bid
        
        return {
            'own_dice': own_dice_values,
            'own_value_counts': own_value_counts,
            'total_dice': total_dice,
            'probabilities': probabilities,
            'expected_counts': {
                value: own_value_counts.get(value, 0) + (total_dice - own_dice_count) * (1 / self.dice_faces)
                for value in range(1, self.dice_faces + 1)
            }
        }
    
    def _calculate_probability(self, total_dice: int, target_quantity: int, 
                              known_count: int, dice_faces: int) -> float:
        """
        Calculate the probability of at least target_quantity dice showing a specific value.
        
        This uses a binomial probability calculation for the unknown dice.
        
        Args:
            total_dice: Total number of dice in the game
            target_quantity: Target quantity in the bid
            known_count: Number of matching dice in own hand
            dice_faces: Number of faces on each die
            
        Returns:
            Probability as a float between 0 and 1
        """
        # If we already have enough, probability is 1
        if known_count >= target_quantity:
            return 1.0
        
        # Number of unknown dice
        unknown_dice = total_dice - known_count
        
        # Number of additional successes needed
        needed = target_quantity - known_count
        
        # If we need more successes than there are unknown dice, probability is 0
        if needed > unknown_dice:
            return 0.0
        
        # Probability of success for each unknown die
        p = 1 / dice_faces
        
        # Calculate probability using cumulative binomial distribution
        probability = 0.0
        for k in range(needed, unknown_dice + 1):
            # Binomial probability: C(n,k) * p^k * (1-p)^(n-k)
            binomial_coef = comb(unknown_dice, k)
            probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
        
        return probability
        
    # Implement required abstract methods from RLAgent
    
    def update(self, *args, **kwargs) -> float:
        """
        Rule agents don't update through learning.
        
        Returns:
            0.0 (no loss)
        """
        return 0.0
        
    def add_experience(self, obs: np.ndarray, action: Dict[str, Any], 
                      reward: float, next_obs: np.ndarray, done: bool):
        """
        Rule agents don't store experiences.
        """
        pass
        
    def save(self, path: str):
        """
        Save agent parameters to the specified path.
        
        Args:
            path: Directory to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        # Save agent type and parameters
        config = {
            'agent_type': self.agent_type,
            'dice_faces': self.dice_faces
        }
        
        # Add specific parameters for different agent types
        if hasattr(self, 'challenge_threshold'):
            config['challenge_threshold'] = self.challenge_threshold
            
        if hasattr(self, 'bluff_frequency'):
            config['bluff_frequency'] = self.bluff_frequency
            
        with open(os.path.join(path, 'rule_agent_config.json'), 'w') as f:
            json.dump(config, f)
            
    def load(self, path: str):
        """
        Load agent parameters from the specified path.
        
        Args:
            path: Directory to load the agent from
        """
        config_path = os.path.join(path, 'rule_agent_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Set parameters
            if 'dice_faces' in config:
                self.dice_faces = config['dice_faces']
                
            if 'challenge_threshold' in config and hasattr(self, 'challenge_threshold'):
                self.challenge_threshold = config['challenge_threshold']
                
            if 'bluff_frequency' in config and hasattr(self, 'bluff_frequency'):
                self.bluff_frequency = config['bluff_frequency']
                
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent for logging.
        
        Returns:
            Dictionary of agent statistics
        """
        stats = {
            'agent_type': self.agent_type,
            'dice_faces': self.dice_faces
        }
        
        # Add specific parameters for different agent types
        if hasattr(self, 'challenge_threshold'):
            stats['challenge_threshold'] = self.challenge_threshold
            
        if hasattr(self, 'bluff_frequency'):
            stats['bluff_frequency'] = self.bluff_frequency
            
        return stats



class RandomAgent(RuleAgent):
    """
    A slightly more sensible random agent that mimics a complete beginner.
    
    This agent still makes mostly random choices but with a slight preference
    for bidding values it actually has in its hand.
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the random agent."""
        super().__init__(agent_type='random', dice_faces=dice_faces)
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with minimal strategy - like a first-time player.
        
        Strategy:
        - 80% random choice
        - 20% choose a value we actually have in our dice
        """
        analysis = self.analyze_game_state(observation)
        
        # 20% of the time, try to be slightly strategic
        if random.random() < 0.2 and analysis['own_dice']:
            # Challenge actions
            challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
            
            # Bid actions
            bid_actions = [a for a in valid_actions if a['type'] == 'bid']
            
            if bid_actions:
                # Try to bid a value we actually have
                own_values = analysis['own_dice']
                value_to_bid = random.choice(own_values)
                
                # Look for bids with this value
                matching_bids = [a for a in bid_actions if a['value'] == value_to_bid]
                if matching_bids:
                    return random.choice(matching_bids)
        
        # Otherwise, completely random choice
        return random.choice(valid_actions)


class NaiveAgent(RuleAgent):
    """
    Naive agent that mimics a human beginner learning the basic rules.
    
    This agent:
    - Primarily bids based on dice it can see (own hand)
    - Has a basic understanding of probability but makes mistakes
    - Makes predictable raises
    - Rarely bluffs
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the naive agent."""
        super().__init__(agent_type='naive', dice_faces=dice_faces)
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action using a beginner's approach.
        
        Strategy:
        - Focus heavily on own dice (beginner tunnel vision)
        - Challenge if the bid seems impossible based on own hand
        - Predictable bid increases
        - Occasional beginner mistakes
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Beginner mistake: occasional random action (5% chance)
        if random.random() <= 0.1:
            return random.choice(valid_actions)
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Challenge if what we see makes it seem highly unlikely
            own_count = analysis['own_value_counts'].get(bid_value, 0)
            total_dice = analysis['total_dice']
            
            # Naive logic: "If I don't see many, there probably aren't many"
            # Challenges if the bid quantity is more than twice what they have
            if own_count * 2 < bid_quantity:
                # Especially likely to challenge if the bid quantity is close to total dice
                if bid_quantity > total_dice * 0.7:
                    return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # First bid strategy
        if current_bid is None:
            # Beginners focus on what they have the most of
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Simple: bid exactly what we have
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == best_count:
                        return action
                
                # If exact bid not available, try close options
                for action in bid_actions:
                    if action['value'] == best_value:
                        return action
            
            # If we don't have any dice or all values are equally common
            return random.choice(bid_actions)
        
        # Subsequent bid strategy
        bid_quantity, bid_value = current_bid
        
        # Very likely to increment quantity of the same value (beginner pattern)
        for action in bid_actions:
            if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                return action
        
        # Check if we have other values we can bid
        own_value_counts = analysis['own_value_counts']
        if own_value_counts:
            # Try to switch to a value we actually have
            for value, count in own_value_counts.items():
                if count > 0 and value > bid_value:
                    for action in bid_actions:
                        if action['value'] == value and action['quantity'] == bid_quantity:
                            return action
        
        # Otherwise just make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class ConservativeAgent(RuleAgent):
    """
    Conservative agent that mimics a cautious human player, but simplified for easier learning.
    
    This agent:
    - Makes "safe" bids based primarily on their own dice
    - Rarely bluffs
    - Challenges when bids seem unlikely but less aggressively
    - More consistent and predictable than the original version
    """
    
    def __init__(self, dice_faces: int = 6, challenge_threshold: float = 0.35):
        """
        Initialize the conservative agent.
        
        Args:
            dice_faces: Number of faces on each die
            challenge_threshold: Probability threshold for challenging (higher = less challenges)
        """
        super().__init__(agent_type='conservative', dice_faces=dice_faces)
        self.challenge_threshold = challenge_threshold
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with a cautious human approach, but simplified.
        
        Strategy:
        - Mostly bid values in own hand
        - Challenge when uncertain but less aggressively
        - Consistent small bid increases
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Check for challenge actions - simplified challenge logic
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Simple probability check for challenging
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Challenge if below threshold (no adjustments for dice count)
            if probability < self.challenge_threshold:
                return challenge_actions[0]
            
            # Also challenge if bid quantity is very close to total dice
            total_dice = analysis['total_dice']
            if bid_quantity > total_dice * 0.7:
                # And we don't have many of this value
                own_count = analysis['own_value_counts'].get(bid_value, 0)
                if own_count < bid_quantity * 0.3:
                    return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # First bid strategy - simplified
        if current_bid is None:
            # Conservative players bid what they have
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                # Bid the most common value in our dice
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Bid exactly what we have (not less)
                bid_quantity = best_count
                
                # Find matching bid
                matching_bids = [a for a in bid_actions if a['value'] == best_value]
                if matching_bids:
                    return min(matching_bids, key=lambda a: abs(a['quantity'] - bid_quantity))
            
            # If no good option, bid the lowest quantity
            return min(bid_actions, key=lambda a: a['quantity'])
        
        # Subsequent bid strategy - more consistent
        bid_quantity, bid_value = current_bid
        
        # Try to raise quantity by 1 if we have this value
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        if own_count > 0:
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Try to switch to a value we have 
        for value, count in analysis['own_value_counts'].items():
            if count > 0 and value > bid_value:
                # Switch to this value at same quantity
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # If we must increase, do it as conservatively as possible
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class AggressiveAgent(RuleAgent):
    """
    Aggressive agent that mimics a bold human player.
    
    This agent:
    - Makes big jumps in bids
    - Frequently bluffs
    - Challenges less often
    - Takes calculated risks
    - Sometimes makes emotional decisions
    """
    
    def __init__(self, dice_faces: int = 6, bluff_frequency: float = 0.4, challenge_threshold: float = 0.25):
        """
        Initialize the aggressive agent.
        
        Args:
            dice_faces: Number of faces on each die
            bluff_frequency: How often to bluff when bidding (0-1)
            challenge_threshold: Probability threshold for challenging
        """
        super().__init__(agent_type='aggressive', dice_faces=dice_faces)
        self.bluff_frequency = bluff_frequency
        self.challenge_threshold = challenge_threshold
        
        # Keep track of consecutive losses for "tilt" behavior
        self.consecutive_losses = 0
        self.last_round = 0
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with an aggressive human approach.
        
        Strategy:
        - Make bold bids and big jumps
        - Bluff frequently
        - Challenge less often
        - Take bigger risks when "on tilt" after losing
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        round_num = observation['round_num']
        
        # Check if a new round has started
        if round_num > self.last_round:
            # Check if we lost a die
            if len(analysis['own_dice']) < self.consecutive_losses + 2:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            self.last_round = round_num
        
        # "On tilt" behavior - more aggressive with consecutive losses
        tilt_factor = min(0.3, self.consecutive_losses * 0.1)
        effective_bluff_frequency = min(0.8, self.bluff_frequency + tilt_factor)
        
        # Decide whether to bluff this turn
        should_bluff = random.random() < effective_bluff_frequency
        
        # Check for challenge actions - aggressive players challenge less
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Only challenge when very confident
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Even more reluctant to challenge when on tilt
            effective_threshold = max(0.1, self.challenge_threshold - tilt_factor)
            
            if probability < effective_threshold:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # First bid strategy
        if current_bid is None:
            if should_bluff:
                # Bold opening bid
                high_bids = sorted(bid_actions, key=lambda a: a['quantity'], reverse=True)
                idx = min(2, len(high_bids) - 1)
                return high_bids[idx]
            
            # Otherwise bid based on our strongest value
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Aggressive players often bid more than they have
                target_quantity = best_count + 1
                
                # Find matching or close bid
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] >= best_count:
                        return action
            
            # Default to a reasonably high bid
            sorted_bids = sorted(bid_actions, key=lambda a: a['quantity'])
            high_idx = min(len(sorted_bids) - 1, int(len(sorted_bids) * 0.7))
            return sorted_bids[high_idx]
        
        # Subsequent bid strategy - aggressive increases
        bid_quantity, bid_value = current_bid
        
        if should_bluff:
            # Big jump in quantity (aggressive move)
            jump_size = random.randint(2, 3)  # 2-3 unit jump
            target_quantity = bid_quantity + jump_size
            
            # Find a valid bid with this jump or closest available
            jump_bids = [a for a in bid_actions if a['quantity'] >= target_quantity]
            if jump_bids:
                return min(jump_bids, key=lambda a: a['quantity'])
            
            # If no big jump available, try highest value
            high_value_bids = [a for a in bid_actions if a['value'] > bid_value]
            if high_value_bids:
                return max(high_value_bids, key=lambda a: a['value'])
        
        # Default increase (still somewhat aggressive)
        # Try to bid one or two more of the same value
        for increase in [2, 1]:
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + increase:
                    return action
        
        # Try higher values
        for value in range(self.dice_faces, bid_value, -1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If all else fails, make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class StrategicAgent(RuleAgent):
    """
    Strategic agent that mimics a thoughtful human player.
    
    This agent:
    - Balances risk and reward
    - Considers game state and opponent tendencies
    - Uses calculated bluffs
    - Makes decisions based on probability and psychology
    - Adapts strategy based on game phase
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the strategic agent."""
        super().__init__(agent_type='strategic', dice_faces=dice_faces)
        self.strategies = {
            'early_game': {'bluff_frequency': 0.2, 'challenge_threshold': 0.3},
            'mid_game': {'bluff_frequency': 0.35, 'challenge_threshold': 0.35},
            'late_game': {'bluff_frequency': 0.4, 'challenge_threshold': 0.4}
        }
        self.bid_history = []
        self.recent_challenges = []  # Track outcome of recent challenges
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with a strategic human approach.
        
        Strategy:
        - Adapt to game phase
        - Consider bid history
        - Balance between aggressive and conservative play
        - Bluff strategically
        - Consider total dice in play for probabilistic decisions
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        dice_counts = observation['dice_counts']
        history = observation.get('history', [])
        
        # Update bid history
        if current_bid is not None and (not self.bid_history or self.bid_history[-1] != current_bid):
            self.bid_history.append(current_bid)
        
        # Track challenges
        if history:
            for entry in history[-3:]:  # Look at recent history
                if entry['action']['type'] == 'challenge' and entry not in self.recent_challenges:
                    self.recent_challenges.append(entry)
                    if len(self.recent_challenges) > 5:  # Keep only recent
                        self.recent_challenges.pop(0)
        
        # Determine game phase based on dice distribution
        total_dice = sum(dice_counts)
        max_possible_dice = self.num_players * max(dice_counts)
        game_progress = 1 - (total_dice / max_possible_dice)
        
        if game_progress < 0.3:
            phase = 'early_game'
        elif game_progress < 0.7:
            phase = 'mid_game'
        else:
            phase = 'late_game'
        
        # Get strategy parameters for current phase
        strategy = self.strategies[phase]
        bluff_frequency = strategy['bluff_frequency']
        challenge_threshold = strategy['challenge_threshold']
        
        # Adjust strategy based on opponent behavior
        if self.recent_challenges:
            challenger_win_rate = sum(1 for c in self.recent_challenges if c.get('success', False)) / len(self.recent_challenges)
            if challenger_win_rate > 0.6:  # Opponents often catch bluffs
                bluff_frequency *= 0.7  # Bluff less
            elif challenger_win_rate < 0.3:  # Opponents rarely challenge successfully
                bluff_frequency *= 1.3  # Bluff more
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Calculate probability and adjust based on game context
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Consider bidding patterns for psychological reads
            if len(self.bid_history) >= 3:
                last_bidder_pattern = []
                # Check if same player has been bidding frequently
                for i in range(len(self.bid_history) - 3, len(self.bid_history)):
                    if i >= 0:
                        last_bidder_pattern.append(self.bid_history[i])
                
                # Look for suspicious patterns (unusual jumps)
                suspicious = False
                for i in range(1, len(last_bidder_pattern)):
                    prev_q, prev_v = last_bidder_pattern[i-1]
                    curr_q, curr_v = last_bidder_pattern[i]
                    # Big quantity jump or unusual value jump
                    if curr_q > prev_q + 2 or (curr_q == prev_q and curr_v > prev_v + 2):
                        suspicious = True
                
                if suspicious:
                    challenge_threshold *= 0.8  # More likely to challenge
            
            # Assess how "realistic" the bid is relative to total dice
            if bid_quantity > total_dice * 0.5 and probability < 0.5:
                # More likely to challenge high quantity bids
                challenge_threshold *= 0.9
            
            # Challenge if probability is below threshold
            if probability < challenge_threshold:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # Decide whether to bluff based on strategy and context
        should_bluff = random.random() < bluff_frequency
        
        # Strategy for first bid
        if current_bid is None:
            # Strategic opening based on our dice
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                # Find our best values
                sorted_values = sorted(own_value_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Strategic players sometimes bid their second-best value to mislead
                value_idx = 1 if len(sorted_values) > 1 and should_bluff else 0
                value_idx = min(value_idx, len(sorted_values) - 1)
                
                best_value, best_count = sorted_values[value_idx]
                
                # Calculate a reasonable bid quantity
                bid_quantity = best_count
                if should_bluff:
                    bid_quantity += 1
                
                # Find matching or close bid
                matching_bids = [a for a in bid_actions if a['value'] == best_value]
                if matching_bids:
                    closest = min(matching_bids, key=lambda a: abs(a['quantity'] - bid_quantity))
                    return closest
            
            # If no good option based on our dice, use balanced strategy
            return sorted(bid_actions, key=lambda a: a['quantity'])[len(bid_actions) // 2]
        
        # Strategy for subsequent bids - more nuanced
        bid_quantity, bid_value = current_bid
        
        # Strategic considerations for educated guesses
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        expected_count = analysis['expected_counts'][bid_value]
        total_dice = analysis['total_dice']
        
        # Find our strongest values
        best_values = sorted([(v, c) for v, c in analysis['own_value_counts'].items() if c > 0], 
                            key=lambda x: x[1], reverse=True)
        
        if should_bluff:
            # Strategic bluffing - calculated risks
            if expected_count >= bid_quantity:
                # Safe-ish raise on current value
                for action in bid_actions:
                    if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                        return action
            
            # Or try value switching to our strength
            if best_values:
                best_value = best_values[0][0]
                if best_value > bid_value:
                    for action in bid_actions:
                        if action['value'] == best_value and action['quantity'] == bid_quantity:
                            return action
            
            # Calculated risk - bid just beyond expected
            max_plausible = int(expected_count * 1.3)
            plausible_actions = [a for a in bid_actions if a['quantity'] <= max_plausible]
            if plausible_actions:
                return max(plausible_actions, key=lambda a: (a['quantity'], a['value']))
        
        # Standard strategic bidding - focus on values we have
        if best_values:
            best_value = best_values[0][0]
            best_count = best_values[0][1]
            
            # If our best value is better than current, switch to it
            if best_value > bid_value:
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == bid_quantity:
                        return action
            
            # If we have current value, consider small raise
            if own_count > 0:
                for action in bid_actions:
                    if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                        return action
        
        # Safe fallback - increment value instead of quantity
        for value in range(bid_value + 1, self.dice_faces + 1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If no good bid found, make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class AdaptiveAgent(RuleAgent):
    """
    Adaptive agent that mimics an expert human player.
    
    This agent:
    - Builds detailed models of opponent tendencies
    - Adapts strategy based on opponent patterns
    - Uses sophisticated probabilistic reasoning
    - Employs psychological tactics
    - Excels at endgame strategy
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the adaptive agent."""
        super().__init__(agent_type='adaptive', dice_faces=dice_faces)
        # Track opponent behaviors
        self.opponent_models = {}  # Models for each opponent
        self.current_round_bids = []  # Bids in current round
        self.round_history = []  # History of completed rounds
        
        # Default strategy parameters
        self.bluff_frequency = 0.3
        self.challenge_threshold = 0.35
        
        # Game phase and position tracking
        self.game_phase = 'early'  # early, mid, late, endgame
        self.is_leading = True
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with an expert human-like approach.
        
        Strategy:
        - Adaptive to opponent patterns
        - Psychological gameplay
        - Strategic use of position (chip leader vs. trailing)
        - Advanced probability assessment
        - Sophisticated bluffing
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        dice_counts = observation['dice_counts']
        round_num = observation['round_num']
        player_id = observation['player_id']
        history = observation.get('history', [])
        total_dice = sum(dice_counts)
        
        # Update game state information
        self._update_game_state(observation)
        
        # Update opponent models if we have enough data
        if round_num > 2:
            self._update_opponent_models(observation)
        
        # Determine if we're leading
        our_dice = dice_counts[self.player_id]
        max_opponent_dice = max([dice_counts[i] for i in range(len(dice_counts)) if i != self.player_id])
        self.is_leading = our_dice >= max_opponent_dice
        
        # Determine game phase
        max_total_dice = self.num_players * max(dice_counts)
        progress = 1 - (total_dice / max_total_dice)
        
        if progress < 0.3:
            self.game_phase = 'early'
        elif progress < 0.6:
            self.game_phase = 'mid'
        elif progress < 0.85:
            self.game_phase = 'late'
        else:
            self.game_phase = 'endgame'
        
        # Adapt strategy based on game phase and position
        self._adapt_strategy()
        
        # Get current opponent
        last_bidder = None
        if history and current_bid:
            # Find who made the current bid
            for entry in reversed(history):
                if entry['action']['type'] == 'bid':
                    last_bidder = entry['player']
                    break
        
        # Challenge decision - expert reasoning
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Calculate probability with expert-level assessment
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Adjust threshold based on various factors
            adjusted_threshold = self.challenge_threshold
            
            # Factor 1: Opponent bluffing tendency
            if last_bidder is not None and last_bidder in self.opponent_models:
                opponent = self.opponent_models[last_bidder]
                if opponent['bluff_ratio'] > 0.6:
                    adjusted_threshold *= 1.3  # Much more likely to challenge
                elif opponent['bluff_ratio'] < 0.2:
                    adjusted_threshold *= 0.7  # Less likely to challenge
            
            # Factor 2: Game phase
            if self.game_phase == 'endgame':
                adjusted_threshold *= 1.2  # More aggressive challenging in endgame
            
            # Factor 3: Leading position
            if not self.is_leading:
                adjusted_threshold *= 1.1  # Slightly more likely to challenge when behind
            
            # Factor 4: Bid plausibility relative to total dice
            implausibility = max(0, bid_quantity / total_dice - 0.5) * 2  # 0 to 1 scale
            adjusted_threshold *= (1 + implausibility * 0.5)
            
            # Make challenge decision
            if probability < adjusted_threshold:
                return challenge_actions[0]
        
        # Bidding decision
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # Sophisticated bluffing decision
        should_bluff = random.random() < self.bluff_frequency
        
        # Expert strategy for first bid
        if current_bid is None:
            # Expert opening strategy
            return self._select_opening_bid(analysis, bid_actions, should_bluff)
        
        # Expert strategy for subsequent bids
        return self._select_subsequent_bid(analysis, bid_actions, should_bluff, current_bid)
    
    def _update_game_state(self, observation: Dict[str, Any]):
        """Update the agent's understanding of the game state."""
        current_bid = observation['current_bid']
        history = observation.get('history', [])
        round_num = observation['round_num']
        
        # Track bids in current round
        if current_bid is not None and (not self.current_round_bids or self.current_round_bids[-1] != current_bid):
            self.current_round_bids.append(current_bid)
        
        # Check if a new round started
        if round_num > len(self.round_history) and self.current_round_bids:
            # Store previous round
            last_challenge = None
            for entry in reversed(history):
                if entry['action']['type'] == 'challenge':
                    last_challenge = entry
                    break
            
            # Record round outcome
            if last_challenge:
                round_data = {
                    'bids': self.current_round_bids.copy(),
                    'challenge': {
                        'player': last_challenge['player'],
                        'success': last_challenge['action'].get('success', False)
                    },
                    'final_bid': self.current_round_bids[-1] if self.current_round_bids else None
                }
                self.round_history.append(round_data)
                self.current_round_bids = []
    
    def _update_opponent_models(self, observation: Dict[str, Any]):
        """Update models of opponent tendencies."""
        history = observation.get('history', [])
        dice_counts = observation['dice_counts']
        
        # Initialize models for each player
        for player_id in range(self.num_players):
            if player_id != self.player_id and player_id not in self.opponent_models:
                self.opponent_models[player_id] = {
                    'bluff_ratio': 0.5,  # Initial estimate (0-1)
                    'challenge_frequency': 0.5,  # Initial estimate (0-1)
                    'bid_patterns': {
                        'value_preferences': {},  # Preferred values
                        'quantity_jumps': []  # Size of quantity increases
                    },
                    'last_actions': []  # Recent actions
                }
        
        # Analyze round history for bluffing tendencies
        if self.round_history:
            for player_id in self.opponent_models:
                bluffs = 0
                honest_bids = 0
                challenges = 0
                total_actions = 0
                
                for round_data in self.round_history:
                    final_bid = round_data.get('final_bid')
                    challenge = round_data.get('challenge')
                    
                    if not final_bid or not challenge:
                        continue
                    
                    # Count challenges by this player
                    if challenge['player'] == player_id:
                        challenges += 1
                    
                    # For each bid, analyze patterns
                    for i, bid in enumerate(round_data['bids']):
                        bid_quantity, bid_value = bid
                        
                        # Count times this player was the final bidder
                        if i == len(round_data['bids']) - 1 and player_id == challenge['player'] - 1:
                            # Final bid was challenged
                            if challenge['success']:  # Challenge succeeded, bid was a bluff
                                bluffs += 1
                            else:  # Challenge failed, bid was honest
                                honest_bids += 1
                
                # Update bluff ratio
                total_bids = bluffs + honest_bids
                if total_bids > 0:
                    self.opponent_models[player_id]['bluff_ratio'] = bluffs / total_bids
                
                # Update challenge frequency
                for entry in history:
                    if entry['player'] == player_id:
                        total_actions += 1
                        if entry['action']['type'] == 'challenge':
                            challenges += 1
                
                if total_actions > 0:
                    self.opponent_models[player_id]['challenge_frequency'] = challenges / total_actions
        
        # Analyze bid patterns
        for entry in history:
            player_id = entry['player']
            if player_id != self.player_id and player_id in self.opponent_models:
                action = entry['action']
                
                # Track recent actions
                self.opponent_models[player_id]['last_actions'].append(action)
                if len(self.opponent_models[player_id]['last_actions']) > 10:
                    self.opponent_models[player_id]['last_actions'].pop(0)
                
                # Analyze bidding patterns
                if action['type'] == 'bid':
                    # Value preferences
                    value = action['value']
                    if value not in self.opponent_models[player_id]['bid_patterns']['value_preferences']:
                        self.opponent_models[player_id]['bid_patterns']['value_preferences'][value] = 0
                    self.opponent_models[player_id]['bid_patterns']['value_preferences'][value] += 1
                    
                    # Quantity jumps (if we have previous bids to compare)
                    if self.current_round_bids and len(self.current_round_bids) >= 2:
                        prev_bid = self.current_round_bids[-2] if len(self.current_round_bids) > 1 else None
                        if prev_bid:
                            prev_quantity, prev_value = prev_bid
                            curr_quantity = action['quantity']
                            
                            if action['value'] == prev_value:
                                jump = curr_quantity - prev_quantity
                                self.opponent_models[player_id]['bid_patterns']['quantity_jumps'].append(jump)
                                
                                # Keep only recent jumps
                                if len(self.opponent_models[player_id]['bid_patterns']['quantity_jumps']) > 5:
                                    self.opponent_models[player_id]['bid_patterns']['quantity_jumps'].pop(0)
    
    def _adapt_strategy(self):
        """Adapt strategy based on game state and opponent models."""
        # Base strategy by game phase
        if self.game_phase == 'early':
            self.bluff_frequency = 0.25
            self.challenge_threshold = 0.3
        elif self.game_phase == 'mid':
            self.bluff_frequency = 0.35
            self.challenge_threshold = 0.35
        elif self.game_phase == 'late':
            self.bluff_frequency = 0.4
            self.challenge_threshold = 0.4
        else:  # endgame
            self.bluff_frequency = 0.45
            self.challenge_threshold = 0.45
        
        # Adjust based on position
        if self.is_leading:
            # When leading, more conservative
            self.bluff_frequency *= 0.8
            self.challenge_threshold *= 1.1
        else:
            # When behind, more aggressive
            self.bluff_frequency *= 1.2
            self.challenge_threshold *= 0.9
        
        # Cap values to reasonable ranges
        self.bluff_frequency = max(0.1, min(0.7, self.bluff_frequency))
        self.challenge_threshold = max(0.15, min(0.6, self.challenge_threshold))
    
    def _select_opening_bid(self, analysis, bid_actions, should_bluff):
        """Expert strategy for opening bids."""
        own_value_counts = analysis['own_value_counts']
        total_dice = analysis['total_dice']
        
        # Expert opening bid focuses on strongest values
        if own_value_counts:
            sorted_values = sorted(own_value_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Experts sometimes use psychological tactics
            if should_bluff and len(sorted_values) > 1:
                # Sometimes bid second best to mislead (expert tactic)
                value_idx = random.randint(0, 1)
            else:
                value_idx = 0
            
            best_value, best_count = sorted_values[min(value_idx, len(sorted_values) - 1)]
            
            # Calculate strategic bid quantity
            if should_bluff:
                # Experts make believable bluffs
                expected_additional = (total_dice - len(analysis['own_dice'])) / self.dice_faces
                plausible_max = int(best_count + expected_additional + 0.5)
                bid_quantity = min(plausible_max, best_count + 2)
            else:
                bid_quantity = best_count
            
            # Find closest bid
            matching_bids = [a for a in bid_actions if a['value'] == best_value]
            if matching_bids:
                return min(matching_bids, key=lambda a: abs(a['quantity'] - bid_quantity))
        
        # Fallback to balanced bid
        mid_idx = len(bid_actions) // 2
        return sorted(bid_actions, key=lambda a: a['quantity'])[mid_idx]
    
    def _select_subsequent_bid(self, analysis, bid_actions, should_bluff, current_bid):
        """Expert strategy for subsequent bids."""
        bid_quantity, bid_value = current_bid
        total_dice = analysis['total_dice']
        
        # Expert-level bid selection
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        expected_count = analysis['expected_counts'][bid_value]
        
        # Find our strongest values
        best_values = [(v, c) for v, c in analysis['own_value_counts'].items() if c > 0]
        best_values.sort(key=lambda x: (x[1], x[0]), reverse=True)  # Sort by count then by value
        
        # Expert endgame tactics
        if self.game_phase == 'endgame' and total_dice <= 4:
            # In endgame with few dice, make very accurate bids
            if best_values:
                best_value, best_count = best_values[0]
                
                # Switch to our best value if better
                if best_value > bid_value:
                    for action in bid_actions:
                        if action['value'] == best_value and action['quantity'] == bid_quantity:
                            return action
                
                # Or increase quantity by 1 if we have current value
                if own_count > 0:
                    for action in bid_actions:
                        if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                            return action
        
        # Sophisticated bluffing - with psychology
        if should_bluff:
            if random.random() < 0.7:  # 70% of expert bluffs are subtle
                # Calculate a plausible bluff
                plausible_quantity = min(bid_quantity + 2, int(expected_count * 1.3))
                
                # Look for bids in this plausible range
                for q in range(bid_quantity + 1, plausible_quantity + 1):
                    for action in bid_actions:
                        if action['value'] == bid_value and action['quantity'] == q:
                            return action
            else:  # 30% are bold psychological plays
                # Make a bold but not impossible bid
                max_quantity = min(total_dice, bid_quantity + 3)
                bold_bids = [a for a in bid_actions if a['quantity'] <= max_quantity]
                if bold_bids:
                    # Choose one of the bolder bids
                    bold_bids.sort(key=lambda a: (a['quantity'], a['value']), reverse=True)
                    return bold_bids[0] if bold_bids else min(bid_actions, key=lambda a: (a['quantity'], a['value']))
        
        # Strategic value switching - based on our hand
        if best_values:
            # Try our best value if it's higher
            for value, count in best_values:
                if value > bid_value:
                    for action in bid_actions:
                        if action['value'] == value and action['quantity'] == bid_quantity:
                            return action
            
            # Expert incrementing with current value
            if own_count > 0 or expected_count > bid_quantity:
                for action in bid_actions:
                    if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                        return action
        
        # Strategic value climbing
        for value in range(bid_value + 1, self.dice_faces + 1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If forced, make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))

class OptimalAgent(RuleAgent):
    """
    Game-theoretically optimal agent for Liar's Dice with improved efficiency.
    
    This agent uses probability calculations, targeted Bayesian inference,
    and expected value maximization with caching and approximation techniques
    to maintain strong play while reducing computational overhead.
    """
    
    def __init__(self, dice_faces: int = 6, search_depth: int = 2, 
                 exact_calculation_threshold: int = 12):
        """Initialize the optimized optimal agent."""
        super().__init__(agent_type='optimal', dice_faces=dice_faces)
        self.search_depth = search_depth
        self.exact_calculation_threshold = exact_calculation_threshold
        
        # Belief model parameters
        self.belief_model = None  # Will be initialized once we know dice counts
        self.opponent_dice_probs = {}  # Maps player_id -> {value -> probability}
        self.bid_history = []     # History of bids in current round
        self.round_history = []   # History of completed rounds
        
        # Mixed strategy parameters
        self.randomization_freq = 0.15  # How often to use mixed strategies
        self.risk_tolerance = 0.5  # 0=risk averse, 1=risk seeking
        
        # Value weights for decision utility
        self.weights = {
            'expected_value': 1.0,      # Weight for mathematical EV
            'positional_value': 0.7,    # Weight for game state
            'strategic_deception': 0.5,  # Weight for deception
            'information_gain': 0.3     # Weight for information
        }
        
        # Caching to avoid redundant calculations
        self.cache = {
            'probabilities': {},  # Cache for probability calculations
            'challenge_ev': {},   # Cache for challenge EV
            'bid_ev': {},         # Cache for bid EV
            'bid_utils': {}       # Cache for bid utilities
        }
        
        # Opponent models
        self.opponent_models = {}  # Maps player_id -> model dict
        
        # Current round information
        self.current_round = 0
        self.update_frequency = 2  # Only update beliefs every N bids
        self.bid_counter = 0
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action using optimized game-theoretic reasoning.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            Selected action as a dictionary
        """
        # Clear caches when round changes
        round_num = observation.get('round_num', 0)
        if round_num != self.current_round:
            self._clear_caches()
            self.current_round = round_num
        
        # Analyze game state
        analysis = self.analyze_game_state(observation)
        
        # Only update beliefs periodically or on first turn
        self.bid_counter += 1
        if self.bid_counter % self.update_frequency == 0 or not self.opponent_dice_probs:
            self._update_belief_model(observation, analysis)
        
        # Get current bid and available actions
        current_bid = observation['current_bid']
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        # If we can only challenge, we must do so
        if not bid_actions and challenge_actions:
            return challenge_actions[0]
        
        # Decision whether to challenge or bid
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Check cache for challenge decision
            cache_key = (bid_quantity, bid_value, tuple(observation['dice_counts']))
            if cache_key in self.cache['challenge_ev']:
                ev_challenge = self.cache['challenge_ev'][cache_key]
            else:
                # Calculate probability of bid being valid
                probability = self._calculate_probability(analysis['total_dice'], bid_quantity, 
                                                         analysis['own_value_counts'].get(bid_value, 0), 
                                                         self.dice_faces, analysis, bid_value)
                
                # Calculate expected value of challenging vs making best bid
                ev_challenge = self._calculate_challenge_ev(probability, observation)
                self.cache['challenge_ev'][cache_key] = ev_challenge
            
            # Calculate best bid EV (with caching)
            if 'best_bid_ev' in self.cache:
                ev_best_bid = self.cache['best_bid_ev']
            else:
                ev_best_bid = self._calculate_best_bid_ev(bid_actions, observation, analysis)
                self.cache['best_bid_ev'] = ev_best_bid
            
            # Choose to challenge if it has better expected value
            should_randomize = random.random() < self.randomization_freq
            if ev_challenge > ev_best_bid or (should_randomize and ev_challenge > ev_best_bid * 0.85):
                return challenge_actions[0]
        
        # If we decide to bid, find the optimal bid
        if not bid_actions:
            return challenge_actions[0]  # Must challenge if no valid bids
        
        # Optimal bid selection (with pruning for efficiency)
        if current_bid is None:
            return self._select_opening_bid(analysis, bid_actions, observation)
        else:
            return self._select_subsequent_bid(analysis, bid_actions, observation, current_bid)
    
    def _clear_caches(self):
        """Clear all caches when entering a new round."""
        self.cache = {
            'probabilities': {},
            'challenge_ev': {},
            'bid_ev': {},
            'bid_utils': {}
        }
        self.bid_counter = 0
    
    def _update_belief_model(self, observation: Dict[str, Any], analysis: Dict[str, Any]):
        """
        Update belief model of opponent dice distributions.
        
        This implements Bayesian updating of our beliefs about what dice
        other players might have, based on their betting patterns.
        
        Args:
            observation: Current game observation
            analysis: Results from analyze_game_state
        """
        dice_counts = observation['dice_counts']
        current_bid = observation['current_bid']
        history = observation.get('history', [])
        round_num = observation.get('round_num', 0)
        total_dice = analysis['total_dice']
        
        # Initialize belief model if needed
        if self.belief_model is None or len(self.belief_model) != self.num_players:
            self._initialize_belief_model(dice_counts)
        
        # Initialize opponent dice probabilities
        if not self.opponent_dice_probs:
            self._initialize_opponent_probs(dice_counts)
        
        # Track bids in the current round
        if current_bid is not None and (not self.bid_history or self.bid_history[-1][0] != current_bid):
            # Record who made this bid
            last_bidder = None
            if history:
                for entry in reversed(history):
                    if entry['action']['type'] == 'bid' and entry['action'].get('quantity') == current_bid[0] and entry['action'].get('value') == current_bid[1]:
                        last_bidder = entry['player']
                        break
            
            self.bid_history.append((current_bid, last_bidder))
            
            # Only update beliefs for significant bids to reduce computation
            if len(self.bid_history) % 2 == 0:  # Update every other bid
                self._update_beliefs_from_bids(total_dice)
        
        # Check if a new round started
        if round_num > len(self.round_history) and self.bid_history:
            # Record completed round
            last_challenge = None
            for entry in reversed(history):
                if entry['action']['type'] == 'challenge':
                    last_challenge = entry
                    break
            
            if last_challenge:
                # Extract true dice counts if available in the challenge result
                true_counts = last_challenge['action'].get('true_counts')
                
                round_data = {
                    'bids': self.bid_history.copy(),
                    'challenge': {
                        'player': last_challenge['player'],
                        'success': last_challenge['action'].get('success', False)
                    },
                    'final_bid': self.bid_history[-1][0] if self.bid_history else None,
                    'true_counts': true_counts
                }
                
                self.round_history.append(round_data)
                
                # Use round outcome to calibrate our belief model
                if true_counts:
                    self._calibrate_beliefs(true_counts)
                
                # Reset bid history for new round
                self.bid_history = []
    
    def _initialize_belief_model(self, dice_counts: List[int]):
        """
        Initialize probabilistic belief model of all players' dice.
        
        Args:
            dice_counts: List of dice counts for each player
        """
        self.belief_model = []
        
        for player in range(self.num_players):
            if player == self.player_id:
                # For our own dice, belief is certainty
                self.belief_model.append(None)
            else:
                # For each opponent, initialize uniform belief over possible dice values
                dice_count = dice_counts[player]
                if dice_count > 0:
                    # Probabilities for each die face for each die
                    player_belief = np.ones((dice_count, self.dice_faces)) / self.dice_faces
                    self.belief_model.append(player_belief)
                else:
                    self.belief_model.append(None)
    
    def _initialize_opponent_probs(self, dice_counts: List[int]):
        """
        Initialize simplified belief model of opponent dice.
        
        Args:
            dice_counts: List of dice counts for each player
        """
        for player_id in range(len(dice_counts)):
            if player_id != self.player_id and dice_counts[player_id] > 0:
                # Simple uniform distribution for each value
                self.opponent_dice_probs[player_id] = {
                    value: 1.0 / self.dice_faces 
                    for value in range(1, self.dice_faces + 1)
                }
    
    def _update_beliefs_from_bids(self, total_dice: int):
        """Update beliefs based on observed bids."""
        if not self.bid_history or len(self.bid_history) < 2:
            return
        
        # Focus only on the most recent bid for efficiency
        (bid, bidder) = self.bid_history[-1]
        if bidder is None or bidder == self.player_id:
            return
            
        bid_quantity, bid_value = bid
        
        # Simple Bayesian update: increase probability for bid value
        if bidder in self.opponent_dice_probs:
            # Strength of the update (adjust based on how strong the bid is)
            # The stronger the bid, the less we trust it
            bid_strength = bid_quantity / total_dice
            
            # Weaker update for stronger bids (might be bluffing)
            update_strength = max(0.1, 0.5 - bid_strength * 0.3)
            
            # Update probabilities
            probs = self.opponent_dice_probs[bidder]
            
            # Increase probability for bid value
            for value in range(1, self.dice_faces + 1):
                if value == bid_value:
                    # Increase probability for this value
                    probs[value] = probs[value] * (1 - update_strength) + update_strength * 1.5
                else:
                    # Slightly decrease other values
                    probs[value] = probs[value] * (1 - update_strength) + update_strength * 0.7
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                for value in probs:
                    probs[value] /= total
        
        # Also update the detailed belief model for backward compatibility
        if self.belief_model and bidder < len(self.belief_model) and self.belief_model[bidder] is not None:
            self._update_belief_for_bid(bid, bidder, total_dice)
    
    def _update_belief_for_bid(self, bid: Tuple[int, int], bidder: int, total_dice: int):
        """
        Update detailed beliefs based on an opponent's bid using Bayesian inference.
        
        Args:
            bid: Tuple of (quantity, value)
            bidder: Player ID who made the bid
            total_dice: Total dice in the game
        """
        bid_quantity, bid_value = bid
        
        # Skip if we don't have a belief model for this player
        if bidder >= len(self.belief_model) or self.belief_model[bidder] is None:
            return
        
        # Get bidder's model to infer their strategy
        bidder_model = self._get_opponent_model(bidder)
        bluff_likelihood = bidder_model.get('bluff_likelihood', 0.5)
        
        # Calculate how "strong" the bid is
        bid_strength = bid_quantity / total_dice
        
        # Stronger bids are more likely to be bluffs
        if bid_strength > 0.7:
            bluff_probability = min(0.9, bluff_likelihood * 1.5)
        elif bid_strength > 0.5:
            bluff_probability = bluff_likelihood
        else:
            bluff_probability = max(0.1, bluff_likelihood * 0.7)
        
        # Update belief model using Bayesian inference
        player_belief = self.belief_model[bidder]
        
        # If player is bluffing, their dice distribution doesn't change our beliefs much
        # If player is honest, they're more likely to have the bid value
        
        for die_idx in range(len(player_belief)):
            for face in range(self.dice_faces):
                face_value = face + 1  # Faces are 1-indexed
                
                # Likelihood of this face given the bid
                if face_value == bid_value:
                    # Higher likelihood if honest bid
                    honest_likelihood = 1.5  # More likely to have bid value
                    bluff_likelihood = 0.7   # Less likely if bluffing
                    likelihood = (1 - bluff_probability) * honest_likelihood + bluff_probability * bluff_likelihood
                else:
                    # Lower likelihood for other values
                    honest_likelihood = 0.7  # Less likely to have other values
                    bluff_likelihood = 1.0   # Normal distribution if bluffing
                    likelihood = (1 - bluff_probability) * honest_likelihood + bluff_probability * bluff_likelihood
                
                # Update probability (Bayes' rule)
                player_belief[die_idx, face] *= likelihood
            
            # Normalize to ensure probabilities sum to 1
            if player_belief[die_idx].sum() > 0:
                player_belief[die_idx] /= player_belief[die_idx].sum()
    
    def _calibrate_beliefs(self, true_counts: Dict[int, int]):
        """
        Calibrate our belief model based on revealed true dice counts.
        
        Args:
            true_counts: Dictionary mapping die values to counts
        """
        # Reset the belief models
        dice_counts = [len(self.belief_model[i]) if self.belief_model[i] is not None else 0 
                      for i in range(len(self.belief_model))]
        self._initialize_belief_model(dice_counts)
        
        # Reset opponent dice probabilities
        self.opponent_dice_probs = {}
        for player_id in range(self.num_players):
            if player_id != self.player_id and player_id < len(dice_counts) and dice_counts[player_id] > 0:
                self.opponent_dice_probs[player_id] = {
                    value: 1.0 / self.dice_faces 
                    for value in range(1, self.dice_faces + 1)
                }
    
    def _get_opponent_model(self, player_id: int) -> Dict[str, Any]:
        """
        Get or create a behavioral model for an opponent.
        
        Args:
            player_id: Player ID to get model for
            
        Returns:
            Dictionary with opponent behavioral parameters
        """
        if player_id not in self.opponent_models:
            # Initialize with default values
            self.opponent_models[player_id] = {
                'bluff_likelihood': 0.5,      # Initial estimate of bluffing frequency
                'challenge_threshold': 0.5,   # When they're likely to challenge
                'value_preferences': {},      # Any bias toward specific die values
                'bid_patterns': [],           # Recent bid patterns
                'aggression': 0.5,            # Bidding aggression (0-1)
                'consistency': 0.5,           # How consistent their strategy is (0-1)
                'adaptability': 0.5,          # How quickly they adapt to situations (0-1)
                'skill_estimate': 0.5         # Estimated skill level (0-1)
            }
        
        # If we have enough history, update the model
        if len(self.round_history) > 2:
            self._update_opponent_model(player_id)
        
        return self.opponent_models[player_id]
    
    def _update_opponent_model(self, player_id: int):
        """
        Update behavioral model for an opponent based on game history.
        
        Args:
            player_id: Player ID to update model for
        """
        model = self.opponent_models[player_id]
        
        # Analyze bidding patterns
        bids_by_player = []
        bluffs = 0
        honest_bids = 0
        challenges = 0
        
        for round_data in self.round_history:
            bids = round_data['bids']
            challenge = round_data.get('challenge', {})
            
            # Count challenges by this player
            if challenge.get('player') == player_id:
                challenges += 1
            
            # Analyze bids by this player
            player_bids = [bid for bid, bidder in bids if bidder == player_id]
            bids_by_player.extend(player_bids)
            
            # Check if their final bid was challenged
            if bids and bids[-1][1] == player_id and challenge:
                if challenge.get('success', False):
                    bluffs += 1
                else:
                    honest_bids += 1
        
        # Update bluff likelihood
        total_final_bids = bluffs + honest_bids
        if total_final_bids > 0:
            model['bluff_likelihood'] = 0.3 * model['bluff_likelihood'] + 0.7 * (bluffs / total_final_bids)
        
        # Update bidding aggression based on bid strengths
        if bids_by_player:
            bid_strengths = [bid[0] / (self.num_players * 5) for bid in bids_by_player]  # Normalize by approximate max
            avg_strength = sum(bid_strengths) / len(bid_strengths)
            model['aggression'] = 0.3 * model['aggression'] + 0.7 * avg_strength
        
        # Update value preferences
        value_counts = {}
        for bid in bids_by_player:
            value = bid[1]
            value_counts[value] = value_counts.get(value, 0) + 1
        
        if value_counts:
            total_bids = sum(value_counts.values())
            model['value_preferences'] = {value: count / total_bids for value, count in value_counts.items()}
    
    # Original interface method kept for compatibility
    def _calculate_probability(self, total_dice: int, bid_quantity: int, own_count: int, 
                             dice_faces: int, analysis: Dict[str, Any] = None, bid_value: int = None) -> float:
        """
        Calculate probability of a bid being valid, original interface for compatibility.
        
        Args:
            total_dice: Total number of dice in the game
            bid_quantity: Quantity in the bid
            own_count: Count of this value among our own dice
            dice_faces: Number of faces on each die
            analysis: (Optional) Results from analyze_game_state
            bid_value: (Optional) Value in the bid
            
        Returns:
            Probability that the bid is valid (0-1)
        """
        # If we have analysis and bid_value, use the optimized method
        if analysis is not None and bid_value is not None:
            return self._calculate_optimized_probability(bid_quantity, bid_value, analysis)
        
        # Otherwise, use the original algorithm
        # If we already have enough dice, probability is 1
        if own_count >= bid_quantity:
            return 1.0
        
        # Number of unknown dice and additional successes needed
        unknown_dice = total_dice - (analysis['own_dice'] if analysis else 0)
        needed = bid_quantity - own_count
        
        # If we need more successes than there are unknown dice, probability is 0
        if needed > unknown_dice:
            return 0.0
        
        # Fallback to simpler binomial probability
        probability = 0.0
        p = 1 / dice_faces  # Probability of rolling the bid value
        
        for k in range(needed, unknown_dice + 1):
            binomial_coef = comb(unknown_dice, k)
            probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
        
        return probability
    
    def _calculate_optimized_probability(self, bid_quantity: int, bid_value: int, analysis: Dict[str, Any]) -> float:
        """
        Calculate exact probability of a bid being valid with optimizations.
        
        Args:
            bid_quantity: Quantity in the bid
            bid_value: Value in the bid
            analysis: Results from analyze_game_state
            
        Returns:
            Probability that the bid is valid (0-1)
        """
        # Check cache first
        cache_key = (bid_quantity, bid_value, tuple(analysis.get('dice_counts', [0])))
        if cache_key in self.cache['probabilities']:
            return self.cache['probabilities'][cache_key]
        
        # What we know for certain (our own dice)
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        
        # If we already have enough dice, probability is 1
        if own_count >= bid_quantity:
            self.cache['probabilities'][cache_key] = 1.0
            return 1.0
        
        # Number of unknown dice and additional successes needed
        total_dice = analysis['total_dice']
        unknown_dice = total_dice - len(analysis['own_dice'])
        needed = bid_quantity - own_count
        
        # If we need more successes than there are unknown dice, probability is 0
        if needed > unknown_dice:
            self.cache['probabilities'][cache_key] = 0.0
            return 0.0
        
        # Use belief model if available
        if self.opponent_dice_probs:
            # Use faster binomial approximation with adjusted probabilities
            p_success = self._calculate_binomial_probability(unknown_dice, needed, bid_value)
            self.cache['probabilities'][cache_key] = p_success
            return p_success
        
        # Fallback to original method if no optimized belief model
        if self.belief_model and all(self.belief_model[i] is not None for i in range(self.num_players) if i != self.player_id):
            # Use original belief model
            p_success = self._calculate_probability_from_belief_model(bid_value, needed, analysis, bid_quantity)
            self.cache['probabilities'][cache_key] = p_success
            return p_success
        else:
            # Use simple binomial as last resort
            probability = 0.0
            p = 1 / self.dice_faces
            
            for k in range(needed, unknown_dice + 1):
                binomial_coef = comb(unknown_dice, k)
                probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
            
            self.cache['probabilities'][cache_key] = probability
            return probability
    
    def _calculate_binomial_probability(self, n_dice: int, needed: int, value: int) -> float:
        """
        Calculate binomial probability with optimizations.
        
        Args:
            n_dice: Number of unknown dice
            needed: Number of successes needed
            value: Die value we're looking for
            
        Returns:
            Probability of at least 'needed' successes
        """
        # Base probability of rolling the value
        p_base = 1.0 / self.dice_faces
        
        # Adjust based on our belief model if available
        if self.opponent_dice_probs:
            # Use average probability across opponents for efficiency
            belief_probs = []
            for player_id, probs in self.opponent_dice_probs.items():
                if value in probs:
                    belief_probs.append(probs[value])
            
            if belief_probs:
                p_base = sum(belief_probs) / len(belief_probs)
        
        # Fast path: if probability is very high or very low
        if needed <= 0:
            return 1.0
        if needed > n_dice:
            return 0.0
        if p_base >= 0.99:
            return 1.0 if needed <= n_dice else 0.0
        if p_base <= 0.01:
            return 0.0 if needed > 0 else 1.0
        
        # Optimize for common cases
        if needed == 1:
            return 1.0 - (1.0 - p_base) ** n_dice
        
        # Binomial probability computation (getting at least 'needed' successes)
        probability = 0.0
        
        # Optimize for larger numbers using normal approximation
        if n_dice >= 20:
            # Normal approximation to binomial
            import math
            
            # Mean and standard deviation of binomial
            mean = n_dice * p_base
            std_dev = math.sqrt(n_dice * p_base * (1 - p_base))
            
            # Z-score for the needed-0.5 (continuity correction)
            z = (needed - 0.5 - mean) / std_dev
            
            # Use simplified approximation of complementary error function
            def approx_erfc(x):
                if x < 0:
                    return 2.0 - approx_erfc(-x)
                a = 1.0 / (1.0 + 0.47047 * x)
                return a * math.exp(-x*x) * (0.3480242 + a * (0.0958798 + a * 0.7478556))
            
            # Probability from normal CDF
            probability = 1.0 - 0.5 * approx_erfc(z / math.sqrt(2))
            
        else:
            # For smaller numbers, use direct computation
            for k in range(needed, n_dice + 1):
                binomial_coef = comb(n_dice, k)
                probability += binomial_coef * (p_base ** k) * ((1 - p_base) ** (n_dice - k))
        
        return probability
    
    # Keep original method for compatibility
    def _calculate_probability_from_belief_model(self, bid_value: int, needed: int, 
                                               analysis: Dict[str, Any] = None, 
                                               bid_quantity: int = None) -> float:
        """
        Calculate probability using our Bayesian belief model.
        
        Args:
            bid_value: The die value we're calculating for
            needed: Number of additional dice with this value needed
            analysis: (Optional) Results from analyze_game_state
            bid_quantity: (Optional) Quantity in the bid
            
        Returns:
            Probability that at least 'needed' unknown dice have the bid value
        """
        # Convert to 0-indexed for numpy arrays
        face_idx = bid_value - 1
        
        # Calculate probability distribution for each player's dice
        player_distributions = []
        
        for player_id in range(self.num_players):
            if player_id == self.player_id or self.belief_model[player_id] is None:
                continue
            
            # Get this player's belief model
            player_belief = self.belief_model[player_id]
            
            # Calculate probability distribution of the number of dice with bid_value
            num_dice = len(player_belief)
            
            # Extract probabilities for the specific face from each die
            face_probs = player_belief[:, face_idx]
            
            # Calculate the probability distribution using convolution
            # This is a key operation - it computes the probability distribution
            # for the number of successes across all of this player's dice
            dist = np.zeros(num_dice + 1)
            dist[0] = 1.0  # Start with 0 successes
            
            for die_idx in range(num_dice):
                p = face_probs[die_idx]
                new_dist = np.zeros(num_dice + 1)
                
                # Convolve the distributions
                for i in range(len(dist)):
                    # Probability of no success from this die
                    new_dist[i] += dist[i] * (1 - p)
                    
                    # Probability of success from this die
                    if i > 0:
                        new_dist[i] += dist[i-1] * p
                
                dist = new_dist
            
            player_distributions.append(dist)
        
        # If we don't have any player distributions, use simpler method
        if not player_distributions:
            if analysis is not None and bid_quantity is not None:
                # Use the optimized calculation method if we have analysis and bid_quantity
                total_dice = analysis['total_dice']
                own_count = analysis['own_value_counts'].get(bid_value, 0)
                return self._calculate_probability(total_dice, bid_quantity, own_count, self.dice_faces)
            else:
                # Fall back to a basic probability estimate
                unknown_dice = 10  # Default assumption
                if analysis:
                    unknown_dice = analysis['total_dice'] - len(analysis['own_dice'])
                p = 1 / self.dice_faces
                probability = 0.0
                for k in range(needed, unknown_dice + 1):
                    binomial_coef = comb(unknown_dice, k)
                    probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
                return probability
        
        # Convolve all player distributions to get the overall distribution
        combined_dist = player_distributions[0]
        for dist in player_distributions[1:]:
            combined_dist = np.convolve(combined_dist, dist)
        
        # Calculate probability that we have at least 'needed' successes
        probability = combined_dist[needed:].sum()
        
        return probability
    
    def _calculate_challenge_ev(self, probability: float, observation: Dict[str, Any]) -> float:
        """
        Calculate expected value of challenging.
        
        Args:
            probability: Probability the current bid is valid
            observation: Current game observation
            
        Returns:
            Expected value of challenging
        """
        # Simple EV calculation
        p_invalid = 1 - probability
        
        # Expected value is net gain/loss
        basic_ev = p_invalid - probability
        
        # Consider positional factors
        dice_counts = observation['dice_counts']
        our_dice = dice_counts[self.player_id]
        
        # Find who made the current bid
        last_bidder = None
        if self.bid_history and len(self.bid_history) > 0:
            _, last_bidder = self.bid_history[-1]
        else:
            # Try to find it in the history
            history = observation.get('history', [])
            current_bid = observation['current_bid']
            
            if history and current_bid:
                for entry in reversed(history):
                    if entry['action']['type'] == 'bid':
                        last_bidder = entry['player']
                        break
        
        # Adjust for critical situations
        if last_bidder is not None and last_bidder < len(dice_counts):
            opponent_dice = dice_counts[last_bidder]
            
            # Critical adjustments
            if our_dice == 1:
                basic_ev -= 1.0  # Survival priority
            if opponent_dice == 1:
                basic_ev += 0.8  # Elimination opportunity
            if our_dice < opponent_dice:
                basic_ev += 0.3  # Catching up bonus
        
        return basic_ev
    
    def _calculate_best_bid_ev(self, bid_actions: List[Dict[str, Any]], observation: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """
        Calculate EV of best bid we could make (optimized).
        
        Args:
            bid_actions: List of valid bid actions
            observation: Current game observation
            analysis: Results from analyze_game_state
            
        Returns:
            Expected value of the best bid
        """
        best_ev = -float('inf')
        
        # Only evaluate a subset of actions for efficiency
        if len(bid_actions) > 10:
            # Evaluate a diverse sample of bid actions
            sampled_actions = self._sample_diverse_bids(bid_actions, 10)
        else:
            sampled_actions = bid_actions
        
        for action in sampled_actions:
            bid_quantity = action['quantity']
            bid_value = action['value']
            
            # Check cache
            cache_key = (bid_quantity, bid_value, observation.get('round_num', 0))
            if cache_key in self.cache['bid_ev']:
                bid_ev = self.cache['bid_ev'][cache_key]
            else:
                # Calculate probability
                probability = self._calculate_optimized_probability(bid_quantity, bid_value, analysis)
                
                # Quick EV calculation
                challenge_likelihood = self._estimate_challenge_likelihood(bid_quantity, bid_value, observation)
                bid_ev = challenge_likelihood * (2 * probability - 1) + (1 - challenge_likelihood) * 0.1
                
                # Add positional and strategic values
                bid_ev += self._calculate_bid_positional_value(bid_quantity, bid_value, observation)
                bid_ev += self._calculate_deception_value(bid_quantity, bid_value, analysis)
                
                # Cache result
                self.cache['bid_ev'][cache_key] = bid_ev
            
            if bid_ev > best_ev:
                best_ev = bid_ev
        
        return best_ev
    
    def _sample_diverse_bids(self, bid_actions: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """
        Sample a diverse subset of bid actions to evaluate.
        
        Args:
            bid_actions: List of bid actions
            sample_size: Number of actions to sample
            
        Returns:
            Sampled actions
        """
        if not bid_actions:
            return []
        
        # Get unique values and quantities
        values = sorted(set(a['value'] for a in bid_actions))
        quantities = sorted(set(a['quantity'] for a in bid_actions))
        
        # Ensure we include range of values and quantities
        samples = []
        
        # Include min, max, and middle values
        for value in [values[0], values[-1]]:
            for quantity in [quantities[0], quantities[-1]]:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == quantity:
                        samples.append(action)
                        break
        
        # Include some random ones for diversity
        remaining = sample_size - len(samples)
        if remaining > 0 and len(bid_actions) > len(samples):
            # Filter out actions already in samples
            remaining_actions = [a for a in bid_actions if a not in samples]
            # Add random selections
            samples.extend(random.sample(remaining_actions, min(remaining, len(remaining_actions))))
        
        return samples
    
    def _estimate_challenge_likelihood(self, bid_quantity: int, bid_value: int, observation: Dict[str, Any]) -> float:
        """
        Estimate likelihood of being challenged (optimized).
        
        Args:
            bid_quantity: Quantity in the bid
            bid_value: Value in the bid
            observation: Current game observation
            
        Returns:
            Probability of being challenged (0-1)
        """
        # Faster approximation based on bid strength
        dice_counts = observation['dice_counts']
        total_dice = sum(dice_counts)
        bid_strength = bid_quantity / total_dice
        
        # Simple mapping from bid strength to challenge likelihood
        if bid_strength > 0.8:
            return 0.8  # Very likely to be challenged
        elif bid_strength > 0.6:
            return 0.5
        elif bid_strength > 0.4:
            return 0.3
        else:
            return 0.1  # Unlikely to be challenged
    
    def _calculate_bid_positional_value(self, bid_quantity: int, bid_value: int, observation: Dict[str, Any]) -> float:
        """
        Calculate positional value of a bid based on game state.
        
        Args:
            bid_quantity: Quantity in the bid
            bid_value: Value in the bid
            observation: Current game observation
            
        Returns:
            Positional value adjustment for the bid
        """
        # Consider game state factors that affect bid value
        dice_counts = observation['dice_counts']
        our_dice = dice_counts[self.player_id]
        
        # Leading vs trailing position
        max_opponent_dice = max([dice_counts[i] for i in range(len(dice_counts)) if i != self.player_id])
        is_leading = our_dice >= max_opponent_dice
        
        # Calculate game phase
        total_dice = sum(dice_counts)
        max_possible_dice = self.num_players * max(dice_counts)
        game_progress = 1 - (total_dice / max_possible_dice)
        
        # Default adjustment
        adjustment = 0.0
        
        # In early game, prefer safer bids
        if game_progress < 0.3:
            # Lower value for risky bids
            if bid_quantity > total_dice * 0.6:
                adjustment -= 0.3
        
        # In late game, more aggressive
        elif game_progress > 0.7:
            # Higher value for bids that might eliminate opponents
            if bid_quantity > total_dice * 0.5:
                adjustment += 0.2
        
        # When leading, play more conservatively
        if is_leading:
            if bid_quantity > total_dice * 0.6:
                adjustment -= 0.2
        # When trailing, more aggressive
        else:
            if bid_quantity > total_dice * 0.5:
                adjustment += 0.2
        
        # If we have only one die left, prioritize survival
        if our_dice == 1:
            adjustment -= bid_quantity / total_dice * 0.5
        
        return adjustment
    
    def _calculate_deception_value(self, bid_quantity: int, bid_value: int, analysis: Dict[str, Any]) -> float:
        """
        Calculate strategic deception value of a bid.
        
        Args:
            bid_quantity: Quantity in the bid
            bid_value: Value in the bid
            analysis: Results from analyze_game_state
            
        Returns:
            Deception value of the bid
        """
        # Consider the value of misleading opponents about our dice
        
        own_value_counts = analysis['own_value_counts']
        own_count = own_value_counts.get(bid_value, 0)
        
        # Default deception value
        deception_value = 0.0
        
        # If we have none of this value, it's maximum deception
        if own_count == 0:
            deception_value = 0.3
        # If we have some but are bidding more, moderate deception
        elif bid_quantity > own_count:
            deception_value = 0.2
        # If we're bidding less than we have, we're hiding information
        elif bid_quantity < own_count:
            deception_value = 0.15
        
        # Balance deception with risk
        deception_value *= self.risk_tolerance
        
        return deception_value
    
    def _select_opening_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]], 
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select optimal opening bid (optimized).
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            observation: Current game observation
            
        Returns:
            Selected bid action
        """
        own_value_counts = analysis['own_value_counts']
        
        # Fast path for common cases
        if own_value_counts:
            # Find our strongest value
            best_value, best_count = max(own_value_counts.items(), key=lambda x: x[1])
            
            # Adjust based on risk tolerance
            if random.random() < self.risk_tolerance:
                target_quantity = best_count
            else:
                # Safer bid
                target_quantity = max(1, best_count - 1)
            
            # Find closest matching bid
            for action in bid_actions:
                if action['value'] == best_value and action['quantity'] == target_quantity:
                    return action
            
            # Find close match
            matching_bids = [a for a in bid_actions if a['value'] == best_value]
            if matching_bids:
                return min(matching_bids, key=lambda a: abs(a['quantity'] - target_quantity))
        
        # If no good match, calculate utilities for all
        bid_utilities = []
        
        for action in bid_actions:
            bid_quantity = action['quantity']
            bid_value = action['value']
            
            # Simple utility calculation
            probability = self._calculate_optimized_probability(bid_quantity, bid_value, analysis)
            utility = 2 * probability - 1  # Base utility
            
            # Add minimal adjustments
            own_count = own_value_counts.get(bid_value, 0)
            if own_count > 0:
                utility += 0.2  # Bonus for values we have
            
            bid_utilities.append((action, utility))
        
        # Choose best bid or use mixed strategy
        if random.random() > self.randomization_freq or not bid_utilities:
            if bid_utilities:
                best_bid = max(bid_utilities, key=lambda x: x[1])[0]
                return best_bid
            return random.choice(bid_actions)  # Fallback
        else:
            # Mixed strategy - sort and take top 3
            bid_utilities.sort(key=lambda x: x[1], reverse=True)
            top_bids = [bid for bid, _ in bid_utilities[:min(3, len(bid_utilities))]]
            return random.choice(top_bids)
    
    def _select_subsequent_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]],
                             observation: Dict[str, Any], current_bid: Tuple[int, int]) -> Dict[str, Any]:
        """
        Select optimal subsequent bid (optimized).
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            observation: Current game observation
            current_bid: Current bid as (quantity, value) tuple
            
        Returns:
            Selected bid action
        """
        bid_quantity, bid_value = current_bid
        own_value_counts = analysis['own_value_counts']
        
        # Fast path for common cases
        # 1. Increment current value if we have it
        if own_value_counts.get(bid_value, 0) > 0:
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # 2. Switch to a value we have
        for value, count in sorted(own_value_counts.items(), key=lambda x: (x[1], x[0]), reverse=True):
            if count > 0 and value > bid_value:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # 3. Calculate utilities for all actions
        bid_utilities = []
        
        for action in bid_actions:
            new_quantity = action['quantity']
            new_value = action['value']
            
            # Check cache
            cache_key = (new_quantity, new_value, bid_quantity, bid_value)
            if cache_key in self.cache['bid_utils']:
                utility = self.cache['bid_utils'][cache_key]
            else:
                # Calculate probability
                probability = self._calculate_optimized_probability(new_quantity, new_value, analysis)
                utility = self.weights['expected_value'] * (2 * probability - 1)
                
                # Add positional value
                positional_value = self._calculate_bid_positional_value(new_quantity, new_value, observation)
                utility += self.weights['positional_value'] * positional_value
                
                # Add deception value
                deception_value = self._calculate_deception_value(new_quantity, new_value, analysis)
                utility += self.weights['strategic_deception'] * deception_value
                
                # Cache result
                self.cache['bid_utils'][cache_key] = utility
            
            bid_utilities.append((action, utility))
        
        # Choose best bid or use mixed strategy
        if random.random() > self.randomization_freq or not bid_utilities:
            if bid_utilities:
                best_bid = max(bid_utilities, key=lambda x: x[1])[0]
                return best_bid
            # Fall back to minimal valid bid
            return min(bid_actions, key=lambda a: (a['quantity'], a['value']))
        else:
            # Mixed strategy
            bid_utilities.sort(key=lambda x: x[1], reverse=True)
            top_bids = [bid for bid, _ in bid_utilities[:min(3, len(bid_utilities))]]
            return random.choice(top_bids)
        
class CounterStrategyAgent(RuleAgent):
    """
    Counter-Strategy agent that directly targets and exploits the RL agent's patterns.
    
    This agent:
    - Keeps a detailed history of the RL agent's actions
    - Identifies patterns in bidding and challenging
    - Develops specific counter-strategies to exploit these patterns
    - Adapts more quickly than the adaptive agent, focusing specifically on RL behavior
    """
    
    def __init__(self, dice_faces: int = 6, adaptation_rate: float = 0.7):
        """
        Initialize the counter-strategy agent.
        
        Args:
            dice_faces: Number of faces on each die
            adaptation_rate: How quickly to adapt to identified patterns (0-1)
        """
        super().__init__(agent_type='counter_strategy', dice_faces=dice_faces)
        self.adaptation_rate = adaptation_rate
        
        # Track RL agent's behavior
        self.rl_agent_bids = []  # History of RL agent bids
        self.rl_agent_challenges = []  # When the RL agent challenges
        self.rl_bid_values = {}  # Frequency of bid values by RL agent
        self.rl_bid_quantities = {}  # Frequency of bid quantities by RL agent
        self.rl_challenge_thresholds = []  # Estimated thresholds for RL challenges
        
        # Counter-strategy state
        self.current_round_bids = []
        self.current_counter_strategy = None  # Strategy selected for this game
        self.strategy_success_rates = {
            'value_targeting': 0.5,  # Counter RL's value preferences
            'challenge_baiting': 0.5,  # Bait RL into bad challenges
            'challenge_avoidance': 0.5,  # Avoid RL's good challenges
            'pattern_breaking': 0.5,  # Break expected patterns
        }
        
        # Default strategy parameters
        self.bluff_frequency = 0.3
        self.challenge_threshold = 0.35
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action specifically designed to counter the RL agent's patterns.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            Selected action as a dictionary
        """
        # Update our understanding of the game state
        analysis = self.analyze_game_state(observation)
        self._update_rl_agent_model(observation)
        
        # Extract current state information
        current_bid = observation['current_bid']
        dice_counts = observation['dice_counts']
        
        # Every 2 rounds, select a counter-strategy based on success rates
        round_num = observation['round_num']
        if round_num % 2 == 0 or self.current_counter_strategy is None:
            self._select_counter_strategy()
        
        # Challenge decision
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Execute strategy-specific challenge logic
            if self._should_challenge(current_bid, analysis, observation):
                return challenge_actions[0]
        
        # Bidding decision
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        if not bid_actions:
            return challenge_actions[0]  # Must challenge if no valid bids
        
        # First bid vs. subsequent bid
        if current_bid is None:
            return self._select_initial_bid(analysis, bid_actions, observation)
        else:
            return self._select_counter_bid(analysis, bid_actions, observation, current_bid)
    
    def _update_rl_agent_model(self, observation: Dict[str, Any]):
        """
        Update our model of the RL agent's behavior.
        
        This tracks bid patterns, challenge patterns, and value preferences.
        
        Args:
            observation: Current game observation
        """
        history = observation.get('history', [])
        current_bid = observation['current_bid']
        
        # Track current round bids
        if current_bid is not None and (not self.current_round_bids or self.current_round_bids[-1] != current_bid):
            # Record who made this bid
            last_bidder = None
            if history:
                for entry in reversed(history):
                    if entry['action']['type'] == 'bid' and entry['action'].get('quantity') == current_bid[0] and entry['action'].get('value') == current_bid[1]:
                        last_bidder = entry['player']
                        break
            
            self.current_round_bids.append((current_bid, last_bidder))
            
            # Track RL agent's bid patterns (assuming RL agent is player 0)
            if last_bidder == 0:
                self.rl_agent_bids.append(current_bid)
                
                # Track value and quantity preferences
                bid_quantity, bid_value = current_bid
                
                # Update value frequency
                self.rl_bid_values[bid_value] = self.rl_bid_values.get(bid_value, 0) + 1
                
                # Update quantity frequency (relative to total dice)
                total_dice = sum(observation['dice_counts'])
                quantity_pct = bid_quantity / total_dice
                quantity_bin = round(quantity_pct * 10) / 10  # Round to nearest 0.1
                self.rl_bid_quantities[quantity_bin] = self.rl_bid_quantities.get(quantity_bin, 0) + 1
        
        # Track RL agent's challenge patterns
        for entry in history:
            if entry['action']['type'] == 'challenge' and entry['player'] == 0:
                # Find the bid that was challenged
                if self.current_round_bids:
                    challenged_bid, _ = self.current_round_bids[-1]
                    bid_quantity, bid_value = challenged_bid
                    
                    # Calculate approximate probability at time of challenge
                    total_dice = sum(observation['dice_counts'])
                    probability = min(1.0, bid_quantity / (total_dice * (1/self.dice_faces)))
                    
                    # Track the probability threshold that triggered a challenge
                    self.rl_challenge_thresholds.append(probability)
                    
                    # Track challenge result
                    success = entry['action'].get('success', False)
                    self.rl_agent_challenges.append((challenged_bid, probability, success))
        
        # New round detection
        if observation.get('round_num', 0) > len(self.rl_agent_challenges):
            # Clear current round tracking
            self.current_round_bids = []
    
    def _select_counter_strategy(self):
        """
        Select the most effective counter-strategy based on past performance.
        """
        # Default to random selection for the first few rounds
        if sum(self.strategy_success_rates.values()) == 2.0:  # Initial values sum to 2.0
            strategies = list(self.strategy_success_rates.keys())
            self.current_counter_strategy = random.choice(strategies)
            return
        
        # Select strategy probabilistically based on success rates
        strategies = list(self.strategy_success_rates.keys())
        weights = [self.strategy_success_rates[s] for s in strategies]
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            norm_weights = [w/total for w in weights]
            
            # Select strategy
            self.current_counter_strategy = random.choices(
                strategies, weights=norm_weights, k=1
            )[0]
        else:
            # Fallback to random if all strategies have failed
            self.current_counter_strategy = random.choice(strategies)
    
    def _should_challenge(self, current_bid: Tuple[int, int], analysis: Dict[str, Any], observation: Dict[str, Any]) -> bool:
        """
        Decide whether to challenge based on counter-strategy.
        
        Args:
            current_bid: Current bid as (quantity, value) tuple
            analysis: Results from analyze_game_state
            observation: Current game observation
            
        Returns:
            True if should challenge, False otherwise
        """
        bid_quantity, bid_value = current_bid
        probability = analysis['probabilities'].get(bid_value, 1.0)
        
        # Default challenge threshold
        challenge_threshold = 0.35
        
        # Adjust based on selected counter-strategy
        if self.current_counter_strategy == 'challenge_avoidance':
            # More conservative challenging to avoid RL agent's value traps
            if bid_value in self.rl_bid_values and self.rl_bid_values[bid_value] >= 3:
                # RL agent frequently bids this value - be more conservative
                challenge_threshold = 0.25
        
        elif self.current_counter_strategy == 'value_targeting':
            # Target values RL rarely bids (they might not have them)
            if bid_value not in self.rl_bid_values or self.rl_bid_values[bid_value] <= 1:
                # RL agent rarely bids this value - more likely to challenge
                challenge_threshold = 0.45
        
        # Calculate estimated RL challenge threshold if we have data
        elif self.rl_challenge_thresholds:
            # If we have enough data, estimate RL agent's challenge threshold
            if len(self.rl_challenge_thresholds) >= 3:
                # Use median to be robust to outliers
                rl_threshold = sorted(self.rl_challenge_thresholds)[len(self.rl_challenge_thresholds)//2]
                
                # Adjust our threshold to just below theirs if using challenge_baiting
                if self.current_counter_strategy == 'challenge_baiting':
                    challenge_threshold = max(0.25, rl_threshold - 0.1)  # Bait them into bad challenges
        
        # Make challenge decision based on adjusted threshold
        return probability < challenge_threshold
    
    def _select_initial_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]], observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select initial bid to counter RL agent patterns.
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            observation: Current game observation
            
        Returns:
            Selected bid action
        """
        # Get own dice information
        own_value_counts = analysis['own_value_counts']
        
        # Default strategy is to bid strongest value
        if own_value_counts:
            best_value, best_count = max(own_value_counts.items(), key=lambda x: x[1])
            
            # Modify based on counter-strategy
            if self.current_counter_strategy == 'value_targeting':
                # Avoid RL agent's preferred values
                preferred_values = []
                if self.rl_bid_values:
                    # Get top values by frequency
                    sorted_values = sorted(self.rl_bid_values.items(), key=lambda x: x[1], reverse=True)
                    preferred_values = [v for v, _ in sorted_values[:2]]  # Top 2 preferred values
                
                # Choose a different value if our best one is an RL preference
                if best_value in preferred_values and len(own_value_counts) > 1:
                    # Get second best value
                    second_best = sorted([(v, c) for v, c in own_value_counts.items() if v != best_value], 
                                        key=lambda x: x[1], reverse=True)
                    if second_best:
                        best_value, best_count = second_best[0]
            
            elif self.current_counter_strategy == 'challenge_baiting':
                # Bid slightly higher to bait challenges
                best_count += 1  # Bid one more than we have
            
            elif self.current_counter_strategy == 'pattern_breaking':
                # Occasionally bid a random value to break expectations
                if random.random() < 0.4:  # 40% of the time
                    best_value = random.randint(1, self.dice_faces)
            
            # Find closest bid
            for action in bid_actions:
                if action['value'] == best_value and action['quantity'] == best_count:
                    return action
            
            # Find close match if exact isn't available
            matching_bids = [a for a in bid_actions if a['value'] == best_value]
            if matching_bids:
                return min(matching_bids, key=lambda a: abs(a['quantity'] - best_count))
        
        # Fallback to a balanced middle bid
        sorted_bids = sorted(bid_actions, key=lambda a: a['quantity'])
        middle_idx = len(sorted_bids) // 2
        return sorted_bids[middle_idx]
    
    def _select_counter_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]], 
                          observation: Dict[str, Any], current_bid: Tuple[int, int]) -> Dict[str, Any]:
        """
        Select subsequent bid to counter RL agent patterns.
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            observation: Current game observation
            current_bid: Current bid as (quantity, value) tuple
            
        Returns:
            Selected bid action
        """
        bid_quantity, bid_value = current_bid
        own_value_counts = analysis['own_value_counts']
        
        # Default to standard bid increases
        target_value = bid_value
        target_quantity = bid_quantity + 1
        
        # Adjust based on counter-strategy
        if self.current_counter_strategy == 'value_targeting':
            # Check if RL agent has preferred values
            if self.rl_bid_values:
                # Identify RL agent's least favorite values
                sorted_values = sorted([(v, self.rl_bid_values.get(v, 0)) 
                                      for v in range(1, self.dice_faces + 1)], 
                                     key=lambda x: x[1])
                
                # Try to switch to a value RL agent rarely bids
                rare_values = [v for v, freq in sorted_values[:3] if v > bid_value]
                
                if rare_values and own_value_counts.get(rare_values[0], 0) > 0:
                    # We have some of this value and it's rare for RL agent
                    target_value = rare_values[0]
                    target_quantity = bid_quantity  # Keep same quantity when switching value
        
        elif self.current_counter_strategy == 'challenge_baiting':
            # Make bids that appear risky but are actually safe
            if sum(own_value_counts.get(bid_value, 0) for bid_value in range(1, self.dice_faces + 1)) >= 2:
                # We have at least 2 of the value we're bidding
                target_quantity = bid_quantity + 2  # Make a bigger jump to bait a challenge
        
        elif self.current_counter_strategy == 'challenge_avoidance':
            # Make small, safe bids to avoid challenges
            own_count = own_value_counts.get(bid_value, 0)
            if own_count > 0:
                # We have some of this value
                target_quantity = bid_quantity + 1  # Small increment
            else:
                # Try to switch to a value we actually have
                for value, count in own_value_counts.items():
                    if count > 0 and value > bid_value:
                        target_value = value
                        target_quantity = bid_quantity  # Keep quantity same when switching
                        break
        
        elif self.current_counter_strategy == 'pattern_breaking':
            # Break predictable patterns in bidding
            if self.rl_agent_bids and len(self.rl_agent_bids) >= 2:
                # Check if RL agent makes predictable raises
                raises = []
                for i in range(1, len(self.rl_agent_bids)):
                    prev_q, prev_v = self.rl_agent_bids[i-1]
                    curr_q, curr_v = self.rl_agent_bids[i]
                    
                    if curr_v == prev_v:
                        # Same value, record quantity increase
                        raises.append(curr_q - prev_q)
                
                if raises and all(r == raises[0] for r in raises):
                    # RL agent always raises by the same amount
                    # Make a different raise to break the pattern
                    if raises[0] == 1:
                        target_quantity = bid_quantity + 2  # RL raises by 1, we raise by 2
                    else:
                        target_quantity = bid_quantity + 1  # RL raises by more, we raise by 1
        
        # Find closest available bid to our target
        for action in bid_actions:
            if action['value'] == target_value and action['quantity'] == target_quantity:
                return action
        
        # Find the closest valid bid
        # First by value, then by quantity
        value_matches = [a for a in bid_actions if a['value'] == target_value]
        if value_matches:
            return min(value_matches, key=lambda a: abs(a['quantity'] - target_quantity))
        
        # If no value match, find closest overall
        return min(bid_actions, key=lambda a: (a['value'] - target_value)**2 + (a['quantity'] - target_quantity)**2)
    
    def update_strategy_success(self, won_game: bool):
        """
        Update success rates for the counter-strategy used this game.
        
        Args:
            won_game: Whether this agent won the game
        """
        if self.current_counter_strategy:
            # Update the success rate with adaptation rate
            current_rate = self.strategy_success_rates[self.current_counter_strategy]
            new_rate = current_rate * (1 - self.adaptation_rate) + won_game * self.adaptation_rate
            self.strategy_success_rates[self.current_counter_strategy] = new_rate

class AntiExploitationAgent(RuleAgent):
    """
    Anti-Exploitation agent that prevents the RL agent from learning narrow strategies.
    
    This agent:
    - Uses randomized strategies that change unpredictably
    - Intentionally varies its bidding patterns and value preferences
    - Introduces controlled unpredictability to force robust learning
    - Punishes simple exploitation attempts by detecting and countering them
    
    The goal is to make the RL agent learn generalizable strategies rather than
    specific exploits of the rule-based agents' patterns.
    """
    
    def __init__(self, dice_faces: int = 6, randomization_level: float = 0.6):
        """
        Initialize the anti-exploitation agent.
        
        Args:
            dice_faces: Number of faces on each die
            randomization_level: Level of randomness in strategy (0-1)
        """
        super().__init__(agent_type='anti_exploitation', dice_faces=dice_faces)
        self.randomization_level = randomization_level
        
        # Tracking RL agent's exploitation attempts
        self.rl_agent_exploits = {
            'value_exploit': {v: 0 for v in range(1, dice_faces + 1)},  # Track if RL repeatedly bids certain values
            'quantity_pattern': [],  # Track patterns in quantity increases
            'challenge_pattern': [],  # Track when RL agent challenges 
        }
        
        # Various strategy parameters that change periodically
        self.current_strategy = None
        self.strategy_duration = 0  # Rounds remaining with current strategy
        self.bid_value_preferences = None  # Will be randomized
        self.challenge_threshold = 0.4  # Base threshold
        self.bluff_frequency = 0.3  # Base frequency
        
        # Game state tracking
        self.current_round = 0
        self.round_bids = []
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action that prevents exploitation by the RL agent.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            Selected action as a dictionary
        """
        # Update our model of the game and RL agent's behavior
        analysis = self.analyze_game_state(observation)
        self._track_rl_behavior(observation)
        
        # Check if we need to rotate strategies
        round_num = observation['round_num']
        if round_num != self.current_round:
            self.current_round = round_num
            self.strategy_duration -= 1
            
            # Randomly change strategies periodically
            if self.strategy_duration <= 0:
                self._rotate_strategies()
                
        # Get current game state
        current_bid = observation['current_bid']
        
        # Challenge decision
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            if self._should_challenge(current_bid, analysis, observation):
                return challenge_actions[0]
        
        # Bidding decision
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        if not bid_actions:
            return challenge_actions[0]  # Must challenge if no valid bids
        
        # Initial or subsequent bid
        if current_bid is None:
            return self._select_initial_bid(analysis, bid_actions, observation)
        else:
            return self._select_subsequent_bid(analysis, bid_actions, observation, current_bid)
    
    def _track_rl_behavior(self, observation: Dict[str, Any]):
        """
        Track RL agent's behavior to detect exploitation patterns.
        
        Args:
            observation: Current game observation
        """
        history = observation.get('history', [])
        current_bid = observation['current_bid']
        
        # Track current round bids
        if current_bid is not None and (not self.round_bids or self.round_bids[-1][0] != current_bid):
            # Find who made this bid
            last_bidder = None
            if history:
                for entry in reversed(history):
                    if (entry['action']['type'] == 'bid' and 
                        entry['action'].get('quantity') == current_bid[0] and 
                        entry['action'].get('value') == current_bid[1]):
                        last_bidder = entry['player']
                        break
            
            self.round_bids.append((current_bid, last_bidder))
            
            # Track RL agent's bid patterns (assuming RL agent is player 0)
            if last_bidder == 0:
                bid_quantity, bid_value = current_bid
                
                # Track value exploit attempts
                self.rl_agent_exploits['value_exploit'][bid_value] += 1
                
                # Track quantity patterns
                if len(self.rl_agent_exploits['quantity_pattern']) >= 10:
                    self.rl_agent_exploits['quantity_pattern'].pop(0)
                
                # Only add if previous bid exists and from RL agent
                if len(self.round_bids) >= 2 and self.round_bids[-2][1] == 0:
                    prev_bid, _ = self.round_bids[-2]
                    prev_quantity, prev_value = prev_bid
                    
                    if prev_value == bid_value:
                        # Same value, track quantity increase
                        self.rl_agent_exploits['quantity_pattern'].append(bid_quantity - prev_quantity)
        
        # Track challenge patterns
        for entry in history:
            if entry['action']['type'] == 'challenge' and entry['player'] == 0:
                # Find what was challenged
                if self.round_bids:
                    challenged_bid, _ = self.round_bids[-1]
                    bid_quantity, bid_value = challenged_bid
                    
                    # Calculate probability at time of challenge
                    total_dice = sum(observation['dice_counts'])
                    probability = min(1.0, bid_quantity / (total_dice * (1/self.dice_faces)))
                    
                    # Track challenge
                    if len(self.rl_agent_exploits['challenge_pattern']) >= 10:
                        self.rl_agent_exploits['challenge_pattern'].pop(0)
                    
                    # Store (value, probability) to see if RL agent challenges specific values
                    self.rl_agent_exploits['challenge_pattern'].append((bid_value, probability))
        
        # Handle new round
        if observation.get('round_num', 0) > self.current_round:
            self.round_bids = []
    
    def _rotate_strategies(self):
        """
        Randomly change strategies to prevent exploitation.
        """
        # Set new strategy duration (1-3 rounds)
        self.strategy_duration = random.randint(1, 3)
        
        # Select a new challenge threshold (vary around base)
        base_threshold = 0.4
        self.challenge_threshold = max(0.2, min(0.6, base_threshold + 
                                               (random.random() - 0.5) * self.randomization_level))
        
        # Select a new bluff frequency
        base_bluff = 0.3
        self.bluff_frequency = max(0.1, min(0.7, base_bluff + 
                                           (random.random() - 0.5) * self.randomization_level))
        
        # Randomize value preferences
        self.bid_value_preferences = []
        
        # Check if RL agent has a value exploit
        value_exploit = None
        if self.rl_agent_exploits['value_exploit']:
            # Find if any value is disproportionately preferred (3x or more than average)
            avg_freq = sum(self.rl_agent_exploits['value_exploit'].values()) / self.dice_faces
            for value, freq in self.rl_agent_exploits['value_exploit'].items():
                if freq >= 3 * avg_freq and freq >= 3:  # At least 3 occurrences
                    value_exploit = value
                    break
        
        # Counter value exploit by changing preferences
        if value_exploit:
            # Avoid the exploited value 
            self.bid_value_preferences = [v for v in range(1, self.dice_faces + 1) 
                                        if v != value_exploit]
            random.shuffle(self.bid_value_preferences)
        else:
            # Otherwise just use random preferences
            self.bid_value_preferences = list(range(1, self.dice_faces + 1))
            random.shuffle(self.bid_value_preferences)
        
        # Select a strategy approach
        strategies = ['balanced', 'aggressive', 'conservative', 'unpredictable']
        self.current_strategy = random.choice(strategies)
        
        # Strategy-specific adjustments
        if self.current_strategy == 'aggressive':
            self.bluff_frequency += 0.2
            self.challenge_threshold -= 0.1
        elif self.current_strategy == 'conservative':
            self.bluff_frequency -= 0.2
            self.challenge_threshold += 0.1
        elif self.current_strategy == 'unpredictable':
            # Even more randomness in this mode
            self.randomization_level = min(1.0, self.randomization_level * 1.5)
    
    def _should_challenge(self, current_bid: Tuple[int, int], analysis: Dict[str, Any], observation: Dict[str, Any]) -> bool:
        """
        Decide whether to challenge, with anti-exploitation logic.
        
        Args:
            current_bid: Current bid as (quantity, value) tuple
            analysis: Results from analyze_game_state
            observation: Current game observation
            
        Returns:
            True if should challenge, False otherwise
        """
        bid_quantity, bid_value = current_bid
        probability = analysis['probabilities'].get(bid_value, 1.0)
        
        # Base decision on adjusted threshold
        adjusted_threshold = self.challenge_threshold
        
        # Check if RL agent is exploiting specific values
        if self.rl_agent_exploits['value_exploit'].get(bid_value, 0) >= 3:
            # RL agent frequently bids this value, adjust threshold randomly
            if random.random() < 0.7:  # 70% of the time
                # Make it harder to exploit our challenging pattern by randomizing
                adjusted_threshold = max(0.2, min(0.6, adjusted_threshold + 
                                               (random.random() - 0.5) * 0.2))
        
        # Random chance to add unpredictability
        if random.random() < self.randomization_level * 0.3:
            # Occasionally make "wrong" challenge decisions to prevent exploitation
            if probability < adjusted_threshold:
                # Should challenge but randomly don't (20% chance)
                if random.random() < 0.2:
                    return False
            else:
                # Shouldn't challenge but randomly do (10% chance)
                if random.random() < 0.1:
                    return True
        
        # Base challenge decision on probability
        return probability < adjusted_threshold
    
    def _select_initial_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]], observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select initial bid with anti-exploitation measures.
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            observation: Current game observation
            
        Returns:
            Selected bid action
        """
        own_value_counts = analysis['own_value_counts']
        
        # Default to bidding our strongest value
        best_value = None
        best_count = 0
        
        if own_value_counts:
            # Find our best value
            best_value, best_count = max(own_value_counts.items(), key=lambda x: x[1])
        
        # Apply randomization and strategy variation
        should_bluff = random.random() < self.bluff_frequency
        
        # Randomly vary the value based on our preferences (if we have them)
        if self.bid_value_preferences and random.random() < self.randomization_level:
            # Check our dice for preferred values
            available_values = [v for v in self.bid_value_preferences 
                               if v in own_value_counts and own_value_counts[v] > 0]
            
            if available_values:
                # Choose one of our preferred values that we have
                best_value = random.choice(available_values)
                best_count = own_value_counts[best_value]
            elif should_bluff and self.bid_value_preferences:
                # Bluff with a preferred value we don't have
                best_value = self.bid_value_preferences[0]  # Top preference
                best_count = 1  # Start with a minimal bid
        
        # Adjust quantity based on strategy
        if should_bluff:
            if self.current_strategy == 'aggressive':
                best_count += random.randint(1, 2)
            elif self.current_strategy == 'unpredictable':
                best_count = random.randint(max(1, best_count - 1), best_count + 2)
        elif self.current_strategy == 'conservative':
            # Sometimes bid less than we have to hide information
            if random.random() < 0.3 and best_count > 1:
                best_count -= 1
        
        # If we couldn't determine a value, choose randomly
        if best_value is None:
            best_value = random.randint(1, self.dice_faces)
            best_count = 1
        
        # Find a matching or close bid
        matching_bids = [a for a in bid_actions if a['value'] == best_value]
        if matching_bids:
            # Find closest quantity match
            return min(matching_bids, key=lambda a: abs(a['quantity'] - best_count))
        
        # If no match on value, pick a bid with closest value
        return min(bid_actions, key=lambda a: abs(a['value'] - best_value))
    
    def _select_subsequent_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]], 
                             observation: Dict[str, Any], current_bid: Tuple[int, int]) -> Dict[str, Any]:
        """
        Select subsequent bid with anti-exploitation measures.
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            observation: Current game observation
            current_bid: Current bid as (quantity, value) tuple
            
        Returns:
            Selected bid action
        """
        bid_quantity, bid_value = current_bid
        own_value_counts = analysis['own_value_counts']
        
        # Check if RL agent has quantity pattern that we can break
        quantity_pattern = False
        if len(self.rl_agent_exploits['quantity_pattern']) >= 3:
            # Check if last 3 increases are the same
            if (len(set(self.rl_agent_exploits['quantity_pattern'][-3:])) == 1 and
                self.rl_agent_exploits['quantity_pattern'][-1] > 0):
                quantity_pattern = True
                pattern_value = self.rl_agent_exploits['quantity_pattern'][-1]
        
        # Decide on bid approach
        should_bluff = random.random() < self.bluff_frequency
        should_randomize = random.random() < self.randomization_level
        
        # Set default targets
        target_value = bid_value
        target_quantity = bid_quantity + 1  # Default increment
        
        # Strategy-specific adjustments
        if self.current_strategy == 'aggressive':
            # More aggressive quantity jumps
            target_quantity = bid_quantity + random.randint(1, 2)
            
            # Sometimes switch to higher value
            if should_randomize and random.random() < 0.4:
                higher_values = [v for v in range(bid_value + 1, self.dice_faces + 1)]
                if higher_values:
                    target_value = random.choice(higher_values)
                    target_quantity = bid_quantity  # Reset quantity when changing value
        
        elif self.current_strategy == 'conservative':
            # Minimal increases, prefer values we have
            if own_value_counts:
                # Check if we have values higher than current bid
                higher_values = [(v, c) for v, c in own_value_counts.items() 
                                if v > bid_value and c > 0]
                
                if higher_values and random.random() < 0.6:
                    # Switch to a higher value we actually have
                    target_value, _ = random.choice(higher_values)
                    target_quantity = bid_quantity  # Reset quantity when switching values
                elif own_value_counts.get(bid_value, 0) > 0:
                    # We have current value, make small increase
                    target_quantity = bid_quantity + 1
        
        elif self.current_strategy == 'unpredictable' or should_randomize:
            # Break patterns to prevent exploitation
            if quantity_pattern:
                # RL agent uses consistent quantity increases, so use a different one
                pattern_value = self.rl_agent_exploits['quantity_pattern'][-1]
                if pattern_value == 1:
                    target_quantity = bid_quantity + random.randint(2, 3)  # Skip by 2 or 3 instead
                else:
                    target_quantity = bid_quantity + 1  # Use 1 instead of their pattern
            
            # Randomize value sometimes
            if random.random() < self.randomization_level * 0.5:
                # Pick a random valid value that's not the current one
                valid_values = [v for v in range(1, self.dice_faces + 1) if v != bid_value]
                valid_values = [v for v in valid_values if v > bid_value]  # Must be higher
                
                if valid_values:
                    target_value = random.choice(valid_values)
                    target_quantity = bid_quantity  # Reset quantity when changing value
            
            # Occasionally make a large jump to disrupt patterns
            if random.random() < self.randomization_level * 0.2:
                target_quantity = bid_quantity + random.randint(2, 3)
        
        # When bluffing, adjust the target
        if should_bluff:
            # If we don't have this value, be more aggressive
            if own_value_counts.get(target_value, 0) == 0:
                # 30% chance to increase quantity more if bluffing
                if random.random() < 0.3:
                    target_quantity += 1
        
        # Find the closest valid bid to our target
        matching_value_bids = [a for a in bid_actions if a['value'] == target_value]
        if matching_value_bids:
            # Find closest quantity match for this value
            return min(matching_value_bids, key=lambda a: abs(a['quantity'] - target_quantity))
        
        # If no match for value, find closest valid bid
        return min(bid_actions, key=lambda a: 
                  (abs(a['value'] - target_value) * 10) + abs(a['quantity'] - target_quantity))
    
class BluffPunisherAgent(RuleAgent):
    """
    BluffCatcher agent that detects and punishes bluffing behavior generally.
    
    This agent:
    - Has a strong bias toward challenging suspicious bids
    - Makes conservative bids based on its own dice
    - Learns and adapts to bluffing patterns for all values
    - Becomes more aggressive in punishing bluffs as the game progresses
    """
    
    def __init__(self, dice_faces: int = 6, base_challenge_threshold: float = 0.45):
        """
        Initialize the BluffCatcher agent.
        
        Args:
            dice_faces: Number of faces on each die
            base_challenge_threshold: Base threshold for challenging (0-1)
        """
        super().__init__(agent_type='bluff_punisher', dice_faces=dice_faces)
        self.base_challenge_threshold = base_challenge_threshold
        
        # Track bidding patterns
        self.bid_history = []
        self.value_frequency = {i: 0 for i in range(1, dice_faces + 1)}
        self.bluff_score = {i: 0.5 for i in range(1, dice_faces + 1)}  # Starting belief about each value
        
        # Track player behavior
        self.player_bluff_tendency = {}  # Maps player_id to bluff tendency
        
        # Game state tracking
        self.round_num = 0
        
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action that detects and punishes bluffing.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
        Returns:
            Selected action as a dictionary
        """
        # Update game state tracking
        if observation.get('round_num', 0) > self.round_num:
            self.round_num = observation.get('round_num', 0)
            self.bid_history = []  # Reset bid history for new round
            
        # Analyze game state
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Challenge decision
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Calculate probability of bid being valid
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Adjust threshold based on game state and bluff history
            adjusted_threshold = self._get_challenge_threshold(bid_value, observation)
            
            # Update bid history and bluff score
            self._update_bid_tracking(current_bid, observation)
            
            # Challenge if probability is below adjusted threshold
            if probability < adjusted_threshold:
                return challenge_actions[0]
        
        # Bidding decision
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        if not bid_actions:
            return challenge_actions[0]  # Must challenge if no valid bids
        
        # First bid vs. subsequent bid
        if current_bid is None:
            return self._select_initial_bid(analysis, bid_actions)
        else:
            return self._select_subsequent_bid(analysis, bid_actions, current_bid)
    
    def _get_challenge_threshold(self, bid_value: int, observation: Dict[str, Any]) -> float:
        """
        Get challenge threshold adjusted for game state and bluff history.
        
        Args:
            bid_value: Value of the current bid
            observation: Current game observation
            
        Returns:
            Adjusted challenge threshold
        """
        # Start with base threshold
        threshold = self.base_challenge_threshold
            
        # Adjust based on game phase
        dice_counts = observation['dice_counts']
        total_dice = sum(dice_counts)
        max_dice = len(dice_counts) * max(dice_counts)
        game_progress = 1 - (total_dice / max_dice)
        
        # More aggressive challenging in late game
        if game_progress > 0.6:
            threshold += 0.1
            
        # More aggressive when opponent is down to last die
        if min(dice_counts) == 1 and dice_counts[self.player_id] > 1:
            threshold += 0.15
        
        # Adjust based on how commonly this value is bid
        value_count = self.value_frequency.get(bid_value, 0)
        total_bids = sum(self.value_frequency.values())
        if total_bids > 0:
            frequency = value_count / total_bids
            # If a value is bid unusually frequently, be more suspicious
            if frequency > 0.25:  # More than 25% of bids are this value
                threshold += 0.1 * min(frequency * 2, 1.0)
            
        # Use bluff score to adjust threshold if we have data
        if self.bluff_score.get(bid_value, 0.5) > 0.6:  # We suspect this value is often bluffed
            threshold += 0.1 * (self.bluff_score[bid_value] - 0.5) * 2  # Scale with confidence
            
        # Adjust based on bid quantity relative to total dice
        bid_quantity, _ = observation['current_bid']
        if bid_quantity > total_dice * 0.5:  # High quantity bid
            threshold += 0.05 * min((bid_quantity / total_dice) * 2, 1.0)
            
        return threshold
    
    def _update_bid_tracking(self, current_bid: Tuple[int, int], observation: Dict[str, Any]):
        """
        Update bid tracking to detect bluffing patterns.
        
        Args:
            current_bid: Current bid as (quantity, value) tuple
            observation: Current game observation
        """
        bid_quantity, bid_value = current_bid
        
        # Add bid to history
        self.bid_history.append(current_bid)
        
        # Update value frequency
        self.value_frequency[bid_value] = self.value_frequency.get(bid_value, 0) + 1
        
        # Identify last bidder
        last_bidder = None
        history = observation.get('history', [])
        if history:
            for entry in reversed(history):
                if entry['action']['type'] == 'bid':
                    last_bidder = entry['player']
                    break
        
        # Look for challenge results in history to update bluff scores
        if history:
            for entry in reversed(history):
                if entry['action']['type'] == 'challenge':
                    # Find the challenged bid
                    if self.bid_history:
                        challenged_value = self.bid_history[-1][1]
                        challenge_success = entry['action'].get('success', False)
                        
                        # Update bluff score for this value
                        old_score = self.bluff_score.get(challenged_value, 0.5)
                        if challenge_success:  # It was a bluff
                            # Increase bluff score (more likely to be bluffed)
                            self.bluff_score[challenged_value] = min(1.0, old_score + 0.15)
                        else:  # It was not a bluff
                            # Decrease bluff score (less likely to be bluffed)
                            self.bluff_score[challenged_value] = max(0.0, old_score - 0.1)
                        
                        # Update player bluff tendency
                        if last_bidder is not None:
                            if last_bidder not in self.player_bluff_tendency:
                                self.player_bluff_tendency[last_bidder] = 0.5
                            
                            if challenge_success:  # Player was bluffing
                                self.player_bluff_tendency[last_bidder] = min(
                                    1.0, self.player_bluff_tendency[last_bidder] + 0.1)
                            else:  # Player was honest
                                self.player_bluff_tendency[last_bidder] = max(
                                    0.0, self.player_bluff_tendency[last_bidder] - 0.05)
                    break  # Only look at most recent challenge
    
    def _select_initial_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select initial bid - conservative approach.
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            
        Returns:
            Selected bid action
        """
        own_value_counts = analysis['own_value_counts']
        
        # Default to bidding our strongest value
        if own_value_counts:
            # Always bid what we have - focus on punishing bluffs, not making them
            best_value, best_count = max(own_value_counts.items(), key=lambda x: x[1])
            
            # Occasionally bid below what we have to be extra safe
            if random.random() < 0.3 and best_count > 1:
                best_count -= 1
                
            # Find closest matching bid
            for action in bid_actions:
                if action['value'] == best_value and action['quantity'] == best_count:
                    return action
                    
            # Find close match
            matching_bids = [a for a in bid_actions if a['value'] == best_value]
            if matching_bids:
                return min(matching_bids, key=lambda a: abs(a['quantity'] - best_count))
        
        # Fallback to a modest bid
        sorted_bids = sorted(bid_actions, key=lambda a: a['quantity'])
        idx = min(len(sorted_bids) // 3, len(sorted_bids) - 1)  # Low 1/3 of bids
        return sorted_bids[idx]
    
    def _select_subsequent_bid(self, analysis: Dict[str, Any], bid_actions: List[Dict[str, Any]], 
                             current_bid: Tuple[int, int]) -> Dict[str, Any]:
        """
        Select subsequent bid - focus on safe increases.
        
        Args:
            analysis: Results from analyze_game_state
            bid_actions: List of valid bid actions
            current_bid: Current bid as (quantity, value) tuple
            
        Returns:
            Selected bid action
        """
        bid_quantity, bid_value = current_bid
        own_value_counts = analysis['own_value_counts']
        
        # Default to minimum increase
        target_value = bid_value
        target_quantity = bid_quantity + 1
        
        # Check if current value has high bluff score
        if self.bluff_score.get(bid_value, 0.5) > 0.6:
            # Try to switch to a value we actually have
            for value, count in sorted(own_value_counts.items(), key=lambda x: (x[1], x[0]), reverse=True):
                if count > 0 and value > bid_value:
                    valid_actions = [a for a in bid_actions if a['value'] == value]
                    if valid_actions:
                        return min(valid_actions, key=lambda a: a['quantity'])
        
        # If we have the current value, make a small increase
        if own_value_counts.get(bid_value, 0) > 0:
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Switch to a value we have, if possible
        for value, count in sorted(own_value_counts.items(), key=lambda x: (x[1], x[0]), reverse=True):
            if count > 0 and value > bid_value:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # If forced to increase, make the smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))

# Dictionary of available agents by difficulty level
RULE_AGENTS = {
    'random': RandomAgent,
    'naive': NaiveAgent,
    'anti_exploitation': AntiExploitationAgent,
    'conservative': ConservativeAgent,
    'bluff_punisher': BluffPunisherAgent,
    'aggressive': AggressiveAgent,
    'strategic': StrategicAgent,
    'adaptive': AdaptiveAgent,
    'counter_strategy': CounterStrategyAgent,
    'optimal': OptimalAgent
}

# Ordered list of agent difficulties for curriculum learning
CURRICULUM_LEVELS = [
    'random',       # Level 0: Mostly random with slight preference for own dice
    'naive',        # Level 1: Beginner human focusing on own dice
    'conservative', # Level 2: Cautious human with minimal risk-taking
    'bluff_punisher', # Level 3: Bluff-punishing human that challenges suspicious bids
    'anti_exploitation', # Level 4: Anti-exploitation human that adapts to RL's behavior
    'aggressive',   # Level 5: Bold human with frequent bluffing
    'strategic',    # Level 6: Strategic human that adapts to game state
    'adaptive',      # Level 7: Expert human with advanced tactics
    'counter_strategy', # Level 8: Counter-strategy human that adapts to RL's behavior
    'optimal'       # Level 9: Optimal human that uses optimal strategy
]


def create_agent(agent_type: str, dice_faces: int = 6, **kwargs) -> RuleAgent:
    """
    Factory function to create a rule-based agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        dice_faces: Number of faces on each die
        **kwargs: Additional parameters for the agent
        
    Returns:
        Instantiated agent of the requested type
    """
    if agent_type not in RULE_AGENTS:
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(RULE_AGENTS.keys())}")
    
    # Create and return the requested agent
    return RULE_AGENTS[agent_type](dice_faces=dice_faces, **kwargs)


def create_curriculum_agent(level: int, dice_faces: int = 6, **kwargs) -> RuleAgent:
    """
    Create an agent for the specified curriculum level.
    
    Args:
        level: Curriculum level (0-5)
        dice_faces: Number of faces on each die
        **kwargs: Additional parameters for the agent
        
    Returns:
        Instantiated agent for the specified level
    """
    if level < 0 or level >= len(CURRICULUM_LEVELS):
        raise ValueError(f"Invalid curriculum level: {level}. Must be between 0 and {len(CURRICULUM_LEVELS) - 1}")
    
    agent_type = CURRICULUM_LEVELS[level]
    return create_agent(agent_type, dice_faces, **kwargs)