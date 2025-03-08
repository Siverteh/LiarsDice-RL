"""
Specialized evaluation agents for Liar's Dice.

These agents are specifically designed to test different aspects of Liar's Dice
strategy and provide consistent benchmarks for evaluating self-play trained agents.
"""

import random
import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

from agents.base_agent import RLAgent
from environment.game import LiarsDiceGame


class EvaluationAgent(RLAgent):
    """Base class for all evaluation agents."""
    
    def __init__(self, agent_type: str = "evaluation", dice_faces: int = 6):
        """Initialize the evaluation agent."""
        super(EvaluationAgent, self).__init__()
        self.agent_type = agent_type
        self.player_id = None
        self.num_players = None
        self.dice_faces = dice_faces
        self.round_history = []
        
    def set_player_id(self, player_id: int, num_players: int):
        """Set the player ID and number of players."""
        self.player_id = player_id
        self.num_players = num_players
    
    def analyze_state(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the game state to extract useful information."""
        dice = observation['dice']
        dice_counts = observation['dice_counts']
        current_bid = observation['current_bid']
        history = observation.get('history', [])
        
        # Count our dice
        own_dice = dice[self.player_id]
        own_dice_values = [d for d in own_dice if d > 0]
        own_dice_count = len(own_dice_values)
        
        # Calculate total dice in game
        total_dice = sum(dice_counts)
        
        # Count occurrences of each value in our dice
        own_value_counts = {}
        for value in range(1, self.dice_faces + 1):
            own_value_counts[value] = own_dice_values.count(value)
        
        # Calculate expected counts for each value
        expected_counts = {}
        for value in range(1, self.dice_faces + 1):
            # Known dice (ours)
            known = own_value_counts.get(value, 0)
            # Unknown dice
            unknown = total_dice - own_dice_count
            # Expected additional dice with this value (assuming uniform distribution)
            expected_additional = unknown * (1 / self.dice_faces)
            # Total expected
            expected_counts[value] = known + expected_additional
        
        # Calculate probabilities for exceeding current bid
        probabilities = {}
        if current_bid:
            bid_quantity, bid_value = current_bid
            for value in range(1, self.dice_faces + 1):
                if value == bid_value:
                    # Probability this many dice have this value
                    own_count = own_value_counts.get(value, 0)
                    probabilities[value] = self._calculate_probability(
                        total_dice, bid_quantity, own_count, self.dice_faces
                    )
        
        # Track history for this round
        if history and (not self.round_history or history[-1] not in self.round_history):
            self.round_history.append(history[-1])
        
        # Did we move to a new round?
        round_num = observation.get('round_num', 0)
        if round_num > len(self.round_history):
            self.round_history = []
        
        # Get player positions
        player_positions = {}
        for i, count in enumerate(dice_counts):
            player_positions[i] = count
        
        # Is this our first action in the round?
        is_first_action = current_bid is None
        
        # Get last bidder if available
        last_bidder = None
        if history and current_bid:
            for entry in reversed(history):
                if entry['action']['type'] == 'bid' and tuple(entry['action'].values())[1:] == current_bid:
                    last_bidder = entry['player']
                    break
        
        return {
            'own_dice': own_dice_values,
            'own_value_counts': own_value_counts,
            'total_dice': total_dice,
            'expected_counts': expected_counts,
            'probabilities': probabilities,
            'player_positions': player_positions,
            'is_first_action': is_first_action,
            'last_bidder': last_bidder,
            'round_num': round_num
        }
    
    def _calculate_probability(self, total_dice: int, bid_quantity: int, 
                              known_count: int, dice_faces: int) -> float:
        """
        Calculate probability of at least bid_quantity dice showing bid_value.
        
        Args:
            total_dice: Total number of dice in play
            bid_quantity: Quantity stated in the bid
            known_count: Number of matching dice in our hand
            dice_faces: Number of faces on each die
        
        Returns:
            Probability (0-1) that the bid is valid
        """
        # If we already have enough, probability is 1
        if known_count >= bid_quantity:
            return 1.0
        
        # How many more do we need?
        needed = bid_quantity - known_count
        
        # How many unknown dice?
        unknown_dice = total_dice - known_count
        
        # If we need more than exist, probability is 0
        if needed > unknown_dice:
            return 0.0
        
        # Probability of success for each unknown die
        p = 1 / dice_faces
        
        # Calculate probability using cumulative binomial distribution
        probability = 0.0
        for k in range(needed, unknown_dice + 1):
            # Binomial probability: C(n,k) * p^k * (1-p)^(n-k)
            binomial_coef = math.comb(unknown_dice, k)
            probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
        
        return probability
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Base implementation - random action."""
        return random.choice(valid_actions)
    
    # Implement required abstract methods for RLAgent base class
    def update(self, *args, **kwargs) -> float:
        """Rule agents don't learn, so no loss."""
        return 0.0
        
    def add_experience(self, *args, **kwargs):
        """Rule agents don't store experiences."""
        pass
        
    def save(self, path: str):
        """Save agent to disk."""
        pass
            
    def load(self, path: str):
        """Load agent from disk."""
        pass
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


class BeginnerAgent(EvaluationAgent):
    """
    A beginner-level agent that follows simple, predictable patterns.
    
    This agent:
    - Primarily focuses on its own dice
    - Makes straightforward, easily predictable bids
    - Challenges only when the bid seems very unlikely
    - Makes mistakes at a consistent rate
    
    Great for evaluating an agent's ability to exploit simple patterns
    and capitalize on predictable mistakes.
    """
    
    def __init__(self, dice_faces: int = 6, mistake_probability: float = 0.15):
        """
        Initialize the beginner agent.
        
        Args:
            dice_faces: Number of faces on each die
            mistake_probability: Probability of making a random mistake
        """
        super().__init__(agent_type="beginner", dice_faces=dice_faces)
        self.mistake_probability = mistake_probability
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select an action with beginner strategy."""
        # Occasional mistake - completely random action
        if random.random() < self.mistake_probability:
            return random.choice(valid_actions)
        
        # Analyze game state
        analysis = self.analyze_state(observation)
        current_bid = observation['current_bid']
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        
        # Challenge if the current bid requires too many dice
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            own_count = analysis['own_value_counts'].get(bid_value, 0)
            total_dice = analysis['total_dice']
            
            # Simplistic challenging logic - challenge if:
            # 1. I have less than 1/3 of the required dice AND
            # 2. The bid requires more than half the total dice
            if own_count < bid_quantity / 3 and bid_quantity > total_dice * 0.5:
                return challenge_actions[0]
            
            # Also challenge if the bid is impossibly high
            if bid_quantity > total_dice * 0.8 and own_count < bid_quantity * 0.2:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        if not bid_actions:
            # Must challenge if no valid bids
            return challenge_actions[0]
        
        # First bid strategy
        if current_bid is None:
            # Beginners just bid what they have the most of
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                # Find our best value
                best_value = max(own_value_counts, key=own_value_counts.get)
                best_count = own_value_counts[best_value]
                
                # Bid exactly what we have
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == best_count:
                        return action
                
                # If exact bid not available, find closest
                for action in bid_actions:
                    if action['value'] == best_value:
                        return action
            
            # Fallback to random bid
            return random.choice(bid_actions)
        
        # Subsequent bid strategy
        bid_quantity, bid_value = current_bid
        
        # Simple increment of quantity when we have that value
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        if own_count > 0:
            # Try to increase quantity by 1
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Or switch to a higher value we have
        for value in range(bid_value + 1, self.dice_faces + 1):
            if analysis['own_value_counts'].get(value, 0) > 0:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # If we must increase quantity without having that value, make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class ConservativeEvaluator(EvaluationAgent):
    """
    A conservative agent that plays extremely safely and strictly by probabilities.
    
    This agent:
    - Bids primarily based on mathematical probability
    - Avoids bluffing almost entirely
    - Has a very low threshold for challenging
    - Prefers low-risk actions consistently
    
    Great for evaluating an agent's ability to identify and exploit
    overly-cautious opponents, and to force more aggressive bidding.
    """
    
    def __init__(self, dice_faces: int = 6, challenge_threshold: float = 0.4):
        """
        Initialize the conservative agent.
        
        Args:
            dice_faces: Number of faces on each die
            challenge_threshold: Probability threshold for challenging (higher = less challenging)
        """
        super().__init__(agent_type="conservative", dice_faces=dice_faces)
        self.challenge_threshold = challenge_threshold
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select an action with conservative strategy."""
        # Analyze game state
        analysis = self.analyze_state(observation)
        current_bid = observation['current_bid']
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Get probability of this bid being valid
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Challenge if probability is below threshold
            if probability < self.challenge_threshold:
                return challenge_actions[0]
            
            # Also challenge if bid quantity is very close to total dice
            total_dice = analysis['total_dice']
            if bid_quantity > total_dice * 0.75:
                own_count = analysis['own_value_counts'].get(bid_value, 0)
                if own_count < bid_quantity * 0.3:  # We don't have many
                    return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        if not bid_actions:
            # Must challenge if no valid bids
            return challenge_actions[0]
        
        # First bid strategy - super conservative
        if current_bid is None:
            # Bid only values we have, and only the exact amount
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                # Find our best value
                best_value = max(own_value_counts, key=own_value_counts.get)
                best_count = own_value_counts[best_value]
                
                # Bid exactly what we have
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == best_count:
                        return action
                
                # If exact bid not available, find closest
                matching_bids = [a for a in bid_actions if a['value'] == best_value]
                if matching_bids:
                    # Conservative - don't exceed what we have
                    valid_bids = [a for a in matching_bids if a['quantity'] <= best_count]
                    if valid_bids:
                        return max(valid_bids, key=lambda a: a['quantity'])
                    return min(matching_bids, key=lambda a: a['quantity'])
            
            # If we don't have any values, bid lowest quantity
            return min(bid_actions, key=lambda a: a['quantity'])
        
        # Subsequent bid strategy - conservative increments
        bid_quantity, bid_value = current_bid
        
        # Only increment quantity if we have this value
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        if own_count > 0:
            # Check if the expected count supports a small increase
            expected = analysis['expected_counts'].get(bid_value, 0)
            if expected >= bid_quantity + 1:
                for action in bid_actions:
                    if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                        return action
        
        # Try to switch to a higher value we have
        for value in range(bid_value + 1, self.dice_faces + 1):
            if analysis['own_value_counts'].get(value, 0) > 0:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # If we must increase quantity, choose safest option
        safe_bids = []
        for action in bid_actions:
            action_value = action['value']
            action_quantity = action['quantity']
            
            # Calculate probability this bid would be valid
            own_count = analysis['own_value_counts'].get(action_value, 0)
            probability = self._calculate_probability(
                analysis['total_dice'], action_quantity, own_count, self.dice_faces
            )
            
            # Only consider bids with >50% probability
            if probability > 0.5:
                safe_bids.append((action, probability))
        
        # Choose the safest bid
        if safe_bids:
            return max(safe_bids, key=lambda x: x[1])[0]
        
        # If no safe bids, make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class AggressiveEvaluator(EvaluationAgent):
    """
    An aggressive agent that bluffs frequently and makes bold bids.
    
    This agent:
    - Makes frequent, substantial bluffs
    - Rarely challenges unless extremely confident
    - Makes large bid jumps
    - Consistently pushes the limits of plausibility
    
    Great for evaluating an agent's ability to detect and punish
    bluffs, and to avoid being intimidated by aggressive bidding.
    """
    
    def __init__(self, dice_faces: int = 6, bluff_frequency: float = 0.6, 
                challenge_threshold: float = 0.25):
        """
        Initialize the aggressive agent.
        
        Args:
            dice_faces: Number of faces on each die
            bluff_frequency: How often to bluff when bidding (0-1)
            challenge_threshold: Probability threshold for challenging (lower = less challenges)
        """
        super().__init__(agent_type="aggressive", dice_faces=dice_faces)
        self.bluff_frequency = bluff_frequency
        self.challenge_threshold = challenge_threshold
        self.consecutive_losses = 0
        self.last_round = 0
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select an action with aggressive strategy."""
        # Analyze game state
        analysis = self.analyze_state(observation)
        current_bid = observation['current_bid']
        round_num = observation.get('round_num', 0)
        
        # Check if a new round has started to track consecutive losses
        if round_num > self.last_round:
            # Check if we lost a die
            own_dice_count = len(analysis['own_dice'])
            if own_dice_count < self.consecutive_losses + 2:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            self.last_round = round_num
        
        # "On tilt" - increase aggression with consecutive losses
        tilt_factor = min(0.3, self.consecutive_losses * 0.1)
        effective_bluff_frequency = min(0.8, self.bluff_frequency + tilt_factor)
        
        # Decide whether to bluff this turn
        should_bluff = random.random() < effective_bluff_frequency
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Get probability of this bid being valid
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Aggressive players rarely challenge, only when very confident
            # Adjust for "on tilt" state - even less challenging when losing
            effective_threshold = max(0.1, self.challenge_threshold - tilt_factor)
            
            if probability < effective_threshold:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        if not bid_actions:
            # Must challenge if no valid bids
            return challenge_actions[0]
        
        # First bid strategy - aggressive
        if current_bid is None:
            if should_bluff:
                # Make a bold opening bid
                high_bids = sorted(bid_actions, key=lambda a: a['quantity'], reverse=True)
                # Choose one of the higher bids, but not the absolute highest
                idx = min(2, len(high_bids) - 1)
                return high_bids[idx]
            
            # If not bluffing, base bid on our best value but be aggressive
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts, key=own_value_counts.get)
                best_count = own_value_counts[best_value]
                
                # Aggressive players often bid more than they have
                target_quantity = best_count + 1
                
                # Find matching or close bid
                matching_bids = [a for a in bid_actions if a['value'] == best_value]
                if matching_bids:
                    closest = min(matching_bids, key=lambda a: abs(a['quantity'] - target_quantity))
                    return closest
            
            # Default to a reasonably high bid
            sorted_bids = sorted(bid_actions, key=lambda a: a['quantity'])
            high_idx = min(len(sorted_bids) - 1, int(len(sorted_bids) * 0.7))
            return sorted_bids[high_idx]
        
        # Subsequent bid strategy - aggressive increases
        bid_quantity, bid_value = current_bid
        
        if should_bluff:
            # Make a big jump in quantity (signature move)
            jump_size = random.randint(2, 3)  # 2-3 unit jump
            target_quantity = bid_quantity + jump_size
            
            # Find valid bids with this jump
            jump_bids = [a for a in bid_actions if a['quantity'] >= target_quantity]
            if jump_bids:
                return min(jump_bids, key=lambda a: a['quantity'])
            
            # If no big jump available, try highest value
            high_value_bids = [a for a in bid_actions if a['value'] > bid_value]
            if high_value_bids:
                return min(high_value_bids, key=lambda a: a['quantity'])
        
        # If not bluffing, still make aggressive but plausible bids
        # Try to increase quantity by 2 first, then 1
        for increase in [2, 1]:
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + increase:
                    return action
        
        # Try higher values with same quantity
        for value in range(self.dice_faces, bid_value, -1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If all else fails, make smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class AdaptiveEvaluator(EvaluationAgent):
    """
    An adaptive expert agent that builds models of opponent behavior.
    
    This agent:
    - Builds detailed models of opponent tendencies
    - Adapts strategy based on game state and opponents
    - Uses sophisticated probability calculations
    - Combines game theory with psychological elements
    
    Great for evaluating an agent's ability to handle opponents
    that adapt to its strategy and exploit its weaknesses.
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the adaptive agent."""
        super().__init__(agent_type="adaptive", dice_faces=dice_faces)
        # Track opponent behaviors
        self.opponent_models = {}  # Models for each opponent
        self.current_round_bids = []  # Bids in current round
        self.round_history = []  # History of completed rounds
        
        # Default strategy parameters
        self.bluff_frequency = 0.3
        self.challenge_threshold = 0.35
        
        # Game phase tracking
        self.game_phase = 'early'  # early, mid, late, endgame
        self.is_leading = True
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select an action with adaptive expert strategy."""
        # Analyze game state
        analysis = self.analyze_state(observation)
        current_bid = observation['current_bid']
        dice_counts = observation['dice_counts']
        round_num = analysis['round_num']
        history = observation.get('history', [])
        
        # Update game state information
        self._update_game_state(observation, history, round_num)
        
        # Update opponent models
        self._update_opponent_models(observation, history)
        
        # Determine if we're leading
        our_dice = dice_counts[self.player_id]
        max_opponent_dice = max([dice_counts[i] for i in range(len(dice_counts)) if i != self.player_id])
        self.is_leading = our_dice >= max_opponent_dice
        
        # Determine game phase
        total_dice = analysis['total_dice']
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
        last_bidder = analysis['last_bidder']
        
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
            return challenge_actions[0]  # Must challenge if no valid bids
        
        # Sophisticated bluffing decision
        should_bluff = random.random() < self.bluff_frequency
        
        # First bid strategy
        if current_bid is None:
            return self._select_opening_bid(analysis, bid_actions, should_bluff)
        
        # Subsequent bid strategy
        return self._select_subsequent_bid(analysis, bid_actions, should_bluff, current_bid)
    
    def _update_game_state(self, observation, history, round_num):
        """Update the agent's understanding of the game state."""
        current_bid = observation['current_bid']
        
        # Track bids in current round
        if current_bid is not None and (not self.current_round_bids or self.current_round_bids[-1][0] != current_bid):
            # Find who made this bid
            last_bidder = None
            if history:
                for entry in reversed(history):
                    if entry['action']['type'] == 'bid' and entry['action'].get('quantity') == current_bid[0] and entry['action'].get('value') == current_bid[1]:
                        last_bidder = entry['player']
                        break
            
            self.current_round_bids.append((current_bid, last_bidder))
            
        # Check if a new round started
        if round_num > len(self.round_history) and self.current_round_bids:
            # Reset bids for new round
            self.current_round_bids = []
    
    def _update_opponent_models(self, observation, history):
        """Update models of opponent tendencies."""
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
        
        # Analyze recent actions
        for entry in history[-10:]:  # Only analyze recent history
            player_id = entry['player']
            if player_id != self.player_id and player_id in self.opponent_models:
                action = entry['action']
                
                # Track recent actions
                self.opponent_models[player_id]['last_actions'].append(action)
                if len(self.opponent_models[player_id]['last_actions']) > 5:
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
                        prev_bid = self.current_round_bids[-2][0] if len(self.current_round_bids) > 1 else None
                        if prev_bid:
                            prev_quantity, prev_value = prev_bid
                            curr_quantity = action['quantity']
                            
                            if action['value'] == prev_value:
                                jump = curr_quantity - prev_quantity
                                self.opponent_models[player_id]['bid_patterns']['quantity_jumps'].append(jump)
                                
                                # Keep only recent jumps
                                if len(self.opponent_models[player_id]['bid_patterns']['quantity_jumps']) > 3:
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
                # Sometimes bid second best to mislead
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
        
        # Sophisticated bluffing with psychology
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
        
        # Strategic value switching based on our hand
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


class OptimalEvaluator(EvaluationAgent):
    """
    A near-optimal agent using game theoretic principles and advanced probability.
    
    This agent:
    - Uses theoretically optimal decisions based on probability
    - Employs precise mathematical analyses
    - Bluffs at optimal frequencies
    - Makes optimal challenges based on EV calculations
    
    Great for evaluating an agent's ability to approach optimal play
    and handle opponents that make few exploitable mistakes.
    """
    
    def __init__(self, dice_faces: int = 6, mixed_strategy_freq: float = 0.15):
        """
        Initialize the optimal agent.
        
        Args:
            dice_faces: Number of faces on each die
            mixed_strategy_freq: How often to use mixed strategy (randomization)
        """
        super().__init__(agent_type="optimal", dice_faces=dice_faces)
        self.mixed_strategy_freq = mixed_strategy_freq
        
        # Probability cache for efficiency
        self.probability_cache = {}
        
        # Game state tracking
        self.belief_model = {}  # Maps player_id -> {value -> probability}
        self.bid_history = []  # Tracks bids in current round
        
        # Value weights for decision utility (used in bidding decisions)
        self.weights = {
            'expected_value': 1.0,      # Weight for mathematical EV
            'positional_value': 0.7,    # Weight for game state
            'deception_value': 0.3      # Weight for deception
        }
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select an action using optimal strategy."""
        # Analyze game state
        analysis = self.analyze_state(observation)
        current_bid = observation['current_bid']
        round_num = analysis['round_num']
        
        # Update belief model
        self._update_belief_model(observation, analysis)
        
        # Get available actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        # If we can only challenge, we must do so
        if not bid_actions and challenge_actions:
            return challenge_actions[0]
        
        # Challenge decision using expected value
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Calculate probability of bid being valid
            probability = self._calculate_enhanced_probability(bid_quantity, bid_value, analysis)
            
            # Calculate EV of challenging
            challenge_ev = self._calculate_challenge_ev(probability, observation)
            
            # Calculate EV of best bid
            bid_ev = self._calculate_best_bid_ev(bid_actions, analysis, observation)
            
            # Optimal decision rule: choose action with highest EV
            # But occasionally use mixed strategy for harder-to-exploit play
            should_randomize = random.random() < self.mixed_strategy_freq
            
            if challenge_ev > bid_ev or (should_randomize and challenge_ev > 0.9 * bid_ev):
                return challenge_actions[0]
        
        # If we decide to bid, find the optimal bid
        if not bid_actions:
            return challenge_actions[0]  # Must challenge if no valid bids
        
        # Optimal bid selection
        if current_bid is None:
            return self._select_optimal_opening_bid(analysis, bid_actions)
        else:
            return self._select_optimal_subsequent_bid(analysis, bid_actions, current_bid, observation)
    
    def _update_belief_model(self, observation, analysis):
        """Update belief model of opponent dice distributions."""
        dice_counts = observation['dice_counts']
        current_bid = observation['current_bid']
        history = observation.get('history', [])
        
        # Initialize belief model if needed
        for player_id in range(self.num_players):
            if player_id != self.player_id and player_id not in self.belief_model:
                self.belief_model[player_id] = {
                    value: 1.0 / self.dice_faces
                    for value in range(1, self.dice_faces + 1)
                }
        
        # Track bids in this round
        if current_bid is not None and (not self.bid_history or self.bid_history[-1][0] != current_bid):
            last_bidder = None
            if history:
                for entry in reversed(history):
                    if entry['action']['type'] == 'bid' and entry['action'].get('quantity') == current_bid[0] and entry['action'].get('value') == current_bid[1]:
                        last_bidder = entry['player']
                        break
            
            self.bid_history.append((current_bid, last_bidder))
            
            # Update belief based on this bid
            if last_bidder is not None and last_bidder != self.player_id and last_bidder in self.belief_model:
                bid_quantity, bid_value = current_bid
                
                # Simple Bayesian update
                # Increase probability for the bid value
                probs = self.belief_model[last_bidder]
                
                # Strength of the update depends on bid plausibility
                bid_strength = bid_quantity / analysis['total_dice']
                update_strength = max(0.1, 0.5 - bid_strength * 0.3)
                
                # Update probabilities
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
    
    def _calculate_enhanced_probability(self, bid_quantity, bid_value, analysis):
        """Calculate probability of a bid being valid, using belief model."""
        # Check cache first
        cache_key = (bid_quantity, bid_value, tuple(analysis.get('own_dice', [])))
        if cache_key in self.probability_cache:
            return self.probability_cache[cache_key]
        
        # What we know for certain (our own dice)
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        
        # If we already have enough dice, probability is 1
        if own_count >= bid_quantity:
            self.probability_cache[cache_key] = 1.0
            return 1.0
        
        # How many more do we need?
        needed = bid_quantity - own_count
        
        # How many unknown dice?
        total_dice = analysis['total_dice']
        unknown_dice = total_dice - len(analysis['own_dice'])
        
        # If we need more than exist, probability is 0
        if needed > unknown_dice:
            self.probability_cache[cache_key] = 0.0
            return 0.0
        
        # Calculate using belief model if available
        if self.belief_model:
            # Use more accurate probability based on belief model
            p_success = self._calculate_belief_based_probability(bid_value, needed, unknown_dice)
            self.probability_cache[cache_key] = p_success
            return p_success
        
        # Fallback to basic probability
        p = 1 / self.dice_faces
        probability = 0.0
        
        for k in range(needed, unknown_dice + 1):
            binomial_coef = math.comb(unknown_dice, k)
            probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
        
        self.probability_cache[cache_key] = probability
        return probability
    
    def _calculate_belief_based_probability(self, value, needed, unknown_dice):
        """Calculate probability using our belief model."""
        # Get average probability for this value across all opponents
        belief_probs = []
        for player_id, probs in self.belief_model.items():
            if value in probs:
                belief_probs.append(probs[value])
        
        if not belief_probs:
            # Default to uniform if no beliefs
            p = 1 / self.dice_faces
        else:
            # Average belief probability
            p = sum(belief_probs) / len(belief_probs)
        
        # Calculate using binomial distribution
        probability = 0.0
        for k in range(needed, unknown_dice + 1):
            binomial_coef = math.comb(unknown_dice, k)
            probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
        
        return probability
    
    def _calculate_challenge_ev(self, probability, observation):
        """Calculate expected value of challenging."""
        # Simple EV calculation based on probability
        p_lose = probability  # Probability we lose the challenge
        p_win = 1 - probability  # Probability we win the challenge
        
        # Base EV is win probability minus lose probability
        ev = p_win - p_lose
        
        # Consider positional factors
        dice_counts = observation['dice_counts']
        our_dice = dice_counts[self.player_id]
        
        # Adjust for critical situations
        for player_id, dice in enumerate(dice_counts):
            if player_id != self.player_id:
                # If opponent has only one die, higher EV for successfully challenging
                if dice == 1:
                    ev += 0.3 * p_win  # Bonus for possibly eliminating opponent
                
                # If we have only one die, lower EV for unsuccessful challenge
                if our_dice == 1:
                    ev -= 0.3 * p_lose  # Penalty for risking elimination
        
        return ev
    
    def _calculate_best_bid_ev(self, bid_actions, analysis, observation):
        """Calculate EV of best bid we could make."""
        best_ev = float('-inf')
        
        # Evaluate up to 10 bids for efficiency
        if len(bid_actions) > 10:
            # Sample diverse bids
            sorted_bids = sorted(bid_actions, key=lambda a: (a['quantity'], a['value']))
            sample_bids = []
            
            # Take bids spread evenly across the range
            for i in range(10):
                idx = i * len(sorted_bids) // 10
                sample_bids.append(sorted_bids[idx])
        else:
            sample_bids = bid_actions
        
        # Evaluate each bid
        for action in sample_bids:
            bid_quantity = action['quantity']
            bid_value = action['value']
            
            # Calculate probability
            probability = self._calculate_enhanced_probability(bid_quantity, bid_value, analysis)
            
            # Calculate challenge likelihood based on bid strength
            bid_strength = bid_quantity / analysis['total_dice']
            challenge_likelihood = self._estimate_challenge_likelihood(bid_strength)
            
            # Calculate expected value
            bid_ev = challenge_likelihood * (2 * probability - 1)  # EV if challenged
            bid_ev += (1 - challenge_likelihood) * 0.1  # Small positive EV if not challenged
            
            # Add positional and deception values
            bid_ev += self._calculate_positional_value(bid_quantity, bid_value, analysis, observation)
            bid_ev += self._calculate_deception_value(bid_quantity, bid_value, analysis)
            
            if bid_ev > best_ev:
                best_ev = bid_ev
        
        return best_ev
    
    def _estimate_challenge_likelihood(self, bid_strength):
        """Estimate likelihood of being challenged based on bid strength."""
        # Simple mapping from bid strength to challenge likelihood
        if bid_strength > 0.8:
            return 0.8  # Very likely to be challenged
        elif bid_strength > 0.6:
            return 0.5
        elif bid_strength > 0.4:
            return 0.3
        else:
            return 0.1  # Unlikely to be challenged
    
    def _calculate_positional_value(self, bid_quantity, bid_value, analysis, observation):
        """Calculate positional value of a bid based on game state."""
        # Consider game state factors
        dice_counts = observation['dice_counts']
        our_dice = dice_counts[self.player_id]
        total_dice = analysis['total_dice']
        
        # Default adjustment
        adjustment = 0.0
        
        # Calculate how conservative/aggressive to be based on position
        max_opponent_dice = max([dice_counts[i] for i in range(len(dice_counts)) if i != self.player_id])
        is_leading = our_dice >= max_opponent_dice
        
        # When leading, play more conservatively
        if is_leading:
            if bid_quantity > total_dice * 0.6:
                adjustment -= 0.2
        # When trailing, play more aggressively
        else:
            if bid_quantity > total_dice * 0.5:
                adjustment += 0.2
        
        # If we have only one die left, prioritize survival
        if our_dice == 1:
            adjustment -= bid_quantity / total_dice * 0.5
        
        return adjustment * self.weights['positional_value']
    
    def _calculate_deception_value(self, bid_quantity, bid_value, analysis):
        """Calculate strategic deception value of a bid."""
        # Consider the value of misleading opponents
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
        
        return deception_value * self.weights['deception_value']
    
    def _select_optimal_opening_bid(self, analysis, bid_actions):
        """Select optimal opening bid."""
        own_value_counts = analysis['own_value_counts']
        total_dice = analysis['total_dice']
        
        # Calculate total dice count
        dice_per_player = len(analysis['own_dice'])
        num_players = analysis['total_dice'] / dice_per_player
        expected_per_value = total_dice / self.dice_faces
        
        # If we have any dice, start with them
        if own_value_counts:
            best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
            own_count = own_value_counts[best_value]
            
            # Calculate optimal opening quantity
            target_quantity = own_count
            
            # Optionally use mixed strategy
            if random.random() < self.mixed_strategy_freq:
                # Mix strategies to avoid being predictable
                # Either more conservative or more aggressive
                if random.random() < 0.5:
                    target_quantity = max(1, own_count - 1)  # More conservative
                else:
                    target_quantity = min(own_count + 1, int(expected_per_value * 1.2))  # More aggressive
            
            # Find matching bid
            for action in bid_actions:
                if action['value'] == best_value and action['quantity'] == target_quantity:
                    return action
            
            # Find close match
            matching_bids = [a for a in bid_actions if a['value'] == best_value]
            if matching_bids:
                return min(matching_bids, key=lambda a: abs(a['quantity'] - target_quantity))
        
        # If no good option based on our dice
        bid_utilities = []
        
        for action in bid_actions:
            bid_quantity = action['quantity']
            bid_value = action['value']
            
            # Calculate probability of this bid being valid
            probability = self._calculate_enhanced_probability(bid_quantity, bid_value, analysis)
            
            # Simple expected value if challenged
            ev = 2 * probability - 1
            
            # Challenge likelihood based on bid strength
            bid_strength = bid_quantity / total_dice
            challenge_likelihood = self._estimate_challenge_likelihood(bid_strength)
            
            # Overall utility
            utility = challenge_likelihood * ev + (1 - challenge_likelihood) * 0.1
            
            # Add bonus for values we have
            own_count = own_value_counts.get(bid_value, 0)
            if own_count > 0:
                utility += 0.2
            
            bid_utilities.append((action, utility))
        
        # Choose best bid or use mixed strategy
        if random.random() > self.mixed_strategy_freq or not bid_utilities:
            if bid_utilities:
                best_bid = max(bid_utilities, key=lambda x: x[1])[0]
                return best_bid
            return random.choice(bid_actions)  # Fallback
        else:
            # Mixed strategy - sort and take top 3
            bid_utilities.sort(key=lambda x: x[1], reverse=True)
            top_bids = [bid for bid, _ in bid_utilities[:min(3, len(bid_utilities))]]
            return random.choice(top_bids)
    
    def _select_optimal_subsequent_bid(self, analysis, bid_actions, current_bid, observation):
        """Select optimal subsequent bid."""
        bid_quantity, bid_value = current_bid
        own_value_counts = analysis['own_value_counts']
        total_dice = analysis['total_dice']
        
        # Calculate bid utilities
        bid_utilities = []
        
        for action in bid_actions:
            new_quantity = action['quantity']
            new_value = action['value']
            
            # Calculate probability of this bid being valid
            probability = self._calculate_enhanced_probability(new_quantity, new_value, analysis)
            
            # Calculate challenge likelihood
            bid_strength = new_quantity / total_dice
            challenge_likelihood = self._estimate_challenge_likelihood(bid_strength)
            
            # Calculate expected value if challenged
            ev_if_challenged = 2 * probability - 1
            
            # Base utility
            utility = challenge_likelihood * ev_if_challenged + (1 - challenge_likelihood) * 0.1
            
            # Add positional value 
            positional_value = self._calculate_positional_value(new_quantity, new_value, analysis, observation)
            utility += positional_value
            
            # Add deception value
            deception_value = self._calculate_deception_value(new_quantity, new_value, analysis)
            utility += deception_value
            
            bid_utilities.append((action, utility))
        
        # Choose bid with highest utility
        if not bid_utilities:
            # Fallback to minimum valid bid
            return min(bid_actions, key=lambda a: (a['quantity'], a['value']))
        
        # Choose best bid or use mixed strategy
        if random.random() > self.mixed_strategy_freq:
            best_bid = max(bid_utilities, key=lambda x: x[1])[0]
            return best_bid
        else:
            # Mixed strategy - sort and take top 3
            bid_utilities.sort(key=lambda x: x[1], reverse=True)
            top_bids = [bid for bid, _ in bid_utilities[:min(3, len(bid_utilities))]]
            return random.choice(top_bids)


def create_evaluation_agent(agent_type: str, dice_faces: int = 6, **kwargs) -> EvaluationAgent:
    """
    Create an evaluation agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        dice_faces: Number of faces on each die
        **kwargs: Additional parameters for the agent
        
    Returns:
        Instantiated agent of the requested type
    """
    agent_map = {
        'beginner': BeginnerAgent,
        'conservative': ConservativeEvaluator,
        'aggressive': AggressiveEvaluator,
        'adaptive': AdaptiveEvaluator,
        'optimal': OptimalEvaluator
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(agent_map.keys())}")
    
    # Create and return the requested agent
    return agent_map[agent_type](dice_faces=dice_faces, **kwargs)

# Dictionary of available agents by difficulty level
RULE_AGENTS = {
    'beginner': BeginnerAgent,
    'conservative': ConservativeEvaluator,
    'aggressive': AggressiveEvaluator,
    'adaptive': AdaptiveEvaluator,
    'optimal': OptimalEvaluator
}


def create_agent(agent_type: str, dice_faces: int = 6, **kwargs) -> EvaluationAgent:
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
