"""
Human-like agents for Liar's Dice.

This module implements various agents with different difficulty levels and strategies,
designed to better mimic human play patterns for more effective learning.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


class RuleAgent:
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
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current game observation
            valid_actions: List of valid actions
            
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
            binomial_coef = np.math.comb(unknown_dice, k)
            probability += binomial_coef * (p ** k) * ((1 - p) ** (unknown_dice - k))
        
        return probability


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
        if random.random() < 0.05:
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
    Conservative agent that mimics a cautious human player.
    
    This agent:
    - Makes "safe" bids based primarily on their own dice
    - Rarely bluffs, and when does, stays close to known values
    - Is quick to challenge when bids seem even slightly unlikely
    - Gets more conservative as they lose dice
    """
    
    def __init__(self, dice_faces: int = 6, challenge_threshold: float = 0.4):
        """
        Initialize the conservative agent.
        
        Args:
            dice_faces: Number of faces on each die
            challenge_threshold: Probability threshold for challenging
        """
        super().__init__(agent_type='conservative', dice_faces=dice_faces)
        self.challenge_threshold = challenge_threshold
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with a cautious human approach.
        
        Strategy:
        - Mostly bid values in own hand
        - Challenge when uncertain (even moderate doubt)
        - Very small bid increases
        - Gets more conservative with fewer dice
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Get our dice count - be more conservative with fewer dice
        dice_counts = observation['dice_counts']
        own_dice_count = dice_counts[self.player_id]
        
        # Adjust challenge threshold based on dice count
        adjusted_threshold = self.challenge_threshold
        if own_dice_count == 1:  # Last die
            adjusted_threshold = self.challenge_threshold * 1.5  # Much more likely to challenge
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Cautious players challenge more easily
            own_count = analysis['own_value_counts'].get(bid_value, 0)
            total_dice = analysis['total_dice']
            
            # Challenge if the probability seems low to them
            probability = analysis['probabilities'].get(bid_value, 1.0)
            if probability < adjusted_threshold:
                return challenge_actions[0]
            
            # Also challenge if bid quantity is close to total dice
            if bid_quantity > total_dice * 0.6:
                # And we don't have many of this value
                if own_count < bid_quantity * 0.3:
                    return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # First bid strategy
        if current_bid is None:
            # Conservative players bid what they're sure of
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                # Bid the most common value in our dice
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Usually bid exactly what we have or less
                bid_quantity = best_count
                
                # Find matching bid or closest under
                matching_bids = [a for a in bid_actions if a['value'] == best_value and a['quantity'] <= bid_quantity]
                if matching_bids:
                    return max(matching_bids, key=lambda a: a['quantity'])
            
            # If no good option, bid the lowest quantity
            return min(bid_actions, key=lambda a: a['quantity'])
        
        # Subsequent bid strategy - very cautious increases
        bid_quantity, bid_value = current_bid
        
        # Check if we have the current bid value
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        
        if own_count > 0:
            # Small increase if we have this value
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Look for values we actually have
        for value, count in analysis['own_value_counts'].items():
            if count > 0 and value > bid_value:
                # Try to switch to this value at same quantity
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # If forced, make smallest valid bid
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


# Dictionary of available agents by difficulty level
RULE_AGENTS = {
    'random': RandomAgent,
    'naive': NaiveAgent,
    'conservative': ConservativeAgent,
    'aggressive': AggressiveAgent,
    'strategic': StrategicAgent,
    'adaptive': AdaptiveAgent
}

# Ordered list of agent difficulties for curriculum learning
CURRICULUM_LEVELS = [
    'random',       # Level 0: Mostly random with slight preference for own dice
    'naive',        # Level 1: Beginner human focusing on own dice
    'conservative', # Level 2: Cautious human with minimal risk-taking
    'aggressive',   # Level 3: Bold human with frequent bluffing
    'strategic',    # Level 4: Strategic human that adapts to game state
    'adaptive'      # Level 5: Expert human with advanced tactics
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