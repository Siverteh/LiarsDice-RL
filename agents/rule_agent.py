"""
Rule-based agents for Liar's Dice.

This module implements various rule-based agents with different
difficulty levels and strategies, designed for curriculum learning
to train the DQN agent.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


class RuleAgent:
    """
    Base class for rule-based Liar's Dice agents.
    
    This class defines the interface for all rule-based agents and
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
    Agent that selects actions completely randomly.
    
    This is the simplest agent and serves as the baseline for curriculum learning.
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the random agent."""
        super().__init__(agent_type='random', dice_faces=dice_faces)
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a random action from valid actions."""
        return random.choice(valid_actions)


class NaiveAgent(RuleAgent):
    """
    Naive agent that follows simple rules without strategic depth.
    
    This agent:
    - Always bids based on dice it can see (no bluffing)
    - Challenges when the bid seems unlikely based on its own dice
    - Doesn't consider game history or adapt to opponents
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the naive agent."""
        super().__init__(agent_type='naive', dice_faces=dice_faces)
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action based on a simple strategy.
        
        Strategy:
        - If no current bid, bid the most common value in own dice
        - If current bid, bid one more of the same value if we have any
        - Challenge if the current bid quantity seems too high
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Challenge if the expected count is significantly less than the bid
            expected_count = analysis['expected_counts'][bid_value]
            if expected_count * 1.5 < bid_quantity:  # Naive threshold
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # If no current bid, bid the most common value in our dice
        if current_bid is None:
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == own_value_counts[best_value]:
                        return action
            
            # If we don't have any dice or all values are equally common, bid randomly
            return random.choice(bid_actions)
        
        # With current bid, try to bid one more of the same value if we have any
        bid_quantity, bid_value = current_bid
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        
        if own_count > 0:
            # Try to bid one more of the same value
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Otherwise, try to bid the same quantity but higher value
        for value in range(bid_value + 1, self.dice_faces + 1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If no good bid found, pick a random bid
        return random.choice(bid_actions)


class ConservativeAgent(RuleAgent):
    """
    Conservative agent that takes minimal risks.
    
    This agent:
    - Only bids quantities it is confident about
    - Rarely challenges unless very confident
    - Prefers to stick with values it has in its own dice
    """
    
    def __init__(self, dice_faces: int = 6, challenge_threshold: float = 0.2):
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
        Select an action with a conservative strategy.
        
        Strategy:
        - Prefer to bid values we have in our own dice
        - Only challenge when very confident (probability < threshold)
        - Make cautious bid increases
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Only challenge if probability is very low
            probability = analysis['probabilities'].get(bid_value, 1.0)
            if probability < self.challenge_threshold:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # Strategy for first bid
        if current_bid is None:
            # Bid the most common value in our dice
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Find a bid close to our actual count
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == best_count:
                        return action
            
            # If we don't have any dice or all values are equally common, bid conservatively
            min_quantity_actions = sorted(bid_actions, key=lambda a: a['quantity'])
            if min_quantity_actions:
                return min_quantity_actions[0]
        
        # Strategy for subsequent bids
        bid_quantity, bid_value = current_bid
        
        # Check if we have the current bid value in our dice
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        
        if own_count > 0:
            # Try to bid one more of the same value
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Try to bid the same quantity but higher value, preferring values we have
        for value in range(bid_value + 1, self.dice_faces + 1):
            has_value = analysis['own_value_counts'].get(value, 0) > 0
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity and has_value:
                    return action
        
        # If no good bid found, make the smallest valid bid
        min_bid = min(bid_actions, key=lambda a: (a['quantity'], a['value']))
        return min_bid


class AggressiveAgent(RuleAgent):
    """
    Aggressive agent that takes risks and bluffs often.
    
    This agent:
    - Makes bold bids to intimidate opponents
    - Challenges frequently to catch bluffs
    - Bluffs strategically to mislead opponents
    """
    
    def __init__(self, dice_faces: int = 6, bluff_frequency: float = 0.4, challenge_threshold: float = 0.4):
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
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with an aggressive strategy.
        
        Strategy:
        - Make bold bids (higher quantities)
        - Challenge more frequently
        - Bluff strategically
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Challenge more aggressively
            probability = analysis['probabilities'].get(bid_value, 1.0)
            if probability < self.challenge_threshold:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # Decide whether to bluff
        should_bluff = random.random() < self.bluff_frequency
        
        # Strategy for first bid
        if current_bid is None:
            if should_bluff:
                # Make a bold first bid
                high_quantity_actions = sorted(bid_actions, key=lambda a: a['quantity'], reverse=True)
                if high_quantity_actions:
                    # Don't pick the absolute highest, but something high
                    idx = min(2, len(high_quantity_actions) - 1)
                    return high_quantity_actions[idx]
            
            # Bid the most common value in our dice
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Find a bid slightly higher than our actual count
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == best_count + 1:
                        return action
            
            # If no good bid found, pick a somewhat high bid
            bids_by_quantity = sorted(bid_actions, key=lambda a: a['quantity'])
            idx = min(len(bids_by_quantity) - 1, len(bids_by_quantity) // 2)
            return bids_by_quantity[idx]
        
        # Strategy for subsequent bids
        bid_quantity, bid_value = current_bid
        
        if should_bluff:
            # Make a bold increase
            for action in bid_actions:
                # Increase quantity by 2 or more
                if action['value'] == bid_value and action['quantity'] >= bid_quantity + 2:
                    return action
            
            # Or try a high value with the same quantity
            for value in range(self.dice_faces, bid_value, -1):
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # More moderate increase
        # Try to bid one more of the same value
        for action in bid_actions:
            if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                return action
        
        # Try to bid the same quantity but higher value
        for value in range(bid_value + 1, self.dice_faces + 1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If no good bid found, make a somewhat aggressive valid bid
        return max(bid_actions, key=lambda a: (a['quantity'] * 0.8 + a['value'] * 0.2))


class StrategicAgent(RuleAgent):
    """
    Strategic agent that adapts to the game state and uses various tactics.
    
    This agent:
    - Tracks bid history to detect patterns
    - Adapts strategy based on game phase and dice counts
    - Balances bluffing and safe plays
    - Uses probabilities to make informed decisions
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the strategic agent."""
        super().__init__(agent_type='strategic', dice_faces=dice_faces)
        self.strategies = {
            'early_game': {'bluff_frequency': 0.2, 'challenge_threshold': 0.25},
            'mid_game': {'bluff_frequency': 0.3, 'challenge_threshold': 0.3},
            'late_game': {'bluff_frequency': 0.4, 'challenge_threshold': 0.35}
        }
        self.bid_history = []
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with an adaptive strategic approach.
        
        Strategy:
        - Adapt to game phase (early, mid, late)
        - Consider bid history and player patterns
        - Balance between aggressive and conservative play
        - Use probabilistic reasoning for challenges
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        dice_counts = observation['dice_counts']
        
        # Update bid history
        if current_bid is not None and (not self.bid_history or self.bid_history[-1] != current_bid):
            self.bid_history.append(current_bid)
        
        # Determine game phase
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
        
        # Check for challenge actions
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Calculate probability and adjust based on game context
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Adjust threshold based on stake (how many dice we have left)
            our_dice = dice_counts[self.player_id]
            if our_dice == 1:  # Last die - be more conservative
                challenge_threshold *= 0.8
            
            # Adjust based on opponent behavior pattern
            if len(self.bid_history) >= 3:
                # Check if opponent has been making increasingly risky bids
                risky_pattern = True
                for i in range(len(self.bid_history) - 3, len(self.bid_history)):
                    if i >= 0 and i < len(self.bid_history) - 1:
                        prev_q, prev_v = self.bid_history[i]
                        curr_q, curr_v = self.bid_history[i + 1]
                        if not (curr_q > prev_q + 1 or (curr_q == prev_q and curr_v > prev_v + 1)):
                            risky_pattern = False
                            break
                
                if risky_pattern:
                    challenge_threshold *= 1.2  # More likely to challenge
            
            # Challenge if probability is below threshold
            if probability < challenge_threshold:
                return challenge_actions[0]
        
        # Filter bid actions
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # Decide whether to bluff based on game context
        should_bluff = random.random() < bluff_frequency
        
        # Adjust bluffing based on other players' dice counts
        other_players_dice = [dice_counts[i] for i in range(self.num_players) if i != self.player_id]
        if max(other_players_dice) <= 1:  # Opponents have few dice
            should_bluff = should_bluff and random.random() < 0.5  # Less bluffing
        
        # Strategy for first bid
        if current_bid is None:
            # Bid based on our own dice
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Decide bid quantity based on strategy
                bid_quantity = best_count
                if should_bluff:
                    bid_quantity += 1
                
                # Find matching bid or closest alternative
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == bid_quantity:
                        return action
            
            # If no good bid found, pick a balanced bid
            return sorted(bid_actions, key=lambda a: a['quantity'])[len(bid_actions) // 2]
        
        # Strategy for subsequent bids
        bid_quantity, bid_value = current_bid
        
        # Check if we have the current bid value in our dice
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        expected_count = analysis['expected_counts'][bid_value]
        
        if should_bluff:
            # Make a bold but plausible bid
            max_plausible = int(expected_count * 1.5)
            for action in bid_actions:
                if action['value'] == bid_value and bid_quantity < action['quantity'] <= max_plausible:
                    return action
            
            # Try switching to a value we have more of
            best_value = max(analysis['own_value_counts'].items(), key=lambda x: x[1])[0] if analysis['own_value_counts'] else 0
            if best_value > 0:
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == bid_quantity:
                        return action
        
        # Standard bid increase
        if own_count > 0 or expected_count > bid_quantity:
            # Increase quantity by 1 for same value
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # Try to bid the same quantity but higher value
        for value in range(bid_value + 1, self.dice_faces + 1):
            has_value = analysis['own_value_counts'].get(value, 0) > 0
            # Prefer values we have
            if has_value:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # Try any higher value
        for value in range(bid_value + 1, self.dice_faces + 1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If no good bid found, make the smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))


class AdaptiveAgent(RuleAgent):
    """
    Adaptive agent that learns from opponent behavior and adjusts strategy.
    
    This agent:
    - Tracks opponent's bidding and challenge patterns
    - Adapts bluffing and challenging thresholds based on opponent behavior
    - Uses different strategies against different opponent types
    - Represents the most challenging opponent for curriculum learning
    """
    
    def __init__(self, dice_faces: int = 6):
        """Initialize the adaptive agent."""
        super().__init__(agent_type='adaptive', dice_faces=dice_faces)
        self.opponent_model = {
            'bluff_ratio': 0.5,  # Initial estimate (0-1)
            'challenge_frequency': 0.5,  # Initial estimate (0-1)
            'bids_before_challenge': []
        }
        self.memory = {
            'actions': [],  # Last actions by any player
            'bids': [],  # Sequence of bids in current round
            'bid_outcomes': [],  # Outcomes of previous bids (challenged/successful)
            'round_history': []  # History of completed rounds
        }
        # Initial strategy parameters
        self.bluff_frequency = 0.3
        self.challenge_threshold = 0.3
    
    def select_action(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action with adaptive strategy based on opponent modeling.
        
        Strategy:
        - Update opponent model based on observed actions
        - Adjust strategy parameters to counter opponent tendencies
        - Use tailored strategies against different opponent types
        """
        analysis = self.analyze_game_state(observation)
        current_bid = observation['current_bid']
        dice_counts = observation['dice_counts']
        round_num = observation['round_num']
        player_id = observation['player_id']
        history = observation['history']
        
        # Update memory based on new observations
        self._update_memory(observation)
        
        # Update opponent model if we have enough data
        if len(self.memory['actions']) > 5:
            self._update_opponent_model()
        
        # Adapt strategy parameters based on opponent model
        self._adapt_strategy()
        
        # Challenge decision
        challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
        if challenge_actions and current_bid is not None:
            bid_quantity, bid_value = current_bid
            
            # Calculate probability and adjust based on opponent model
            probability = analysis['probabilities'].get(bid_value, 1.0)
            
            # Adjust threshold based on opponent bluffing tendency
            adjusted_threshold = self.challenge_threshold
            if self.opponent_model['bluff_ratio'] > 0.7:
                adjusted_threshold *= 1.3  # More likely to challenge a bluffer
            elif self.opponent_model['bluff_ratio'] < 0.3:
                adjusted_threshold *= 0.7  # Less likely to challenge an honest player
            
            # Consider the stake
            our_dice = dice_counts[self.player_id]
            if our_dice == 1:  # Last die
                adjusted_threshold *= 0.8  # More conservative
            
            # Consider how "suspicious" the bid is
            if current_bid in self.memory['bids']:
                bid_idx = self.memory['bids'].index(current_bid)
                if bid_idx > 0:
                    prev_quantity, prev_value = self.memory['bids'][bid_idx - 1]
                    if bid_quantity > prev_quantity + 2:  # Big jump in quantity
                        adjusted_threshold *= 1.2  # More suspicious
            
            # Challenge if probability is below threshold
            if probability < adjusted_threshold:
                return challenge_actions[0]
        
        # Bidding decision
        bid_actions = [a for a in valid_actions if a['type'] == 'bid']
        
        if not bid_actions:
            # If no valid bids, must challenge
            return challenge_actions[0]
        
        # Decide whether to bluff based on opponent tendencies
        should_bluff = random.random() < self.bluff_frequency
        
        # Adjust bluffing based on opponent challenging behavior
        if self.opponent_model['challenge_frequency'] > 0.7:
            should_bluff = should_bluff and random.random() < 0.3  # Bluff less often
        
        # Strategy for first bid
        if current_bid is None:
            # Bid based on our own dice with possible bluffing
            own_value_counts = analysis['own_value_counts']
            if own_value_counts:
                best_value = max(own_value_counts.items(), key=lambda x: x[1])[0]
                best_count = own_value_counts[best_value]
                
                # Decide bid quantity
                bid_quantity = best_count
                if should_bluff:
                    # Bluff amount depends on opponent challenging tendency
                    if self.opponent_model['challenge_frequency'] < 0.3:
                        bid_quantity += 2  # Bolder bluff against passive opponents
                    else:
                        bid_quantity += 1  # Modest bluff
                
                # Find matching bid
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == bid_quantity:
                        return action
                
                # Try closest alternative
                sorted_by_quantity = sorted(bid_actions, key=lambda a: abs(a['quantity'] - bid_quantity))
                for action in sorted_by_quantity:
                    if action['value'] == best_value:
                        return action
            
            # If no good bid found based on our dice, use a balanced bid
            return sorted(bid_actions, key=lambda a: a['quantity'])[len(bid_actions) // 2]
        
        # Strategy for subsequent bids
        bid_quantity, bid_value = current_bid
        own_count = analysis['own_value_counts'].get(bid_value, 0)
        expected_count = analysis['expected_counts'][bid_value]
        
        # Find values we have the most of
        best_value = max(analysis['own_value_counts'].items(), key=lambda x: x[1])[0] if analysis['own_value_counts'] else 0
        best_count = analysis['own_value_counts'].get(best_value, 0)
        
        if should_bluff:
            # Strategic bluffing
            if self.opponent_model['challenge_frequency'] < 0.3:
                # Against passive opponents, make aggressive bids
                max_plausible = int(expected_count * 1.7)
                for action in bid_actions:
                    if action['value'] == bid_value and bid_quantity < action['quantity'] <= max_plausible:
                        return action
            else:
                # Against active challengers, make subtle bluffs
                max_plausible = int(expected_count * 1.3)
                for action in bid_actions:
                    if action['value'] == bid_value and bid_quantity < action['quantity'] <= max_plausible:
                        return action
            
            # Try switching to a value we have more of
            if best_value > 0 and best_count > own_count:
                for action in bid_actions:
                    if action['value'] == best_value and action['quantity'] == bid_quantity:
                        return action
        
        # Standard bidding strategy
        if own_count > 0 or expected_count > bid_quantity:
            # Increase quantity by 1 for same value
            for action in bid_actions:
                if action['value'] == bid_value and action['quantity'] == bid_quantity + 1:
                    return action
        
        # If we have a better value, try to switch
        if best_value > 0 and best_count > own_count:
            for action in bid_actions:
                if action['value'] == best_value and action['quantity'] == bid_quantity:
                    return action
        
        # Try to bid the same quantity but higher value
        for value in range(bid_value + 1, self.dice_faces + 1):
            has_value = analysis['own_value_counts'].get(value, 0) > 0
            # Prefer values we have
            if has_value:
                for action in bid_actions:
                    if action['value'] == value and action['quantity'] == bid_quantity:
                        return action
        
        # Try any higher value
        for value in range(bid_value + 1, self.dice_faces + 1):
            for action in bid_actions:
                if action['value'] == value and action['quantity'] == bid_quantity:
                    return action
        
        # If no good bid found, make the smallest valid bid
        return min(bid_actions, key=lambda a: (a['quantity'], a['value']))
    
    def _update_memory(self, observation: Dict[str, Any]):
        """
        Update agent's memory with new observations.
        
        Args:
            observation: Current game observation
        """
        current_bid = observation['current_bid']
        history = observation['history']
        
        # Track bids in the current round
        if current_bid is not None and (not self.memory['bids'] or self.memory['bids'][-1] != current_bid):
            self.memory['bids'].append(current_bid)
        
        # Track actions by all players
        if history:
            new_actions = [entry for entry in history if entry not in self.memory['actions']]
            self.memory['actions'].extend(new_actions)
        
        # Check if a new round has started
        if len(history) > 0 and history[-1]['action']['type'] == 'challenge':
            # A challenge means the end of a round
            if len(self.memory['round_history']) < observation['round_num'] - 1:
                # Store the previous round data
                self.memory['round_history'].append({
                    'bids': self.memory['bids'].copy(),
                    'outcome': history[-1]['action'].get('success', None)
                })
                # Reset bids for new round
                self.memory['bids'] = []
    
    def _update_opponent_model(self):
        """Update the opponent model based on observed behaviors."""
        actions = self.memory['actions']
        
        # Count challenges and bids by opponents
        opponent_challenges = 0
        opponent_bids = 0
        
        for action in actions:
            if action['player'] != self.player_id:
                if action['action']['type'] == 'challenge':
                    opponent_challenges += 1
                elif action['action']['type'] == 'bid':
                    opponent_bids += 1
        
        total_opponent_actions = opponent_challenges + opponent_bids
        if total_opponent_actions > 0:
            self.opponent_model['challenge_frequency'] = opponent_challenges / total_opponent_actions
        
        # Analyze bidding behavior for bluffing estimation
        if len(self.memory['round_history']) > 0:
            bluff_count = 0
            honest_count = 0
            
            for round_data in self.memory['round_history']:
                last_bid_idx = len(round_data['bids']) - 1
                if last_bid_idx >= 0:
                    last_bid = round_data['bids'][last_bid_idx]
                    outcome = round_data.get('outcome')
                    
                    if outcome is not None:
                        if outcome == 'failed':  # Bid was a bluff
                            bluff_count += 1
                        else:  # Bid was honest
                            honest_count += 1
            
            total_analyzed_bids = bluff_count + honest_count
            if total_analyzed_bids > 0:
                self.opponent_model['bluff_ratio'] = bluff_count / total_analyzed_bids
    
    def _adapt_strategy(self):
        """Adapt strategy parameters based on opponent model."""
        # Adapt bluff frequency
        if self.opponent_model['challenge_frequency'] < 0.2:
            # Against passive opponents, bluff more
            self.bluff_frequency = min(0.7, self.opponent_model['challenge_frequency'] + 0.5)
        elif self.opponent_model['challenge_frequency'] > 0.6:
            # Against aggressive challengers, bluff less
            self.bluff_frequency = max(0.1, 0.8 - self.opponent_model['challenge_frequency'])
        else:
            # Balanced approach
            self.bluff_frequency = 0.3
        
        # Adapt challenge threshold
        if self.opponent_model['bluff_ratio'] < 0.2:
            # Against honest opponents, challenge less
            self.challenge_threshold = 0.2
        elif self.opponent_model['bluff_ratio'] > 0.6:
            # Against frequent bluffers, challenge more
            self.challenge_threshold = 0.4
        else:
            # Balanced approach
            self.challenge_threshold = 0.3


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
    'random',       # Level 0: Random agent (completely random actions)
    'naive',        # Level 1: Naive agent (simple rules, no bluffing)
    'conservative', # Level 2: Conservative agent (cautious play)
    'aggressive',   # Level 3: Aggressive agent (bold bids and challenges)
    'strategic',    # Level 4: Strategic agent (adaptive to game state)
    'adaptive'      # Level 5: Adaptive agent (models opponent behavior)
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