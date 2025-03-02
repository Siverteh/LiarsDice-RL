"""
Refined Reward Calculation for Liar's Dice.

This module provides multiple reward schemes, each designed to guide
learning agents in Liar’s Dice. It balances end-game rewards (winning
or losing dice) with small shaping incentives for strategic behavior
(bidding, bluffing, challenging).

Usage:
  1. Create a RewardCalculator with desired parameters:
       >>> rc = RewardCalculator(num_players=4, reward_scheme='strategic')
  2. Call `calculate_rewards(prev_state, action, next_state, info)` each time
     the environment transitions from `prev_state` to `next_state`.
  3. Use the returned dictionary of {player_id: reward} in your RL algorithm.
"""

import numpy as np
from typing import Dict, Any


class RewardCalculator:
    """
    Calculates rewards for Liar's Dice players.

    Offers different reward schemes to encourage:
      - Standard end-game objectives (win vs. lose dice)
      - Strategic bidding, bluffing, and challenging
      - Sparse vs. shaped reward signals

    Attributes:
        num_players (int): Number of players in the game.
        reward_scheme (str): 'standard', 'shaped', 'sparse', or 'strategic'.
        win_reward (float): Reward for winning the game.
        lose_dice_penalty (float): Penalty for losing a die.
        successful_bluff_reward (float): Small reward for successful bluffing.
        successful_challenge_reward (float): Small reward for successful challenge.
        action_penalty (float): Small penalty per step to encourage game speed.
    """

    def __init__(
        self,
        num_players: int,
        reward_scheme: str = "standard",
        win_reward: float = 10.0,
        lose_dice_penalty: float = -1.0,
        successful_bluff_reward: float = 0.5,
        successful_challenge_reward: float = 0.5,
        action_penalty: float = -0.01
    ):
        """
        Initialize the RewardCalculator.

        Args:
            num_players: Number of players in the game.
            reward_scheme: Which reward scheme to use:
                           ['standard', 'shaped', 'sparse', 'strategic'].
            win_reward: Reward given to the winner at game-over.
            lose_dice_penalty: Penalty each time a player loses a die.
            successful_bluff_reward: Extra reward for a successful bluff.
            successful_challenge_reward: Extra reward for a successful challenge.
            action_penalty: Small penalty applied every step (time penalty).
        """
        self.num_players = num_players
        self.reward_scheme = reward_scheme
        self.win_reward = win_reward
        self.lose_dice_penalty = lose_dice_penalty
        self.successful_bluff_reward = successful_bluff_reward
        self.successful_challenge_reward = successful_challenge_reward
        self.action_penalty = action_penalty

    def calculate_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate rewards for all players after a state transition.

        Args:
            prev_state: State dictionary before the action.
            action: The action dict taken by the current player.
            next_state: State dictionary after the action.
            info: Additional info about the transition (e.g., 'GAME_OVER').

        Returns:
            A dict {player_id: reward}, indicating the immediate reward
            for each player at this transition.
        """
        if self.reward_scheme == "standard":
            return self._standard_rewards(prev_state, action, next_state, info)
        elif self.reward_scheme == "shaped":
            return self._shaped_rewards(prev_state, action, next_state, info)
        elif self.reward_scheme == "sparse":
            return self._sparse_rewards(prev_state, action, next_state, info)
        elif self.reward_scheme == "strategic":
            return self._strategic_rewards(prev_state, action, next_state, info)
        else:
            raise ValueError(f"Unknown reward scheme: {self.reward_scheme}")

    # -----------------------------------------------------------------------
    # 1) Standard Rewards
    # -----------------------------------------------------------------------
    def _standard_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Standard rewards based on:
          - Win (+self.win_reward)
          - Lose a die (self.lose_dice_penalty)
          - Small time penalty each step (self.action_penalty)

        Encourages shorter games, punishes dice loss, rewards only the winner.
        """
        rewards = {player: 0.0 for player in range(self.num_players)}

        # 1. Per-step action penalty (time penalty).
        for player in range(self.num_players):
            rewards[player] += self.action_penalty

        # 2. Dice loss penalty.
        prev_counts = prev_state["dice_counts"]
        next_counts = next_state["dice_counts"]
        for player in range(self.num_players):
            if next_counts[player] < prev_counts[player]:
                # Player lost at least one die.
                rewards[player] += self.lose_dice_penalty

        # 3. Game-over bonus for the winner.
        if info.get("state") == "GAME_OVER":
            # Typically, the winner is the last player with dice > 0,
            # or the one with the most dice if the game ended abruptly.
            winner = np.argmax(next_counts)
            rewards[winner] += self.win_reward

        return rewards

    # -----------------------------------------------------------------------
    # 2) Shaped Rewards
    # -----------------------------------------------------------------------
    def _shaped_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Builds upon the standard scheme, adding:
          - Small reward for "risky" bid increments.
          - Reward for successful challenge or bluff.
          - Still overshadowed by final win/loss to avoid reward hacking.
        """
        # Start with base standard rewards.
        rewards = self._standard_rewards(prev_state, action, next_state, info)

        current_player = prev_state["current_player"]
        prev_bid = prev_state.get("current_bid", None)
        prev_dice_counts = prev_state["dice_counts"]
        next_dice_counts = next_state["dice_counts"]

        if action["type"] == "bid":
            # Reward for riskier, bigger bids (compared to previous bid).
            if prev_bid is not None:
                prev_qty, prev_val = prev_bid
                new_qty, new_val = action["quantity"], action["value"]
                dice_total = sum(prev_dice_counts)

                # 1. Incrementing quantity
                if new_qty > prev_qty:
                    risk_factor = (new_qty - prev_qty) / dice_total
                    # Keep risk_factor small, scale by 0.1
                    rewards[current_player] += 0.1 * risk_factor

                # 2. Incrementing value
                elif new_val > prev_val:
                    value_diff = new_val - prev_val
                    # Scale by 0.1 to keep it modest.
                    rewards[current_player] += 0.1 * (value_diff / 6.0)

        elif action["type"] == "challenge":
            # If a challenge causes someone else to lose a die, it's successful.
            for player in range(self.num_players):
                if next_dice_counts[player] < prev_dice_counts[player]:
                    if player != current_player:
                        # The current_player's challenge was correct.
                        rewards[current_player] += self.successful_challenge_reward
                    else:
                        # The challenger lost a die => failed challenge => the
                        # last bidder was actually bluffing successfully.
                        prev_player = prev_state.get("previous_player", None)
                        if prev_player is not None:
                            rewards[prev_player] += self.successful_bluff_reward

        return rewards

    # -----------------------------------------------------------------------
    # 3) Sparse Rewards
    # -----------------------------------------------------------------------
    def _sparse_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Provides rewards *only* at game end:
          - Winner gets self.win_reward
          - Others get 0

        Useful for pure RL approaches where shaping might bias exploration.
        """
        rewards = {player: 0.0 for player in range(self.num_players)}

        if info.get("state") == "GAME_OVER":
            winner = np.argmax(next_state["dice_counts"])
            rewards[winner] = self.win_reward

        return rewards

    # -----------------------------------------------------------------------
    # 4) Strategic Rewards
    # -----------------------------------------------------------------------
    def _strategic_rewards(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Rewards advanced strategic considerations:
          - Base standard rewards (win, lose dice, time penalty).
          - Additional bonus/penalty for bold or overly conservative bids,
            relative to expected dice counts from the player's perspective.
          - Additional bonus/penalty for challenge decisions based on
            estimated probability.

        NOTE: The partial observability assumption means we do *not* assume
        knowledge of all dice. The example logic below uses uniform
        distribution for unknown dice, which might not be perfect but
        serves as a heuristic.
        """
        # Start with standard.
        rewards = self._standard_rewards(prev_state, action, next_state, info)

        current_player = prev_state["current_player"]
        prev_bid = prev_state.get("current_bid", None)
        prev_dice_counts = prev_state["dice_counts"]
        next_dice_counts = next_state["dice_counts"]

        # Attempt to get the current player's dice if it exists (partial obs).
        # If not, we skip advanced shaping to avoid unrealistic knowledge.
        player_dice = None
        if "dice" in prev_state and current_player in prev_state["dice"]:
            player_dice = prev_state["dice"][current_player]

        # Bidding logic: reward or penalize bold or conservative bids
        if action["type"] == "bid" and player_dice is not None:
            new_qty, new_val = action["quantity"], action["value"]
            # Count how many dice in the player's hand match new_val.
            own_count = sum(d == new_val for d in player_dice)
            total_dice = sum(prev_dice_counts)
            other_dice = total_dice - len(player_dice)

            # Estimate how many might exist outside player's hand.
            expected_outside = other_dice * (1.0 / 6.0)
            expected_total = own_count + expected_outside

            # Compare new_qty to expected_total:
            # If new_qty >> expected_total, it's a bold bluff.
            if new_qty > 1.2 * expected_total:
                # A moderate reward for big risk (that might pay off).
                rewards[current_player] += 0.2
            elif new_qty < 0.8 * expected_total:
                # Mild penalty for being too timid if there's likely more dice out there.
                rewards[current_player] -= 0.1

        # Challenge logic: reward or penalize based on probability
        if action["type"] == "challenge" and player_dice is not None:
            # The current bid under challenge:
            if prev_bid is not None:
                bid_quantity, bid_value = prev_bid
                # Probability check from the challenger's perspective
                own_count = sum(d == bid_value for d in player_dice)
                total_dice = sum(prev_dice_counts)
                other_dice = total_dice - len(player_dice)
                expected_outside = other_dice * (1.0 / 6.0)
                expected_total = own_count + expected_outside

                # If the bid seems too high vs. expected_total => good challenge
                if bid_quantity > 1.3 * expected_total:
                    rewards[current_player] += 0.3
                # If the bid is likely true => penalty for a poor challenge
                elif bid_quantity < 0.7 * expected_total:
                    rewards[current_player] -= 0.2

        # After challenge, see if it succeeded or failed to award bluff/challenge:
        if action["type"] == "challenge":
            for player in range(self.num_players):
                if next_dice_counts[player] < prev_dice_counts[player]:
                    if player != current_player:
                        # This means the challenge was successful (someone else lost a die).
                        rewards[current_player] += self.successful_challenge_reward
                    else:
                        # The challenger lost a die => previous bidder was bluffing well.
                        prev_player = prev_state.get("previous_player", None)
                        if prev_player is not None:
                            rewards[prev_player] += self.successful_bluff_reward

        return rewards
