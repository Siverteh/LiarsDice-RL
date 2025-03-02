"""
Base interface for reinforcement learning agents in Liar's Dice.

This module defines a common interface that all RL algorithms must implement
to work with the curriculum learning framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch


class RLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    Any algorithm (DQN, PPO, AlphaZero, etc.) must implement this interface
    to work with the curriculum learning framework.
    """
    
    def __init__(self):
        """Initialize the agent."""
        self.action_to_game_action = None
    
    @abstractmethod
    def select_action(self, obs: np.ndarray, valid_actions: List[Dict[str, Any]], training: bool = True) -> Dict[str, Any]:
        """
        Select an action given the current observation and valid actions.
        
        Args:
            obs: Current observation
            valid_actions: List of valid actions in game format
            training: Whether the agent is in training mode
            
        Returns:
            Selected action in game format
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """
        Update the agent's policy or value function.
        
        Returns:
            Loss value for logging
        """
        pass
    
    @abstractmethod
    def add_experience(self, obs: np.ndarray, action: Dict[str, Any], 
                      reward: float, next_obs: np.ndarray, done: bool):
        """
        Add experience to the agent's memory.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save the agent to the specified path.
        
        Args:
            path: Directory to save the agent
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load the agent from the specified path.
        
        Args:
            path: Directory to load the agent from
        """
        pass
    
    def set_action_mapping(self, action_mapping: List[Dict[str, Any]]):
        """
        Set the mapping from action indices to game actions.
        
        Args:
            action_mapping: List of game actions where the index corresponds to the action index
        """
        self.action_to_game_action = action_mapping
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent for logging.
        
        Returns:
            Dictionary of agent statistics
        """
        pass
    
    def _actions_equal(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """
        Check if two actions are equal.
        
        Args:
            action1: First action
            action2: Second action
            
        Returns:
            True if actions are equal, False otherwise
        """
        if action1['type'] != action2['type']:
            return False
        
        if action1['type'] == 'challenge':
            return True
        
        return (action1['quantity'] == action2['quantity'] and 
                action1['value'] == action2['value'])