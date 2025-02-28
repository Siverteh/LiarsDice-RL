"""
Abstract base class for Liar's Dice agents.

This module defines the BaseAgent abstract class that all other agents
should inherit from, ensuring a consistent interface.
"""

import abc
from typing import Dict, List, Any, Optional, Tuple


class BaseAgent(abc.ABC):
    """
    Abstract base class for all Liar's Dice agents.
    
    This class defines the interface that all agents must implement,
    including methods for action selection, training, and evaluation.
    
    Attributes:
        player_id (int): ID of the player this agent controls
        name (str): Name of the agent for display purposes
    """
    
    def __init__(self, player_id: int, name: str = None):
        """
        Initialize the agent.
        
        Args:
            player_id: ID of the player this agent controls
            name: Optional name for the agent
        """
        self.player_id = player_id
        self.name = name if name is not None else f"Agent-{player_id}"
    
    @abc.abstractmethod
    def act(self, observation: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current game observation from the environment
            valid_actions: List of valid actions in the current state
            
        Returns:
            Selected action as a dictionary
        """
        pass
    
    def update(self, 
               observation: Dict[str, Any], 
               action: Dict[str, Any], 
               reward: float, 
               next_observation: Dict[str, Any], 
               done: bool, 
               valid_actions: List[Dict[str, Any]]) -> None:
        """
        Update the agent's internal state based on experience.
        
        By default, this does nothing. Reinforcement learning agents
        should override this method to implement learning.
        
        Args:
            observation: Observation that led to the action
            action: Action that was taken
            reward: Reward that was received
            next_observation: Observation after the action
            done: Whether the episode is done
            valid_actions: Valid actions for the observation
        """
        pass
    
    def train(self) -> Optional[float]:
        """
        Execute a training step.
        
        By default, this does nothing. Reinforcement learning agents
        should override this method to implement model updates.
        
        Returns:
            Optional loss value from training
        """
        return None
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to disk.
        
        By default, this does nothing. Reinforcement learning agents
        should override this method to save model weights.
        
        Args:
            path: Path to save the agent to
        """
        pass
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from disk.
        
        By default, this does nothing. Reinforcement learning agents
        should override this method to load model weights.
        
        Args:
            path: Path to load the agent from
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the agent's state for a new episode.
        
        By default, this does nothing. Agents that maintain episode-specific
        state should override this method.
        """
        pass