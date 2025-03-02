"""
Agent factory for creating reinforcement learning agents.

This module provides factory functions for instantiating different types of RL agents
that all conform to the RLAgent interface.
"""

from typing import Dict, Any, List, Optional

from agents.base_agent import RLAgent
from agents.dqn_agent import DQNAgent  # Import your actual DQN implementation
from agents.ppo_agent import PPOAgent


def create_agent(
    agent_type: str,
    obs_dim: int,
    action_dim: int,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RLAgent:
    """
    Create a reinforcement learning agent of the specified type.
    
    Args:
        agent_type: Type of agent to create ('dqn', 'ppo', etc.)
        obs_dim: Dimension of the observation space
        action_dim: Dimension of the action space
        config: Dictionary of configuration parameters
        **kwargs: Additional keyword arguments to pass to the agent constructor
        
    Returns:
        An agent implementing the RLAgent interface
    """
    # Default configuration
    if config is None:
        config = {}
    
    # Combine config and kwargs, with kwargs taking precedence
    full_config = {**config, **kwargs}
    
    # Create agent based on type
    if agent_type.lower() == 'dqn':
        return create_dqn_agent(obs_dim, action_dim, full_config)
    elif agent_type.lower() == 'ppo':
        return create_ppo_agent(obs_dim, action_dim, full_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_dqn_agent(
    obs_dim: int,
    action_dim: int,
    config: Dict[str, Any]
) -> DQNAgent:
    """
    Create a DQN agent with the specified configuration.
    
    Args:
        obs_dim: Dimension of the observation space
        action_dim: Dimension of the action space
        config: Dictionary of configuration parameters
        
    Returns:
        A configured DQN agent
    """
    # Extract only the parameters that your DQN agent actually accepts
    kwargs = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'learning_rate': config.get('learning_rate', 0.0005),
        'gamma': config.get('gamma', 0.99),
        'epsilon_start': config.get('epsilon_start', 1.0),
        'epsilon_end': config.get('epsilon_end', 0.05),
        'target_update_freq': config.get('target_update_freq', 500),
        'buffer_size': config.get('buffer_size', 100000),
        'batch_size': config.get('batch_size', 64),
        'hidden_dims': config.get('hidden_dims', [256, 128, 64]),
        'device': config.get('device', 'auto')
    }
    
    # Handle epsilon decay parameter based on which version of DQN you're using
    if hasattr(DQNAgent, 'epsilon_decay'):
        # Original DQN with multiplicative decay
        kwargs['epsilon_decay'] = config.get('epsilon_decay', 0.995)
    elif hasattr(DQNAgent, 'epsilon_decay_steps'):
        # Enhanced DQN with linear decay
        kwargs['epsilon_decay_steps'] = config.get('epsilon_decay_steps', 100000)
    
    # Create and return DQN agent using only the parameters it accepts
    return DQNAgent(**kwargs)


def create_ppo_agent(
    obs_dim: int,
    action_dim: int,
    config: Dict[str, Any]
) -> PPOAgent:
    """
    Create a PPO agent with the specified configuration.
    
    Args:
        obs_dim: Dimension of the observation space
        action_dim: Dimension of the action space
        config: Dictionary of configuration parameters
        
    Returns:
        A configured PPO agent
    """
    # Only include parameters that your PPO agent actually accepts
    kwargs = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'learning_rate': config.get('learning_rate', 0.0003),
        'gamma': config.get('gamma', 0.99),
        'gae_lambda': config.get('gae_lambda', 0.95),
        'policy_clip': config.get('policy_clip', 0.2),
        'value_coef': config.get('value_coef', 0.5),
        'entropy_coef': config.get('entropy_coef', 0.01),
        'ppo_epochs': config.get('ppo_epochs', 4),
        'batch_size': config.get('batch_size', 64),
        'hidden_dims': config.get('hidden_dims', [256, 128, 64]),
        'update_frequency': config.get('update_frequency', 2048),
        'device': config.get('device', 'auto')
    }
    
    # Create and return PPO agent with parameters it accepts
    return PPOAgent(**kwargs)