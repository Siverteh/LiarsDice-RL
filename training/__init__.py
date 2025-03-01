"""
Training package for Liar's Dice DQN agent.

This package contains training utilities, evaluation functions,
and environment wrappers for training DQN agents to play Liar's Dice.
"""

from .train import train_dqn_episode, train_dqn
from .evaluate import evaluate_agent, evaluate_against_curriculum
from .environment_wrapper import LiarsDiceEnvWrapper
from .utils import (
    setup_logger, 
    plot_training_results, 
    save_training_data, 
    load_training_data,
    get_action_mapping
)

__all__ = [
    'train_dqn_episode',
    'train_dqn',
    'evaluate_agent',
    'evaluate_against_curriculum',
    'LiarsDiceEnvWrapper',
    'setup_logger',
    'plot_training_results',
    'save_training_data',
    'load_training_data',
    'get_action_mapping'
]