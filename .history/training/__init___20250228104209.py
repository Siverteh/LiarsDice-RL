"""
Training package for Liar's Dice reinforcement learning.

This package contains modules for training, evaluating, and managing
RL agents for playing Liar's Dice.
"""

from .trainer import Trainer, TrainingConfig
from .evaluate import evaluate_agents, tournament
from .utils import plot_training_curves, save_metrics

__all__ = [
    'Trainer',
    'TrainingConfig',
    'evaluate_agents',
    'tournament',
    'plot_training_curves',
    'save_metrics'
]