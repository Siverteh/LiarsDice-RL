"""
Utility functions for training and evaluation.

This module provides various helper functions for managing training,
visualizing results, and handling data.
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    Prevents duplicate handlers when called multiple times with the same name.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Prevent propagation to the root logger
    logger.propagate = False
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_training_data(data: Dict[str, Any], filepath: str):
    """
    Save training data to a file.
    
    Args:
        data: Dictionary containing training data
        filepath: Path to save the data
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_training_data(filepath: str) -> Dict[str, Any]:
    """
    Load training data from a file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Dictionary containing the training data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_training_results(
    data: Dict[str, Any],
    window_size: int = 20,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot training results with smoothing.
    
    Args:
        data: Dictionary containing training data
        window_size: Window size for smoothing
        save_path: Path to save the plot (if None, don't save)
        show_plot: Whether to display the plot
    """
    rewards = data.get('rewards', [])
    episode_lengths = data.get('episode_lengths', [])
    losses = data.get('losses', [])
    eval_rewards = data.get('eval_rewards', [])
    
    # Apply smoothing
    def smooth(y, window_size):
        if len(y) < window_size:
            return y
        box = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth
    
    rewards_smooth = smooth(rewards, window_size)
    episode_lengths_smooth = smooth(episode_lengths, window_size)
    losses_smooth = smooth(losses, window_size) if losses else []
    
    # Create x-axes
    x_rewards = np.arange(len(rewards_smooth)) + window_size - 1
    x_lengths = np.arange(len(episode_lengths_smooth)) + window_size - 1
    x_losses = np.arange(len(losses_smooth)) + window_size - 1
    
    # Number of subplots
    num_plots = 3 if len(losses) > 0 else 2
    if len(eval_rewards) > 0:
        num_plots += 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3*num_plots), sharex=True)
    
    # Plot rewards
    axes[0].plot(x_rewards, rewards_smooth, 'b-')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards (Smoothed)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1].plot(x_lengths, episode_lengths_smooth, 'g-')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('Episode Lengths (Smoothed)')
    axes[1].grid(True, alpha=0.3)
    
    plot_idx = 2
    
    # Plot losses if available
    if len(losses) > 0:
        axes[plot_idx].plot(x_losses, losses_smooth, 'r-')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Training Loss (Smoothed)')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot evaluation rewards if available
    if len(eval_rewards) > 0:
        eval_episodes = [i * (len(rewards) // len(eval_rewards)) for i in range(len(eval_rewards))]
        axes[plot_idx].plot(eval_episodes, eval_rewards, 'm-', marker='o')
        axes[plot_idx].set_ylabel('Eval Reward')
        axes[plot_idx].set_title('Evaluation Rewards')
        axes[plot_idx].grid(True, alpha=0.3)
    
    # Set x-label for bottom subplot
    axes[-1].set_xlabel('Episode')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def get_action_mapping(
    num_players: int,
    num_dice: int,
    dice_faces: int
) -> List[Dict[str, Any]]:
    """
    Generate a complete action mapping for Liar's Dice.
    
    Args:
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        
    Returns:
        List of all possible game actions
    """
    action_mapping = []
    
    # Add challenge action
    action_mapping.append({'type': 'challenge'})
    
    # Add all possible bid actions
    max_quantity = num_players * num_dice
    for quantity in range(1, max_quantity + 1):
        for value in range(1, dice_faces + 1):
            action_mapping.append({
                'type': 'bid',
                'quantity': quantity,
                'value': value
            })
    
    return action_mapping


def generate_curriculum_schedule(
    total_episodes: int,
    num_levels: int,
    distribution: str = 'linear'
) -> List[int]:
    """
    Generate a schedule for curriculum learning.
    
    Args:
        total_episodes: Total number of training episodes
        num_levels: Number of difficulty levels
        distribution: How to distribute episodes ('linear', 'exp', or 'front_loaded')
        
    Returns:
        List where index i contains the number of episodes for level i
    """
    if distribution == 'linear':
        # Equal distribution
        base = total_episodes // num_levels
        remainder = total_episodes % num_levels
        schedule = [base] * num_levels
        for i in range(remainder):
            schedule[i] += 1
    
    elif distribution == 'exp':
        # Exponential distribution (more episodes for harder levels)
        weights = [2 ** i for i in range(num_levels)]
        total_weight = sum(weights)
        schedule = [int(total_episodes * w / total_weight) for w in weights]
        # Adjust for rounding errors
        diff = total_episodes - sum(schedule)
        schedule[-1] += diff
    
    elif distribution == 'front_loaded':
        # Front-loaded distribution (more episodes for easier levels)
        weights = [2 ** (num_levels - i) for i in range(num_levels)]
        total_weight = sum(weights)
        schedule = [int(total_episodes * w / total_weight) for w in weights]
        # Adjust for rounding errors
        diff = total_episodes - sum(schedule)
        schedule[0] += diff
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return schedule