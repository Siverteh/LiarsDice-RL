"""
Utility functions for training Liar's Dice agents.

This module provides utility functions for plotting, saving metrics,
and other helpers for the training process.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def smooth_data(data: List[float], window_size: int = 10) -> np.ndarray:
    """
    Apply moving average smoothing to data.
    
    Args:
        data: Data to smooth
        window_size: Size of the smoothing window
        
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return np.array(data)
    
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_training_curves(metrics: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot training curves and save the figure.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the figure
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot rewards
    axs[0].plot(metrics['episode_rewards'], alpha=0.3, label='Raw')
    if len(metrics['episode_rewards']) > 10:
        smoothed = smooth_data(metrics['episode_rewards'], 100)
        axs[0].plot(range(len(metrics['episode_rewards']) - len(smoothed), 
                          len(metrics['episode_rewards'])), 
                   smoothed, label='Smoothed')
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    
    # Plot episode lengths
    axs[1].plot(metrics['episode_lengths'], alpha=0.3, label='Raw')
    if len(metrics['episode_lengths']) > 10:
        smoothed = smooth_data(metrics['episode_lengths'], 100)
        axs[1].plot(range(len(metrics['episode_lengths']) - len(smoothed), 
                         len(metrics['episode_lengths'])), 
                  smoothed, label='Smoothed')
    axs[1].set_title('Episode Lengths')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')
    axs[1].legend()
    
    # Plot win rates
    if 'win_rates' in metrics and metrics['win_rates']:
        eval_interval = len(metrics['episode_rewards']) // len(metrics['win_rates'])
        episodes = np.arange(eval_interval, len(metrics['episode_rewards']) + 1, eval_interval)
        axs[2].plot(episodes, metrics['win_rates'], marker='o')
        axs[2].set_title('Evaluation Win Rate')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Win Rate')
        axs[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_metrics(metrics: Dict[str, List[float]], save_path: str) -> None:
    """
    Save training metrics to a JSON file.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the metrics
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)


def load_metrics(load_path: str) -> Dict[str, List[float]]:
    """
    Load training metrics from a JSON file.
    
    Args:
        load_path: Path to load the metrics from
        
    Returns:
        Dictionary of training metrics
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def create_agent_mapping(agents: List[Any], names: Optional[List[str]] = None) -> Dict[int, Any]:
    """
    Create a mapping from player IDs to agents.
    
    Args:
        agents: List of agent objects
        names: Optional list of agent names
        
    Returns:
        Dictionary mapping player IDs to agents
    """
    agent_mapping = {}
    
    for i, agent in enumerate(agents):
        agent.player_id = i
        if names and i < len(names):
            agent.name = names[i]
        agent_mapping[i] = agent
    
    return agent_mapping