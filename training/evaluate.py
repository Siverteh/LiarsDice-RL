"""
Evaluation module for DQN agent in Liar's Dice.

This module provides functions for evaluating trained agents
against rule-based agents of different difficulty levels.
"""

import os
import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

from agents.base_agent import RLAgent
from agents.rule_agent import create_agent, CURRICULUM_LEVELS
from environment.game import LiarsDiceGame
from .environment_wrapper import LiarsDiceEnvWrapper
from .utils import setup_logger


def evaluate_agent(
    agent: RLAgent,  # Change to generic RLAgent instead of DQNAgent
    opponent_type: str,
    num_episodes: int = 100,
    num_players: int = 2,
    num_dice: int = 2,
    dice_faces: int = 6,
    epsilon: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = False,
    render_every: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a reinforcement learning agent against a specific opponent type.
    
    Args:
        agent: RL agent to evaluate (DQN, PPO, etc.)
        opponent_type: Type of rule-based opponent
        num_episodes: Number of episodes to evaluate
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        epsilon: Exploration rate during evaluation (for DQN agents)
        seed: Random seed for reproducibility
        verbose: Whether to print detailed evaluation information
        render_every: Render every N episodes (None for no rendering)
        
    Returns:
        Dictionary with evaluation results
    """
    # Set up environment
    env = LiarsDiceEnvWrapper(
        num_players=num_players,
        num_dice=num_dice,
        dice_faces=dice_faces,
        seed=seed,
        rule_agent_types=[opponent_type]
    )
    
    # Set exploration rate for agents that use epsilon (e.g., DQN)
    original_epsilon = None
    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = epsilon
    
    # Set up action mapping if not already set
    if agent.action_to_game_action is None:
        agent.set_action_mapping(env.action_mapping)
    
    # Track results
    wins = 0
    rewards = []
    episode_lengths = []
    
    try:
        # Evaluation loop
        for i in tqdm(range(num_episodes), desc=f"Evaluating vs {opponent_type}", disable=not verbose):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Render the first state if requested
            if render_every is not None and i % render_every == 0:
                env.render()
                time.sleep(0.5)
            
            while not done:
                # Get valid actions
                valid_action_indices = env.get_valid_actions()
                valid_actions = [env.action_mapping[idx] for idx in valid_action_indices]
                
                # Select action
                action = agent.select_action(obs, valid_actions, training=False)
                
                # Find the index of the selected action
                action_idx = None
                for idx, valid_action in enumerate(valid_actions):
                    if agent._actions_equal(action, valid_action):
                        action_idx = valid_action_indices[idx]
                        break
                
                if action_idx is None:
                    raise ValueError(f"Selected action {action} not found in valid actions")
                
                # Take step in environment
                obs, reward, done, info = env.step(action_idx)
                total_reward += reward
                steps += 1
                
                # Render if requested
                if render_every is not None and i % render_every == 0:
                    env.render()
                    time.sleep(0.5)
            
            # Check if agent won
            dice_counts = info['dice_counts']
            if dice_counts[0] > 0 and all(dice_counts[j] == 0 for j in range(1, len(dice_counts))):
                wins += 1
            
            rewards.append(total_reward)
            episode_lengths.append(steps)
    finally:
        # Restore original epsilon if agent has it
        if original_epsilon is not None and hasattr(agent, 'epsilon'):
            agent.epsilon = original_epsilon
    
    # Calculate results
    win_rate = wins / num_episodes
    mean_reward = np.mean(rewards)
    mean_length = np.mean(episode_lengths)
    
    if verbose:
        print(f"Evaluation vs {opponent_type}:")
        print(f"  Win rate: {win_rate:.2f}")
        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Mean episode length: {mean_length:.2f}")
    
    return {
        'win_rate': win_rate,
        'mean_reward': mean_reward,
        'mean_episode_length': mean_length,
        'rewards': rewards,
        'episode_lengths': episode_lengths,
        'opponent_type': opponent_type,
        'num_episodes': num_episodes
    }


def evaluate_against_curriculum(
    agent: RLAgent,
    num_episodes_per_level: int = 50,
    num_players: int = 2,
    num_dice: int = 2,
    dice_faces: int = 6,
    epsilon: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = True,
    render_every: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a DQN agent against all levels of the curriculum.
    
    Args:
        agent: DQN agent to evaluate
        num_episodes_per_level: Number of episodes per curriculum level
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        epsilon: Exploration rate during evaluation
        seed: Random seed for reproducibility
        verbose: Whether to print detailed evaluation information
        render_every: Render every N episodes (None for no rendering)
        
    Returns:
        Dictionary with evaluation results for all levels
    """
    results = {}
    
    for opponent_type in CURRICULUM_LEVELS:
        result = evaluate_agent(
            agent=agent,
            opponent_type=opponent_type,
            num_episodes=num_episodes_per_level,
            num_players=num_players,
            num_dice=num_dice,
            dice_faces=dice_faces,
            epsilon=epsilon,
            seed=seed,
            verbose=verbose,
            render_every=render_every
        )
        results[opponent_type] = result
    
    # Calculate overall performance
    overall_win_rate = np.mean([r['win_rate'] for r in results.values()])
    overall_reward = np.mean([r['mean_reward'] for r in results.values()])
    
    if verbose:
        print("\nOverall Performance:")
        print(f"  Average win rate: {overall_win_rate:.2f}")
        print(f"  Average reward: {overall_reward:.2f}")
        
        print("\nWin rates by opponent:")
        for opponent, result in results.items():
            print(f"  vs {opponent}: {result['win_rate']:.2f}")
    
    results['overall'] = {
        'win_rate': overall_win_rate,
        'mean_reward': overall_reward
    }
    
    return results


def visualize_evaluation_results(
    results: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Visualize evaluation results.
    
    Args:
        results: Evaluation results from evaluate_against_curriculum
        save_path: Path to save the visualization (if None, show the plot)
    """
    # Extract opponent types and win rates
    opponents = [opp for opp in results.keys() if opp != 'overall']
    win_rates = [results[opp]['win_rate'] for opp in opponents]
    rewards = [results[opp]['mean_reward'] for opp in opponents]
    
    # Set up figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot win rates
    bars1 = ax1.bar(opponents, win_rates, color='skyblue')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('DQN Agent Performance Against Different Opponents')
    ax1.set_ylim(0, 1.0)
    
    # Add win rate values on top of bars
    for bar, win_rate in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{win_rate:.2f}', ha='center', va='bottom')
    
    # Plot mean rewards
    bars2 = ax2.bar(opponents, rewards, color='lightgreen')
    ax2.set_ylabel('Mean Reward')
    ax2.set_xlabel('Opponent Type')
    
    # Add reward values on top of bars
    for bar, reward in zip(bars2, rewards):
        height = bar.get_height()
        y_pos = height + 0.05 if height >= 0 else height - 0.5
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{reward:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def evaluate_self_play(
    agent: RLAgent,
    num_episodes: int = 50,
    num_dice: int = 2,
    dice_faces: int = 6,
    epsilon: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the agent in self-play (against a copy of itself).
    
    Args:
        agent: DQN agent to evaluate
        num_episodes: Number of episodes for evaluation
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        epsilon: Exploration rate during evaluation
        seed: Random seed for reproducibility
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with evaluation results
    """
    # Create a copy of the agent for the opponent
    device = agent.device
    opponent_agent = RLAgent(agent.obs_dim, agent.action_dim, device=device)
    opponent_agent.q_network.load_state_dict(agent.q_network.state_dict())
    opponent_agent.target_network.load_state_dict(agent.target_network.state_dict())
    opponent_agent.epsilon = epsilon
    
    # Set up the game
    game = LiarsDiceGame(
        num_players=2,
        num_dice=num_dice,
        dice_faces=dice_faces,
        seed=seed
    )
    
    observation_encoder = ObservationEncoder(
        num_players=2,
        num_dice=num_dice,
        dice_faces=dice_faces
    )
    
    # Set action mappings
    all_actions = []
    # Challenge action
    all_actions.append({'type': 'challenge'})
    # Bid actions
    max_quantity = 2 * num_dice
    for quantity in range(1, max_quantity + 1):
        for value in range(1, dice_faces + 1):
            all_actions.append({
                'type': 'bid',
                'quantity': quantity,
                'value': value
            })
    
    agent.set_action_mapping(all_actions)
    opponent_agent.set_action_mapping(all_actions)
    
    # Track results
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    episode_lengths = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating self-play", disable=not verbose):
        # Reset the game
        observations = game.reset(seed=seed + episode if seed else None)
        done = False
        steps = 0
        
        while not done:
            current_player = game.current_player
            obs = observation_encoder.encode(observations[current_player])
            
            valid_actions = game.get_valid_actions(current_player)
            
            # Select action based on which player's turn it is
            if current_player == 0:
                action = agent.select_action(obs, valid_actions, training=False)
            else:
                action = opponent_agent.select_action(obs, valid_actions, training=False)
            
            # Take step in environment
            observations, rewards, done, info = game.step(action)
            steps += 1
            
            # Prevent infinite games
            if steps > 100:
                break
        
        # Determine winner
        if done:
            dice_counts = info['dice_counts']
            if dice_counts[0] > 0 and dice_counts[1] == 0:
                agent1_wins += 1
            elif dice_counts[1] > 0 and dice_counts[0] == 0:
                agent2_wins += 1
            else:
                draws += 1
        else:
            # If game hit step limit, count as a draw
            draws += 1
        
        episode_lengths.append(steps)
    
    # Calculate statistics
    agent1_win_rate = agent1_wins / num_episodes
    agent2_win_rate = agent2_wins / num_episodes
    draw_rate = draws / num_episodes
    mean_length = np.mean(episode_lengths)
    
    if verbose:
        print(f"Self-play Evaluation ({num_episodes} episodes):")
        print(f"  Agent 1 win rate: {agent1_win_rate:.2f}")
        print(f"  Agent 2 win rate: {agent2_win_rate:.2f}")
        print(f"  Draw rate: {draw_rate:.2f}")
        print(f"  Mean episode length: {mean_length:.2f}")
    
    return {
        'agent1_win_rate': agent1_win_rate,
        'agent2_win_rate': agent2_win_rate,
        'draw_rate': draw_rate,
        'mean_episode_length': mean_length,
        'episode_lengths': episode_lengths,
        'num_episodes': num_episodes
    }