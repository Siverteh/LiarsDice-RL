"""
Evaluation utilities for Liar's Dice agents.

This module provides functions for evaluating and comparing
different agents through direct matchups and tournaments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import torch

from environment.game import LiarsDiceGame
from agents.base_agent import BaseAgent
from .utils import create_agent_mapping


def evaluate_agents(
    agent1: BaseAgent,
    agent2: BaseAgent,
    num_games: int = 100,
    num_dice: int = 5,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Evaluate two agents against each other.
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_games: Number of games to play
        num_dice: Number of dice per player
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (agent1 win rate, agent2 win rate, average game length)
    """
    # Set player IDs
    agent1.player_id = 0
    agent2.player_id = 1
    
    # Create environment
    game = LiarsDiceGame(num_players=2, num_dice=num_dice, seed=seed)
    
    # Initialize metrics
    wins = [0, 0]
    total_steps = 0
    
    # Play games
    for game_num in range(num_games):
        # Reset game
        observations = game.reset(seed=seed + game_num if seed is not None else None)
        
        # Reset agents if they have reset method
        if hasattr(agent1, 'reset'):
            agent1.reset()
        if hasattr(agent2, 'reset'):
            agent2.reset()
        
        done = False
        steps = 0
        
        # Play until game is done
        while not done:
            # Get current player
            current_player = game.current_player
            
            # Get valid actions
            valid_actions = game.get_valid_actions(current_player)
            
            # Get agent for current player
            agent = agent1 if current_player == 0 else agent2
            
            # Select action
            with torch.no_grad():  # Disable gradient computation for evaluation
                action = agent.act(observations[current_player], valid_actions)
            
            # Take action
            observations, rewards, done, info = game.step(action)
            steps += 1
        
        # Record winner
        winner = np.argmax(game.dice_counts)
        wins[winner] += 1
        total_steps += steps
    
    # Calculate metrics
    win_rate1 = wins[0] / num_games
    win_rate2 = wins[1] / num_games
    avg_steps = total_steps / num_games
    
    return win_rate1, win_rate2, avg_steps


def tournament(
    agents: List[BaseAgent],
    agent_names: List[str],
    num_games: int = 50,
    num_dice: int = 5,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run a round-robin tournament between agents.
    
    Args:
        agents: List of agents to compete
        agent_names: Names of the agents
        num_games: Number of games for each matchup
        num_dice: Number of dice per player
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        
    Returns:
        DataFrame with tournament results
    """
    n_agents = len(agents)
    
    # Initialize results matrix
    results = np.zeros((n_agents, n_agents))
    steps = np.zeros((n_agents, n_agents))
    
    # Run all matchups
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            print(f"Matchup: {agent_names[i]} vs {agent_names[j]}")
            
            win_rate_i, win_rate_j, avg_steps = evaluate_agents(
                agents[i], agents[j], num_games, num_dice, seed
            )
            
            results[i, j] = win_rate_i
            results[j, i] = win_rate_j
            steps[i, j] = steps[j, i] = avg_steps
            
            print(f"  {agent_names[i]} win rate: {win_rate_i:.2f}")
            print(f"  {agent_names[j]} win rate: {win_rate_j:.2f}")
            print(f"  Average game length: {avg_steps:.1f} steps")
    
    # Create DataFrames
    results_df = pd.DataFrame(
        results, 
        index=agent_names, 
        columns=agent_names
    )
    
    steps_df = pd.DataFrame(
        steps, 
        index=agent_names, 
        columns=agent_names
    )
    
    # Calculate Elo ratings
    elo_ratings = calculate_elo(results_df, agent_names)
    
    # Save results if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrames
        results_df.to_csv(os.path.join(output_dir, 'win_rates.csv'))
        steps_df.to_csv(os.path.join(output_dir, 'avg_steps.csv'))
        
        # Create a heatmap for win rates
        plt.figure(figsize=(10, 8))
        plt.imshow(results, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    plt.text(j, i, f"{results[i, j]:.2f}", 
                            ha="center", va="center", 
                            color="black" if 0.3 <= results[i, j] <= 0.7 else "white")
        
        plt.colorbar(label='Win Rate')
        plt.title('Tournament Results (Row vs Column)')
        plt.xticks(range(n_agents), agent_names, rotation=45)
        plt.yticks(range(n_agents), agent_names)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tournament_results.png'))
        plt.close()
        
        # Plot Elo ratings
        plt.figure(figsize=(10, 6))
        bars = plt.bar(agent_names, [elo_ratings[name] for name in agent_names])
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                    f"{elo_ratings[agent_names[i]]:.0f}",
                    ha='center', va='bottom')
        plt.title('Elo Ratings')
        plt.ylabel('Elo Rating')
        plt.ylim(top=max(elo_ratings.values()) + 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'elo_ratings.png'))
        plt.close()
    
    # Return results DataFrame
    return results_df


def calculate_elo(results_df: pd.DataFrame, agent_names: List[str], K: int = 32) -> Dict[str, float]:
    """
    Calculate Elo ratings based on tournament results.
    
    Args:
        results_df: DataFrame with win rates
        agent_names: Names of the agents
        K: Elo K-factor
        
    Returns:
        Dictionary mapping agent names to Elo ratings
    """
    n_agents = len(agent_names)
    elo_ratings = {name: 1000 for name in agent_names}
    
    # Simulate a series of games between each pair of agents
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            name_i = agent_names[i]
            name_j = agent_names[j]
            
            win_rate_i = results_df.loc[name_i, name_j]
            win_rate_j = results_df.loc[name_j, name_i]
            
            # Skip if no games were played
            if win_rate_i == 0 and win_rate_j == 0:
                continue
            
            # Calculate expected win probabilities based on current Elo
            r_i = 10 ** (elo_ratings[name_i] / 400)
            r_j = 10 ** (elo_ratings[name_j] / 400)
            
            e_i = r_i / (r_i + r_j)
            e_j = r_j / (r_i + r_j)
            
            # Update Elo ratings
            elo_ratings[name_i] += K * (win_rate_i - e_i)
            elo_ratings[name_j] += K * (win_rate_j - e_j)
    
    return elo_ratings


def analyze_agent_behavior(
    agent: BaseAgent,
    opponent: BaseAgent,
    num_games: int = 20,
    num_dice: int = 5,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze an agent's behavior in terms of bidding and challenging patterns.
    
    Args:
        agent: Agent to analyze
        opponent: Opponent agent
        num_games: Number of games to play
        num_dice: Number of dice per player
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        
    Returns:
        Dictionary with behavior statistics
    """
    # Set player IDs
    agent.player_id = 0
    opponent.player_id = 1
    
    # Create environment
    game = LiarsDiceGame(num_players=2, num_dice=num_dice, seed=seed)
    
    # Tracking metrics
    bid_values = []  # Bid values chosen
    bid_quantities = []  # Bid quantities chosen
    challenge_frequencies = []  # Frequency of challenges
    bluff_frequencies = []  # Frequency of bluffing
    
    # Play games
    for game_num in range(num_games):
        # Reset game
        observations = game.reset(seed=seed + game_num if seed is not None else None)
        
        # Reset agents if they have reset method
        if hasattr(agent, 'reset'):
            agent.reset()
        if hasattr(opponent, 'reset'):
            opponent.reset()
        
        done = False
        game_history = []
        
        # Play until game is done
        while not done:
            # Get current player
            current_player = game.current_player
            
            # Get valid actions
            valid_actions = game.get_valid_actions(current_player)
            
            # Get agent for current player
            current_agent = agent if current_player == 0 else opponent
            
            # Select action
            with torch.no_grad():  # Disable gradient computation for evaluation
                action = current_agent.act(observations[current_player], valid_actions)
            
            # Record action if it's the agent we're analyzing
            if current_player == agent.player_id:
                game_history.append({
                    'state': observations[current_player],
                    'action': action,
                    'dice': game.dice[current_player, :game.dice_counts[current_player]].copy()
                })
            
            # Take action
            observations, rewards, done, info = game.step(action)
        
        # Analyze game history
        for i, entry in enumerate(game_history):
            if entry['action']['type'] == 'bid':
                bid_values.append(entry['action']['value'])
                bid_quantities.append(entry['action']['quantity'])
                
                # Check if this was a bluff
                dice = entry['dice']
                bid_value = entry['action']['value']
                bid_quantity = entry['action']['quantity']
                
                # Count dice matching the bid value
                matching_dice = np.sum(dice == bid_value)
                
                # If the bid quantity exceeds what the agent has, it's a bluff
                if bid_quantity > matching_dice + (len(dice) * 0.16):  # 0.16 = 1/6 probability
                    bluff_frequencies.append(1)
                else:
                    bluff_frequencies.append(0)
            
            elif entry['action']['type'] == 'challenge':
                challenge_frequencies.append(1)
            else:
                challenge_frequencies.append(0)
    
    # Calculate statistics
    stats = {
        'avg_bid_value': np.mean(bid_values) if bid_values else 0,
        'avg_bid_quantity': np.mean(bid_quantities) if bid_quantities else 0,
        'challenge_rate': np.mean(challenge_frequencies) if challenge_frequencies else 0,
        'bluff_rate': np.mean(bluff_frequencies) if bluff_frequencies else 0,
        'bid_value_distribution': np.bincount(bid_values, minlength=7)[1:] / len(bid_values) if bid_values else np.zeros(6),
    }
    
    # Plot statistics if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot bid value distribution
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, 7), stats['bid_value_distribution'])
        plt.title('Bid Value Distribution')
        plt.xlabel('Dice Value')
        plt.ylabel('Frequency')
        plt.xticks(range(1, 7))
        plt.savefig(os.path.join(output_dir, 'bid_value_distribution.png'))
        plt.close()
        
        # Plot other statistics
        plt.figure(figsize=(10, 6))
        metrics = ['challenge_rate', 'bluff_rate']
        values = [stats[m] for m in metrics]
        plt.bar(metrics, values)
        plt.title('Agent Behavior Metrics')
        plt.ylabel('Rate')
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, 'behavior_metrics.png'))
        plt.close()
    
    return stats    