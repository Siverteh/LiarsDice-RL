"""
Self-play training for reinforcement learning agents in Liar's Dice.

This module implements a self-play training approach where an agent learns
by playing against copies of itself, with periodic evaluation against rule-based agents.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import deque

from agents.base_agent import RLAgent
from agents.agent_factory import create_agent
from agents.rule_agent import CURRICULUM_LEVELS, create_agent as create_rule_agent
from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from training.environment_wrapper import LiarsDiceEnvWrapper
from training.train import train_episode, evaluate_agent
from training.evaluate import evaluate_against_curriculum, visualize_evaluation_results
from training.utils import setup_logger, save_training_data, plot_training_results

# Configuration presets for different game setups
CONFIG_PRESETS = {
    # 2 players, 3 dice, good for initial learning
    'basic': {
        'num_players': 2,
        'num_dice': 3,
        'self_play_episodes': 100000,
        'network_size': [256, 128, 64],
        'learning_rate': 0.0005,
        'win_rate_threshold': 0.6
    },
    # 2 players, 5 dice, standard game
    'standard': {
        'num_players': 2,
        'num_dice': 5,
        'self_play_episodes': 250000,
        'network_size': [1024, 512, 256, 128, 64],
        'learning_rate': 0.0002,
        'win_rate_threshold': 0.6
    },
    # 4 players, 5 dice, complex game
    'advanced': {
        'num_players': 4,
        'num_dice': 5,
        'self_play_episodes': 300000,
        'network_size': [2048, 1024, 512, 256],
        'learning_rate': 0.00015,
        'win_rate_threshold': 0.55
    }
}

def train_self_play(
    # Base settings
    agent_type: str = 'ppo',
    preset: str = 'standard',
    results_path: str = 'results',
    seed: Optional[int] = None,
    device: str = 'auto',
    render_training: bool = False,
    
    # Game setup - override preset values if specified
    num_players: Optional[int] = None,
    num_dice: Optional[int] = None,
    dice_faces: int = 6,
    
    # Training schedule
    self_play_episodes: Optional[int] = None,
    evaluation_opponents: List[str] = ['conservative', 'bluff_punisher', 'adaptive', 'optimal'],
    
    # Agent configuration
    learning_rate: Optional[float] = None,
    network_size: Optional[List[int]] = None,
    custom_agent_config: Optional[Dict[str, Any]] = None,
    
    # Training control
    checkpoint_frequency: int = 1000,
    evaluation_frequency: int = 1000,
    enable_early_stopping: bool = True,  # Changed to True by default
    win_rate_threshold: Optional[float] = None,
    early_stopping_patience: int = 5,
    
    # Self-play settings
    opponent_pool_size: int = 5,
    update_pool_frequency: int = 5000,  # How often to update the pool of opponents
    newest_model_freq: float = 0.7      # Frequency of playing against the newest model
) -> Tuple[RLAgent, Dict[str, Any]]:
    """
    Train an agent using self-play, where it learns by playing against copies of itself.
    
    Args:
        agent_type: Type of agent to train ('dqn' or 'ppo')
        preset: Configuration preset ('basic', 'standard', or 'advanced')
        results_path: Directory to save results, checkpoints, and logs
        seed: Random seed for reproducibility
        device: Device to run on ('cpu', 'cuda', or 'auto')
        render_training: Whether to render gameplay during training
        
        # Game settings
        num_players: Number of players in the game (overrides preset if specified)
        num_dice: Number of dice per player (overrides preset if specified)
        dice_faces: Number of faces on each die
        
        # Training schedule
        self_play_episodes: Total episodes for self-play training
        evaluation_opponents: List of rule-based agents to evaluate against
        
        # Agent configuration
        learning_rate: Learning rate for agent (overrides preset if specified)
        network_size: Neural network architecture (overrides preset if specified)
        custom_agent_config: Additional agent-specific parameters
        
        # Training control
        checkpoint_frequency: How often to save checkpoints (in episodes)
        evaluation_frequency: How often to evaluate the agent (in episodes)
        enable_early_stopping: Whether to stop training early when win rate is high (default True)
        win_rate_threshold: Win rate threshold for early stopping (overrides preset if specified)
        early_stopping_patience: Number of evaluations above threshold to trigger early stopping
        
        # Self-play settings
        opponent_pool_size: Number of past model versions to keep in the opponent pool
        update_pool_frequency: How often to update the pool of opponents
        newest_model_freq: Frequency of playing against the newest model
        
    Returns:
        Tuple of (trained_agent, training_results)
    """
    # Apply preset configuration
    preset_config = CONFIG_PRESETS[preset]
    
    # Apply values from preset unless explicitly overridden
    _num_players = num_players if num_players is not None else preset_config['num_players']
    _num_dice = num_dice if num_dice is not None else preset_config['num_dice']
    _self_play_episodes = self_play_episodes if self_play_episodes is not None else preset_config['self_play_episodes']
    _win_rate_threshold = win_rate_threshold if win_rate_threshold is not None else preset_config['win_rate_threshold']
    _learning_rate = learning_rate if learning_rate is not None else preset_config['learning_rate']
    _network_size = network_size if network_size is not None else preset_config['network_size']
    
    # Set up paths for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"selfplay_{agent_type}_{preset}_{_num_players}p{_num_dice}d_{timestamp}"
    base_path = os.path.join(results_path, run_name)
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'logs')
    
    # Create directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('selfplay', os.path.join(log_dir, 'selfplay.log'))
    logger.info(f"Starting self-play training for Liar's Dice with {agent_type} agent")
    logger.info(f"Game setup: {_num_players} players, {_num_dice} dice, {dice_faces} faces")
    logger.info(f"Self-play: {_self_play_episodes} episodes")
    
    # Resolve device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
    
    # Create test environment to get dimensions
    env = LiarsDiceEnvWrapper(
        num_players=_num_players,
        num_dice=_num_dice,
        dice_faces=dice_faces,
        seed=seed
    )
    
    obs_shape = env.get_observation_shape()
    obs_dim = obs_shape[0]
    action_dim = env.get_action_dim()
    logger.info(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # Create agent configuration
    agent_config = {
        'learning_rate': _learning_rate,
        'hidden_dims': _network_size,
        'device': device
    }
    
    # Add agent-specific parameters
    if agent_type.lower() == 'dqn':
        agent_config.update({
            'epsilon_start': 0.5,  # Lower initial exploration for self-play
            'epsilon_end': 0.05,
            'epsilon_decay': 0.99995  # Slower decay for more stable learning
        })
    elif agent_type.lower() == 'ppo':
        agent_config.update({
            'entropy_coef': 0.04,        # Higher entropy for more exploration
            'entropy_min': 0.01,         # Minimum entropy coefficient
            'update_frequency': 2048,    # Update frequency
            'gae_lambda': 0.95,          # Standard GAE parameter
            'max_grad_norm': 0.5,        # Clip gradients
            'policy_clip': 0.15          # Tighter clipping
        })
    
    # Apply any custom agent config
    if custom_agent_config:
        agent_config.update(custom_agent_config)
    
    # Create primary learning agent
    primary_agent = create_agent(
        agent_type=agent_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config
    )
    
    # Set action mapping
    primary_agent.set_action_mapping(env.action_mapping)
    logger.info(f"Created {agent_type} agent with {len(env.action_mapping)} actions")
    
    # Setup opponent pool for self-play
    opponent_pool = []  # Will contain (path, model_version, episodes_trained)
    latest_opponent_path = None
    
    # Tracking variables
    best_eval_reward = float('-inf')
    best_model_path = None
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_win_rates = []
    early_stopping_counter = 0
    high_win_rate_count = 0
    
    # Training loop
    logger.info("Starting self-play training...")
    start_time = time.time()
    
    for episode in range(1, _self_play_episodes + 1):
        # Periodically update the opponent pool
        if episode % update_pool_frequency == 1 or not opponent_pool:
            # Save current model state
            model_path = os.path.join(checkpoint_dir, f"model_ep{episode}")
            primary_agent.save(model_path)
            
            # Add to opponent pool
            opponent_pool.append((model_path, episode // update_pool_frequency, episode))
            latest_opponent_path = model_path
            
            # Keep pool at desired size
            if len(opponent_pool) > opponent_pool_size:
                # Remove oldest model (except for the very first one)
                if len(opponent_pool) > 2:
                    opponent_pool.pop(1)  # Keep index 0 (first) and remove second oldest
                    
            logger.info(f"Updated opponent pool. Current size: {len(opponent_pool)}")
        
        # Select opponent from pool
        if random.random() < newest_model_freq and latest_opponent_path:
            # Use the latest model
            opponent_path = latest_opponent_path
        else:
            # Use a random model from the pool
            opponent_path = random.choice(opponent_pool)[0]
        
        # Create opponent agent
        opponent_agent = create_agent(
            agent_type=agent_type,
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=agent_config
        )
        opponent_agent.load(opponent_path)
        
        # Create self-play environment
        selfplay_env = LiarsDiceEnvWrapper(
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            seed=seed + episode if seed else None,
            rl_agent_as_opponent=opponent_agent
        )
        
        # Set action mappings for agents
        primary_agent.set_action_mapping(selfplay_env.action_mapping)
        opponent_agent.set_action_mapping(selfplay_env.action_mapping)
        
        # Train for one episode
        reward, episode_length, _ = train_episode(
            env=selfplay_env,
            agent=primary_agent,
            evaluation=False,
            render=render_training and episode % 100 == 0,
            reward_shaping=True
        )
        
        episode_rewards.append(reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            time_taken = time.time() - start_time
            logger.info(f"Episode {episode}/{_self_play_episodes} | "
                       f"Avg Reward: {avg_reward:.2f} | "
                       f"Avg Length: {avg_length:.2f} | "
                       f"Time: {time_taken:.2f}s")
        
        # Periodic evaluation against rule-based agents
        if episode % evaluation_frequency == 0:
            logger.info(f"Evaluating agent at episode {episode}...")
            
            eval_results = {}
            total_wins = 0
            total_games = 0
            
            # Evaluate against each opponent type
            for opponent_type in evaluation_opponents:
                # Create evaluation environment
                eval_env = LiarsDiceEnvWrapper(
                    num_players=_num_players,
                    num_dice=_num_dice,
                    dice_faces=dice_faces,
                    seed=seed + 10000 if seed else None,
                    rule_agent_types=[opponent_type]
                )
                
                primary_agent.set_action_mapping(eval_env.action_mapping)
                
                # Run evaluation
                num_eval_episodes = 100
                eval_reward = 0
                wins = 0
                
                for _ in range(num_eval_episodes):
                    episode_reward, _, _ = train_episode(eval_env, primary_agent, evaluation=True)
                    eval_reward += episode_reward
                    if episode_reward > 0:
                        wins += 1
                
                win_rate = wins / num_eval_episodes
                avg_reward = eval_reward / num_eval_episodes
                
                total_wins += wins
                total_games += num_eval_episodes
                
                eval_results[opponent_type] = {
                    'win_rate': win_rate,
                    'avg_reward': avg_reward
                }
                
                logger.info(f"  vs {opponent_type}: Win Rate = {win_rate:.2f}, Reward = {avg_reward:.2f}")
            
            # Calculate overall win rate
            overall_win_rate = total_wins / total_games
            overall_eval_reward = sum(r['avg_reward'] for r in eval_results.values()) / len(eval_results)
            
            logger.info(f"Overall Evaluation: Win Rate = {overall_win_rate:.2f}, Reward = {overall_eval_reward:.2f}")
            
            eval_rewards.append(overall_eval_reward)
            eval_win_rates.append(overall_win_rate)
            
            # Check if this is the best model so far
            if overall_eval_reward > best_eval_reward:
                best_eval_reward = overall_eval_reward
                best_model_path = os.path.join(checkpoint_dir, f"best_model_ep{episode}")
                primary_agent.save(best_model_path)
                logger.info(f"New best model saved with reward {best_eval_reward:.2f}")
            
            # Check early stopping if enabled
            if enable_early_stopping and overall_win_rate >= _win_rate_threshold:
                high_win_rate_count += 1
                logger.info(f"Win rate {overall_win_rate:.2f} above threshold {_win_rate_threshold:.2f} "
                           f"({high_win_rate_count}/{early_stopping_patience})")
                if high_win_rate_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at episode {episode}")
                    break
            else:
                high_win_rate_count = 0
            
            # Save full evaluation results
            eval_data = {
                'episode': episode,
                'results': eval_results,
                'overall_win_rate': overall_win_rate,
                'overall_reward': overall_eval_reward
            }
            
            with open(os.path.join(log_dir, f"eval_ep{episode}.json"), 'w') as f:
                json.dump(eval_data, f, indent=2)
        
        # Save checkpoint
        if episode % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}")
            primary_agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint at episode {episode}")
            
            # Save training progress
            training_data = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'eval_rewards': eval_rewards,
                'eval_win_rates': eval_win_rates
            }
            
            with open(os.path.join(log_dir, "training_progress.json"), 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # Plot training progress
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot rewards
            ax1.plot(range(1, len(episode_rewards) + 1), episode_rewards, alpha=0.3, color='blue')
            ax1.plot(range(1, len(episode_rewards) + 1, 100), 
                    [np.mean(episode_rewards[max(0, i-100):i]) for i in range(100, len(episode_rewards) + 1, 100)],
                    color='blue', linewidth=2)
            ax1.set_title('Training Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
            
            # Plot eval win rates
            if eval_win_rates:  # Only plot if we have evaluation data
                eval_episodes = range(evaluation_frequency, episode + 1, evaluation_frequency)
                ax2.plot(eval_episodes, eval_win_rates, marker='o', color='green', linewidth=2)
                ax2.set_title('Evaluation Win Rates')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Win Rate')
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "training_progress.png"))
            plt.close()
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"Self-play training completed after {episode} episodes in {total_time:.2f} seconds")
    
    # Final evaluation against all rule-based agents
    logger.info("\n=== Final Comprehensive Evaluation ===")
    final_eval_results = evaluate_against_curriculum(
        agent=primary_agent,
        num_episodes_per_level=250,
        num_players=_num_players,
        num_dice=_num_dice,
        dice_faces=dice_faces,
        epsilon=0.05,
        seed=seed,
        verbose=True
    )
    
    # Save and visualize final evaluation
    with open(os.path.join(log_dir, "final_evaluation.json"), 'w') as f:
        json.dump(final_eval_results, f, indent=2)
    
    vis_path = os.path.join(log_dir, "final_evaluation.png")
    visualize_evaluation_results(final_eval_results, save_path=vis_path)
    logger.info(f"Saved final evaluation visualization to {vis_path}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    primary_agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # If we found a best model, load it
    if best_model_path and os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        primary_agent.load(best_model_path)
    
    # Save model to models directory with metadata
    models_dir = os.path.join(results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model name with performance info
    overall_win_rate = final_eval_results['overall']['win_rate']
    model_name = f"selfplay_{agent_type}_{_num_players}p{_num_dice}d_{overall_win_rate:.2f}wr_{timestamp}"
    model_path = os.path.join(models_dir, model_name)
    primary_agent.save(model_path)
    
    # Save model metadata
    model_metadata = {
        'timestamp': timestamp,
        'agent_type': agent_type,
        'num_players': _num_players,
        'num_dice': _num_dice,
        'dice_faces': dice_faces,
        'win_rates': {k: v['win_rate'] for k, v in final_eval_results.items() if k != 'overall'},
        'overall_win_rate': overall_win_rate,
        'network_size': _network_size,
        'training_parameters': {
            'self_play_episodes': _self_play_episodes,
            'opponent_pool_size': opponent_pool_size
        }
    }
    
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Saved final model with metadata to {model_path}")
    
    # Prepare and return training results
    training_results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'eval_win_rates': eval_win_rates,
        'final_evaluation': final_eval_results,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
        'training_time': total_time,
        'parameters': {
            'agent_type': agent_type,
            'preset': preset,
            'num_players': _num_players,
            'num_dice': _num_dice,
            'dice_faces': dice_faces,
            'self_play_episodes': _self_play_episodes,
            'network_size': _network_size,
            'win_rate_threshold': _win_rate_threshold
        }
    }
    
    return primary_agent, training_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-play training for Liar's Dice")
    parser.add_argument('--agent', type=str, default='ppo', choices=['dqn', 'ppo'],
                        help='Type of agent to train')
    parser.add_argument('--preset', type=str, default='standard', choices=['basic', 'standard', 'advanced'],
                        help='Configuration preset to use')
    parser.add_argument('--path', type=str, default='results/self_play_learning/ppo_5dice', help='Base path for results')
    parser.add_argument('--episodes', type=int, default=None, help='Number of self-play episodes')
    parser.add_argument('--pool-size', type=int, default=5, help='Size of opponent pool')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping (enabled by default)')
    
    args = parser.parse_args()
    
    # Run training with parsed arguments
    train_self_play(
        agent_type=args.agent,
        preset=args.preset,
        results_path=args.path,
        self_play_episodes=args.episodes,
        opponent_pool_size=args.pool_size,
        seed=args.seed,
        render_training=args.render,
        enable_early_stopping=not args.no_early_stopping,  # Enable by default, disable with flag
        evaluation_frequency=1000
    )