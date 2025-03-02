"""
Curriculum learning for reinforcement learning agents in Liar's Dice.

This module implements a model-agnostic curriculum learning approach that progressively
trains agents against opponents of increasing difficulty.
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
from training.train import train_agent, train_self_play, evaluate_agent
from training.evaluate import evaluate_against_curriculum, visualize_evaluation_results
from training.utils import (
    setup_logger, 
    save_training_data, 
    load_training_data,
    plot_training_results
)


# Configuration presets for different game setups
CONFIG_PRESETS = {
    # 2 players, 3 dice, good for initial learning
    'basic': {
        'num_players': 2,
        'num_dice': 3,
        'curriculum_episodes': 100000,
        'self_play_episodes': 10000,
        'network_size': [256, 128, 64],
        'learning_rate': 0.0005,
        'win_rate_threshold': 0.75
    },
    # 2 players, 5 dice, standard game
    'standard': {
        'num_players': 2,
        'num_dice': 5,
        'curriculum_episodes': 50000,
        'self_play_episodes': 15000,
        'network_size': [512, 256, 128, 64],
        'learning_rate': 0.0003,
        'win_rate_threshold': 0.70
    },
    # 4 players, 5 dice, complex game
    'advanced': {
        'num_players': 4,
        'num_dice': 5,
        'curriculum_episodes': 100000,
        'self_play_episodes': 30000,
        'network_size': [1024, 512, 256, 128],
        'learning_rate': 0.0002,
        'win_rate_threshold': 0.65
    }
}


def train_curriculum(
    # Base settings
    agent_type: str = 'dqn',
    preset: str = 'basic',
    results_path: str = 'results',
    seed: Optional[int] = None,
    device: str = 'auto',
    render_training: bool = False,
    
    # Game setup - override preset values if specified
    num_players: Optional[int] = None,
    num_dice: Optional[int] = None,
    dice_faces: int = 6,
    
    # Training schedule
    curriculum_episodes: Optional[int] = None,
    self_play_episodes: Optional[int] = None,
    curriculum_distribution: Dict[str, float] = None,  # Maps level names to percentages
    enable_self_play: bool = True,
    enable_remedial: bool = True,
    
    # Agent configuration
    learning_rate: Optional[float] = None,
    network_size: Optional[List[int]] = None,
    custom_agent_config: Optional[Dict[str, Any]] = None,
    
    # Training control
    checkpoint_frequency: int = 1000,
    evaluation_frequency: int = 500,
    enable_early_stopping: bool = True,
    win_rate_threshold: Optional[float] = None,
    early_stopping_patience: int = 3,
    
    # Exploration settings
    initial_epsilon: float = 0.9,
    min_epsilon: float = 0.05,
    epsilon_decay_per_level: float = 0.2  # How much to reduce epsilon per level
) -> Tuple[RLAgent, Dict[str, Any]]:
    """
    Train an agent using curriculum learning with progressively more difficult opponents.
    
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
        curriculum_episodes: Total episodes for curriculum phase (overrides preset if specified)
        self_play_episodes: Episodes for self-play phase (overrides preset if specified)
        curriculum_distribution: Custom distribution of episodes across levels (default: progressive)
        enable_self_play: Whether to run self-play after curriculum
        enable_remedial: Whether to run remedial training for weak areas
        
        # Agent configuration
        learning_rate: Learning rate for agent (overrides preset if specified)
        network_size: Neural network architecture (overrides preset if specified)
        custom_agent_config: Additional agent-specific parameters
        
        # Training control
        checkpoint_frequency: How often to save checkpoints (in episodes)
        evaluation_frequency: How often to evaluate the agent (in episodes)
        enable_early_stopping: Whether to stop training early on a level if win rate is high
        win_rate_threshold: Win rate threshold for early stopping (overrides preset if specified)
        early_stopping_patience: Number of evaluations above threshold to trigger early stopping
        
        # Exploration settings
        initial_epsilon: Starting exploration rate (for DQN)
        min_epsilon: Minimum exploration rate
        epsilon_decay_per_level: How much to reduce epsilon per curriculum level
        
    Returns:
        Tuple of (trained_agent, training_results)
    """
    # Apply preset configuration
    preset_config = CONFIG_PRESETS[preset]
    
    # Apply values from preset unless explicitly overridden
    _num_players = num_players if num_players is not None else preset_config['num_players']
    _num_dice = num_dice if num_dice is not None else preset_config['num_dice']
    _curriculum_episodes = curriculum_episodes if curriculum_episodes is not None else preset_config['curriculum_episodes']
    _self_play_episodes = self_play_episodes if self_play_episodes is not None else preset_config['self_play_episodes']
    _win_rate_threshold = win_rate_threshold if win_rate_threshold is not None else preset_config['win_rate_threshold']
    _learning_rate = learning_rate if learning_rate is not None else preset_config['learning_rate']
    _network_size = network_size if network_size is not None else preset_config['network_size']
    
    # Set up paths for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{agent_type}_{preset}_{_num_players}p{_num_dice}d_{timestamp}"
    base_path = os.path.join(results_path, run_name)
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'logs')
    
    # Create directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('curriculum', os.path.join(log_dir, 'curriculum.log'))
    logger.info(f"Starting curriculum learning for Liar's Dice with {agent_type} agent")
    logger.info(f"Game setup: {_num_players} players, {_num_dice} dice, {dice_faces} faces")
    logger.info(f"Curriculum: {_curriculum_episodes} episodes, Self-play: {_self_play_episodes} episodes")
    
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
            'epsilon_start': initial_epsilon,
            'epsilon_end': min_epsilon,
            'epsilon_decay': 0.99995  # Slower decay for more stable learning
        })
    elif agent_type.lower() == 'ppo':
        agent_config.update({
            'entropy_coef': 0.02,  # Higher entropy for more exploration
            'update_frequency': 2048,  # Update less frequently for more stable learning
            'gae_lambda': 0.95  # Standard GAE parameter
        })
    
    # Apply any custom agent config
    if custom_agent_config:
        agent_config.update(custom_agent_config)
    
    # Create agent
    agent = create_agent(
        agent_type=agent_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config
    )
    
    # Set action mapping
    agent.set_action_mapping(env.action_mapping)
    logger.info(f"Created {agent_type} agent with {len(env.action_mapping)} actions")
    
    # Generate curriculum distribution
    if curriculum_distribution is None:
        # Default progressive distribution (more episodes for harder opponents)
        curriculum_distribution = {
            'random': 0.05,       # 5%
            'naive': 0.05,        # 10%
            'conservative': 0.3, # 15%
            'aggressive': 0.05,   # 20%
            'strategic': 0.35,    # 25%
            'adaptive': 0.25      # 25%
        }
    
    # Convert distribution to episode counts
    episode_distribution = {}
    remaining_episodes = _curriculum_episodes
    
    # First pass: allocate based on percentages
    for level, percentage in curriculum_distribution.items():
        episode_distribution[level] = int(_curriculum_episodes * percentage)
        remaining_episodes -= episode_distribution[level]
    
    # Add any remaining episodes to the most challenging level
    if remaining_episodes > 0:
        episode_distribution['adaptive'] += remaining_episodes
    
    logger.info("Curriculum schedule:")
    for level, episodes in episode_distribution.items():
        logger.info(f"  {level}: {episodes} episodes ({episodes/_curriculum_episodes:.1%})")
    
    # Track training progress and models
    all_results = {}
    best_models = {}
    best_overall_model = None
    best_overall_winrate = 0.0
    current_epsilon = initial_epsilon
    
    # Curriculum learning loop
    for level_idx, level_name in enumerate(CURRICULUM_LEVELS):
        episodes_for_level = episode_distribution.get(level_name, 0)
        if episodes_for_level <= 0:
            continue
            
        logger.info(f"\n=== Training against {level_name} ({episodes_for_level} episodes) ===")
        
        # Set exploration rate for this level (DQN only)
        if agent_type.lower() == 'dqn':
            # Gradually reduce exploration as opponents get harder
            if level_idx > 0:
                current_epsilon = max(min_epsilon, current_epsilon - epsilon_decay_per_level)
                
            logger.info(f"Setting epsilon to {current_epsilon:.2f} for level {level_name}")
            agent.epsilon = current_epsilon
            
        # Create environment with current opponent
        env = LiarsDiceEnvWrapper(
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            seed=seed,
            rule_agent_types=[level_name]
        )
        
        # Make sure action mapping is set
        agent.set_action_mapping(env.action_mapping)
        
        # Adjust learning rate for harder opponents if using DQN
        if level_idx >= 3 and agent_type.lower() == 'dqn' and hasattr(agent, 'optimizer'):
            # Reduce learning rate by 30% for harder opponents
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.7
            logger.info(f"Adjusted learning rate to {param_group['lr']:.6f} for level {level_name}")
        
        # Best model tracking for this level
        best_level_winrate = 0.0
        best_level_model_path = None
        above_threshold_count = 0
        
        # Define callback for early stopping and model saving
        def checkpoint_callback(episode, data, current_agent):
            nonlocal best_level_winrate, best_level_model_path, best_overall_winrate, best_overall_model
            nonlocal above_threshold_count
            
            if 'last_win_rate' in data:
                current_winrate = data['last_win_rate']
                
                # Update best model for this level if better
                if current_winrate > best_level_winrate:
                    best_level_winrate = current_winrate
                    best_level_model_path = os.path.join(checkpoint_dir, f"best_{level_name}")
                    current_agent.save(best_level_model_path)
                    logger.info(f"New best model for {level_name}: {current_winrate:.2f} win rate")
                
                # Update best overall model if better
                if current_winrate > best_overall_winrate:
                    best_overall_winrate = current_winrate
                    best_overall_model = os.path.join(checkpoint_dir, "best_overall")
                    current_agent.save(best_overall_model)
                    logger.info(f"New best overall model: {current_winrate:.2f} win rate")
                
                # Check for early stopping
                if enable_early_stopping and current_winrate >= _win_rate_threshold:
                    above_threshold_count += 1
                    logger.info(f"Win rate {current_winrate:.2f} above threshold {_win_rate_threshold:.2f} "
                               f"({above_threshold_count}/{early_stopping_patience})")
                    if above_threshold_count >= early_stopping_patience:
                        return True  # Signal to stop training
                else:
                    above_threshold_count = 0  # Reset counter if win rate drops
            
            return False  # Continue training
        
        # Train against current opponent
        level_start_time = time.time()
        level_results = train_agent(
            env=env,
            agent=agent,
            num_episodes=episodes_for_level,
            log_interval=min(100, episodes_for_level // 10),
            save_interval=checkpoint_frequency,
            eval_interval=evaluation_frequency,
            checkpoint_dir=os.path.join(checkpoint_dir, f"level_{level_idx}_{level_name}"),
            log_dir=os.path.join(log_dir, f"level_{level_idx}_{level_name}"),
            render_interval=100 if render_training else None,
            eval_episodes=100,  # More episodes for reliable evaluation
            early_stopping=enable_early_stopping,
            win_rate_threshold=_win_rate_threshold,
            patience=early_stopping_patience,
            callback=checkpoint_callback,
            reward_shaping=True  # Enable reward shaping for better learning signals
        )
        
        level_duration = time.time() - level_start_time
        
        # Store best model info
        best_models[level_name] = {
            "path": best_level_model_path,
            "win_rate": best_level_winrate
        }
        
        # Use best model for next level if good enough
        if best_level_model_path and best_level_winrate >= 0.6:
            logger.info(f"Loading best model for {level_name} (win rate: {best_level_winrate:.2f})")
            agent.load(best_level_model_path)
        
        # Track results
        all_results[level_name] = level_results
        
        # Log level completion
        logger.info(f"Completed training against {level_name}")
        logger.info(f"Episodes: {episodes_for_level}, Duration: {level_duration:.2f} seconds")
        logger.info(f"Final evaluation reward: {level_results['final_eval_reward']:.2f}")
        logger.info(f"Best win rate: {best_level_winrate:.2f}")
        
        # Evaluate against all previous levels to check for forgetting
        if level_idx > 0:
            logger.info("Evaluating against all previous levels...")
            previous_levels = CURRICULUM_LEVELS[:level_idx+1]
            all_levels_win_rate = []
            
            for prev_level in previous_levels:
                # Create evaluation environment
                eval_env = LiarsDiceEnvWrapper(
                    num_players=_num_players,
                    num_dice=_num_dice,
                    dice_faces=dice_faces,
                    seed=seed,
                    rule_agent_types=[prev_level]
                )
                
                # Set action mapping for evaluation
                agent.set_action_mapping(eval_env.action_mapping)
                
                # Run evaluation
                eval_reward = evaluate_agent(eval_env, agent, num_episodes=50)
                
                # Test win rate with separate episodes
                wins = 0
                for _ in range(30):
                    eval_reward, _, _ = train_episode(eval_env, agent, evaluation=True)
                    if eval_reward > 0:  # Win
                        wins += 1
                
                level_win_rate = wins / 30
                all_levels_win_rate.append(level_win_rate)
                logger.info(f"  vs {prev_level}: win rate = {level_win_rate:.2f}")
            
            # Log average win rate across all levels
            avg_win_rate = sum(all_levels_win_rate) / len(all_levels_win_rate)
            logger.info(f"  Average win rate across all levels: {avg_win_rate:.2f}")
    
    # Remedial training phase for any weak spots
    if enable_remedial and best_overall_model:
        logger.info("\n=== Starting Remedial Training ===")
        
        # Load best overall model
        logger.info(f"Loading best overall model with win rate {best_overall_winrate:.2f}")
        agent.load(best_overall_model)
        
        # Evaluate against all levels to find weaknesses
        eval_results = evaluate_against_curriculum(
            agent=agent,
            num_episodes_per_level=50,
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            epsilon=0.05,
            seed=seed,
            verbose=True
        )
        
        # Identify levels that need remedial training (below win threshold)
        remedial_levels = []
        for level_name in CURRICULUM_LEVELS:
            if level_name in eval_results and eval_results[level_name]['win_rate'] < _win_rate_threshold:
                remedial_levels.append(level_name)
        
        if remedial_levels:
            logger.info(f"Identified {len(remedial_levels)} levels needing remedial training: {remedial_levels}")
            
            # Train on weak levels (starting with easiest)
            remedial_levels.sort(key=lambda x: CURRICULUM_LEVELS.index(x))
            
            for level_name in remedial_levels:
                logger.info(f"Starting remedial training against {level_name}")
                
                # Set moderate exploration for remedial training (DQN only)
                if agent_type.lower() == 'dqn':
                    remedial_epsilon = 0.3
                    logger.info(f"Setting epsilon to {remedial_epsilon:.2f} for remedial training")
                    agent.epsilon = remedial_epsilon
                
                # Create remedial environment
                remedial_env = LiarsDiceEnvWrapper(
                    num_players=_num_players,
                    num_dice=_num_dice,
                    dice_faces=dice_faces,
                    seed=seed,
                    rule_agent_types=[level_name]
                )
                
                # Set action mapping
                agent.set_action_mapping(remedial_env.action_mapping)
                
                # Limit remedial episodes
                remedial_episodes = 3000
                
                # Define remedial callback for early stopping
                remedial_best_winrate = 0.0
                remedial_best_path = None
                
                def remedial_callback(episode, data, current_agent):
                    nonlocal remedial_best_winrate, remedial_best_path
                    
                    if 'last_win_rate' in data:
                        current_winrate = data['last_win_rate']
                        
                        # Update best model
                        if current_winrate > remedial_best_winrate:
                            remedial_best_winrate = current_winrate
                            remedial_best_path = os.path.join(checkpoint_dir, f"remedial_best_{level_name}")
                            current_agent.save(remedial_best_path)
                            logger.info(f"Remedial training new best: {current_winrate:.2f}")
                        
                        # Stop early if we reached threshold
                        if current_winrate >= _win_rate_threshold:
                            logger.info(f"Remedial training goal achieved with {current_winrate:.2f}")
                            return True
                    
                    return False
                
                # Run remedial training
                train_agent(
                    env=remedial_env,
                    agent=agent,
                    num_episodes=remedial_episodes,
                    log_interval=100,
                    save_interval=checkpoint_frequency,
                    eval_interval=evaluation_frequency,
                    checkpoint_dir=os.path.join(checkpoint_dir, f"remedial_{level_name}"),
                    log_dir=os.path.join(log_dir, f"remedial_{level_name}"),
                    render_interval=100 if render_training else None,
                    eval_episodes=50,
                    early_stopping=True,
                    win_rate_threshold=_win_rate_threshold,
                    patience=2,  # Less patience for remedial
                    callback=remedial_callback,
                    reward_shaping=True
                )
                
                # Log remedial completion
                logger.info(f"Completed remedial training against {level_name}")
                logger.info(f"Final remedial win rate: {remedial_best_winrate:.2f}")
                
                # Use the best remedial model if it's better
                if remedial_best_path and remedial_best_winrate > best_models[level_name]["win_rate"]:
                    agent.load(remedial_best_path)
                    
                    # Update model info
                    best_models[level_name] = {
                        "path": remedial_best_path,
                        "win_rate": remedial_best_winrate
                    }
                    
                    # Update overall best if needed
                    if remedial_best_winrate > best_overall_winrate:
                        best_overall_winrate = remedial_best_winrate
                        best_overall_model = remedial_best_path
                        logger.info(f"New overall best from remedial: {best_overall_winrate:.2f}")
        else:
            logger.info("No levels need remedial training - all above threshold!")
    
    # Self-play phase for final improvement
    if enable_self_play and _self_play_episodes > 0:
        logger.info("\n=== Starting Self-Play Phase ===")
        
        # Start with the best model
        if best_overall_model:
            logger.info(f"Loading best overall model with win rate {best_overall_winrate:.2f}")
            agent.load(best_overall_model)
        
        # Set moderate exploration for self-play (DQN only)
        if agent_type.lower() == 'dqn':
            agent.epsilon = 0.3
        
        # Create self-play directory
        self_play_checkpoint_dir = os.path.join(checkpoint_dir, "self_play")
        os.makedirs(self_play_checkpoint_dir, exist_ok=True)
        
        # Run self-play training
        self_play_results = train_self_play(
            agent=agent,
            num_episodes=_self_play_episodes,
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            checkpoint_dir=self_play_checkpoint_dir,
            log_dir=os.path.join(log_dir, "self_play"),
            seed=seed,
            eval_interval=evaluation_frequency,
            update_opponent_interval=1000,  # Update opponent model every 1000 episodes
            reward_shaping=True
        )
        
        # Add self-play results
        all_results['self_play'] = self_play_results
        
        # Save final self-play model
        final_self_play_path = os.path.join(self_play_checkpoint_dir, "final_self_play")
        agent.save(final_self_play_path)
        logger.info(f"Saved final self-play model to {final_self_play_path}")
        
        # Update best model if self-play improved it
        if self_play_results.get('final_win_rate', 0) > best_overall_winrate:
            best_overall_winrate = self_play_results['final_win_rate']
            best_overall_model = final_self_play_path
            logger.info(f"Self-play produced new best model: {best_overall_winrate:.2f}")
    
    # Final evaluation against all levels
    logger.info("\n=== Final Evaluation ===")
    final_eval_results = evaluate_against_curriculum(
        agent=agent,
        num_episodes_per_level=200,  # More episodes for final evaluation
        num_players=_num_players,
        num_dice=_num_dice,
        dice_faces=dice_faces,
        epsilon=0.05,  # Low exploration for evaluation
        seed=seed,
        verbose=True
    )
    
    # Visualize final evaluation
    vis_path = os.path.join(log_dir, "final_evaluation.png")
    visualize_evaluation_results(final_eval_results, save_path=vis_path)
    logger.info(f"Saved final evaluation visualization to {vis_path}")
    
    # Save combined training data
    combined_results = {
        'curriculum_distribution': episode_distribution,
        'level_results': all_results,
        'final_evaluation': final_eval_results,
        'best_models': best_models,
        'parameters': {
            'agent_type': agent_type,
            'preset': preset,
            'num_players': _num_players,
            'num_dice': _num_dice,
            'dice_faces': dice_faces,
            'curriculum_episodes': _curriculum_episodes,
            'self_play_episodes': _self_play_episodes,
            'network_size': _network_size,
            'win_rate_threshold': _win_rate_threshold
        }
    }
    save_training_data(combined_results, os.path.join(log_dir, 'curriculum_results.pkl'))
    
    # Save final model with metadata
    models_dir = os.path.join(results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model name with performance info
    overall_win_rate = final_eval_results['overall']['win_rate']
    model_name = f"{agent_type}_{_num_players}p{_num_dice}d_{overall_win_rate:.2f}wr_{timestamp}"
    model_path = os.path.join(models_dir, model_name)
    agent.save(model_path)
    
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
            'curriculum_episodes': _curriculum_episodes,
            'self_play_episodes': _self_play_episodes
        }
    }
    
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Saved final model to {model_path}")
    logger.info("\nCurriculum learning completed!")
    
    return agent, combined_results


# Ensure this function is defined
from training.train import train_episode

# For direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum learning for Liar's Dice")
    parser.add_argument('--agent', type=str, default='ppo', choices=['dqn', 'ppo'],
                        help='Type of agent to train')
    parser.add_argument('--preset', type=str, default='basic', choices=['basic', 'standard', 'advanced'],
                        help='Configuration preset to use')
    parser.add_argument('--path', type=str, default='results/ppo_3dice', help='Base path for results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
    parser.add_argument('--no-self-play', action='store_true', help='Disable self-play phase')
    parser.add_argument('--no-remedial', action='store_true', help='Disable remedial training')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    
    args = parser.parse_args()
    
    # Run training with parsed arguments
    train_curriculum(
        agent_type=args.agent,
        preset=args.preset,
        results_path=args.path,
        seed=args.seed,
        render_training=args.render,
        enable_self_play=not args.no_self_play,
        enable_remedial=not args.no_remedial,
        enable_early_stopping=not args.no_early_stopping,
        evaluation_frequency=100,
    )