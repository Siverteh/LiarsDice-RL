"""
Enhanced curriculum learning for DQN agent in Liar's Dice.

This script implements an improved curriculum learning approach with
self-play and better learning dynamics for training a robust DQN agent.
"""

import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import random
from typing import List, Dict, Any, Optional, Union
from collections import deque

from agents.dqn_agent import DQNAgent
from agents.rule_agent import CURRICULUM_LEVELS, create_agent
from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from training.environment_wrapper import LiarsDiceEnvWrapper
from training.train import train_dqn, train_self_play, shape_reward
from training.evaluate import evaluate_against_curriculum, visualize_evaluation_results
from training.utils import (
    setup_logger, 
    save_training_data, 
    load_training_data,
    plot_training_results,
    generate_curriculum_schedule
)


def enhanced_shape_reward(
    original_reward: float,
    observation: np.ndarray,
    action: Dict[str, Any],
    next_observation: np.ndarray,
    info: Dict[str, Any],
    game_stage: str = 'mid',
    dice_counts: Optional[np.ndarray] = None
) -> float:
    """
    Enhanced reward shaping function that considers game stage and strategic elements.
    
    Args:
        original_reward: Original reward from environment
        observation: Current observation
        action: Action taken
        next_observation: Next observation
        info: Additional information
        game_stage: Current stage of the game ('early', 'mid', 'late')
        dice_counts: Number of dice for each player
        
    Returns:
        Shaped reward
    """
    reward = original_reward
    
    # Game outcome rewards
    if info.get('state') == 'GAME_OVER':
        if reward > 0:  # Win
            # Scale win reward by difficulty of opponent
            opponent_type = info.get('opponent_type', 'unknown')
            difficulty_multiplier = {
                'random': 1.0,
                'naive': 1.5,
                'conservative': 2.0,
                'aggressive': 2.5,
                'strategic': 3.0,
                'adaptive': 3.5,
                'self_play': 3.0,
                'unknown': 2.0
            }.get(opponent_type, 2.0)
            
            reward = reward * difficulty_multiplier
            
            # Extra reward for winning with more dice remaining
            if 'dice_counts' in info:
                own_dice = info['dice_counts'][0]  # Assuming agent is player 0
                reward += own_dice * 0.5
        else:  # Loss
            # Reduce penalty for losing to strong opponents
            opponent_type = info.get('opponent_type', 'unknown')
            difficulty_factor = {
                'random': 1.0,
                'naive': 0.9,
                'conservative': 0.8,
                'aggressive': 0.7,
                'strategic': 0.6,
                'adaptive': 0.5,
                'self_play': 0.7,
                'unknown': 0.75
            }.get(opponent_type, 0.75)
            
            reward = reward * difficulty_factor
    
    # Strategic bidding rewards
    if action['type'] == 'bid':
        # Get current game state information
        last_bid = info.get('last_bid')
        
        if last_bid is None:
            # First bid of the game - small reward for reasonable opening bid
            reward += 0.1
        else:
            last_quantity, last_value = last_bid
            current_quantity, current_value = action['quantity'], action['value']
            
            # Reward for minimal valid increases (strategic play)
            if current_quantity == last_quantity + 1 and current_value == last_value:
                reward += 0.15
            elif current_quantity == last_quantity and current_value > last_value:
                reward += 0.15
            
            # Mild penalty for unnecessarily large jumps (may signal weak hand)
            if current_quantity > last_quantity + 2 and current_value == last_value:
                reward -= 0.1
    
    # Challenge action rewards
    elif action['type'] == 'challenge':
        # Successful challenge (already captured in original reward)
        if reward > 0:
            # Enhanced reward based on game stage
            if game_stage == 'late':  # Critical challenges late game are more valuable
                reward += 0.2
        # Failed challenge
        else:
            # Less penalty for reasonable failed challenges
            challenge_probability = info.get('challenge_probability', 0.5)
            if challenge_probability < 0.4:  # It seemed like a good challenge based on probability
                reward += 0.1  # Reduce penalty for a reasonable challenge
    
    # Survival rewards - encourage preserving dice
    if 'dice_counts' in info and dice_counts is not None:
        agent_dice = info['dice_counts'][0]  # Assuming agent is player 0
        total_dice = sum(info['dice_counts'])
        
        # Reward for having more dice than average
        avg_dice = total_dice / len(info['dice_counts'])
        if agent_dice > avg_dice:
            reward += 0.05 * (agent_dice - avg_dice)
        
        # Small continuous reward for surviving each step with dice
        reward += 0.01 * agent_dice
    
    # Encourage diversity in actions with a small exploration bonus
    action_hash = hash(str(action))
    if abs(action_hash) % 5 == 0:  # Pseudo-random exploration bonus
        reward += 0.02
    
    return reward


def create_enhanced_agent(
    obs_dim: int,
    action_dim: int,
    learning_rate: float = 0.0005,
    network_size: str = 'medium',
    dropout_rate: float = 0.2,
    use_prioritized_replay: bool = False,  # Not used, but kept for parameter compatibility
    device: str = 'auto'
) -> DQNAgent:
    """
    Create an enhanced DQN agent with better architecture and hyperparameters.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        learning_rate: Learning rate
        network_size: Size of neural network ('small', 'medium', 'large')
        dropout_rate: Dropout rate for regularization (note: already set in QNetwork)
        use_prioritized_replay: Not used with existing implementation (kept for compatibility)
        device: Device to run on ('cpu', 'cuda', 'auto')
        
    Returns:
        Enhanced DQN agent
    """
    # Determine network architecture based on size
    if network_size == 'small':
        hidden_dims = [128, 64]
    elif network_size == 'medium':
        hidden_dims = [256, 128, 64]
    else:  # large
        hidden_dims = [512, 256, 128, 64]
    
    # Try to create agent with specified device, fall back to CPU if CUDA issues occur
    try:
        # Create DQN agent with improved parameters
        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            target_update_freq=500,
            buffer_size=150000,
            batch_size=128,
            hidden_dims=hidden_dims,
            device=device
        )
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error encountered: {e}")
            print("Falling back to CPU...")
            # Try again with CPU
            agent = DQNAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.9999,
                target_update_freq=500,
                buffer_size=150000,
                batch_size=128,
                hidden_dims=hidden_dims,
                device='cpu'
            )
        else:
            raise e  # Re-raise if it's not a CUDA error
    
    return agent


def progressive_curriculum_learning(
    base_path: str = 'results',
    num_players: int = 2,
    num_dice: int = 2,
    dice_faces: int = 6,
    total_episodes: int = 30000,
    distribution: str = 'front_loaded',
    eval_interval: int = 500,
    learning_rate: float = 0.0003,
    seed: Optional[int] = None,
    render_interval: Optional[int] = None,
    device: str = 'auto',
    use_early_stopping: bool = True,
    win_rate_threshold: float = 0.85,
    early_stopping_patience: int = 5,
    enable_self_play: bool = True,
    self_play_episodes: int = 10000,
    network_size: str = 'large',
    use_prioritized_replay: bool = False,
    population_size: int = 5,
    use_exploration_bonus: bool = True,
    progressive_opponent_mixing: bool = True,
    checkpoint_frequency: int = 1000,
    learning_rate_schedule: str = 'step',
    reward_shaping_intensity: float = 1.0
):
    """
    Enhanced curriculum learning with progressive difficulty and better exploration.
    
    Args:
        base_path: Base directory for saving results
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        total_episodes: Total number of training episodes for curriculum phase
        distribution: How to distribute episodes among difficulty levels
        eval_interval: Interval for evaluation
        learning_rate: Initial learning rate for the DQN agent
        seed: Random seed for reproducibility
        render_interval: Interval for rendering (if None, no rendering)
        device: Device to run on ('cpu', 'cuda', or 'auto')
        use_early_stopping: Whether to enable early stopping based on win rate
        win_rate_threshold: Win rate threshold for early stopping
        early_stopping_patience: Number of consecutive evaluations above threshold
        enable_self_play: Whether to enable self-play phase after curriculum
        self_play_episodes: Number of episodes for self-play
        network_size: Size of neural network ('small', 'medium', 'large')
        use_prioritized_replay: Whether to use prioritized experience replay
        population_size: Size of the population for self-play
        use_exploration_bonus: Whether to add exploration bonuses to rewards
        progressive_opponent_mixing: Whether to mix in previous opponents progressively
        checkpoint_frequency: How often to save checkpoints
        learning_rate_schedule: Learning rate schedule type ('step', 'cosine', 'none')
        reward_shaping_intensity: Intensity of reward shaping (0.0-1.0)
    """
    # Fix logger to prevent duplicate handlers
    import logging
    
    # Remove all existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Set up paths
    os.makedirs(base_path, exist_ok=True)
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('curriculum', os.path.join(log_dir, 'curriculum.log'))
    logger.info("Starting enhanced curriculum learning for Liar's Dice")
    logger.info(f"Parameters: players={num_players}, dice={num_dice}, faces={dice_faces}")
    logger.info(f"Total curriculum episodes: {total_episodes}, distribution: {distribution}")
    logger.info(f"Self-play episodes: {self_play_episodes}, Population size: {population_size}")
    logger.info(f"Network size: {network_size}, Learning rate: {learning_rate}")
    
    # Set default device if auto specified
    if device == 'auto':
        if torch.cuda.is_available():
            try:
                # Try to create a small tensor on GPU to check if it's working
                test_tensor = torch.zeros(1, device='cuda')
                device = 'cuda'
            except RuntimeError:
                logger.info("CUDA device detected but unavailable, using CPU instead")
                device = 'cpu'
        else:
            device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Create environment to get observation and action dimensions
    env = LiarsDiceEnvWrapper(
        num_players=num_players,
        num_dice=num_dice,
        dice_faces=dice_faces,
        seed=seed
    )
    
    # Use the actual observation dimension from the environment
    obs_shape = env.get_observation_shape()
    obs_dim = obs_shape[0]
    action_dim = env.get_action_dim()
    
    logger.info(f"Observation dimension: {obs_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Create enhanced DQN agent
    agent = create_enhanced_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        network_size=network_size,
        dropout_rate=0.2,
        use_prioritized_replay=False,  # Not supported with current implementation
        device=device
    )
    
    # Set action mapping
    agent.set_action_mapping(env.action_mapping)
    logger.info(f"Action mapping set with {len(env.action_mapping)} possible actions")
    
    # Generate curriculum schedule with more episodes for difficult levels
    num_levels = len(CURRICULUM_LEVELS)
    if distribution == 'progressive':
        # Custom progressive distribution with more episodes for difficult levels
        schedule = [
            int(total_episodes * 0.05),  # random: 5%
            int(total_episodes * 0.15),  # naive: 15%
            int(total_episodes * 0.2),   # conservative: 20%
            int(total_episodes * 0.2),   # aggressive: 20%
            int(total_episodes * 0.2),   # strategic: 20%
            int(total_episodes * 0.2)    # adaptive: 20%
        ]
    else:
        schedule = generate_curriculum_schedule(total_episodes, num_levels, distribution)
    
    # Ensure the schedule sums up to total_episodes
    if sum(schedule) != total_episodes:
        diff = total_episodes - sum(schedule)
        schedule[-1] += diff
    
    logger.info("Curriculum schedule:")
    for i, level in enumerate(CURRICULUM_LEVELS):
        logger.info(f"  Level {i} ({level}): {schedule[i]} episodes")
    
    # Training results for each level
    all_results = {}
    total_trained_episodes = 0
    
    # Track best models
    best_models = {}
    best_overall_model = None
    best_overall_winrate = 0.0
    
    # List of mastered opponents for progressive opponent mixing
    mastered_opponents = []
    
    # Perform curriculum learning
    for level_idx, level_name in enumerate(CURRICULUM_LEVELS):
        episodes_for_level = schedule[level_idx]
        if episodes_for_level <= 0:
            continue
        
        logger.info(f"\n=== Training against {level_name} ({episodes_for_level} episodes) ===")
        
        # For progressive opponent mixing, include previous opponents
        opponent_types = []
        if progressive_opponent_mixing and mastered_opponents:
            # Log that we're aware of the limitation but can't implement it currently
            logger.info(f"Note: Would mix in previous opponents {mastered_opponents}, but API only supports one opponent type")
            logger.info(f"Consider adding support for multiple opponent types in future versions")
            
        # Just use the current level opponent
        logger.info(f"Training against {level_name} for {episodes_for_level} episodes")
        
        # Create environment with current opponent
        if progressive_opponent_mixing and mastered_opponents:
            # Since create_progressive_env only supports one opponent type,
            # we'll just use the current level opponent but note the limitation
            logger.info(f"Note: Progressive mixing with previous opponents is limited due to API constraints")
            logger.info(f"Using primary opponent: {level_name}")
            
        env = LiarsDiceEnvWrapper.create_progressive_env(
            num_players=num_players,
            num_dice=num_dice,
            dice_faces=dice_faces,
            seed=seed,
            opponent_type=level_name,
            episodes_for_level=episodes_for_level
        )
        
        # Apply dynamic learning rate if enabled
        if learning_rate_schedule == 'step':
            # Reduce learning rate for later curriculum stages
            if level_idx >= 3:  # More difficult opponents
                adjusted_lr = learning_rate * (0.7 ** (level_idx - 2))
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = adjusted_lr
                logger.info(f"Adjusted learning rate to {adjusted_lr} for level {level_idx}")
        
        # Best model tracking for this level
        best_level_winrate = 0.0
        best_level_model_path = None
        above_threshold_count = 0
        
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
                
                # Check if win rate is above threshold for early stopping
                if use_early_stopping and current_winrate >= win_rate_threshold:
                    above_threshold_count += 1
                    logger.info(f"Win rate {current_winrate:.2f} above threshold {win_rate_threshold:.2f} "
                               f"({above_threshold_count}/{early_stopping_patience})")
                else:
                    above_threshold_count = 0
        
        # Custom reward shaping is already included in training.py, so just enable it
        # if reward shaping intensity is > 0
        use_reward_shaping = reward_shaping_intensity > 0
        
        # Set info about current level for reward shaping
        env.current_level_name = level_name  # Add this attribute if used inside shape_reward
        
        # Train agent
        level_start_time = time.time()
        level_results = train_dqn(
            env=env,
            agent=agent,
            num_episodes=episodes_for_level,
            log_interval=min(100, episodes_for_level // 10),
            save_interval=min(checkpoint_frequency, episodes_for_level // 5),
            eval_interval=min(eval_interval, episodes_for_level // 2),
            checkpoint_dir=os.path.join(checkpoint_dir, f"level_{level_idx}_{level_name}"),
            log_dir=os.path.join(log_dir, f"level_{level_idx}_{level_name}"),
            render_interval=render_interval,
            eval_episodes=100,  # More episodes for reliable evaluation
            eval_epsilon=0.05,  # Lower epsilon for better evaluation
            early_stopping=use_early_stopping,
            win_rate_threshold=win_rate_threshold,
            patience=early_stopping_patience,
            callback=checkpoint_callback,
            reward_shaping=use_reward_shaping
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
            
            # Add to mastered opponents if win rate is good
            if best_level_winrate >= 0.7:
                mastered_opponents.append(level_name)
                logger.info(f"Added {level_name} to mastered opponents: {mastered_opponents}")
        
        # Track results
        all_results[level_name] = level_results
        total_trained_episodes += episodes_for_level
        
        # Log level completion
        logger.info(f"Completed training against {level_name}")
        logger.info(f"Episodes: {episodes_for_level}, Duration: {level_duration:.2f} seconds")
        logger.info(f"Final evaluation reward: {level_results['final_eval_reward']:.2f}")
        logger.info(f"Best win rate: {best_level_winrate:.2f}")
        
        # Checkpoint after each level
        checkpoint_path = os.path.join(checkpoint_dir, f"after_level_{level_idx}")
        agent.save(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Plot level results
        plot_path = os.path.join(log_dir, f"level_{level_idx}_{level_name}_results.png")
        plot_training_results(
            level_results,
            window_size=min(20, episodes_for_level // 10),
            save_path=plot_path,
            show_plot=False
        )
        
        # Evaluate against all previous levels
        if level_idx > 0:
            logger.info("Evaluating against all previous levels...")
            previous_levels = CURRICULUM_LEVELS[:level_idx+1]
            all_levels_win_rate = []
            
            for prev_level in previous_levels:
                eval_env = LiarsDiceEnvWrapper(
                    num_players=num_players,
                    num_dice=num_dice,
                    dice_faces=dice_faces,
                    seed=seed,
                    rule_agent_types=[prev_level]
                )
                
                # Test with 30 episodes per evaluation for more stability
                eval_results = evaluate_against_curriculum(
                    agent=agent,
                    num_episodes_per_level=1000,
                    num_players=num_players,
                    num_dice=num_dice,
                    dice_faces=dice_faces,
                    epsilon=0.05,
                    seed=seed,
                    verbose=False,
                )
                
                # Log evaluation results for the current level
                level_win_rate = eval_results[prev_level]['win_rate']
                all_levels_win_rate.append(level_win_rate)
                logger.info(f"  vs {prev_level}: win rate = {level_win_rate:.2f}")
            
            # Log average win rate across all levels
            avg_win_rate = sum(all_levels_win_rate) / len(all_levels_win_rate)
            logger.info(f"  Average win rate across all levels: {avg_win_rate:.2f}")
    
    # Enhanced self-play phase with simple self-play approach
    if enable_self_play and best_overall_model:
        logger.info("\n=== Starting Enhanced Self-Play Phase ===")
        
        # Start with the best model
        logger.info(f"Loading best overall model with win rate {best_overall_winrate:.2f}")
        agent.load(best_overall_model)
        
        # Create self-play checkpoint directory
        self_play_checkpoint_dir = os.path.join(checkpoint_dir, "self_play")
        os.makedirs(self_play_checkpoint_dir, exist_ok=True)
        
        # Create a simple self-play environment
        class SimpleSelfPlayEnv:
            def __init__(self):
                self.game = LiarsDiceGame(
                    num_players=num_players,
                    num_dice=num_dice,
                    dice_faces=dice_faces,
                    seed=seed
                )
                self.observation_encoder = ObservationEncoder(
                    num_players=num_players, 
                    num_dice=num_dice, 
                    dice_faces=dice_faces
                )
                self.action_mapping = agent.action_to_game_action
                self.episode_counter = 0
                self.current_level_name = 'self_play'
                # Use the same agent as opponent but with higher epsilon
                self.epsilon_for_opponent = 0.2
                
            def reset(self):
                self.episode_counter += 1
                observations = self.game.reset()
                return self.observation_encoder.encode(observations[0])
                
            def step(self, action_idx):
                # Convert action index to game action
                action = self.action_mapping[action_idx].copy()
                
                # Execute agent's action
                observations, rewards, done, info = self.game.step(action)
                
                # If game not done and it's opponent's turn, take opponent action
                while not done and self.game.current_player != 0:
                    opponent_player = self.game.current_player
                    obs = self.observation_encoder.encode(observations[opponent_player])
                    valid_actions = self.game.get_valid_actions(opponent_player)
                    
                    # Get action from the same agent but with randomization
                    original_epsilon = agent.epsilon
                    agent.epsilon = self.epsilon_for_opponent  # More randomness for opponent
                    opponent_action = agent.select_action(
                        obs, valid_actions, training=False
                    )
                    agent.epsilon = original_epsilon  # Restore
                    
                    # Execute opponent's action
                    observations, rewards, done, info = self.game.step(opponent_action)
                
                # Return observation for agent
                next_obs = self.observation_encoder.encode(observations[0])
                reward = rewards[0]  # Agent's reward
                
                # Add opponent info to info dict
                info['opponent_type'] = 'self_play'
                
                return next_obs, reward, done, info
            
            def get_valid_actions(self):
                valid_game_actions = self.game.get_valid_actions(0)
                valid_indices = []
                
                for game_action in valid_game_actions:
                    for idx, action in enumerate(self.action_mapping):
                        if self._actions_equal(game_action, action):
                            valid_indices.append(idx)
                            break
                
                return valid_indices
            
            def _actions_equal(self, action1, action2):
                if action1['type'] != action2['type']:
                    return False
                
                if action1['type'] == 'challenge':
                    return True
                
                return (action1['quantity'] == action2['quantity'] and 
                        action1['value'] == action2['value'])
            
            def render(self):
                self.game.render()
        
        # Create self-play environment
        self_play_env = SimpleSelfPlayEnv()
        
        # Define a callback function for self-play training
        def self_play_callback(episode, data, current_agent):
            # Print stats
            if episode % 500 == 0 and 'last_win_rate' in data:
                logger.info(f"Self-play episode {episode}: Win rate = {data['last_win_rate']:.2f}")
        
        # Define training parameters
        self_play_training_params = {
            'env': self_play_env,
            'agent': agent,
            'num_episodes': self_play_episodes,
            'log_interval': 100,
            'save_interval': 500,
            'eval_interval': 500,
            'checkpoint_dir': self_play_checkpoint_dir,
            'log_dir': os.path.join(log_dir, "self_play"),
            'render_interval': render_interval,
            'callback': self_play_callback,
            'reward_shaping': use_reward_shaping
        }
        
        # Train with self-play
        logger.info(f"Starting self-play training for {self_play_episodes} episodes")
        self_play_results = train_dqn(**self_play_training_params)
        
        # Update results with self-play data
        all_results['self_play'] = self_play_results
        
        # Load the final self-play model
        final_self_play_path = os.path.join(self_play_checkpoint_dir, "final_agent")
        agent.save(final_self_play_path)
        logger.info(f"Saved final self-play model to {final_self_play_path}")
    
    # Final evaluation against all levels
    logger.info("\n=== Final Evaluation ===")
    final_eval_results = evaluate_against_curriculum(
        agent=agent,
        num_episodes_per_level=1000,  # More episodes for final evaluation
        num_players=num_players,
        num_dice=num_dice,
        dice_faces=dice_faces,
        epsilon=0.05,
        seed=seed,
        verbose=True
    )
    
    # Visualize final evaluation
    vis_path = os.path.join(log_dir, "final_evaluation.png")
    visualize_evaluation_results(final_eval_results, save_path=vis_path)
    logger.info(f"Saved final evaluation visualization to {vis_path}")
    
    # Save combined training data
    combined_results = {
        'curriculum_schedule': {level: count for level, count in zip(CURRICULUM_LEVELS, schedule)},
        'level_results': all_results,
        'final_evaluation': final_eval_results,
        'best_models': best_models,
        'parameters': {
            'num_players': num_players,
            'num_dice': num_dice,
            'dice_faces': dice_faces,
            'total_episodes': total_episodes,
            'distribution': distribution,
            'network_size': network_size,
            'self_play_episodes': self_play_episodes,
            'use_prioritized_replay': False,  # Not supported with current implementation
            'population_size': population_size
        }
    }
    save_training_data(combined_results, os.path.join(log_dir, 'curriculum_results.pkl'))
    
    # Final save to dedicated models directory with timestamp and performance info
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a descriptive model name with timestamp and performance
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_win_rate = final_eval_results['overall']['win_rate']
    adaptive_win_rate = final_eval_results['adaptive']['win_rate'] if 'adaptive' in final_eval_results else 0
    
    model_name = f"liars_dice_dqn_{timestamp}_p{num_players}d{num_dice}_wr{overall_win_rate:.2f}"
    model_path = os.path.join(models_dir, model_name)
    agent.save(model_path)
    
    # Save model metadata for easier loading
    model_metadata = {
        'timestamp': timestamp,
        'num_players': num_players,
        'num_dice': num_dice,
        'dice_faces': dice_faces,
        'win_rates': {k: v['win_rate'] for k, v in final_eval_results.items() if k != 'overall'},
        'overall_win_rate': overall_win_rate,
        'action_mapping_available': True,
        'network_size': network_size,
        'best_models': best_models,
        'training_parameters': {
            'total_episodes': total_episodes,
            'self_play_episodes': self_play_episodes,
            'distribution': distribution,
            'use_prioritized_replay': False,
            'population_size': population_size
        }
    }
    
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Saved final model to {model_path}")
    
    # Final save to checkpoint dir (keeping this for backward compatibility)
    final_checkpoint = os.path.join(checkpoint_dir, "final_agent")
    agent.save(final_checkpoint)
    logger.info(f"Saved final agent to {final_checkpoint}")
    
    logger.info("\nEnhanced curriculum learning completed!")
    return agent, combined_results


if __name__ == "__main__":
    # Fix setup_logger to handle duplicate logging
    original_setup_logger = setup_logger
    import logging

    def fixed_setup_logger(name, log_file, level=logging.INFO):
        """
        Fixed setup logger function to prevent duplicate handlers
        """
        # Get logger
        logger = logging.getLogger(name)
        
        # If this logger already has handlers, remove them
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # Set level
        logger.setLevel(level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        
        return logger
    
    # Replace the original setup_logger
    import training.utils
    training.utils.setup_logger = fixed_setup_logger
    
    parser = argparse.ArgumentParser(description="Enhanced curriculum learning for Liar's Dice")
    parser.add_argument('--path', type=str, default='results', help='Base path for results')
    parser.add_argument('--players', type=int, default=2, help='Number of players')
    parser.add_argument('--dice', type=int, default=3, help='Number of dice per player')
    parser.add_argument('--faces', type=int, default=6, help='Number of faces per die')
    parser.add_argument('--curriculum_episodes', type=int, default=50000, help='Total number of curriculum episodes')
    parser.add_argument('--distribution', type=str, default='progressive', 
                       choices=['linear', 'exp', 'front_loaded', 'progressive'],
                       help='How to distribute episodes among levels')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--render', type=int, default=None, help='Render every N episodes')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='Device to run on')
    parser.add_argument('--early_stopping', action='store_true', 
                        help='Enable early stopping based on win rate')
    parser.add_argument('--win_threshold', type=float, default=0.75,
                        help='Win rate threshold for early stopping (0-1)')
    parser.add_argument('--patience', type=int, default=1,
                        help='Number of evaluations above threshold to trigger early stopping')
    parser.add_argument('--self_play', action='store_true', help='Enable self-play phase')
    parser.add_argument('--self_play_episodes', type=int, default=10000, 
                        help='Number of self-play episodes')
    parser.add_argument('--network_size', type=str, default='large',
                        choices=['small', 'medium', 'large'],
                        help='Size of neural network architecture')
    parser.add_argument('--prioritized_replay', action='store_true',
                        help='Use prioritized experience replay (note: not supported with current DQNAgent)')
    parser.add_argument('--population_size', type=int, default=5,
                        help='Population size for self-play')
    parser.add_argument('--exploration_bonus', action='store_true',
                        help='Add exploration bonuses to rewards')
    parser.add_argument('--progressive_mixing', action='store_true',
                        help='Progressively mix in previous opponents')
    parser.add_argument('--checkpoint_frequency', type=int, default=1000,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--lr_schedule', type=str, default='step',
                        choices=['step', 'cosine', 'none'],
                        help='Learning rate schedule type')
    parser.add_argument('--reward_shaping', type=float, default=1.0,
                        help='Reward shaping intensity (0.0-1.0)')
    
    args = parser.parse_args()
    
    progressive_curriculum_learning(
        base_path=args.path,
        num_players=args.players,
        num_dice=args.dice,
        dice_faces=args.faces,
        total_episodes=args.curriculum_episodes,
        distribution=args.distribution,
        eval_interval=args.eval_interval,
        learning_rate=args.lr,
        seed=args.seed,
        render_interval=args.render,
        device=args.device,
        use_early_stopping=args.early_stopping,
        win_rate_threshold=args.win_threshold,
        early_stopping_patience=args.patience,
        enable_self_play=args.self_play,
        self_play_episodes=args.self_play_episodes,
        network_size=args.network_size,
        use_prioritized_replay=args.prioritized_replay,
        population_size=args.population_size,
        use_exploration_bonus=args.exploration_bonus,
        progressive_opponent_mixing=args.progressive_mixing,
        checkpoint_frequency=args.checkpoint_frequency,
        learning_rate_schedule=args.lr_schedule,
        reward_shaping_intensity=args.reward_shaping
    )