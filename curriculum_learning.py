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
        'curriculum_episodes':1500000,
        'self_play_episodes': 15000,
        'network_size': [512, 256, 128, 64],
        'learning_rate': 0.0003,
        'win_rate_threshold': 0.9
    },
    # 4 players, 5 dice, complex game
    'advanced': {
        'num_players': 4,
        'num_dice': 5,
        'curriculum_episodes': 6000000,
        'self_play_episodes': 30000,
        'network_size': [2048, 1024, 512, 256, 128],
        'learning_rate': 0.0003,
        'win_rate_threshold': 0.9
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
    randomize_positions: bool = True,
    position_randomization_schedule: Optional[Dict[str, float]] = None,
    
    # Training schedule
    curriculum_episodes: Optional[int] = None,
    curriculum_distribution: Dict[str, float] = None,
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
    early_stopping_patience: int = 1,
    
    # Exploration settings
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.1,
    epsilon_decay_per_level: float = 0.15
) -> Tuple[RLAgent, Dict[str, Any]]:
    """
    Train an agent using curriculum learning with progressively more difficult opponents.
    Self-play functionality has been implemented separately.
    """
    # Apply preset configuration
    preset_config = CONFIG_PRESETS[preset]
    
    # Apply values from preset unless explicitly overridden
    _num_players = num_players if num_players is not None else preset_config['num_players']
    _num_dice = num_dice if num_dice is not None else preset_config['num_dice']
    _curriculum_episodes = curriculum_episodes if curriculum_episodes is not None else preset_config['curriculum_episodes']
    _win_rate_threshold = win_rate_threshold if win_rate_threshold is not None else preset_config['win_rate_threshold']
    _learning_rate = learning_rate if learning_rate is not None else preset_config['learning_rate']
    _network_size = network_size if network_size is not None else preset_config['network_size']

    
    # Default progressive position randomization if not provided
    if position_randomization_schedule is None:
        position_randomization_schedule = {
            'random': 0.0,          # No randomization for easiest opponents
            'naive': 0.0,
            'conservative': 0.25,   # 25% randomization
            'bluff_punisher': 0.5,  # 50% randomization
            'anti_exploitation': 0.5,
            'aggressive': 0.75,     # 75% randomization
            'strategic': 0.75,
            'adaptive': 1.0,        # Full randomization for hardest opponents
            'counter_strategy': 1.0,
            'optimal': 1.0
        }
    
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
    logger.info(f"Curriculum: {_curriculum_episodes} episodes")
    
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
        seed=seed,
        randomize_positions=False  # Start with no randomization for dimension determination
    )
    
    obs_shape = env.get_observation_shape()
    obs_dim = obs_shape[0]
    action_dim = env.get_action_dim()
    logger.info(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # Create agent configuration with improved PPO parameters
    if agent_type.lower() == 'ppo':
        # Preset configs for different game complexities
        if preset == 'standard':  # 2 players, 5 dice (standard game)
            ppo_config = {
                'learning_rate': 0.0001,        # Lower learning rate for stability
                'min_learning_rate': 5e-6,      # Lower minimum LR
                'gamma': 0.99,                 
                'gae_lambda': 0.97,            
                'policy_clip': 0.1,             # Reduced clipping for smoother updates
                'value_coef': 0.8,              # Higher value coefficient
                'entropy_coef': 0.15,           # Higher entropy for better exploration
                'entropy_min': 0.02,            # Higher minimum entropy
                'entropy_decay_steps': 300000,  # Much slower entropy decay
                'ppo_epochs': 10,               # More PPO epochs per update
                'batch_size': 256,              # Larger batch size
                'update_frequency': 2048,       
                'max_grad_norm': 0.3,           # Lower gradient clipping
                'total_training_steps': 3000000 
            }
        elif preset == 'advanced':  # 4 players, 5 dice (complex game)
            ppo_config = {
                'learning_rate': 0.0001,        
                'min_learning_rate': 5e-6,      
                'gamma': 0.99,                 
                'gae_lambda': 0.97,            
                'policy_clip': 0.1,             
                'value_coef': 0.8,             
                'entropy_coef': 0.18,           # Even higher for complex environment
                'entropy_min': 0.03,            # Higher minimum for complex game
                'entropy_decay_steps': 500000,  # Much slower decay for 6M episodes
                'ppo_epochs': 12,               # Even more epochs for complex environment
                'batch_size': 256,              
                'update_frequency': 2048,       
                'max_grad_norm': 0.3,           
                'total_training_steps': 6000000
            }
        else:  # Basic preset
            ppo_config = {
                'learning_rate': 0.0001,        
                'min_learning_rate': 5e-6,     
                'gamma': 0.99,                 
                'gae_lambda': 0.95,            
                'policy_clip': 0.15,            
                'value_coef': 0.7,             
                'entropy_coef': 0.1,          
                'entropy_min': 0.01,          
                'entropy_decay_steps': 50000,  
                'ppo_epochs': 8,               
                'batch_size': 128,              
                'update_frequency': 1024,      
                'max_grad_norm': 0.4,           
                'total_training_steps': 500000
            }
            
        agent_config = ppo_config
    else:
        # For DQN
        agent_config = {
            'learning_rate': _learning_rate,
            'hidden_dims': _network_size,
            'device': device,
            'epsilon_start': initial_epsilon,
            'epsilon_end': min_epsilon,
            'epsilon_decay': 0.99995  # Slower decay for more stable learning
        }
    
    ppo_config['hidden_dims'] = _network_size

    # Store original entropy coefficient for PPO resets
    original_entropy_coef = agent_config.get('entropy_coef', 0.0) if agent_type.lower() == 'ppo' else 0.0
    # Apply any custom agent config
    if custom_agent_config:
        agent_config.update(custom_agent_config)
        # Update original entropy if it was customized
        if agent_type.lower() == 'ppo' and 'entropy_coef' in custom_agent_config:
            original_entropy_coef = custom_agent_config['entropy_coef']
    
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
            'random': 0.01,          # 1% (reduced from 3% since it's quick)
            'naive': 0.03,           # 3% (reduced from 3% since it's quick)
            'conservative': 0.18,    # 18% (slightly reduced to make room)
            'bluff_punisher': 0.20,  # 20% (increased to focus on anti-bluff training)
            'anti_exploitation': 0.15, # 15% (keep the same)
            'aggressive': 0.03,      # 3% (reduced from 4% since it's quick)
            'strategic': 0.12,       # 12% (slightly reduced to make room)
            'adaptive': 0.12,        # 12% (slightly reduced to make room)
            'counter_strategy': 0.12, # 12% (slightly reduced to make room)
            'optimal': 0.04          # 4% (significantly reduced - explained below)
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
    
    # Store all candidate models for final evaluation
    candidate_models = {}
    
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
        
        # Reset/Boost entropy for PPO when facing new opponent level
        if agent_type.lower() == 'ppo' and hasattr(agent, 'entropy_coef'):
            # Calculate entropy boost based on level index and environment complexity
            if level_idx == 0:
                # First level - use the original high entropy
                reset_entropy = original_entropy_coef
            else:
                # Later levels - boost entropy but not higher than original
                # More complex opponents need more exploration
                current_entropy = agent.entropy_coef
                complexity_factor = min(1.0, 0.5 + (level_idx / len(CURRICULUM_LEVELS)))
                # Boost more for harder opponents but don't exceed original
                reset_entropy = min(original_entropy_coef, 
                                   current_entropy * (1.5 + complexity_factor))
                                   
                # Ensure entropy never drops below minimum threshold
                entropy_min = agent_config.get('entropy_min', 0.005)
                reset_entropy = max(reset_entropy, entropy_min * 2)
                
            # Apply the entropy reset
            agent.entropy_coef = reset_entropy
            logger.info(f"Reset entropy coefficient to {reset_entropy:.5f} for level {level_name}")
        
        # Get randomization probability for this level
        randomize_prob = position_randomization_schedule.get(level_name, 1.0 if randomize_positions else 0.0)
        logger.info(f"Using position randomization probability of {randomize_prob:.2f} for {level_name}")
            
        # Create environment with current opponent and appropriate randomization
        env = LiarsDiceEnvWrapper(
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            seed=seed,
            rule_agent_types=[level_name] * (_num_players - 1),
            randomize_positions=randomize_prob  # Pass the probability
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
        
        # Define callback for early stopping, model saving, and metrics tracking
        def checkpoint_callback(episode, data, current_agent):
            nonlocal best_level_winrate, best_level_model_path
            nonlocal best_overall_model, best_overall_winrate
            nonlocal above_threshold_count
            
            if 'last_win_rate' in data:
                current_winrate = data['last_win_rate']
                
                # Log win rate and average reward during important checkpoints
                if 'last_avg_reward' in data:
                    logger.info(f"Episode {episode}: Win rate: {current_winrate:.2f}, " + 
                               f"Avg reward: {data['last_avg_reward']:.2f}")
                
                # Update best model for this level if better
                if current_winrate > best_level_winrate:
                    best_level_winrate = current_winrate
                    best_level_model_path = os.path.join(checkpoint_dir, f"best_{level_name}_model")
                    current_agent.save(best_level_model_path)
                    logger.info(f"New best model for {level_name}: {current_winrate:.2f} win rate")
                
                # Update best overall model
                if current_winrate > best_overall_winrate:
                    best_overall_winrate = current_winrate
                    best_overall_model = os.path.join(checkpoint_dir, "best_overall_model")
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
            eval_episodes=500,  # More episodes for reliable evaluation
            early_stopping=enable_early_stopping,
            win_rate_threshold=_win_rate_threshold,
            patience=early_stopping_patience,
            callback=checkpoint_callback,
            reward_shaping=True  # Enable reward shaping for better learning signals
        )
        
        level_duration = time.time() - level_start_time
        
        # Get best model info from train_agent results if available
        if 'best_win_rate' in level_results and level_results['best_win_rate'] > best_level_winrate:
            best_level_winrate = level_results['best_win_rate']
            logger.info(f"Updated best win rate from training results: {best_level_winrate:.2f}")
            
        if 'best_model_path' in level_results and level_results['best_model_path']:
            best_level_model_path = level_results['best_model_path']
            logger.info(f"Using best model path from training results: {best_level_model_path}")
        
        # Store best model info
        best_models[level_name] = {
            "path": best_level_model_path,
            "win_rate": best_level_winrate
        }
        
        # Add to candidate models for final evaluation
        if best_level_model_path and best_level_winrate >= 0.6:
            candidate_models[f"level_{level_name}"] = best_level_model_path
        
        # Use best model for next level if good enough
        if best_level_model_path and best_level_winrate >= 0.6:
            logger.info(f"Loading best model for {level_name} (win rate: {best_level_winrate:.2f})")
            agent.load(best_level_model_path)
        
        # Track results
        all_results[level_name] = level_results
        
        # Generate training progress graphs
        level_dir = os.path.join(log_dir, f"level_{level_idx}_{level_name}")
        try:
            # Plot training metrics - rewards and episode lengths
            if 'rewards' in level_results and len(level_results['rewards']) > 0:
                plt.figure(figsize=(12, 10))
                
                # Plot 1: Average rewards
                plt.subplot(2, 1, 1)
                rewards = level_results['rewards']
                reward_episodes = [i * level_results.get('log_interval', 100) for i in range(len(rewards))]
                
                plt.plot(reward_episodes, rewards, 'b-', linewidth=2)
                plt.grid(True, alpha=0.3)
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.title(f'Reward Progress vs {level_name.capitalize()} Opponent')
                
                # Add moving average for trend line
                if len(rewards) > 5:
                    window_size = min(5, len(rewards)//5)
                    moving_avg = [sum(rewards[i:i+window_size])/window_size 
                                 for i in range(len(rewards)-window_size+1)]
                    ma_episodes = reward_episodes[:len(moving_avg)]
                    plt.plot(ma_episodes, moving_avg, 'r-', linewidth=1.5, alpha=0.7, 
                            label=f'{window_size}-episode moving average')
                    plt.legend()
                
                # Plot 2: Win rate over time
                plt.subplot(2, 1, 2)
                if 'win_rates' in level_results and len(level_results['win_rates']) > 0:
                    win_rates = level_results['win_rates']
                    eval_points = [i * evaluation_frequency for i in range(len(win_rates))]
                    
                    plt.plot(eval_points, win_rates, 'g-', linewidth=2)
                    plt.axhline(y=_win_rate_threshold, color='r', linestyle='--', 
                              label=f'Threshold ({_win_rate_threshold})')
                    plt.grid(True, alpha=0.3)
                    plt.xlabel('Episode')
                    plt.ylabel('Win Rate')
                    plt.title(f'Win Rate vs {level_name.capitalize()} Opponent')
                    plt.legend()
                    
                    # Add best win rate annotation
                    if best_level_winrate > 0 and win_rates:
                        max_idx = win_rates.index(max(win_rates))
                        max_x = eval_points[max_idx]
                        plt.annotate(f'Best: {max(win_rates):.2f}', 
                                    xy=(max_x, max(win_rates)),
                                    xytext=(max_x+evaluation_frequency*0.2, max(win_rates)+0.05),
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                                    fontsize=10)
                
                plt.tight_layout()
                plt.savefig(os.path.join(level_dir, f"{level_name}_training_progress.png"))
                plt.close()
                
                # Create a dedicated win rate chart
                if 'win_rates' in level_results and len(level_results['win_rates']) > 0:
                    plt.figure(figsize=(10, 6))
                    win_rates = level_results['win_rates']
                    eval_points = [i * evaluation_frequency for i in range(len(win_rates))]
                    
                    # Plot win rate with larger figure
                    plt.plot(eval_points, win_rates, 'g-', linewidth=2.5, label='Win Rate')
                    
                    # Add horizontal threshold line
                    plt.axhline(y=_win_rate_threshold, color='r', linestyle='--', 
                              label=f'Threshold ({_win_rate_threshold})')
                    
                    # Add win rate at key points (start, middle, end)
                    if len(win_rates) > 2:
                        # Mark start point
                        plt.plot(eval_points[0], win_rates[0], 'bo', markersize=8)
                        plt.annotate(f'Start: {win_rates[0]:.2f}', 
                                    xy=(eval_points[0], win_rates[0]),
                                    xytext=(eval_points[0]+evaluation_frequency*0.1, win_rates[0]-0.08),
                                    fontsize=9)
                        
                        # Mark end point
                        plt.plot(eval_points[-1], win_rates[-1], 'go', markersize=8)
                        plt.annotate(f'End: {win_rates[-1]:.2f}', 
                                    xy=(eval_points[-1], win_rates[-1]),
                                    xytext=(eval_points[-1]-evaluation_frequency*0.3, win_rates[-1]+0.08),
                                    fontsize=9)
                        
                        # Mark best point
                        max_idx = win_rates.index(max(win_rates))
                        plt.plot(eval_points[max_idx], win_rates[max_idx], 'ro', markersize=8)
                        plt.annotate(f'Best: {win_rates[max_idx]:.2f}', 
                                    xy=(eval_points[max_idx], win_rates[max_idx]),
                                    xytext=(eval_points[max_idx]+evaluation_frequency*0.1, win_rates[max_idx]+0.08),
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                                    fontsize=10)
                    
                    # Add more formatting
                    plt.grid(True, alpha=0.3)
                    plt.xlabel('Episode', fontsize=12)
                    plt.ylabel('Win Rate', fontsize=12)
                    plt.title(f'Win Rate Progress vs {level_name.capitalize()} Opponent', fontsize=14)
                    plt.legend(fontsize=10)
                    plt.ylim(0, 1.1)  # Set y axis from 0 to 1.1 for clarity
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(level_dir, f"{level_name}_win_rate_chart.png"))
                    plt.close()
            
            logger.info(f"Generated training progress graphs for {level_name} level")
        except Exception as e:
            logger.error(f"Error generating graphs for {level_name}: {e}")

        
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
                    rule_agent_types=[prev_level] * (_num_players - 1)
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
    
    # Add the current agent state as a candidate
    current_model_path = os.path.join(checkpoint_dir, "final_agent")
    agent.save(current_model_path)
    candidate_models["final_agent"] = current_model_path
    
    # Add the best model as a candidate
    if best_overall_model:
        candidate_models["best_overall_model"] = best_overall_model
    
    # First find the best overall model by evaluating all candidates against all opponents
    logger.info("\n=== First Evaluation: Finding Best Overall Model ===")
    logger.info(f"Found {len(candidate_models)} candidate models to evaluate")
    
    # Initialize variables to track the true best overall model
    final_best_model = None
    final_best_winrate = 0.0
    model_evaluation_results = {}
    
    # Define evaluation function to get consistent results
    def evaluate_model_against_all_opponents(model_path):
        # Load the model to evaluate
        agent.load(model_path)
        
        # Reset entropy to a low value for evaluation
        if agent_type.lower() == 'ppo' and hasattr(agent, 'entropy_coef'):
            agent.entropy_coef = agent_config.get('entropy_min', 0.005)
        
        # Evaluate against all curriculum levels
        eval_results = evaluate_against_curriculum(
            agent=agent,
            num_episodes_per_level=1000,  # Use sufficient episodes for stable results
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            epsilon=0.05,  # Low exploration for evaluation
            seed=seed,
            randomize_positions=randomize_positions,
            verbose=False  # Suppress detailed output
        )
        
        # Return the overall win rate and results
        return eval_results['overall']['win_rate'], eval_results
    
    # Evaluate each candidate model
    for model_name, model_path in candidate_models.items():
        logger.info(f"Evaluating candidate model: {model_name}")
        
        try:
            overall_winrate, detailed_results = evaluate_model_against_all_opponents(model_path)
            model_evaluation_results[model_name] = {
                'path': model_path,
                'overall_winrate': overall_winrate,
                'detailed_results': detailed_results
            }
            
            logger.info(f"  {model_name}: Overall win rate = {overall_winrate:.4f}")
            
            # Check if this is the best model so far
            if overall_winrate > final_best_winrate:
                final_best_winrate = overall_winrate
                final_best_model = model_path
                logger.info(f"  New best overall model: {model_name} with {overall_winrate:.4f} win rate")
                
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
    
    if not final_best_model:
        logger.warning("No valid best model found! Using the latest model.")
        final_best_model = current_model_path
        final_best_winrate = 0.0
    
    # Now we have the true best overall model, load it for remedial training
    logger.info(f"\nBest overall model selected: with overall win rate: {final_best_winrate:.4f}")
    agent.load(final_best_model)
    
    # Save detailed evaluation of the best model before remedial training
    pre_remedial_evaluation = model_evaluation_results.get(
        next(name for name, info in model_evaluation_results.items() if info['path'] == final_best_model),
        {'detailed_results': None}
    )['detailed_results']
    
    # REMEDIAL TRAINING ON THE BEST OVERALL MODEL
    if enable_remedial:
        logger.info("\n=== Starting Remedial Training on Best Overall Model ===")
        
        # We already have the evaluation results from the previous evaluation
        if pre_remedial_evaluation:
            eval_results = pre_remedial_evaluation
        else:
            # Re-evaluate if we don't have the results
            _, eval_results = evaluate_model_against_all_opponents(final_best_model)
        
        # Identify levels that need remedial training (below win threshold)
        remedial_levels = []
        for level_name in CURRICULUM_LEVELS:
            if level_name in eval_results and eval_results[level_name]['win_rate'] < _win_rate_threshold:
                remedial_levels.append(level_name)
        
        if remedial_levels:
            logger.info(f"Identified {len(remedial_levels)} levels needing remedial training: {remedial_levels}")
            
            # Create a copy of the best model to preserve it
            pre_remedial_best_model = os.path.join(checkpoint_dir, "pre_remedial_best")
            agent.save(pre_remedial_best_model)
            
            # Train on weak levels (starting with easiest)
            remedial_levels.sort(key=lambda x: CURRICULUM_LEVELS.index(x))
            
            for level_name in remedial_levels:
                logger.info(f"Starting remedial training against {level_name}")
                
                # Set moderate exploration for remedial training
                if agent_type.lower() == 'dqn':
                    remedial_epsilon = 0.3
                    logger.info(f"Setting epsilon to {remedial_epsilon:.2f} for remedial training")
                    agent.epsilon = remedial_epsilon
                
                # Boost entropy for PPO remedial training
                if agent_type.lower() == 'ppo' and hasattr(agent, 'entropy_coef'):
                    # Set higher entropy for remedial training
                    remedial_entropy = original_entropy_coef * 0.8  # 80% of original
                    agent.entropy_coef = remedial_entropy
                    logger.info(f"Boosted entropy to {remedial_entropy:.5f} for remedial training")
                
                # Create remedial environment
                remedial_env = LiarsDiceEnvWrapper(
                    num_players=_num_players,
                    num_dice=_num_dice,
                    dice_faces=dice_faces,
                    seed=seed,
                    rule_agent_types=[level_name] * (_num_players - 1)
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
                    eval_episodes=500,
                    early_stopping=True,
                    win_rate_threshold=_win_rate_threshold,
                    patience=2,  # Less patience for remedial
                    callback=remedial_callback,
                    reward_shaping=True
                )
                
                # Log remedial completion
                logger.info(f"Completed remedial training against {level_name}")
                logger.info(f"Final remedial win rate: {remedial_best_winrate:.2f}")
                
                # Update best models tracking for this level
                if remedial_best_path and remedial_best_winrate > best_models.get(level_name, {}).get("win_rate", 0):
                    best_models[level_name] = {
                        "path": remedial_best_path,
                        "win_rate": remedial_best_winrate
                    }
            
            # After all remedial training, evaluate the newly trained model against all opponents
            logger.info("\n=== Evaluating Remedial-Trained Model ===")
            post_remedial_winrate, post_remedial_results = evaluate_model_against_all_opponents(current_model_path)
            
            logger.info(f"Pre-remedial overall win rate: {final_best_winrate:.4f}")
            logger.info(f"Post-remedial overall win rate: {post_remedial_winrate:.4f}")
            
            # Only keep the remedial-trained model if it improved the overall win rate
            if post_remedial_winrate > final_best_winrate:
                logger.info("Remedial training improved overall performance - keeping remedial model")
                final_best_model = current_model_path
                final_best_winrate = post_remedial_winrate
            else:
                logger.info("Remedial training did not improve overall performance - reverting to pre-remedial model")
                agent.load(pre_remedial_best_model)  # Revert to the pre-remedial model
        else:
            logger.info("No levels need remedial training - all already above threshold!")
    
    # Save the final best model with a descriptive name
    best_model_copy = os.path.join(checkpoint_dir, f"best_overall_{final_best_winrate:.4f}")
    agent.save(best_model_copy)
    logger.info(f"Saved best overall model to {best_model_copy}")
    
    # Final evaluation using the best model
    logger.info("\n=== Final Evaluation of Best Model ===")
    
    # Ensure we're using the best overall model for final evaluation
    if final_best_model:
        logger.info(f"Loading best overall model with win rate {final_best_winrate:.4f} for final evaluation")
        agent.load(final_best_model)
    
    # Set low entropy for final evaluation
    if agent_type.lower() == 'ppo' and hasattr(agent, 'entropy_coef'):
        agent.entropy_coef = agent_config.get('entropy_min', 0.005)
    
    final_eval_results = evaluate_against_curriculum(
        agent=agent,
        num_episodes_per_level=1000,  # More episodes for final evaluation
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
    
    # Create additional comprehensive visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Extract win rates for each level
        levels = [level for level in CURRICULUM_LEVELS if level in final_eval_results]
        win_rates = [final_eval_results[level]['win_rate'] for level in levels]
        
        # Create bar chart
        bars = plt.bar(range(len(levels)), win_rates, color='skyblue')
        
        # Add threshold line
        plt.axhline(y=_win_rate_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({_win_rate_threshold:.2f})')
        
        # Add data labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Format plot
        plt.xlabel('Opponent Level')
        plt.ylabel('Win Rate')
        plt.title('Final Agent Performance Across All Opponent Levels')
        plt.xticks(range(len(levels)), [l.capitalize() for l in levels], rotation=45)
        plt.ylim(0, 1.1)  # 0 to 110% to make room for labels
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        comprehensive_vis_path = os.path.join(log_dir, "comprehensive_evaluation.png")
        plt.savefig(comprehensive_vis_path)
        plt.close()
        
        # Create win rate progression graph across curriculum
        if all_results:
            plt.figure(figsize=(12, 6))
            
            # Extract max win rates for each curriculum level
            curriculum_levels = []
            best_win_rates = []
            
            for level in CURRICULUM_LEVELS:
                if level in all_results and 'win_rates' in all_results[level]:
                    curriculum_levels.append(level)
                    level_win_rates = all_results[level]['win_rates']
                    if level_win_rates:
                        best_win_rates.append(max(level_win_rates))
                    else:
                        best_win_rates.append(0)
            
            # Plot progression
            plt.plot(range(len(curriculum_levels)), best_win_rates, 'bo-', linewidth=2, markersize=8)
            
            # Add threshold line
            plt.axhline(y=_win_rate_threshold, color='r', linestyle='--', 
                      label=f'Threshold ({_win_rate_threshold:.2f})')
            
            # Format plot
            plt.xlabel('Curriculum Level')
            plt.ylabel('Best Win Rate')
            plt.title('Training Progress Across Curriculum Levels')
            plt.xticks(range(len(curriculum_levels)), [l.capitalize() for l in curriculum_levels], rotation=45)
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            progress_vis_path = os.path.join(log_dir, "curriculum_progress.png")
            plt.savefig(progress_vis_path)
            plt.close()
            
            logger.info(f"Created additional visualization charts at {log_dir}")
    except Exception as e:
        logger.error(f"Error creating comprehensive visualizations: {e}")
    
    # Save combined training data
    combined_results = {
        'curriculum_distribution': episode_distribution,
        'level_results': all_results,
        'final_evaluation': final_eval_results,
        'best_models': best_models,
        'model_evaluation_results': model_evaluation_results,
        'best_overall_model': {
            'path': final_best_model,
            'win_rate': final_best_winrate
        },
        'parameters': {
            'agent_type': agent_type,
            'preset': preset,
            'num_players': _num_players,
            'num_dice': _num_dice,
            'dice_faces': dice_faces,
            'curriculum_episodes': _curriculum_episodes,
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
            'curriculum_episodes': _curriculum_episodes
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
    parser.add_argument('--preset', type=str, default='standard', choices=['basic', 'standard', 'advanced'],
                        help='Configuration preset to use')
    parser.add_argument('--path', type=str, default='final_new_results/curriculum_learning/ppo_5dice_2person', help='Base path for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
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
        enable_remedial=not args.no_remedial,
        enable_early_stopping=not args.no_early_stopping,
        evaluation_frequency=500,
        checkpoint_frequency=50000
    )