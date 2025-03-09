"""
Enhanced self-play training for reinforcement learning agents in Liar's Dice.

This module implements an improved self-play training approach where an agent learns
by playing against a diverse pool of its own past versions, with adaptive exploration
and periodic evaluation against specialized agents.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import random
import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import deque

from agents.base_agent import RLAgent
from agents.agent_factory import create_agent
# Import the specialized evaluation agents
from agents.evaluation_agents import create_agent as create_rule_agent
from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from training.environment_wrapper import LiarsDiceEnvWrapper
from training.train import train_episode, evaluate_agent
from training.utils import setup_logger, save_training_data, plot_training_results

# Specialized evaluation agents for better performance assessment
EVALUATION_AGENTS = [
    'beginner',         # Tests basic pattern exploitation
    'conservative',     # Tests exploitation of cautious play
    'aggressive',       # Tests bluff detection and punishment
    'adaptive',         # Tests adaptability and robustness
    'optimal'           # Tests proximity to optimal play
]

# Configuration presets for different game setups
CONFIG_PRESETS = {
    # 2 players, 3 dice, good for initial learning
    'basic': {
        'num_players': 2,
        'num_dice': 3,
        'self_play_episodes': 100000,
        'network_size': [256, 128, 64],
        'learning_rate': 0.0005,
        'win_rate_threshold': 0.9
    },
    # 2 players, 5 dice, standard game
    'standard': {
        'num_players': 2,
        'num_dice': 5,
        'self_play_episodes': 250000,
        'network_size': [512, 256, 128, 64],
        'learning_rate': 0.0002,
        'win_rate_threshold': 0.9
    },
    # 4 players, 5 dice, complex game
    'advanced': {
        'num_players': 4,
        'num_dice': 5,
        'self_play_episodes': 300000,
        'network_size': [1024, 512, 256, 128],
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
    evaluation_opponents: List[str] = EVALUATION_AGENTS,
    
    # Agent configuration
    learning_rate: Optional[float] = None,
    network_size: Optional[List[int]] = None,
    custom_agent_config: Optional[Dict[str, Any]] = None,
    
    # Training control
    checkpoint_frequency: int = 50000,
    evaluation_frequency: int = 1000,
    enable_early_stopping: bool = True,
    win_rate_threshold: Optional[float] = None,
    early_stopping_patience: int = 3,
    
    # Self-play settings - IMPROVED VALUES
    opponent_pool_size: int = 15,         # Increased from 5 to 15
    update_pool_frequency: int = 1000,    # Reduced from 5000 to 1000
    newest_model_freq: float = 0.4,       # Reduced from 0.7 to 0.4
    
    # Adaptive parameters (new)
    enable_adaptive_params: bool = True,
    entropy_schedule: Optional[List[float]] = None,
    learning_rate_schedule: Optional[List[float]] = None,
    
    # Visualization settings
    visualize_progress: bool = True,
    visualize_interval: int = 5000
) -> Tuple[RLAgent, Dict[str, Any]]:
    """
    Train an agent using enhanced self-play, where it learns by playing against diverse versions of itself.
    
    This improved implementation features:
    - Larger and more diverse opponent pool
    - Better pool management strategy
    - Adaptive hyperparameters
    - More frequent opponent pool updates
    - Evaluation against specialized agents
    - Progress visualization
    
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
        enable_early_stopping: Whether to stop training early when win rate is high
        win_rate_threshold: Win rate threshold for early stopping (overrides preset if specified)
        early_stopping_patience: Number of evaluations above threshold to trigger early stopping
        
        # Self-play settings
        opponent_pool_size: Number of past model versions to keep in the opponent pool
        update_pool_frequency: How often to update the pool of opponents
        newest_model_freq: Frequency of playing against the newest model
        
        # Adaptive parameters
        enable_adaptive_params: Whether to use adaptive hyperparameters
        entropy_schedule: Schedule for entropy coefficient (PPO only)
        learning_rate_schedule: Schedule for learning rate
        
        # Visualization settings
        visualize_progress: Whether to visualize training progress
        visualize_interval: How often to update visualizations
        
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
    run_name = f"enhanced_selfplay_{agent_type}_{preset}_{_num_players}p{_num_dice}d_{timestamp}"
    base_path = os.path.join(results_path, run_name)
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'logs')
    visualization_dir = os.path.join(base_path, 'visualizations')
    
    # Create directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('enhanced_selfplay', os.path.join(log_dir, 'selfplay.log'))
    logger.info(f"Starting enhanced self-play training for Liar's Dice with {agent_type} agent")
    logger.info(f"Game setup: {_num_players} players, {_num_dice} dice, {dice_faces} faces")
    logger.info(f"Self-play: {_self_play_episodes} episodes")
    logger.info(f"Opponent pool size: {opponent_pool_size}, Update frequency: {update_pool_frequency}")
    
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
            torch.backends.cudnn.deterministic = True
    
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
    
    # Create agent configuration with improved defaults
    if agent_type.lower() == 'ppo':
        agent_config = {
            'learning_rate': _learning_rate,
            'hidden_dims': _network_size,
            'device': device,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'policy_clip': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.05,        # Higher initial entropy for more exploration
            'entropy_min': 0.01,         # Minimum entropy coefficient
            'entropy_decay_steps': _self_play_episodes // 3,  # Decay over 1/3 of training
            'update_frequency': 2048,
            'batch_size': 64,
            'ppo_epochs': 4,
            'max_grad_norm': 0.5
        }
    else:  # DQN
        agent_config = {
            'learning_rate': _learning_rate,
            'hidden_dims': _network_size,
            'device': device,
            'epsilon_start': 0.5,       # Lower initial exploration for self-play
            'epsilon_end': 0.05,
            'epsilon_decay_steps': _self_play_episodes // 3  # Decay over 1/3 of training
        }
    
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
    
    # Setup opponent pool for self-play with better management
    opponent_pool = []  # Will contain (path, model_version, episodes_trained)
    latest_opponent_path = None
    
    # Tracking variables
    best_eval_reward = float('-inf')
    best_win_rate = 0.0
    best_model_path = None
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_win_rates = []
    detailed_win_rates = {opponent_type: [] for opponent_type in evaluation_opponents}
    high_win_rate_count = 0
    
    # Initialize entropy schedule if using PPO and adaptive parameters
    if agent_type.lower() == 'ppo' and enable_adaptive_params:
        if entropy_schedule is None:
            # Default schedule: gradually decrease entropy
            initial_entropy = agent_config['entropy_coef']
            min_entropy = agent_config['entropy_min']
            # Create a schedule with 10 steps
            entropy_schedule = np.linspace(initial_entropy, min_entropy, 10).tolist()
        logger.info(f"Using entropy schedule: {entropy_schedule}")
    
    # Initialize learning rate schedule if using adaptive parameters
    if enable_adaptive_params:
        if learning_rate_schedule is None:
            # Default schedule: gradually decrease learning rate
            initial_lr = agent_config['learning_rate']
            min_lr = initial_lr * 0.1  # 10% of initial rate
            # Create a schedule with 10 steps
            learning_rate_schedule = np.linspace(initial_lr, min_lr, 10).tolist()
        logger.info(f"Using learning rate schedule: {learning_rate_schedule}")
    
    # Training loop with tqdm progress bar
    logger.info("Starting enhanced self-play training...")
    start_time = time.time()
    
    # Create progress bar
    progress_bar = tqdm.tqdm(total=_self_play_episodes, desc="Self-Play Training", 
                            unit="episode", dynamic_ncols=True)
    
    for episode in range(1, _self_play_episodes + 1):
        # Periodically update the opponent pool
        if episode % update_pool_frequency == 1 or not opponent_pool:
            # Save current model state
            model_path = os.path.join(checkpoint_dir, f"model_ep{episode}")
            primary_agent.save(model_path)
            
            # Add to opponent pool with metadata
            opponent_pool.append((model_path, episode // update_pool_frequency, episode))
            latest_opponent_path = model_path
            
            # Keep pool at desired size with better management
            if len(opponent_pool) > opponent_pool_size:
                # Strategy 1: Keep first model, latest model, and random selection of others
                if len(opponent_pool) > 2:
                    # Keep the first (oldest) and the most recent
                    keepers = [0, len(opponent_pool) - 1]  
                    # Randomly select others to keep
                    available_indices = list(range(1, len(opponent_pool) - 1))
                    num_to_keep = opponent_pool_size - 2
                    if num_to_keep > 0 and available_indices:
                        # Select random indices to keep (up to num_to_keep)
                        random_keepers = random.sample(available_indices, 
                                                    min(num_to_keep, len(available_indices)))
                        keepers.extend(random_keepers)
                    
                    # Create new pool with only the selected models
                    opponent_pool = [opponent_pool[i] for i in sorted(keepers)]
                    
            logger.info(f"Updated opponent pool. Current size: {len(opponent_pool)}")
        
        # Adaptive parameter updates based on training progress
        if enable_adaptive_params:
            # Determine current phase (0-9) based on episode progress
            phase = min(9, int(episode / _self_play_episodes * 10))
            
            # Update learning rate if provided
            if learning_rate_schedule and hasattr(primary_agent, 'optimizer'):
                current_lr = learning_rate_schedule[phase]
                for param_group in primary_agent.optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            # Update entropy coefficient for PPO
            if agent_type.lower() == 'ppo' and entropy_schedule and hasattr(primary_agent, 'entropy_coef'):
                primary_agent.entropy_coef = entropy_schedule[phase]
        
        # Select opponent from pool with improved selection strategy
        if opponent_pool:
            if random.random() < newest_model_freq and latest_opponent_path:
                # Use the latest model (with probability newest_model_freq)
                opponent_path = latest_opponent_path
            else:
                # Select opponent with weighted probability (newer models more likely)
                weights = [1.0 + 0.5 * i for i in range(len(opponent_pool))]  # Increasing weights
                selected_idx = random.choices(range(len(opponent_pool)), weights=weights, k=1)[0]
                opponent_path = opponent_pool[selected_idx][0]
        else:
            # If no opponents yet, play against self
            opponent_path = None
        
        # Create opponent agent
        if opponent_path:
            opponent_agent = create_agent(
                agent_type=agent_type,
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=agent_config
            )
            opponent_agent.load(opponent_path)
        else:
            # Use a copy of the primary agent for first episodes
            opponent_agent = primary_agent
        
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
        
        # Update progress bar
        progress_bar.update(1)
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            progress_bar.set_postfix({
                'Avg Reward': f'{avg_reward:.2f}',
                'Avg Length': f'{avg_length:.2f}',
                'LR': f'{primary_agent.optimizer.param_groups[0]["lr"]:.5f}' if hasattr(primary_agent, 'optimizer') else 'N/A',
                'Entropy': f'{primary_agent.entropy_coef:.4f}' if hasattr(primary_agent, 'entropy_coef') else 'N/A'
            })
        
        # Log progress periodically
        if episode % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            avg_length = np.mean(episode_lengths[-500:])
            time_taken = time.time() - start_time
            logger.info(f"Episode {episode}/{_self_play_episodes} | "
                       f"Avg Reward: {avg_reward:.2f} | "
                       f"Avg Length: {avg_length:.2f} | "
                       f"Time: {time_taken:.2f}s")
        
        # Periodic evaluation against specialized rule-based agents
        if episode % evaluation_frequency == 0:
            logger.info(f"Evaluating agent at episode {episode}...")
            progress_bar.set_description("Evaluating agent")
            
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
                    seed=seed + 10000 + episode if seed else None,
                    rule_agent_types=[opponent_type]
                )
                
                primary_agent.set_action_mapping(eval_env.action_mapping)
                
                # Run evaluation
                num_eval_episodes = 1000
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
                
                # Store detailed win rates
                detailed_win_rates[opponent_type].append(win_rate)
                
                logger.info(f"  vs {opponent_type}: Win Rate = {win_rate:.2f}, Reward = {avg_reward:.2f}")
            
            # Calculate overall win rate
            overall_win_rate = total_wins / total_games
            overall_eval_reward = sum(r['avg_reward'] for r in eval_results.values()) / len(eval_results)
            
            logger.info(f"Overall Evaluation: Win Rate = {overall_win_rate:.2f}, Reward = {overall_eval_reward:.2f}")
            
            # Reset progress bar description
            progress_bar.set_description("Self-Play Training")
            
            eval_rewards.append(overall_eval_reward)
            eval_win_rates.append(overall_win_rate)
            
            # Check if this is the best model so far
            if overall_win_rate > best_win_rate:
                best_win_rate = overall_win_rate
                best_eval_reward = overall_eval_reward
                best_model_path = os.path.join(checkpoint_dir, f"best_model_ep{episode}")
                primary_agent.save(best_model_path)
                logger.info(f"New best model saved with win rate {best_win_rate:.2f}")
            
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
                'eval_win_rates': eval_win_rates,
                'detailed_win_rates': detailed_win_rates
            }
            
            with open(os.path.join(log_dir, "training_progress.json"), 'w') as f:
                json.dump(training_data, f, indent=2)
        
        # Visualize progress
        if visualize_progress and (episode % visualize_interval == 0 or episode == _self_play_episodes):
            # Call visualization function
            create_training_visualizations(
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                eval_rewards=eval_rewards,
                eval_win_rates=eval_win_rates,
                detailed_win_rates=detailed_win_rates,
                episode=episode,
                win_rate_threshold=_win_rate_threshold,
                save_dir=visualization_dir
            )
    
    # Close progress bar when done
    progress_bar.close()
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Self-play training completed after {episode} episodes in "
                f"{int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Final comprehensive evaluation against all evaluation agents
    logger.info("\n=== Final Comprehensive Evaluation ===")
    
    # Set appropriate exploration parameters for evaluation
    if agent_type.lower() == 'dqn' and hasattr(primary_agent, 'epsilon'):
        original_epsilon = primary_agent.epsilon
        primary_agent.epsilon = 0.05  # Low exploration for evaluation
    
    if agent_type.lower() == 'ppo' and hasattr(primary_agent, 'entropy_coef'):
        original_entropy = primary_agent.entropy_coef
        primary_agent.entropy_coef = 0.01  # Low entropy for evaluation
    
    # Evaluate against each specialized agent
    final_eval_results = {}
    total_wins = 0
    total_games = 0
    
    # Create a progress bar for the final evaluation
    eval_progress = tqdm.tqdm(
        total=len(evaluation_opponents) * 250,
        desc="Final Evaluation",
        unit="episode",
        dynamic_ncols=True
    )
    
    for opponent_type in evaluation_opponents:
        eval_env = LiarsDiceEnvWrapper(
            num_players=_num_players,
            num_dice=_num_dice,
            dice_faces=dice_faces,
            seed=seed + 20000 if seed else None,
            rule_agent_types=[opponent_type]
        )
        
        primary_agent.set_action_mapping(eval_env.action_mapping)
        
        # Run more episodes for final evaluation
        num_eval_episodes = 250  # More episodes for better statistics
        eval_reward = 0
        wins = 0
        episode_lengths = []
        
        for _ in range(num_eval_episodes):
            episode_reward, length, _ = train_episode(eval_env, primary_agent, evaluation=True)
            eval_reward += episode_reward
            episode_lengths.append(length)
            if episode_reward > 0:
                wins += 1
            eval_progress.update(1)
        
        win_rate = wins / num_eval_episodes
        avg_reward = eval_reward / num_eval_episodes
        avg_length = sum(episode_lengths) / len(episode_lengths)
        
        final_eval_results[opponent_type] = {
            'win_rate': win_rate,
            'mean_reward': avg_reward,
            'mean_episode_length': avg_length
        }
        
        total_wins += wins
        total_games += num_eval_episodes
        
        logger.info(f"  vs {opponent_type}: Win Rate = {win_rate:.2f}, Reward = {avg_reward:.2f}")
    
    eval_progress.close()
    
    # Calculate overall statistics
    overall_win_rate = total_wins / total_games
    overall_reward = sum(r['mean_reward'] for r in final_eval_results.values()) / len(final_eval_results)
    
    final_eval_results['overall'] = {
        'win_rate': overall_win_rate,
        'mean_reward': overall_reward
    }
    
    logger.info(f"\nFinal Overall: Win Rate = {overall_win_rate:.2f}, Reward = {overall_reward:.2f}")
    
    # Restore original exploration parameters
    if agent_type.lower() == 'dqn' and hasattr(primary_agent, 'epsilon'):
        primary_agent.epsilon = original_epsilon
    
    if agent_type.lower() == 'ppo' and hasattr(primary_agent, 'entropy_coef'):
        primary_agent.entropy_coef = original_entropy
    
    # Save and visualize final evaluation
    with open(os.path.join(log_dir, "final_evaluation.json"), 'w') as f:
        json.dump(final_eval_results, f, indent=2)
    
    # Create final evaluation visualization
    create_final_evaluation_visualization(
        final_eval_results, 
        save_path=os.path.join(visualization_dir, "final_evaluation.png")
    )
    
    # Create final dashboard visualization
    create_final_dashboard(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        eval_win_rates=eval_win_rates,
        detailed_win_rates=detailed_win_rates,
        final_eval_results=final_eval_results,
        agent_type=agent_type,
        preset=preset,
        win_rate_threshold=_win_rate_threshold,
        save_path=os.path.join(visualization_dir, "final_dashboard.png")
    )
    
    # If we found a best model, load it
    if best_model_path and os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        primary_agent.load(best_model_path)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    primary_agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save model to models directory with metadata
    models_dir = os.path.join(results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model name with performance info
    model_name = f"enhanced_selfplay_{agent_type}_{_num_players}p{_num_dice}d_{overall_win_rate:.2f}wr_{timestamp}"
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
            'self_play_episodes': episode,  # Actual episodes trained
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
        'detailed_win_rates': detailed_win_rates,
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
            'win_rate_threshold': _win_rate_threshold,
            'opponent_pool_size': opponent_pool_size
        }
    }
    
    return primary_agent, training_results


def create_training_visualizations(
    episode_rewards: List[float],
    episode_lengths: List[float],
    eval_rewards: List[float],
    eval_win_rates: List[float],
    detailed_win_rates: Dict[str, List[float]],
    episode: int,
    win_rate_threshold: float,
    save_dir: str = 'visualizations'
):
    """
    Create training progress visualizations.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        eval_rewards: List of evaluation rewards
        eval_win_rates: List of overall win rates during evaluation
        detailed_win_rates: Dictionary mapping opponent types to lists of win rates
        episode: Current episode number
        win_rate_threshold: Win rate threshold for early stopping
        save_dir: Directory to save visualizations
    """
    # Set visualization style
    sns.set(style="whitegrid")
    
    # Create rewards plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards with low alpha
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, 
             alpha=0.2, color='blue', label='Individual Episodes')
    
    # Plot smoothed rewards
    window_size = min(500, max(1, len(episode_rewards) // 10))
    if len(episode_rewards) > window_size:
        smoothed_rewards = []
        for i in range(len(episode_rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(episode_rewards[i:i+window_size]))
        plt.plot(range(window_size, len(episode_rewards) + 1), smoothed_rewards, 
                 linewidth=2, color='darkblue', label=f'{window_size}-Episode Moving Average')
    
    plt.title(f'Training Rewards (Episode {episode})', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rewards.png"))
    plt.close()
    
    # Create episode length plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw episode lengths with low alpha
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths, 
             alpha=0.2, color='green', label='Individual Episodes')
    
    # Plot smoothed episode lengths
    if len(episode_lengths) > window_size:
        smoothed_lengths = []
        for i in range(len(episode_lengths) - window_size + 1):
            smoothed_lengths.append(np.mean(episode_lengths[i:i+window_size]))
        plt.plot(range(window_size, len(episode_lengths) + 1), smoothed_lengths, 
                 linewidth=2, color='darkgreen', label=f'{window_size}-Episode Moving Average')
    
    plt.title(f'Episode Lengths (Episode {episode})', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "episode_lengths.png"))
    plt.close()
    
    # Create win rate plot
    if eval_win_rates:
        plt.figure(figsize=(12, 6))
        
        # Plot overall win rate
        eval_episodes = range(len(eval_win_rates))
        plt.plot(eval_episodes, eval_win_rates, 'o-', linewidth=2, color='blue', label='Overall Win Rate')
        
        # Plot threshold line
        plt.axhline(y=win_rate_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({win_rate_threshold:.2f})')
        
        # Plot win rates against individual opponent types
        for opponent, win_rates in detailed_win_rates.items():
            if win_rates:
                plt.plot(range(len(win_rates)), win_rates, 'o-', linewidth=1.5, 
                        alpha=0.7, label=f'vs {opponent}')
        
        plt.title(f'Win Rate Over Time (Episode {episode})', fontsize=14)
        plt.xlabel('Evaluation Index', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "win_rates.png"))
        plt.close()
    
    # Create combined dashboard
    plt.figure(figsize=(18, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    if len(episode_rewards) > window_size:
        plt.plot(range(window_size, len(episode_rewards) + 1), smoothed_rewards, 
                linewidth=2, color='blue')
    plt.title('Training Rewards', fontsize=12)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    if len(episode_lengths) > window_size:
        plt.plot(range(window_size, len(episode_lengths) + 1), smoothed_lengths, 
                linewidth=2, color='green')
    plt.title('Episode Lengths', fontsize=12)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Length', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot win rates
    plt.subplot(2, 2, 3)
    if eval_win_rates:
        plt.plot(eval_episodes, eval_win_rates, 'o-', linewidth=2, color='blue')
        plt.axhline(y=win_rate_threshold, color='red', linestyle='--')
    plt.title('Overall Win Rate', fontsize=12)
    plt.xlabel('Evaluation Index', fontsize=10)
    plt.ylabel('Win Rate', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Plot win rates by opponent type
    plt.subplot(2, 2, 4)
    if detailed_win_rates:
        for opponent, win_rates in detailed_win_rates.items():
            if win_rates:
                plt.plot(range(len(win_rates)), win_rates, 'o-', linewidth=1.5, 
                        label=f'{opponent}')
        plt.title('Win Rate by Opponent Type', fontsize=12)
        plt.xlabel('Evaluation Index', fontsize=10)
        plt.ylabel('Win Rate', fontsize=10)
        plt.ylim(0, 1.05)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_dashboard.png"))
    plt.close()


def create_final_evaluation_visualization(final_eval_results, save_path):
    """
    Create visualization of final evaluation results.
    
    Args:
        final_eval_results: Dictionary containing final evaluation results
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Extract win rates against each agent type
    agent_types = [t for t in final_eval_results.keys() if t != 'overall']
    win_rates = [final_eval_results[t]['win_rate'] for t in agent_types]
    
    # Create color map based on win rates
    colors = []
    for win_rate in win_rates:
        if win_rate < 0.4:
            colors.append('crimson')  # Poor performance
        elif win_rate < 0.5:
            colors.append('darkorange')  # Below average
        elif win_rate < 0.6:
            colors.append('gold')  # Average
        elif win_rate < 0.75:
            colors.append('yellowgreen')  # Good
        else:
            colors.append('darkgreen')  # Excellent
    
    # Create bar chart
    bars = plt.bar(agent_types, win_rates, color=colors)
    
    # Add win rate values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Add overall win rate as a horizontal line
    overall_win_rate = final_eval_results['overall']['win_rate']
    plt.axhline(y=overall_win_rate, color='black', linestyle='--', 
               linewidth=2, label=f'Overall: {overall_win_rate:.2f}')
    
    # Add 50% win rate reference line
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label='50% Baseline')
    
    plt.title('Final Evaluation: Win Rate Against Different Agents', fontsize=16)
    plt.xlabel('Agent Type', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.ylim(0, min(1.1, max(win_rates) + 0.15))  # Set y-axis with some headroom
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()


def create_final_dashboard(
    episode_rewards,
    episode_lengths,
    eval_win_rates,
    detailed_win_rates,
    final_eval_results,
    agent_type,
    preset,
    win_rate_threshold,
    save_path
):
    """
    Create comprehensive final dashboard visualization.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        eval_win_rates: List of overall win rates during evaluation
        detailed_win_rates: Dictionary mapping opponent types to lists of win rates
        final_eval_results: Dictionary containing final evaluation results
        agent_type: Type of agent used ('dqn' or 'ppo')
        preset: Configuration preset used
        win_rate_threshold: Win rate threshold used for early stopping
        save_path: Path to save the visualization
    """
    # Set visualization style
    sns.set(style="whitegrid")
    
    # Create figure
    plt.figure(figsize=(20, 15))
    
    # Define grid layout
    gs = plt.GridSpec(3, 3, height_ratios=[1, 1, 1])
    
    # Panel 1: Training Rewards
    ax1 = plt.subplot(gs[0, 0:2])
    
    # Plot smoothed rewards
    window_size = min(500, max(1, len(episode_rewards) // 10))
    if len(episode_rewards) > window_size:
        smoothed_rewards = []
        for i in range(len(episode_rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(episode_rewards[i:i+window_size]))
        ax1.plot(range(window_size, len(episode_rewards) + 1), smoothed_rewards, 
                linewidth=2, color='blue')
    
    ax1.set_title('Training Rewards', fontsize=14)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Final Evaluation
    ax2 = plt.subplot(gs[0, 2])
    
    # Extract win rates for each agent type
    agent_types = [t for t in final_eval_results.keys() if t != 'overall']
    win_rates = [final_eval_results[t]['win_rate'] for t in agent_types]
    
    # Create color map
    colors = []
    for win_rate in win_rates:
        if win_rate < 0.4:
            colors.append('crimson')
        elif win_rate < 0.5:
            colors.append('darkorange')
        elif win_rate < 0.6:
            colors.append('gold')
        elif win_rate < 0.75:
            colors.append('yellowgreen')
        else:
            colors.append('darkgreen')
    
    # Create horizontal bar chart for better readability
    bars = ax2.barh(agent_types, win_rates, color=colors)
    
    # Add win rate values inside bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(max(0.1, width - 0.15), bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', ha='center', va='center', 
                color='white' if width > 0.3 else 'black',
                fontsize=10, fontweight='bold')
    
    # Add overall win rate as a vertical line
    overall_win_rate = final_eval_results['overall']['win_rate']
    ax2.axvline(x=overall_win_rate, color='black', linestyle='--', 
               linewidth=2, label=f'Overall: {overall_win_rate:.2f}')
    
    ax2.set_title('Final Evaluation', fontsize=14)
    ax2.set_xlabel('Win Rate', fontsize=12)
    ax2.set_xlim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Win Rate Evolution
    ax3 = plt.subplot(gs[1, 0:2])
    
    if eval_win_rates:
        # Calculate evaluation episodes
        eval_episodes = range(len(eval_win_rates))
        
        # Plot overall win rate
        ax3.plot(eval_episodes, eval_win_rates, 'o-', linewidth=2.5, 
                color='blue', label='Overall')
        
        # Add threshold line
        ax3.axhline(y=win_rate_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({win_rate_threshold:.2f})')
        
        # Plot trendline if possible
        if len(eval_win_rates) > 2:
            z = np.polyfit(eval_episodes, eval_win_rates, 1)
            p = np.poly1d(z)
            ax3.plot(eval_episodes, p(eval_episodes), 'b--', alpha=0.7)
        
        # Add annotations for start and end
        if eval_win_rates:
            ax3.annotate(f'Start: {eval_win_rates[0]:.2f}', 
                        xy=(0, eval_win_rates[0]),
                        xytext=(1, eval_win_rates[0] + 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1))
            
            ax3.annotate(f'End: {eval_win_rates[-1]:.2f}', 
                        xy=(len(eval_win_rates)-1, eval_win_rates[-1]),
                        xytext=(len(eval_win_rates)-3, eval_win_rates[-1] + 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    ax3.set_title('Win Rate Evolution', fontsize=14)
    ax3.set_xlabel('Evaluation Index', fontsize=12)
    ax3.set_ylabel('Win Rate', fontsize=12)
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Win Rate by Opponent Type
    # First try a polar subplot for radar chart, with fallback to regular bar chart
    try:
        ax4 = plt.subplot(gs[1, 2], polar=True)
        
        # Radar chart of final performance
        if final_eval_results:
            opponent_types = [t for t in final_eval_results.keys() if t != 'overall']
            win_rates = [final_eval_results[t]['win_rate'] for t in opponent_types]
            
            # Number of variables
            N = len(opponent_types)
            
            # What will be the angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add win rates and close the loop
            win_rates += win_rates[:1]
            
            # Set the plot start at the top and go clockwise
            ax4.set_theta_offset(np.pi / 2)
            ax4.set_theta_direction(-1)
            
            # Draw the polygon and fill area
            ax4.plot(angles, win_rates, 'o-', linewidth=2)
            ax4.fill(angles, win_rates, alpha=0.25)
            
            # Draw threshold circle
            thresholds = [win_rate_threshold] * (N + 1)
            ax4.plot(angles, thresholds, '--', color='red', alpha=0.7, 
                    label=f'Threshold ({win_rate_threshold:.2f})')
            
            # Draw axis lines and labels
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(opponent_types)
            
            # Set y-limits
            ax4.set_ylim(0, 1)
            
            ax4.set_title('Win Rate by Opponent Type', fontsize=14)
    except:
        # Fallback to horizontal bar chart if polar plot fails
        plt.delaxes(ax4)  # Remove failed axis
        ax4 = plt.subplot(gs[1, 2])
        
        if final_eval_results:
            opponent_types = [t for t in final_eval_results.keys() if t != 'overall']
            win_rates = [final_eval_results[t]['win_rate'] for t in opponent_types]
            
            # Create color map
            colors = ['skyblue' if wr < win_rate_threshold else 'mediumseagreen' for wr in win_rates]
            
            # Horizontal bar chart for better readability
            y_pos = range(len(opponent_types))
            ax4.barh(y_pos, win_rates, color=colors)
            
            # Add win rate labels
            for i, wr in enumerate(win_rates):
                ax4.text(wr + 0.02, i, f'{wr:.2f}', va='center')
            
            # Add threshold line
            ax4.axvline(x=win_rate_threshold, color='red', linestyle='--',
                      label=f'Threshold ({win_rate_threshold:.2f})')
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(opponent_types)
            ax4.set_xlim(0, 1.1)
            ax4.set_title('Win Rate by Opponent Type', fontsize=14)
            ax4.set_xlabel('Win Rate', fontsize=12)
            ax4.legend()
    
    # Panel 5: Win Rate Progression by Type
    ax5 = plt.subplot(gs[2, 0:2])
    
    if detailed_win_rates:
        # Get list of opponents
        opponents = list(detailed_win_rates.keys())
        
        # Plot win rate for each opponent type
        for opponent in opponents:
            win_rates = detailed_win_rates[opponent]
            if win_rates:
                ax5.plot(range(len(win_rates)), win_rates, 'o-', linewidth=2, 
                        label=opponent)
        
        # Add threshold line
        ax5.axhline(y=win_rate_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({win_rate_threshold:.2f})')
    
    ax5.set_title('Win Rate Progression by Opponent Type', fontsize=14)
    ax5.set_xlabel('Evaluation Index', fontsize=12)
    ax5.set_ylabel('Win Rate', fontsize=12)
    ax5.set_ylim(0, 1.05)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Training Summary
    ax6 = plt.subplot(gs[2, 2])
    ax6.axis('off')  # Turn off axis for text panel
    
    # Compute final statistics
    num_episodes = len(episode_rewards)
    avg_reward = np.mean(episode_rewards[-1000:]) if episode_rewards else 0
    avg_length = np.mean(episode_lengths[-1000:]) if episode_lengths else 0
    
    # Create training summary text
    summary_text = [
        f"Agent Type: {agent_type.upper()}",
        f"Preset: {preset}",
        f"Episodes Trained: {num_episodes}",
        f"Final Overall Win Rate: {final_eval_results['overall']['win_rate']:.4f}",
        f"Win Rate Threshold: {win_rate_threshold:.2f}",
        f"Avg Final Reward: {avg_reward:.2f}",
        f"Avg Episode Length: {avg_length:.2f}"
    ]
    
    # Add performance details
    summary_text.append("\nPerformance by Opponent:")
    for opponent in agent_types:
        summary_text.append(f"vs {opponent}: {final_eval_results[opponent]['win_rate']:.4f}")
    
    # Add text to panel
    ax6.text(0.05, 0.95, '\n'.join(summary_text), transform=ax6.transAxes,
            fontsize=12, verticalalignment='top')
    
    ax6.set_title('Training Summary', fontsize=14)
    
    # Add overall title
    plt.suptitle(f"Enhanced Self-Play Training Results - {agent_type.upper()} Agent", 
                fontsize=18, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced self-play training for Liar's Dice")
    parser.add_argument('--agent', type=str, default='ppo', choices=['dqn', 'ppo'],
                        help='Type of agent to train')
    parser.add_argument('--preset', type=str, default='standard', choices=['basic', 'standard', 'advanced'],
                        help='Configuration preset to use')
    parser.add_argument('--path', type=str, default='results/enhanced_self_play/ppo_5_2_people_lstm', help='Base path for results')
    parser.add_argument('--episodes', type=int, default=None, help='Number of self-play episodes')
    parser.add_argument('--pool-size', type=int, default=15, help='Size of opponent pool')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render', action='store_true', help='Enable rendering during training')
    parser.add_argument('--no-early-stopping', action='store_false', dest='early_stopping',
                        help='Disable early stopping')
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive',
                        help='Disable adaptive parameters')
    parser.add_argument('--eval-freq', type=int, default=1000, 
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--visualize-freq', type=int, default=10000,
                        help='Visualization frequency (episodes)')
    
    parser.set_defaults(early_stopping=True, adaptive=True)
    
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
        enable_early_stopping=args.early_stopping,
        enable_adaptive_params=args.adaptive,
        evaluation_frequency=args.eval_freq,
        visualize_interval=args.visualize_freq
    )