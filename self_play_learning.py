"""
Population-based self-play training for Liar's Dice with specialized evaluation agents.

This script implements an evolutionary self-play training approach where agents
learn exclusively through playing against copies of themselves, while being evaluated
against specialized rule-based agents to track progress and select the best models.
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
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import deque
import pickle
import copy

from agents.base_agent import RLAgent
from agents.agent_factory import create_agent
# Import the specialized evaluation agents
from agents.evaluation_agents import create_agent as create_rule_agent
from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from training.environment_wrapper import LiarsDiceEnvWrapper
from training.train import train_episode, evaluate_agent
from training.utils import setup_logger, save_training_data, plot_training_results

# Define the specialized evaluation agents to use during evaluation
EVALUATION_AGENTS = [
    'beginner',         # Tests basic pattern exploitation
    'conservative',     # Tests exploitation of cautious play
    'aggressive',       # Tests bluff detection and punishment
    'adaptive',         # Tests adaptability and robustness
    'optimal'           # Tests proximity to optimal play
]

# Agent-specific configuration presets
AGENT_CONFIGS = {
    'dqn': {
        'basic': {
            'hidden_dims': [256, 128, 64],
            'learning_rate': 0.0005,
            'learning_rate_min': 1e-5,
            'lr_decay_steps': 100000,
            'epsilon_start': 0.5,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 80000,
            'buffer_size': 100000,
            'batch_size': 64,
            'target_update_freq': 500,
            'gamma': 0.99,
            'double_dqn': False
        },
        'standard': {
            'hidden_dims': [512, 256, 128, 64],
            'learning_rate': 0.0003,
            'learning_rate_min': 5e-6,
            'lr_decay_steps': 200000,
            'epsilon_start': 0.5,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 120000,
            'buffer_size': 200000,
            'batch_size': 128,
            'target_update_freq': 1000,
            'gamma': 0.99,
            'double_dqn': True
        },
        'advanced': {
            'hidden_dims': [1024, 512, 256, 128, 64],
            'learning_rate': 0.0002,
            'learning_rate_min': 1e-6,
            'lr_decay_steps': 300000,
            'epsilon_start': 0.5,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 150000,
            'buffer_size': 500000,
            'batch_size': 256,
            'target_update_freq': 2000,
            'gamma': 0.99,
            'double_dqn': True,
            'prioritized_replay': True
        }
    },
    'ppo': {
        'basic': {
            'hidden_dims': [256, 128, 64],
            'learning_rate': 0.0003,
            'min_learning_rate': 5e-6,
            'lr_decay_steps': 100000,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'policy_clip': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.04,
            'entropy_min': 0.01,
            'entropy_decay_steps': 80000,
            'update_frequency': 2048,
            'batch_size': 64,
            'ppo_epochs': 4,
            'max_grad_norm': 0.5
        },
        'standard': {
            'hidden_dims': [512, 256, 128, 64],
            'learning_rate': 0.0002,
            'min_learning_rate': 1e-6,
            'lr_decay_steps': 200000,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'policy_clip': 0.15,
            'value_coef': 0.5,
            'entropy_coef': 0.03,
            'entropy_min': 0.01,
            'entropy_decay_steps': 120000,
            'update_frequency': 2048,
            'batch_size': 128,
            'ppo_epochs': 5,
            'max_grad_norm': 0.5
        },
        'advanced': {
            'hidden_dims': [1024, 512, 256, 128],
            'learning_rate': 0.00015,
            'min_learning_rate': 5e-7,
            'lr_decay_steps': 300000,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'policy_clip': 0.1,
            'value_coef': 0.5,
            'entropy_coef': 0.02,
            'entropy_min': 0.005,
            'entropy_decay_steps': 200000,
            'update_frequency': 4096,
            'batch_size': 256,
            'ppo_epochs': 8,
            'max_grad_norm': 0.5
        }
    },
    'crf': {
        'basic': {
            'feature_engineering': 'advanced',
            'learning_rate': 0.001,
            'learning_rate_min': 1e-5,
            'lr_decay_steps': 100000,
            'c1': 0.1,
            'c2': 0.1,
            'max_iterations': 100,
            'buffer_size': 10000,
            'update_frequency': 500,
            'initial_exploration': 1.0,
            'final_exploration': 0.1,
            'exploration_decay': 80000
        },
        'standard': {
            'feature_engineering': 'advanced',
            'learning_rate': 0.0005,
            'learning_rate_min': 5e-6,
            'lr_decay_steps': 200000,
            'c1': 0.05,
            'c2': 0.1,
            'max_iterations': 200,
            'buffer_size': 20000,
            'update_frequency': 250,
            'initial_exploration': 1.0,
            'final_exploration': 0.1,
            'exploration_decay': 100000
        },
        'advanced': {
            'feature_engineering': 'advanced',
            'learning_rate': 0.0001,
            'learning_rate_min': 1e-6,
            'lr_decay_steps': 300000,
            'c1': 0.01,
            'c2': 0.05,
            'max_iterations': 300,
            'buffer_size': 50000,
            'update_frequency': 200,
            'initial_exploration': 1.0,
            'final_exploration': 0.05,
            'exploration_decay': 150000
        }
    }
}

# Game configuration presets
GAME_CONFIGS = {
    'basic': {
        'num_players': 2,
        'num_dice': 3,
        'self_play_episodes': 50000,
        'win_rate_threshold': 0.65,
        'eval_frequency': 500,
        'checkpoint_frequency': 1000
    },
    'standard': {
        'num_players': 2,
        'num_dice': 5,
        'self_play_episodes': 100000,
        'win_rate_threshold': 0.60,
        'eval_frequency': 1000,
        'checkpoint_frequency': 2000
    },
    'advanced': {
        'num_players': 4,
        'num_dice': 5,
        'self_play_episodes': 150000,
        'win_rate_threshold': 0.55,
        'eval_frequency': 1500,
        'checkpoint_frequency': 3000
    }
}

def population_based_self_play(
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
    
    # Population settings
    population_size: int = 10,
    elite_size: int = 3,        # Number of top agents to keep unchanged
    mutation_prob: float = 0.3,  # Probability of mutating hyperparameters
    
    # Evaluation 
    evaluation_frequency: Optional[int] = None,
    comprehensive_eval: bool = True,
    evaluation_episodes: int = 100,  # Episodes per opponent in evaluation
    
    # Training control
    checkpoint_frequency: Optional[int] = None,
    enable_early_stopping: bool = True,
    win_rate_threshold: Optional[float] = None,
    early_stopping_patience: int = 5,
    
    # Hyperparameter mutation ranges (for population-based training)
    mutation_ranges: Optional[Dict[str, Tuple]] = None,
    
    # Visualization settings
    plot_interval: int = 10000,  # How often to update visualizations
    detailed_metrics: bool = True,  # Track detailed metrics
    
    # Misc
    save_game_samples: bool = True,  # Save sample games
    num_game_samples: int = 20      # Number of games to sample
) -> Tuple[RLAgent, Dict[str, Any]]:
    """
    Population-based self-play training for Liar's Dice with specialized evaluation agents.
    
    This method implements population-based training (PBT) with pure self-play where:
    1. A population of agents play against each other
    2. Agents are evaluated against specialized rule-based agents
    3. Low-performing agents are replaced with mutated versions of high-performing ones
    4. The best-performing agent (against the evaluation agents) is tracked throughout training
    
    Args:
        agent_type: Type of agent to train ('dqn', 'ppo', or 'crf')
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
        
        # Population settings
        population_size: Number of agents in the population
        elite_size: Number of top agents to keep unchanged between generations
        mutation_prob: Probability of mutating hyperparameters
        
        # Evaluation
        evaluation_frequency: How often to evaluate the agents against the curriculum
        comprehensive_eval: Whether to perform a comprehensive evaluation at the end
        evaluation_episodes: Episodes per opponent in evaluation
        
        # Training control
        checkpoint_frequency: How often to save checkpoints
        enable_early_stopping: Whether to stop training early when win rate is high
        win_rate_threshold: Win rate threshold for early stopping
        early_stopping_patience: Number of evaluations above threshold to trigger early stopping
        
        # Hyperparameter mutation
        mutation_ranges: Dictionary mapping hyperparameter names to (min, max) ranges
        
        # Visualization settings
        plot_interval: How often to update visualizations
        detailed_metrics: Whether to track detailed metrics
        
        # Misc
        save_game_samples: Whether to save sample games
        num_game_samples: Number of games to sample
        
    Returns:
        Tuple of (best_agent, training_results)
    """
    if agent_type.lower() not in ['dqn', 'ppo', 'crf']:
        raise ValueError(f"Unsupported agent type: {agent_type}. Choose from 'dqn', 'ppo', or 'crf'.")
    
    if preset not in GAME_CONFIGS:
        raise ValueError(f"Unsupported preset: {preset}. Choose from {list(GAME_CONFIGS.keys())}.")
    
    # Apply preset configurations
    game_config = GAME_CONFIGS[preset]
    base_agent_config = AGENT_CONFIGS[agent_type.lower()][preset]
    
    # Apply values from preset unless explicitly overridden
    _num_players = num_players if num_players is not None else game_config['num_players']
    _num_dice = num_dice if num_dice is not None else game_config['num_dice']
    _self_play_episodes = self_play_episodes if self_play_episodes is not None else game_config['self_play_episodes']
    _win_rate_threshold = win_rate_threshold if win_rate_threshold is not None else game_config['win_rate_threshold']
    _evaluation_frequency = evaluation_frequency if evaluation_frequency is not None else game_config['eval_frequency']
    _checkpoint_frequency = checkpoint_frequency if checkpoint_frequency is not None else game_config['checkpoint_frequency']
    
    # Calculate how many generations we'll have based on evaluation frequency
    num_generations = _self_play_episodes // _evaluation_frequency
    episodes_per_generation = _evaluation_frequency
    
    # Set up paths for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pbt_selfplay_{agent_type}_{preset}_{_num_players}p{_num_dice}d_{timestamp}"
    base_path = os.path.join(results_path, run_name)
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'logs')
    population_dir = os.path.join(base_path, 'population')
    visualization_dir = os.path.join(base_path, 'visualizations')
    sample_games_dir = os.path.join(base_path, 'sample_games')
    
    # Create directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(population_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    if save_game_samples:
        os.makedirs(sample_games_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('pbt_selfplay', os.path.join(log_dir, 'training.log'))
    logger.info(f"Starting population-based self-play training for Liar's Dice with {agent_type.upper()} agent")
    logger.info(f"Game setup: {_num_players} players, {_num_dice} dice, {dice_faces} faces")
    logger.info(f"Population size: {population_size}, Elite size: {elite_size}")
    logger.info(f"Self-play episodes: {_self_play_episodes}, Episodes per generation: {episodes_per_generation}")
    logger.info(f"Using specialized evaluation agents: {EVALUATION_AGENTS}")
    
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
            torch.backends.cudnn.benchmark = False
    
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
    
    # Set up default hyperparameter mutation ranges if not provided
    if mutation_ranges is None:
        if agent_type.lower() == 'ppo':
            mutation_ranges = {
                'learning_rate': (1e-6, 5e-3),
                'gamma': (0.95, 0.995),
                'gae_lambda': (0.9, 0.99),
                'policy_clip': (0.05, 0.3),
                'value_coef': (0.3, 0.8),
                'entropy_coef': (0.001, 0.05)
            }
        elif agent_type.lower() == 'dqn':
            mutation_ranges = {
                'learning_rate': (1e-6, 5e-3),
                'gamma': (0.95, 0.995),
                'target_update_freq': (200, 2000)
            }
        elif agent_type.lower() == 'crf':
            mutation_ranges = {
                'learning_rate': (1e-6, 5e-3),
                'c1': (0.01, 0.2),
                'c2': (0.01, 0.2)
            }
    
    # Function to create hyperparameter mutations
    def mutate_config(config):
        new_config = copy.deepcopy(config)
        for param, (min_val, max_val) in mutation_ranges.items():
            if param in new_config and random.random() < mutation_prob:
                if isinstance(new_config[param], int):
                    # For integer parameters
                    new_val = random.randint(int(min_val), int(max_val))
                else:
                    # For float parameters
                    new_val = random.uniform(min_val, max_val)
                
                # Apply mutation with some constraints
                if param == 'learning_rate':
                    # Log-uniform distribution for learning rate
                    log_min = np.log(min_val)
                    log_max = np.log(max_val)
                    new_val = np.exp(random.uniform(log_min, log_max))
                
                # Update the parameter
                new_config[param] = new_val
                
        return new_config
    
    # Initialize population with slightly different hyperparameters
    population = []
    for i in range(population_size):
        # Create a copy of the base config
        agent_config = copy.deepcopy(base_agent_config)
        agent_config['device'] = device
        
        # Apply small mutations to create diversity in initial population
        if i > 0:  # Keep the first agent with default hyperparameters
            agent_config = mutate_config(agent_config)
        
        # Create agent
        agent = create_agent(
            agent_type=agent_type,
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=agent_config
        )
        
        # Set action mapping
        agent.set_action_mapping(env.action_mapping)
        
        # Add to population
        population.append({
            'agent': agent,
            'config': agent_config,
            'fitness': 0.0,  # Will be updated during evaluation
            'id': i
        })
    
    logger.info(f"Initialized population with {len(population)} agents")
    
    # Tracking variables
    generation = 0
    episode = 0
    best_agent = None
    best_fitness = float('-inf')
    best_model_path = None
    
    # Training metrics
    all_rewards = []
    all_episode_lengths = []
    eval_win_rates = {agent_type: [] for agent_type in EVALUATION_AGENTS}
    eval_win_rates['overall'] = []
    generation_stats = []
    population_diversity = []
    
    # For early stopping
    high_win_rate_count = 0
    
    # Detailed metrics
    if detailed_metrics:
        agent_losses = {i: [] for i in range(population_size)}
        agent_entropies = {i: [] for i in range(population_size)} if agent_type.lower() == 'ppo' else None
        agent_exploration = {i: [] for i in range(population_size)} if agent_type.lower() == 'dqn' else None
    
    # Main training loop over generations
    logger.info(f"Starting population-based training for {num_generations} generations")
    start_time = time.time()
    
    while episode < _self_play_episodes:
        generation_start_time = time.time()
        logger.info(f"\n=== Generation {generation + 1}/{num_generations} (Episodes {episode + 1}-{min(episode + episodes_per_generation, _self_play_episodes)}) ===")
        
        # Train each agent in the population for this generation
        for agent_idx, agent_data in enumerate(population):
            agent = agent_data['agent']
            
            # Number of episodes to train this agent in this generation
            agent_episodes = min(episodes_per_generation // population_size, _self_play_episodes - episode)
            if agent_episodes <= 0:
                break
                
            logger.info(f"Training agent {agent_idx} for {agent_episodes} episodes")
            
            for ep in range(agent_episodes):
                current_episode = episode + ep
                
                # Select opponent from population (excluding self)
                opponent_idx = random.choice([i for i in range(population_size) if i != agent_idx])
                opponent = population[opponent_idx]['agent']
                
                # Create self-play environment
                selfplay_env = LiarsDiceEnvWrapper(
                    num_players=_num_players,
                    num_dice=_num_dice,
                    dice_faces=dice_faces,
                    seed=seed + current_episode if seed else None,
                    rl_agent_as_opponent=opponent
                )
                
                # Set action mappings
                agent.set_action_mapping(selfplay_env.action_mapping)
                opponent.set_action_mapping(selfplay_env.action_mapping)
                
                # Train for one episode
                reward, episode_length, _ = train_episode(
                    env=selfplay_env,
                    agent=agent,
                    evaluation=False,
                    render=render_training and current_episode % 100 == 0,
                    reward_shaping=True
                )
                
                all_rewards.append(reward)
                all_episode_lengths.append(episode_length)
                
                # Track detailed metrics if enabled
                if detailed_metrics:
                    if hasattr(agent, 'get_latest_loss'):
                        agent_losses[agent_idx].append(agent.get_latest_loss())
                    
                    if agent_type.lower() == 'ppo' and hasattr(agent, 'get_entropy'):
                        agent_entropies[agent_idx].append(agent.get_entropy())
                    
                    if agent_type.lower() == 'dqn' and hasattr(agent, 'get_exploration_rate'):
                        agent_exploration[agent_idx].append(agent.get_exploration_rate())
                
                # Collect game samples for analysis
                if save_game_samples and current_episode % max(1, _self_play_episodes // num_game_samples) == 0:
                    # Reset environment for a clean game
                    sample_env = LiarsDiceEnvWrapper(
                        num_players=_num_players,
                        num_dice=_num_dice,
                        dice_faces=dice_faces,
                        seed=seed + 10000 + current_episode if seed else None,
                        rl_agent_as_opponent=opponent
                    )
                    
                    # Play and record the game
                    game_history = []
                    obs = sample_env.reset()
                    done = False
                    
                    while not done:
                        # Get valid action indices
                        valid_action_indices = sample_env.get_valid_actions()
                        valid_actions = [sample_env.action_mapping[idx] for idx in valid_action_indices]

                        # Get action from agent (this returns a game action dict)
                        action = agent.select_action(obs, valid_actions, training=False)

                        # Convert game action back to index
                        action_idx = None
                        for idx, valid_idx in enumerate(valid_action_indices):
                            if agent._actions_equal(action, sample_env.action_mapping[valid_idx]):
                                action_idx = valid_idx
                                break

                        if action_idx is None:
                            raise ValueError(f"Selected action {action} not found in valid actions")

                        # Pass the index to step, not the dictionary
                        next_obs, reward, done, info = sample_env.step(action_idx)
                        
                        game_history.append({
                            'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                            'action': action,
                            'reward': reward,
                            'done': done,
                            'info': info
                        })
                        
                        obs = next_obs
                    
                    # Save the game
                    game_path = os.path.join(sample_games_dir, f"game_sample_ep{current_episode}_agent{agent_idx}.pkl")
                    with open(game_path, 'wb') as f:
                        pickle.dump(game_history, f)
                
                # Log progress occasionally
                if (current_episode + 1) % 100 == 0 or current_episode == _self_play_episodes - 1:
                    avg_reward = np.mean(all_rewards[-100:])
                    avg_length = np.mean(all_episode_lengths[-100:])
                    elapsed_time = time.time() - start_time
                    
                    logger.info(f"Episode {current_episode + 1}/{_self_play_episodes} | "
                              f"Agent {agent_idx} | "
                              f"Avg Reward: {avg_reward:.2f} | "
                              f"Avg Length: {avg_length:.2f} | "
                              f"Elapsed: {elapsed_time:.1f}s")
            
            # Update episode counter
            episode += agent_episodes
        
        generation_time = time.time() - generation_start_time
        logger.info(f"Completed generation {generation + 1} training in {generation_time:.1f}s")
        
        # Evaluate all agents against the specialized evaluation agents
        logger.info(f"Evaluating population against specialized evaluation agents...")
        
        # First calculate population diversity before evaluation
        if detailed_metrics:
            # Measure parameter diversity
            param_values = {param: [] for param in mutation_ranges.keys()}
            for agent_data in population:
                for param in param_values:
                    if param in agent_data['config']:
                        param_values[param].append(agent_data['config'][param])
            
            # Calculate coefficient of variation for each parameter
            diversity = {}
            for param, values in param_values.items():
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    # Coefficient of variation (std/mean) as a measure of diversity
                    diversity[param] = std / mean if mean != 0 else 0
            
            population_diversity.append(diversity)
        
        # Evaluate each agent in the population
        for agent_idx, agent_data in enumerate(population):
            agent = agent_data['agent']
            
            agent_results = {}
            total_wins = 0
            total_games = 0
            
            # Evaluate against each specialized evaluation agent
            for opponent_type in EVALUATION_AGENTS:
                # Create evaluation environment
                eval_env = LiarsDiceEnvWrapper(
                    num_players=_num_players,
                    num_dice=_num_dice,
                    dice_faces=dice_faces,
                    seed=seed + 20000 + generation * 100 + agent_idx if seed else None,
                    rule_agent_types=[opponent_type]
                )
                
                agent.set_action_mapping(eval_env.action_mapping)
                
                # Run evaluation
                num_eval_episodes = max(20, evaluation_episodes // len(EVALUATION_AGENTS))  # Faster per-agent evaluation
                wins = 0
                total_reward = 0
                
                for _ in range(num_eval_episodes):
                    episode_reward, _, _ = train_episode(eval_env, agent, evaluation=True)
                    total_reward += episode_reward
                    if episode_reward > 0:  # Win
                        wins += 1
                
                win_rate = wins / num_eval_episodes
                avg_reward = total_reward / num_eval_episodes
                
                agent_results[opponent_type] = {
                    'win_rate': win_rate,
                    'avg_reward': avg_reward
                }
                
                total_wins += wins
                total_games += num_eval_episodes
            
            # Calculate overall fitness as weighted average of win rates
            # Give higher weight to harder opponents
            weighted_win_rate = 0
            weights = {
                'beginner': 0.05,       # Easy opponent
                'conservative': 0.15,   # Medium opponent
                'aggressive': 0.20,     # Medium-hard opponent
                'adaptive': 0.25,       # Hard opponent
                'optimal': 0.35         # Hardest opponent
            }
            
            for opponent, result in agent_results.items():
                if opponent in weights:
                    weighted_win_rate += result['win_rate'] * weights[opponent]
            
            # Normalize if not all opponents were used
            used_weights = sum(weights[op] for op in agent_results if op in weights)
            if used_weights > 0:
                weighted_win_rate /= used_weights
            
            # Simple win rate as fallback
            overall_win_rate = total_wins / total_games if total_games > 0 else 0
            
            # Combine weighted and overall win rates
            fitness = 0.7 * weighted_win_rate + 0.3 * overall_win_rate
            
            # Update agent fitness
            population[agent_idx]['fitness'] = fitness
            population[agent_idx]['eval_results'] = agent_results
            population[agent_idx]['overall_win_rate'] = overall_win_rate
            
            # Check if this is the best agent so far
            if fitness > best_fitness:
                best_fitness = fitness
                best_agent = agent
                best_model_path = os.path.join(checkpoint_dir, f"best_model_gen{generation}")
                agent.save(best_model_path)
                logger.info(f"New best agent found! Agent {agent_idx} with fitness {fitness:.4f}")
            
            logger.info(f"Agent {agent_idx} Evaluation: Overall Win Rate = {overall_win_rate:.4f}, Fitness = {fitness:.4f}")
            
            # Detailed per-opponent win rates
            for opponent, result in agent_results.items():
                logger.info(f"  - vs {opponent}: {result['win_rate']:.4f}")
        
        # Rank population by fitness
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Update global evaluation metrics
        # Use the best agent's results for the evaluation metrics
        best_agent_results = population[0]['eval_results']
        for opponent_type, result in best_agent_results.items():
            eval_win_rates[opponent_type].append(result['win_rate'])
        
        # Overall win rate
        overall_win_rate = population[0]['overall_win_rate']
        eval_win_rates['overall'].append(overall_win_rate)
        
        # Check early stopping
        if enable_early_stopping and overall_win_rate >= _win_rate_threshold:
            high_win_rate_count += 1
            logger.info(f"Win rate {overall_win_rate:.4f} above threshold {_win_rate_threshold:.4f} "
                       f"({high_win_rate_count}/{early_stopping_patience})")
            if high_win_rate_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {episode} episodes")
                break
        else:
            high_win_rate_count = 0
        
        # Save generation stats
        gen_stats = {
            'generation': generation,
            'episode': episode,
            'time': generation_time,
            'population_fitness': [p['fitness'] for p in population],
            'best_fitness': best_fitness,
            'overall_win_rate': overall_win_rate,
            'win_rates': {k: v[-1] for k, v in eval_win_rates.items() if len(v) > 0},
            'opponent_details': {
                opponent: {
                    'win_rate': population[0]['eval_results'][opponent]['win_rate'],
                    'reward': population[0]['eval_results'][opponent]['avg_reward']
                } for opponent in population[0]['eval_results']
            }
        }
        generation_stats.append(gen_stats)
        
        # Save detailed stats to file
        with open(os.path.join(log_dir, f"generation_{generation}.json"), 'w') as f:
            json.dump(gen_stats, f, indent=2)
        
        # Visualize current training progress
        if episode % plot_interval == 0 or episode >= _self_play_episodes:
            create_training_visualizations(
                all_rewards=all_rewards,
                all_episode_lengths=all_episode_lengths,
                eval_win_rates=eval_win_rates,
                population_fitness=[p['fitness'] for p in population],
                population_diversity=population_diversity,
                agent_losses=agent_losses if detailed_metrics else None,
                agent_entropies=agent_entropies if detailed_metrics else None,
                agent_exploration=agent_exploration if detailed_metrics else None,
                generation=generation,
                episode=episode,
                save_dir=visualization_dir
            )
            logger.info(f"Updated visualizations at episode {episode}")
        
        # Save population checkpoint
        if episode % _checkpoint_frequency == 0 or episode >= _self_play_episodes:
            # Save each agent in the population
            for agent_idx, agent_data in enumerate(population):
                agent = agent_data['agent']
                agent_path = os.path.join(population_dir, f"gen{generation}_agent{agent_idx}")
                agent.save(agent_path)
                
                # Save agent configuration
                config_path = os.path.join(population_dir, f"gen{generation}_agent{agent_idx}_config.json")
                with open(config_path, 'w') as f:
                    # Convert any numpy types to Python types
                    config_to_save = {}
                    for k, v in agent_data['config'].items():
                        if isinstance(v, np.ndarray):
                            config_to_save[k] = v.tolist()
                        elif isinstance(v, np.integer):
                            config_to_save[k] = int(v)
                        elif isinstance(v, np.floating):
                            config_to_save[k] = float(v)
                        else:
                            config_to_save[k] = v
                    
                    json.dump(config_to_save, f, indent=2)
            
            logger.info(f"Saved population checkpoint at episode {episode}")
        
        # Population-based training: Replace worst agents with mutated versions of best agents
        if generation < num_generations - 1:  # Skip on the final generation
            # Keep elite_size best agents unchanged
            elite_agents = population[:elite_size]
            
            # Replace the rest with mutations of the elite agents
            for i in range(elite_size, population_size):
                # Select a random elite agent to clone and mutate
                elite_idx = random.randint(0, elite_size - 1)
                elite_agent = elite_agents[elite_idx]['agent']
                elite_config = elite_agents[elite_idx]['config']
                
                # Create a mutated copy of the configuration
                mutated_config = mutate_config(elite_config)
                
                # Create a new agent with the mutated configuration
                new_agent = create_agent(
                    agent_type=agent_type,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    config=mutated_config
                )
                
                # Copy weights from elite agent to new agent
                new_agent.copy_weights_from(elite_agent)
                
                # Update the population with the new agent
                population[i] = {
                    'agent': new_agent,
                    'config': mutated_config,
                    'fitness': 0.0,
                    'id': population[i]['id']  # Preserve ID for tracking
                }
            
            logger.info(f"Updated population with {population_size - elite_size} new agents")
        
        # Increment generation counter
        generation += 1
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed after {episode} episodes in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Make sure we're using the best agent for final evaluation
    if best_agent is None:
        best_agent = population[0]['agent']
        logger.warning("No best agent found during training, using the top agent from the final population")
    
    # Final comprehensive evaluation against all specialized agents
    if comprehensive_eval:
        logger.info("\n=== Final Comprehensive Evaluation ===")
        
        # Set appropriate exploration parameters for evaluation
        if hasattr(best_agent, 'epsilon'):
            original_epsilon = best_agent.epsilon
            best_agent.epsilon = 0.05  # Low exploration for evaluation
        
        if hasattr(best_agent, 'entropy_coef'):
            original_entropy = best_agent.entropy_coef
            best_agent.entropy_coef = 0.01  # Low entropy for evaluation
        
        # Evaluate against each specialized agent
        final_eval_results = {}
        total_wins = 0
        total_games = 0
        
        for opponent_type in EVALUATION_AGENTS:
            eval_env = LiarsDiceEnvWrapper(
                num_players=_num_players,
                num_dice=_num_dice,
                dice_faces=dice_faces,
                seed=seed,
                rule_agent_types=[opponent_type]
            )
            
            best_agent.set_action_mapping(eval_env.action_mapping)
            
            # Run more episodes for final evaluation
            wins = 0
            total_reward = 0
            episode_lengths = []
            
            for _ in range(evaluation_episodes):
                episode_reward, length, _ = train_episode(eval_env, best_agent, evaluation=True)
                total_reward += episode_reward
                episode_lengths.append(length)
                if episode_reward > 0:
                    wins += 1
            
            win_rate = wins / evaluation_episodes
            avg_reward = total_reward / evaluation_episodes
            avg_length = sum(episode_lengths) / len(episode_lengths)
            
            final_eval_results[opponent_type] = {
                'win_rate': win_rate,
                'mean_reward': avg_reward,
                'mean_episode_length': avg_length
            }
            
            logger.info(f"  vs {opponent_type}: Win Rate = {win_rate:.2f}, Reward = {avg_reward:.2f}")
            
            total_wins += wins
            total_games += evaluation_episodes
        
        # Calculate overall statistics
        overall_win_rate = total_wins / total_games
        overall_reward = sum(r['mean_reward'] for r in final_eval_results.values()) / len(final_eval_results)
        
        final_eval_results['overall'] = {
            'win_rate': overall_win_rate,
            'mean_reward': overall_reward
        }
        
        logger.info(f"\nOverall: Win Rate = {overall_win_rate:.2f}, Reward = {overall_reward:.2f}")
        
        # Restore original exploration parameters
        if hasattr(best_agent, 'epsilon'):
            best_agent.epsilon = original_epsilon
        
        if hasattr(best_agent, 'entropy_coef'):
            best_agent.entropy_coef = original_entropy
    else:
        # Use the last generation's evaluation as final results
        final_eval_results = {}
        
        # Get results from the best agent in the population
        best_agent_results = population[0]['eval_results']
        total_wins = 0
        total_games = 0
        
        for opponent_type, result in best_agent_results.items():
            final_eval_results[opponent_type] = {
                'win_rate': result['win_rate'],
                'mean_reward': result['avg_reward']
            }
            
            # Assuming equal episodes per opponent
            episodes_per_opponent = evaluation_episodes // len(EVALUATION_AGENTS)
            total_wins += int(result['win_rate'] * episodes_per_opponent)
            total_games += episodes_per_opponent
        
        # Calculate overall statistics
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        overall_reward = sum(r['mean_reward'] for r in final_eval_results.values()) / len(final_eval_results)
        
        final_eval_results['overall'] = {
            'win_rate': overall_win_rate,
            'mean_reward': overall_reward
        }
        
        logger.info(f"\nOverall: Win Rate = {overall_win_rate:.2f}, Reward = {overall_reward:.2f}")
    
    # Save and visualize final evaluation
    with open(os.path.join(log_dir, "final_evaluation.json"), 'w') as f:
        json.dump(final_eval_results, f, indent=2)
    
    vis_path = os.path.join(visualization_dir, "final_evaluation.png")
    create_final_evaluation_visualization(final_eval_results, save_path=vis_path)
    logger.info(f"Saved final evaluation visualization to {vis_path}")
    
    # Create additional comprehensive visualizations
    create_final_visualizations(
        final_eval_results=final_eval_results,
        training_results={
            'episode_rewards': all_rewards,
            'episode_lengths': all_episode_lengths,
            'eval_win_rates': eval_win_rates,
            'generation_stats': generation_stats,
            'population_diversity': population_diversity
        },
        agent_type=agent_type,
        preset=preset,
        win_rate_threshold=_win_rate_threshold,
        num_generations=generation,
        save_dir=visualization_dir
    )
    
    # Save final best model
    final_model_path = os.path.join(checkpoint_dir, "final_best_model")
    best_agent.save(final_model_path)
    logger.info(f"Saved final best model to {final_model_path}")
    
    # Save model to models directory with metadata
    models_dir = os.path.join(results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model name with performance info
    overall_win_rate = final_eval_results['overall']['win_rate']
    model_name = f"pbt_selfplay_{agent_type}_{_num_players}p{_num_dice}d_{overall_win_rate:.2f}wr_{timestamp}"
    model_path = os.path.join(models_dir, model_name)
    best_agent.save(model_path)
    
    # Find the config of the best agent
    best_agent_config = None
    for agent_data in population:
        if agent_data['agent'] == best_agent:
            best_agent_config = agent_data['config']
            break
    
    if best_agent_config is None:
        best_agent_config = base_agent_config  # Fallback
    
    # Save model metadata
    model_metadata = {
        'timestamp': timestamp,
        'agent_type': agent_type,
        'num_players': _num_players,
        'num_dice': _num_dice,
        'dice_faces': dice_faces,
        'win_rates': {k: v['win_rate'] for k, v in final_eval_results.items() if k != 'overall'},
        'overall_win_rate': overall_win_rate,
        'agent_config': best_agent_config,
        'training_parameters': {
            'self_play_episodes': _self_play_episodes,
            'population_size': population_size,
            'elite_size': elite_size,
            'generations': generation
        }
    }
    
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Saved final model with metadata to {model_path}")
    
    # Prepare and return training results
    training_results = {
        'episode_rewards': all_rewards,
        'episode_lengths': all_episode_lengths,
        'eval_win_rates': eval_win_rates,
        'generation_stats': generation_stats,
        'population_diversity': population_diversity,
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
            'win_rate_threshold': _win_rate_threshold,
            'population_size': population_size,
            'elite_size': elite_size,
            'generations': generation,
            'agent_config': best_agent_config
        }
    }
    
    return best_agent, training_results


def create_training_visualizations(
    all_rewards,
    all_episode_lengths,
    eval_win_rates,
    population_fitness,
    population_diversity=None,
    agent_losses=None,
    agent_entropies=None,
    agent_exploration=None,
    generation=0,
    episode=0,
    save_dir='visualizations'
):
    """
    Create and save visualizations of training progress.
    
    Args:
        all_rewards: List of episode rewards
        all_episode_lengths: List of episode lengths
        eval_win_rates: Dict mapping opponent types to lists of win rates
        population_fitness: List of fitness values for the current population
        population_diversity: List of population diversity metrics
        agent_losses: Dict mapping agent IDs to lists of loss values
        agent_entropies: Dict mapping agent IDs to lists of entropy values
        agent_exploration: Dict mapping agent IDs to lists of exploration rates
        generation: Current generation number
        episode: Current episode number
        save_dir: Directory to save visualizations
    """
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    # 1. Training Rewards Plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards with low alpha
    plt.plot(range(1, len(all_rewards) + 1), all_rewards, 
             alpha=0.2, color='blue', label='Individual Episodes')
    
    # Plot smoothed rewards
    window_size = min(100, max(1, len(all_rewards) // 50))
    if len(all_rewards) > window_size:
        smoothed_rewards = []
        for i in range(len(all_rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(all_rewards[i:i+window_size]))
        plt.plot(range(window_size, len(all_rewards) + 1), smoothed_rewards, 
                 linewidth=2, color='darkblue', label=f'{window_size}-Episode Moving Average')
    
    plt.title('Training Rewards Over Time', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rewards_over_time.png"))
    plt.close()
    
    # 2. Win Rates Against Evaluation Agents
    if eval_win_rates and any(len(rates) > 0 for rates in eval_win_rates.values()):
        plt.figure(figsize=(14, 8))
        
        # Plot win rate for each opponent type
        for opponent, rates in eval_win_rates.items():
            if rates and opponent != 'overall':
                plt.plot(range(len(rates)), rates, marker='o', linewidth=2, label=f'vs {opponent}')
        
        # Plot overall win rate with thicker line
        if 'overall' in eval_win_rates and eval_win_rates['overall']:
            plt.plot(range(len(eval_win_rates['overall'])), eval_win_rates['overall'],
                    marker='*', linewidth=3, color='black', label='Overall')
        
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Win Rate')
        
        plt.title('Win Rates Against Evaluation Agents', fontsize=16)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Win Rate', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "win_rates_by_opponent.png"))
        plt.close()
    
    # 3. Population Fitness Distribution
    if population_fitness:
        plt.figure(figsize=(10, 6))
        
        # Sort fitness values for better visualization
        sorted_fitness = sorted(population_fitness, reverse=True)
        
        # Plot fitness values as bars
        bars = plt.bar(range(len(sorted_fitness)), sorted_fitness, color='green')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f'Population Fitness Distribution (Generation {generation})', fontsize=16)
        plt.xlabel('Agent Rank', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.ylim(0, max(sorted_fitness) * 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "population_fitness.png"))
        plt.close()
    
    # 4. Population Diversity Over Time
    if population_diversity:
        plt.figure(figsize=(14, 8))
        
        # Extract diversity metrics for each parameter
        parameters = list(population_diversity[0].keys())
        
        for param in parameters:
            diversity_values = [d.get(param, 0) for d in population_diversity]
            plt.plot(range(len(diversity_values)), diversity_values, marker='o', linewidth=2, label=param)
        
        plt.title('Population Hyperparameter Diversity Over Time', fontsize=16)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Coefficient of Variation', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "population_diversity.png"))
        plt.close()
        
    # 5. Training Progress Overview Dashboard
    plt.figure(figsize=(16, 12))
    
    # Define grid layout
    gs = plt.GridSpec(2, 2)
    
    # Subplot 1: Rewards Trend
    ax1 = plt.subplot(gs[0, 0])
    if len(all_rewards) > window_size:
        smoothed_rewards = []
        for i in range(0, len(all_rewards) - window_size + 1, max(1, window_size // 10)):
            smoothed_rewards.append(np.mean(all_rewards[i:i+window_size]))
        x_values = range(window_size, len(all_rewards) + 1, max(1, window_size // 10))
        ax1.plot(x_values, smoothed_rewards, linewidth=2.5, color='blue')
    ax1.set_title('Training Rewards (Smoothed)', fontsize=14)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Win Rate Trend
    ax2 = plt.subplot(gs[0, 1])
    if 'overall' in eval_win_rates and eval_win_rates['overall']:
        overall_win_rates = eval_win_rates['overall']
        ax2.plot(range(len(overall_win_rates)), overall_win_rates, 'o-', 
                linewidth=2.5, color='green', label='Overall Win Rate')
        
        # Add trendline
        if len(overall_win_rates) > 1:
            z = np.polyfit(range(len(overall_win_rates)), overall_win_rates, 1)
            p = np.poly1d(z)
            ax2.plot(range(len(overall_win_rates)), p(range(len(overall_win_rates))), 
                    '--', color='darkgreen', alpha=0.7, label='Trend')
    
    ax2.set_title('Overall Win Rate Trend', fontsize=14)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Episode Length Distribution
    ax3 = plt.subplot(gs[1, 0])
    if all_episode_lengths:
        ax3.hist(all_episode_lengths, bins=30, alpha=0.7, color='purple')
        ax3.axvline(np.mean(all_episode_lengths), color='red', linestyle='dashed', linewidth=2,
                   label=f'Mean: {np.mean(all_episode_lengths):.1f}')
        ax3.axvline(np.median(all_episode_lengths), color='blue', linestyle='dashed', linewidth=2,
                   label=f'Median: {np.median(all_episode_lengths):.1f}')
    ax3.set_title('Episode Length Distribution', fontsize=14)
    ax3.set_xlabel('Steps per Episode', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Population Fitness History
    ax4 = plt.subplot(gs[1, 1])
    if population_fitness:
        # Bar chart of current population fitness
        sorted_fitness = sorted(population_fitness, reverse=True)
        ax4.bar(range(len(sorted_fitness)), sorted_fitness, color='lightgreen')
        
        # Add best fitness text
        if sorted_fitness:
            ax4.text(0.05, 0.95, f"Best Fitness: {sorted_fitness[0]:.4f}", 
                    transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    ax4.set_title(f'Population Fitness (Generation {generation})', fontsize=14)
    ax4.set_xlabel('Agent Rank', fontsize=12)
    ax4.set_ylabel('Fitness', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_overview.png"))
    plt.close()


def create_final_evaluation_visualization(final_eval_results, save_path):
    """
    Create a visualization of the final evaluation results against all evaluation agents.
    
    Args:
        final_eval_results: Dictionary containing evaluation results
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
    
    plt.title('Final Evaluation: Win Rate Against Specialized Agents', fontsize=16)
    plt.xlabel('Opponent Agent Type', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.ylim(0, min(1.1, max(win_rates) + 0.15))  # Set y-axis with some headroom
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()


def create_final_visualizations(
    final_eval_results,
    training_results,
    agent_type,
    preset,
    win_rate_threshold,
    num_generations,
    save_dir='visualizations'
):
    """
    Create comprehensive visualizations for the final report.
    
    Args:
        final_eval_results: Dictionary containing final evaluation results
        training_results: Dictionary containing training data
        agent_type: Type of agent used ('dqn', 'ppo', or 'crf')
        preset: Configuration preset used
        win_rate_threshold: Win rate threshold used for early stopping
        num_generations: Number of generations trained
        save_dir: Directory to save visualizations
    """
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    # 1. Win Rates against different opponent types (radar chart)
    plt.figure(figsize=(10, 10))
    
    # Get opponents and win rates
    opponents = [op for op in final_eval_results.keys() if op != 'overall']
    win_rates = [final_eval_results[op]['win_rate'] for op in opponents]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(opponents), endpoint=False).tolist()
    
    # Close the loop
    win_rates.append(win_rates[0])
    angles.append(angles[0])
    opponents.append(opponents[0])
    
    # Plot
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, win_rates, 'o-', linewidth=2, color='blue')
    ax.fill(angles, win_rates, alpha=0.25, color='blue')
    
    # Add threshold circle
    threshold_circle = [win_rate_threshold] * len(angles)
    ax.plot(angles, threshold_circle, '--', linewidth=1.5, color='red', alpha=0.7)
    ax.fill(angles, threshold_circle, alpha=0.1, color='red')
    
    # Add 50% baseline circle
    baseline_circle = [0.5] * len(angles)
    ax.plot(angles, baseline_circle, ':', linewidth=1, color='gray', alpha=0.7)
    
    # Set labels and limits
    ax.set_thetagrids(np.degrees(angles[:-1]), opponents[:-1])
    ax.set_ylim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(0)
    
    # Add title and overall win rate
    plt.title('Performance Profile Against Different Opponents', fontsize=16, y=1.1)
    plt.figtext(0.5, 0.02, f'Overall Win Rate: {final_eval_results["overall"]["win_rate"]:.2f}',
               ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "radar_performance.png"))
    plt.close()
    
    # 2. Win Rate Evolution by Opponent Type
    if 'eval_win_rates' in training_results:
        eval_win_rates = training_results['eval_win_rates']
        
        if eval_win_rates:
            plt.figure(figsize=(14, 8))
            
            # Plot win rate evolution for each opponent type
            for opponent, rates in eval_win_rates.items():
                if rates and opponent != 'overall':
                    plt.plot(range(len(rates)), rates, 'o-', linewidth=2, label=f'vs {opponent}')
            
            # Add threshold line
            plt.axhline(y=win_rate_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({win_rate_threshold:.2f})')
            
            # Add 50% baseline
            plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='50% Baseline')
            
            plt.title('Win Rate Evolution by Opponent Type', fontsize=16)
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Win Rate', fontsize=14)
            plt.ylim(0, 1.05)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_dir, "win_rate_evolution.png"))
            plt.close()
    
    # 3. Comprehensive Final Report Dashboard
    plt.figure(figsize=(16, 16))
    
    # Define grid layout: 2x2 grid
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Panel 1: Win Rate by Opponent (bar chart)
    ax1 = plt.subplot(gs[0, 0])
    opponents = [op for op in final_eval_results.keys() if op != 'overall']
    win_rates = [final_eval_results[op]['win_rate'] for op in opponents]
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
    
    bars = ax1.bar(opponents, win_rates, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax1.axhline(y=win_rate_threshold, color='red', linestyle='--', 
               label=f'Threshold ({win_rate_threshold:.2f})')
    ax1.axhline(y=final_eval_results['overall']['win_rate'], color='black', linestyle='-',
               label=f'Overall ({final_eval_results["overall"]["win_rate"]:.2f})')
    
    ax1.set_title('Final Win Rates by Opponent', fontsize=14)
    ax1.set_xlabel('Opponent', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticklabels(opponents, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    
    # Panel 2: Training Rewards
    ax2 = plt.subplot(gs[0, 1])
    episode_rewards = training_results.get('episode_rewards', [])
    
    if episode_rewards:
        # Plot smoothed rewards
        window_size = min(100, max(1, len(episode_rewards) // 50))
        if len(episode_rewards) > window_size:
            smoothed_rewards = []
            for i in range(0, len(episode_rewards) - window_size + 1, max(1, window_size // 10)):
                smoothed_rewards.append(np.mean(episode_rewards[i:i+window_size]))
            x_values = range(window_size, len(episode_rewards) + 1, max(1, window_size // 10))
            ax2.plot(x_values, smoothed_rewards, linewidth=2, color='blue')
    
    ax2.set_title('Training Rewards (Smoothed)', fontsize=14)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Win Rate Progression by Opponent Difficulty
    ax3 = plt.subplot(gs[1, 0])
    if 'eval_win_rates' in training_results:
        # Group opponents by difficulty
        easy_opponents = ['beginner']
        medium_opponents = ['conservative', 'aggressive']
        hard_opponents = ['adaptive', 'optimal']
        
        # Calculate average win rates by difficulty
        easy_rates = []
        medium_rates = []
        hard_rates = []
        
        for i in range(num_generations):
            # Easy opponents
            easy_gen_rates = []
            for op in easy_opponents:
                if op in eval_win_rates and i < len(eval_win_rates[op]):
                    easy_gen_rates.append(eval_win_rates[op][i])
            if easy_gen_rates:
                easy_rates.append(np.mean(easy_gen_rates))
            
            # Medium opponents
            medium_gen_rates = []
            for op in medium_opponents:
                if op in eval_win_rates and i < len(eval_win_rates[op]):
                    medium_gen_rates.append(eval_win_rates[op][i])
            if medium_gen_rates:
                medium_rates.append(np.mean(medium_gen_rates))
            
            # Hard opponents
            hard_gen_rates = []
            for op in hard_opponents:
                if op in eval_win_rates and i < len(eval_win_rates[op]):
                    hard_gen_rates.append(eval_win_rates[op][i])
            if hard_gen_rates:
                hard_rates.append(np.mean(hard_gen_rates))
        
        # Plot win rates by difficulty
        if easy_rates:
            ax3.plot(range(len(easy_rates)), easy_rates, 'o-', 
                    linewidth=2, color='green', label='Easy')
        if medium_rates:
            ax3.plot(range(len(medium_rates)), medium_rates, 's-', 
                    linewidth=2, color='orange', label='Medium')
        if hard_rates:
            ax3.plot(range(len(hard_rates)), hard_rates, '^-', 
                    linewidth=2, color='red', label='Hard')
    
    ax3.set_title('Win Rate by Opponent Difficulty', fontsize=14)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Win Rate', fontsize=12)
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Population Fitness Evolution
    ax4 = plt.subplot(gs[1, 1])
    if 'generation_stats' in training_results:
        generation_stats = training_results['generation_stats']
        
        if generation_stats:
            # Extract best, average, and worst fitness from each generation
            generations = [stats['generation'] for stats in generation_stats]
            best_fitness = [max(stats['population_fitness']) for stats in generation_stats]
            avg_fitness = [sum(stats['population_fitness'])/len(stats['population_fitness']) 
                          for stats in generation_stats]
            worst_fitness = [min(stats['population_fitness']) for stats in generation_stats]
            
            # Plot fitness evolution
            ax4.plot(generations, best_fitness, 'o-', linewidth=2, 
                    color='green', label='Best')
            ax4.plot(generations, avg_fitness, 's-', linewidth=2, 
                    color='blue', label='Average')
            ax4.plot(generations, worst_fitness, '^-', linewidth=2, 
                    color='red', label='Worst')
            
            # Fill area between best and worst
            ax4.fill_between(generations, best_fitness, worst_fitness, 
                            color='lightblue', alpha=0.3)
    
    ax4.set_title('Population Fitness Evolution', fontsize=14)
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Fitness', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Radar Chart of Performance
    ax5 = plt.subplot(gs[2, 0], polar=True)
    
    # Get opponents and win rates
    opponents = [op for op in final_eval_results.keys() if op != 'overall']
    win_rates = [final_eval_results[op]['win_rate'] for op in opponents]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(opponents), endpoint=False).tolist()
    
    # Close the loop
    win_rates.append(win_rates[0])
    angles.append(angles[0])
    opponents.append(opponents[0])
    
    # Plot
    ax5.plot(angles, win_rates, 'o-', linewidth=2, color='blue')
    ax5.fill(angles, win_rates, alpha=0.25, color='blue')
    
    # Add threshold circle
    threshold_circle = [win_rate_threshold] * len(angles)
    ax5.plot(angles, threshold_circle, '--', linewidth=1.5, color='red', alpha=0.7)
    
    # Set labels and limits
    ax5.set_thetagrids(np.degrees(angles[:-1]), opponents[:-1])
    ax5.set_ylim(0, 1)
    ax5.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax5.set_rlabel_position(0)
    
    ax5.set_title('Performance Profile', fontsize=14)
    
    # Panel 6: Overall Training Information
    ax6 = plt.subplot(gs[2, 1])
    # Remove axes for text-only panel
    ax6.axis('off')
    
    # Create training summary text
    summary_text = [
        f"Agent Type: {agent_type.upper()}",
        f"Preset: {preset}",
        f"Players: {training_results['parameters']['num_players']}, Dice: {training_results['parameters']['num_dice']}",
        f"Episodes: {training_results['parameters']['self_play_episodes']}",
        f"Population Size: {training_results['parameters']['population_size']}",
        f"Elite Size: {training_results['parameters']['elite_size']}",
        f"Generations: {training_results['parameters']['generations']}",
        f"",
        f"Final Overall Win Rate: {final_eval_results['overall']['win_rate']:.4f}",
        f"Best vs Beginner: {final_eval_results.get('beginner', {}).get('win_rate', 0):.4f}",
        f"Best vs Conservative: {final_eval_results.get('conservative', {}).get('win_rate', 0):.4f}",
        f"Best vs Aggressive: {final_eval_results.get('aggressive', {}).get('win_rate', 0):.4f}",
        f"Best vs Adaptive: {final_eval_results.get('adaptive', {}).get('win_rate', 0):.4f}",
        f"Best vs Optimal: {final_eval_results.get('optimal', {}).get('win_rate', 0):.4f}",
    ]
    
    # Add training summary
    ax6.text(0.1, 0.95, '\n'.join(summary_text), fontsize=12, verticalalignment='top')
    ax6.set_title('Training Summary', fontsize=14)
    
    # Add overall title to the figure
    plt.suptitle(f"Final Training Report: {agent_type.upper()} Agent", fontsize=18, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    plt.savefig(os.path.join(save_dir, "final_report.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Population-based self-play training for Liar's Dice agents with specialized evaluation")
    parser.add_argument('--agent', type=str, default='ppo', 
                        choices=['dqn', 'ppo', 'crf', 'all'],
                        help='Type of agent to train (or "all" for all types)')
    parser.add_argument('--preset', type=str, default='basic', 
                        choices=['basic', 'standard', 'advanced'],
                        help='Configuration preset to use')
    parser.add_argument('--path', type=str, default='results/pbt_self_play',
                        help='Base path for results')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of self-play episodes (if None, use preset value)')
    parser.add_argument('--population', type=int, default=10,
                        help='Population size')
    parser.add_argument('--elite', type=int, default=3,
                        help='Number of elite agents to keep unchanged')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--players', type=int, default=None,
                        help='Number of players (if None, use preset value)')
    parser.add_argument('--dice', type=int, default=None,
                        help='Number of dice per player (if None, use preset value)')
    parser.add_argument('--eval-freq', type=int, default=None,
                        help='Evaluation frequency (episodes between evaluations)')
    parser.add_argument('--early-stop', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering during training')
    parser.add_argument('--no-detailed-metrics', action='store_true',
                        help='Disable detailed metrics tracking')
    
    args = parser.parse_args()
    
    # Create base results directory
    os.makedirs(args.path, exist_ok=True)
    
    # Determine which agents to train
    agents_to_train = ['ppo', 'dqn', 'crf'] if args.agent == 'all' else [args.agent]
    
    # Train each agent
    results = {}
    for agent_type in agents_to_train:
        print(f"\n{'='*80}")
        print(f"Training {agent_type.upper()} agent with population-based self-play")
        print(f"Using specialized evaluation agents: {EVALUATION_AGENTS}")
        print(f"{'='*80}\n")
        
        agent_path = os.path.join(args.path, agent_type)
        agent, training_results = population_based_self_play(
            agent_type=agent_type,
            preset=args.preset,
            results_path=agent_path,
            seed=args.seed,
            num_players=args.players,
            num_dice=args.dice,
            self_play_episodes=args.episodes,
            population_size=args.population,
            elite_size=args.elite,
            evaluation_frequency=args.eval_freq,
            render_training=args.render,
            enable_early_stopping=args.early_stop,
            detailed_metrics=not args.no_detailed_metrics
        )
        
        results[agent_type] = {
            'win_rate': training_results['final_evaluation']['overall']['win_rate'],
            'training_time': training_results['training_time'],
            'performance_by_opponent': {
                opponent: training_results['final_evaluation'][opponent]['win_rate'] 
                for opponent in EVALUATION_AGENTS if opponent in training_results['final_evaluation']
            }
        }
    
    # Print summary of results
    print("\n")
    print(f"{'='*80}")
    print(f"Population-Based Self-Play Training Results Summary")
    print(f"{'='*80}")
    
    for agent_type, result in results.items():
        hours, remainder = divmod(result['training_time'], 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{agent_type.upper()} Agent:")
        print(f"  Overall Win Rate: {result['win_rate']:.2f}")
        print(f"  Training Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        print("  Performance against specialized agents:")
        for opponent, win_rate in result['performance_by_opponent'].items():
            print(f"    - vs {opponent}: {win_rate:.2f}")
    
    print("\nTraining complete! Results and visualizations saved to:")
    print(args.path)
    

if __name__ == "__main__":
    main()