"""
Self-play learning for reinforcement learning agents in Liar's Dice.

This module implements a self-play learning approach that trains agents against 
themselves with multiple rounds of increasing difficulty.
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
from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from training.environment_wrapper import LiarsDiceEnvWrapper
from training.train import train_agent, evaluate_agent, train_episode
from training.evaluate import evaluate_against_curriculum, visualize_evaluation_results
from training.utils import (
    setup_logger, 
    save_training_data, 
    load_training_data,
    plot_training_results
)


def self_play_learning(
    # Base settings
    model_path: str,
    results_path: str = 'results/self_play/ppo_5dice',
    seed: Optional[int] = None,
    device: str = 'auto',
    render_training: bool = False,
    
    # Game setup
    num_players: int = 2,
    num_dice: int = 5,
    dice_faces: int = 6,
    
    # Self-play parameters
    num_rounds: int = 10,
    episodes_per_round: int = 10000,
    win_rate_threshold: float = 0.80,
    opponent_update_frequency: int = 1000,
    
    # Agent configuration
    custom_agent_config: Optional[Dict[str, Any]] = None,
    
    # Training control
    checkpoint_frequency: int = 5000,
    evaluation_frequency: int = 100,
    eval_against_rules: bool = True,
    
    # Exploration settings
    initial_epsilon: float = 0.3,
    min_epsilon: float = 0.05,
    epsilon_decay_per_round: float = 0.5
) -> Tuple[RLAgent, Dict[str, Any]]:
    """
    Train an agent using multi-round self-play, with increasing difficulty each round.
    
    Args:
        model_path: Path to the pre-trained model to load
        results_path: Directory to save results, checkpoints, and logs
        seed: Random seed for reproducibility
        device: Device to run on ('cpu', 'cuda', or 'auto')
        render_training: Whether to render gameplay during training
        
        # Game settings
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        
        # Self-play parameters
        num_rounds: Number of self-play rounds
        episodes_per_round: Number of episodes per round
        win_rate_threshold: Win rate to achieve before moving to next round
        opponent_update_frequency: How often to update opponent model (in episodes)
        
        # Agent configuration
        custom_agent_config: Additional agent-specific parameters
        
        # Training control
        checkpoint_frequency: How often to save checkpoints (in episodes)
        evaluation_frequency: How often to evaluate the agent (in episodes)
        eval_against_rules: Whether to also evaluate against rule-based agents
        
        # Exploration settings
        initial_epsilon: Starting exploration rate (for DQN)
        min_epsilon: Minimum exploration rate
        epsilon_decay_per_round: How much to reduce epsilon per round
        
    Returns:
        Tuple of (trained_agent, training_results)
    """
    # Load model metadata to get agent type and configuration
    if os.path.isdir(model_path):
        metadata_path = os.path.join(model_path, 'metadata.json')
    else:
        metadata_path = os.path.join(os.path.dirname(model_path), 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        agent_type = metadata.get('agent_type', 'unknown')
        model_num_players = metadata.get('num_players', num_players)
        model_num_dice = metadata.get('num_dice', num_dice)
        model_network_size = metadata.get('network_size', [256, 128, 64])
    else:
        # Default values if metadata not found
        agent_type = 'unknown'
        model_network_size = [256, 128, 64]
        print(f"Warning: No metadata found for model. Assuming default configuration.")
    
    # Set up paths for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"self_play_{agent_type}_{num_players}p{num_dice}d_{timestamp}"
    base_path = os.path.join(results_path, run_name)
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    log_dir = os.path.join(base_path, 'logs')
    
    # Create directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('self_play', os.path.join(log_dir, 'self_play.log'))
    logger.info(f"Starting self-play learning for Liar's Dice with loaded {agent_type} agent")
    logger.info(f"Game setup: {num_players} players, {num_dice} dice, {dice_faces} faces")
    logger.info(f"Self-play: {num_rounds} rounds with {episodes_per_round} episodes per round")
    
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
        num_players=num_players,
        num_dice=num_dice,
        dice_faces=dice_faces,
        seed=seed
    )
    
    obs_shape = env.get_observation_shape()
    obs_dim = obs_shape[0]
    action_dim = env.get_action_dim()
    logger.info(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # Create agent configuration
    if agent_type.lower() == 'dqn':
        agent_config = {
            'hidden_dims': model_network_size,
            'device': device,
            'epsilon_start': initial_epsilon,
            'epsilon_end': min_epsilon,
            'epsilon_decay': 0.9999  # Slower decay for more stable learning
        }
    elif agent_type.lower() == 'ppo':
        agent_config = {
            'hidden_dims': model_network_size,
            'device': device,
            'entropy_coef': 0.01,  # Lower entropy for more focused learning
            'update_frequency': 2048,
            'gae_lambda': 0.95
        }
    else:
        # Generic config for unknown agent type
        agent_config = {
            'hidden_dims': model_network_size,
            'device': device
        }
    
    # Apply any custom agent config
    if custom_agent_config:
        agent_config.update(custom_agent_config)
    
    # Create agent with the same architecture as the loaded model
    agent = create_agent(
        agent_type=agent_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config
    )
    
    # Set action mapping
    agent.set_action_mapping(env.action_mapping)
    
    # Load the pre-trained model
    logger.info(f"Loading pre-trained model from {model_path}")
    agent.load(model_path)
    
    # Create opponent agent (clone of main agent)
    opponent_agent = create_agent(
        agent_type=agent_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config
    )
    opponent_agent.set_action_mapping(env.action_mapping)
    opponent_agent.load(model_path)
    
    # Create environment with opponent_agent as opponent
    env = LiarsDiceEnvWrapper(
        num_players=num_players,
        num_dice=num_dice,
        dice_faces=dice_faces,
        seed=seed,
        rl_agent_as_opponent=opponent_agent
    )
    
    # Track training progress and models
    all_results = {}
    current_epsilon = initial_epsilon
    best_model_path = os.path.join(checkpoint_dir, "best_model")
    best_winrate = 0.0
    
    # Save initial model
    agent.save(os.path.join(checkpoint_dir, "initial_model"))
    
    # Self-play rounds loop
    for round_idx in range(num_rounds):
        logger.info(f"\n=== Starting Self-Play Round {round_idx+1}/{num_rounds} ===")
        
        # Set exploration rate for this round (DQN only)
        if agent_type.lower() == 'dqn':
            # Gradually reduce exploration as we progress through rounds
            if round_idx > 0:
                current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay_per_round)
                agent.epsilon = current_epsilon
                logger.info(f"Setting epsilon to {current_epsilon:.3f} for round {round_idx+1}")
        
        # Create directory for this round
        round_dir = os.path.join(checkpoint_dir, f"round_{round_idx+1}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Variables to track this round's progress
        round_best_winrate = 0.0
        round_best_model_path = None
        episodes_since_update = 0
        round_complete = False
        
        # Track when we should update the opponent
        episodes_since_opponent_update = 0
        
        # Define callback for model saving and opponent updating
        def self_play_callback(episode, data, current_agent):
            nonlocal round_best_winrate, round_best_model_path, best_winrate, best_model_path
            nonlocal episodes_since_opponent_update, episodes_since_update, round_complete
            
            # Update opponent model periodically
            episodes_since_opponent_update += 1
            if episodes_since_opponent_update >= opponent_update_frequency:
                # Save current agent model
                temp_path = os.path.join(round_dir, "temp_model")
                current_agent.save(temp_path)
                
                # Load into opponent
                opponent_agent.load(temp_path)
                
                # Update the opponent in the environment
                if hasattr(env, 'update_rl_opponent'):
                    env.update_rl_opponent(opponent_agent)
                else:
                    # If the env doesn't have the update method, recreate it
                    # This is a fallback and shouldn't be needed with our updated wrapper
                    logger.warning("Environment doesn't have update_rl_opponent method. Recreation might be needed.")
                
                logger.info(f"Updated opponent model at episode {episode}")
                episodes_since_opponent_update = 0
            
            # Check if we should evaluate win rate
            if 'last_win_rate' in data:
                current_winrate = data['last_win_rate']
                episodes_since_update += 1
                
                # Update best model for this round if better
                if current_winrate > round_best_winrate:
                    round_best_winrate = current_winrate
                    round_best_model_path = os.path.join(round_dir, f"best_model")
                    current_agent.save(round_best_model_path)
                    logger.info(f"New best model for round {round_idx+1}: {current_winrate:.3f} win rate")
                    episodes_since_update = 0
                
                # Update best overall model if better
                if current_winrate > best_winrate:
                    best_winrate = current_winrate
                    best_model_path = os.path.join(checkpoint_dir, "best_overall")
                    current_agent.save(best_model_path)
                    logger.info(f"New best overall model: {current_winrate:.3f} win rate")
                
                # Check if we've reached the win rate threshold
                if current_winrate >= win_rate_threshold:
                    logger.info(f"Win rate threshold {win_rate_threshold:.3f} reached with {current_winrate:.3f}")
                    round_complete = True
                    return True  # Signal to stop training
                
                # If we haven't improved in a while, also move to next round
                if episodes_since_update >= 5 * evaluation_frequency:
                    logger.info(f"No improvement for {episodes_since_update} episodes, moving to next round")
                    round_complete = True
                    return True
            
            return False  # Continue training
        
        # Train with self-play for this round
        round_start_time = time.time()
        
        # Use the best model from previous round if available
        if round_idx > 0 and round_best_model_path and os.path.exists(round_best_model_path):
            logger.info(f"Loading best model from previous round")
            agent.load(round_best_model_path)
            
            # Also update the opponent with the best model
            opponent_agent.load(round_best_model_path)
            if hasattr(env, 'update_rl_opponent'):
                env.update_rl_opponent(opponent_agent)
        
        # Run training for this round
        round_results = train_agent(
            env=env,
            agent=agent,
            num_episodes=episodes_per_round,
            log_interval=min(100, episodes_per_round // 10),
            save_interval=checkpoint_frequency,
            eval_interval=evaluation_frequency,
            checkpoint_dir=round_dir,
            log_dir=os.path.join(log_dir, f"round_{round_idx+1}"),
            render_interval=100 if render_training else None,
            eval_episodes=500,  # More episodes for reliable evaluation
            callback=self_play_callback,
            reward_shaping=True
        )
        
        round_duration = time.time() - round_start_time
        
        # Add round results
        all_results[f"round_{round_idx+1}"] = {
            "episodes": episodes_per_round if not round_complete else round_results.get('episode', 0),
            "duration": round_duration,
            "final_win_rate": round_best_winrate,
            "model_path": round_best_model_path
        }
        
        # Log round completion
        logger.info(f"Completed self-play round {round_idx+1}")
        logger.info(f"Duration: {round_duration:.2f} seconds")
        logger.info(f"Best win rate: {round_best_winrate:.3f}")
        
        # Update opponent with best model from this round for next round
        if round_best_model_path and os.path.exists(round_best_model_path):
            logger.info(f"Updating opponent with best model from round {round_idx+1}")
            opponent_agent.load(round_best_model_path)
            
            # Also update the agent if it's not already using the best model
            agent.load(round_best_model_path)
            
            # Update the opponent in the environment
            if hasattr(env, 'update_rl_opponent'):
                env.update_rl_opponent(opponent_agent)
        
        # Evaluate against rule-based agents if requested
        if eval_against_rules:
            logger.info("Evaluating against rule-based agents...")
            
            eval_results = evaluate_against_curriculum(
                agent=agent,
                num_episodes_per_level=300,
                num_players=num_players,
                num_dice=num_dice,
                dice_faces=dice_faces,
                epsilon=0.05,  # Low exploration for evaluation
                seed=seed,
                verbose=True
            )
            
            # Log evaluation results
            logger.info("Evaluation results vs rule-based agents:")
            for level_name, result in eval_results.items():
                if level_name != 'overall':
                    logger.info(f"  vs {level_name}: win rate = {result['win_rate']:.3f}")
            
            logger.info(f"  Overall win rate: {eval_results['overall']['win_rate']:.3f}")
            
            # Save evaluation visualization
            vis_path = os.path.join(log_dir, f"evaluation_round_{round_idx+1}.png")
            visualize_evaluation_results(eval_results, save_path=vis_path)
            
            # Add to results
            all_results[f"round_{round_idx+1}"]["rule_eval"] = eval_results
    
    # Final evaluation against rule-based agents
    logger.info("\n=== Final Evaluation ===")
    final_eval_results = evaluate_against_curriculum(
        agent=agent,
        num_episodes_per_level=500,  # More episodes for final evaluation
        num_players=num_players,
        num_dice=num_dice,
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
        'round_results': all_results,
        'final_evaluation': final_eval_results,
        'parameters': {
            'agent_type': agent_type,
            'num_players': num_players,
            'num_dice': num_dice,
            'dice_faces': dice_faces,
            'num_rounds': num_rounds,
            'episodes_per_round': episodes_per_round,
            'win_rate_threshold': win_rate_threshold
        }
    }
    save_training_data(combined_results, os.path.join(log_dir, 'self_play_results.pkl'))
    
    # Save final model with metadata
    models_dir = os.path.join(results_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model name with performance info
    overall_win_rate = final_eval_results['overall']['win_rate']
    model_name = f"self_play_{agent_type}_{num_players}p{num_dice}d_{overall_win_rate:.2f}wr_{timestamp}"
    model_path = os.path.join(models_dir, model_name)
    agent.save(model_path)
    
    # Save model metadata
    model_metadata = {
        'timestamp': timestamp,
        'agent_type': agent_type,
        'num_players': num_players,
        'num_dice': num_dice,
        'dice_faces': dice_faces,
        'win_rates': {k: v['win_rate'] for k, v in final_eval_results.items() if k != 'overall'},
        'overall_win_rate': overall_win_rate,
        'network_size': model_network_size,
        'training_parameters': {
            'num_rounds': num_rounds,
            'episodes_per_round': episodes_per_round,
            'win_rate_threshold': win_rate_threshold
        }
    }
    
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"Saved final model to {model_path}")
    logger.info("\nSelf-play learning completed!")
    
    return agent, combined_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-play learning for Liar's Dice")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pre-trained model to load')
    parser.add_argument('--path', type=str, default='results/self_play/ppo_5dice',
                        help='Base path for results')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of self-play rounds')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Episodes per round')
    parser.add_argument('--players', type=int, default=2,
                        help='Number of players')
    parser.add_argument('--dice', type=int, default=5,
                        help='Number of dice per player')
    parser.add_argument('--win-rate', type=float, default=0.85,
                        help='Win rate threshold to advance rounds')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering during training')
    
    args = parser.parse_args()
    
    # Run self-play training with parsed arguments
    self_play_learning(
        model_path=args.model,
        results_path=args.path,
        num_rounds=args.rounds,
        episodes_per_round=args.episodes,
        num_players=args.players,
        num_dice=args.dice,
        win_rate_threshold=args.win_rate,
        seed=args.seed,
        render_training=args.render
    )