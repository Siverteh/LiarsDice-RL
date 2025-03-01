"""
Training module for DQN agent in Liar's Dice.

This module contains functions for training DQN agents, including
training loops, experience collection, and progress tracking.
"""

import os
import time
import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional, Callable
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from environment.game import LiarsDiceGame
from .environment_wrapper import LiarsDiceEnvWrapper
from .utils import setup_logger, save_training_data
from agents.rule_agent import CURRICULUM_LEVELS  # Add this import
from environment.state import ObservationEncoder  # Add this import

def train_dqn_episode(
    env: LiarsDiceEnvWrapper,
    agent: DQNAgent,
    epsilon: Optional[float] = None,
    render: bool = False,
    reward_shaping: bool = False
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Train the DQN agent for a single episode.
    
    Args:
        env: Environment wrapper
        agent: DQN agent
        epsilon: Override epsilon for this episode (if None, use agent's epsilon)
        render: Whether to render the environment
        reward_shaping: Whether to use reward shaping
        
    Returns:
        Tuple of (total_reward, episode_length, episode_info)
    """
    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    loss_values = []
    
    # Set exploration epsilon if provided
    original_epsilon = agent.epsilon
    if epsilon is not None:
        agent.epsilon = epsilon
    
    # Track episode data
    episode_info = {
        'actions': [],
        'rewards': [],
        'observations': [],
        'losses': []
    }
    
    try:
        while not done:
            # Render if requested
            if render:
                env.render()
                time.sleep(0.5)  # Slow down rendering
            
            # Get valid actions
            valid_action_indices = env.get_valid_actions()
            
            # Select action
            valid_actions = [env.action_mapping[idx] for idx in valid_action_indices]
            action = agent.select_action(obs, valid_actions)
            
            # Find the index of the selected action
            action_idx = None
            for idx, valid_action in enumerate(valid_actions):
                if agent._actions_equal(action, valid_action):
                    action_idx = valid_action_indices[idx]
                    break
            
            if action_idx is None:
                raise ValueError(f"Selected action {action} not found in valid actions")
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action_idx)
            
            # Apply reward shaping if enabled
            if reward_shaping:
                shaped_reward = shape_reward(
                    reward, obs, action, next_obs, info, 
                    dice_counts=info.get('dice_counts')
                )
            else:
                shaped_reward = reward
            
            # Add experience to replay buffer
            agent.add_experience(obs, action, shaped_reward, next_obs, done)
            
            # Update agent
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.update()
                loss_values.append(loss)
                episode_info['losses'].append(loss)
            
            # Track episode data
            episode_info['actions'].append(action)
            episode_info['rewards'].append(reward)  # Track original reward
            episode_info['observations'].append(obs.copy())
            
            # Update for next step
            obs = next_obs
            total_reward += reward  # Use original reward for reporting
            episode_steps += 1
            
            # Optional: limit maximum episode length
            if episode_steps >= 100:  # Prevent very long games
                break
    finally:
        # Restore original epsilon
        if epsilon is not None:
            agent.epsilon = original_epsilon
    
    episode_info['total_reward'] = total_reward
    episode_info['episode_steps'] = episode_steps
    episode_info['mean_loss'] = np.mean(loss_values) if loss_values else 0.0
    
    return total_reward, episode_steps, episode_info


def train_dqn(
    env: LiarsDiceEnvWrapper,
    agent: DQNAgent,
    num_episodes: int = 1000,
    log_interval: int = 10,
    save_interval: int = 100,
    eval_interval: int = 50,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    render_interval: Optional[int] = None,
    eval_episodes: int = 20,
    eval_epsilon: float = 0.05,
    early_stopping: bool = False,
    win_rate_threshold: float = 0.9,
    patience: int = 3,
    callback: Optional[Callable[[int, Dict[str, Any], DQNAgent], None]] = None,
    reward_shaping: bool = False
) -> Dict[str, Any]:
    """
    Train a DQN agent on Liar's Dice.
    
    Args:
        env: Environment wrapper
        agent: DQN agent to train
        num_episodes: Number of episodes to train for
        log_interval: Interval (in episodes) for logging
        save_interval: Interval (in episodes) for saving checkpoints
        eval_interval: Interval (in episodes) for evaluation
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        render_interval: Interval for rendering (if None, no rendering)
        eval_episodes: Number of episodes for evaluation
        eval_epsilon: Exploration rate during evaluation
        early_stopping: Whether to enable early stopping based on win rate
        win_rate_threshold: Win rate threshold for early stopping
        patience: Number of consecutive evaluations above threshold to trigger early stopping
        callback: Optional callback function after each evaluation
        reward_shaping: Whether to use reward shaping
        
    Returns:
        Dictionary with training results
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('dqn_train', os.path.join(log_dir, 'train.log'))
    logger.info(f"Starting DQN training for {num_episodes} episodes")
    logger.info(f"Agent config: {agent.get_statistics()}")
    
    # Tracking variables
    rewards = []
    episode_lengths = []
    losses = []
    eval_rewards = []
    win_rates = []
    above_threshold_count = 0  # Counter for early stopping
    
    # Make sure the action mapping is set in the agent
    if agent.action_to_game_action is None:
        agent.set_action_mapping(env.action_mapping)
    
    # Main training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes"):
        # Render this episode?
        render = render_interval is not None and episode % render_interval == 0
        
        # Run a training episode
        reward, steps, info = train_dqn_episode(
            env, agent, render=render, reward_shaping=reward_shaping
        )
        
        # Track metrics
        rewards.append(reward)
        episode_lengths.append(steps)
        if info['mean_loss'] > 0:
            losses.append(info['mean_loss'])
        
        # Log progress
        if episode % log_interval == 0:
            mean_reward = np.mean(rewards[-log_interval:])
            mean_length = np.mean(episode_lengths[-log_interval:])
            mean_loss = np.mean(losses[-log_interval:]) if losses else 0.0
            
            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {mean_reward:.2f} | "
                f"Length: {mean_length:.2f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_episode_{episode}")
            agent.save(checkpoint_path)
            
            # Save training data
            training_data = {
                'rewards': rewards,
                'episode_lengths': episode_lengths,
                'losses': losses,
                'eval_rewards': eval_rewards,
                'win_rates': win_rates,
                'episode': episode,
                'agent_stats': agent.get_statistics()
            }
            save_training_data(training_data, os.path.join(log_dir, 'training_data.pkl'))
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Evaluate
        if episode % eval_interval == 0:
            # Evaluate reward
            eval_reward = evaluate_training(env, agent, eval_episodes, eval_epsilon)
            eval_rewards.append(eval_reward)
            
            # Calculate win rate for early stopping
            wins = 0
            for _ in range(30):  # Use 30 episodes to get a stable win rate estimate
                obs = env.reset()
                done = False
                while not done:
                    valid_action_indices = env.get_valid_actions()
                    valid_actions = [env.action_mapping[idx] for idx in valid_action_indices]
                    action = agent.select_action(obs, valid_actions, training=False)
                    action_idx = None
                    for idx, valid_action in enumerate(valid_actions):
                        if agent._actions_equal(action, valid_action):
                            action_idx = valid_action_indices[idx]
                            break
                    obs, reward, done, info = env.step(action_idx)
                
                # Check if agent won
                if done and reward > 0:
                    wins += 1
            
            win_rate = wins / 30
            win_rates.append(win_rate)
            
            logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f} reward, Win rate: {win_rate:.2f}")
            
            # Check for early stopping
            if early_stopping and win_rate >= win_rate_threshold:
                above_threshold_count += 1
                logger.info(f"Win rate {win_rate:.2f} is above threshold {win_rate_threshold:.2f} "
                           f"({above_threshold_count}/{patience})")
                
                if above_threshold_count >= patience:
                    logger.info(f"Early stopping triggered at episode {episode} with win rate {win_rate:.2f}")
                    # Save a final checkpoint for this level
                    checkpoint_path = os.path.join(checkpoint_dir, f"early_stop_episode_{episode}")
                    agent.save(checkpoint_path)
                    break
            else:
                above_threshold_count = 0  # Reset counter if win rate drops below threshold
            
            # Call callback if provided
            if callback is not None:
                current_data = {
                    'rewards': rewards,
                    'episode_lengths': episode_lengths,
                    'losses': losses,
                    'eval_rewards': eval_rewards,
                    'win_rates': win_rates,
                    'current_episode': episode,
                    'last_eval_reward': eval_reward,
                    'last_win_rate': win_rate
                }
                callback(episode, current_data, agent)
    
    logger.info("Training completed!")
    
    # Final evaluation
    final_eval_reward = evaluate_training(env, agent, eval_episodes * 2, eval_epsilon)
    logger.info(f"Final evaluation: {final_eval_reward:.2f} reward")
    
    # Prepare and return results
    results = {
        'rewards': rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'eval_rewards': eval_rewards + [final_eval_reward],
        'win_rates': win_rates,
        'final_eval_reward': final_eval_reward,
        'agent_stats': agent.get_statistics(),
        'trained_episodes': num_episodes
    }
    
    return results


def evaluate_training(
    env: LiarsDiceEnvWrapper,
    agent: DQNAgent,
    num_episodes: int = 20,
    epsilon: float = 0.05
) -> float:
    """
    Evaluate the agent during training.
    
    Args:
        env: Environment wrapper
        agent: DQN agent to evaluate
        num_episodes: Number of episodes to evaluate
        epsilon: Exploration rate during evaluation
        
    Returns:
        Mean reward across evaluation episodes
    """
    eval_rewards = []
    
    # Store original epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = epsilon
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
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
            obs, reward, done, _ = env.step(action_idx)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Return mean reward
    return np.mean(eval_rewards)

def shape_reward(
    original_reward: float,
    observation: np.ndarray,
    action: Dict[str, Any],
    next_observation: np.ndarray,
    info: Dict[str, Any],
    dice_counts: Optional[np.ndarray] = None
) -> float:
    """
    Shape the reward to provide better learning signals.
    
    Enhances rewards based on game state and action quality.
    
    Args:
        original_reward: Original reward from the environment
        observation: Current observation
        action: Action taken
        next_observation: Next observation
        info: Additional information from the environment
        dice_counts: Dice counts of all players
        
    Returns:
        Shaped reward
    """
    reward = original_reward
    
    # Get game state information
    game_state = info.get('state', 'ONGOING')
    
    # Enhanced rewards for winning/losing
    if game_state == 'GAME_OVER':
        if reward > 0:
            reward *= 2.0  # Double the winning reward
        else:
            reward *= 0.5  # Halve the losing penalty (less discouragement)
    
    # Reward specific actions
    if action['type'] == 'bid':
        # Small reward for making a bid (encourages action)
        reward += 0.05
        
    elif action['type'] == 'challenge':
        # If the challenge was successful (reward is positive)
        if reward > 0:
            reward += 0.5  # Bonus for successful challenges
        # If the challenge failed (reward is negative)
        elif reward < 0:
            reward -= 0.1  # Extra penalty for failed challenges
    
    # Reward for keeping dice (survival)
    if 'dice_counts' in info:
        current_player = 0  # Assumes player 0 is the DQN agent
        if info['dice_counts'][current_player] > 0:
            reward += 0.1  # Small reward for survival
    
    # Reward for outlasting opponents
    if dice_counts is not None and sum(dice_counts) < len(dice_counts) * dice_counts[0]:
        # The agent has more dice than the average player
        reward += 0.1
    
    return reward

def train_self_play(
    agent: DQNAgent,
    num_episodes: int,
    num_players: int,
    num_dice: int,
    dice_faces: int,
    checkpoint_dir: str,
    log_dir: str,
    seed: Optional[int] = None,
    eval_interval: int = 200,
    update_opponent_interval: int = 500,
    initial_epsilon: float = 0.1
):
    """
    Train the agent through self-play against copies of itself.
    
    Periodically updates the opponent with the current agent's policy.
    
    Args:
        agent: DQN agent to train
        num_episodes: Number of self-play episodes
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        seed: Random seed for reproducibility
        eval_interval: Interval for evaluation
        update_opponent_interval: Interval for updating opponent agent
        initial_epsilon: Initial exploration rate for self-play
        
    Returns:
        Dictionary with training results
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('self_play', os.path.join(log_dir, 'self_play.log'))
    logger.info(f"Starting self-play training for {num_episodes} episodes")
    
    # Copy the agent for the opponent
    opponent_agent = DQNAgent(
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        device=agent.device
    )
    # Copy the weights from the agent
    opponent_agent.q_network.load_state_dict(agent.q_network.state_dict())
    opponent_agent.target_network.load_state_dict(agent.target_network.state_dict())
    opponent_agent.set_action_mapping(agent.action_to_game_action)
    opponent_agent.epsilon = initial_epsilon  # Some exploration for opponent
    
    # Define a specialized self-play environment
    class SelfPlayEnv:
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
                
                # Get action from opponent agent
                opponent_action = opponent_agent.select_action(
                    obs, valid_actions, training=False
                )
                
                # Execute opponent's action
                observations, rewards, done, info = self.game.step(opponent_action)
            
            # Return observation for agent
            next_obs = self.observation_encoder.encode(observations[0])
            reward = rewards[0]  # Agent's reward
            
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
    env = SelfPlayEnv()
    
    # Original agent epsilon for recovery
    original_epsilon = agent.epsilon
    agent.epsilon = max(original_epsilon, 0.1)  # Ensure some exploration during self-play
    
    # Training stats
    rewards = []
    episode_lengths = []
    losses = []
    win_rates = []
    
    # Main training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Self-Play Training"):
        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        
        # Episode loop
        while not done:
            # Select action
            valid_action_indices = env.get_valid_actions()
            valid_actions = [agent.action_to_game_action[idx] for idx in valid_action_indices]
            action = agent.select_action(obs, valid_actions)
            
            # Get action index
            action_idx = None
            for idx, valid_action in enumerate(valid_actions):
                if agent._actions_equal(action, valid_action):
                    action_idx = valid_action_indices[idx]
                    break
            
            if action_idx is None:
                raise ValueError(f"Action {action} not found in valid actions")
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action_idx)
            
            # Add experience with reward shaping
            shaped_reward = shape_reward(reward, obs, action, next_obs, info)
            agent.add_experience(obs, action, shaped_reward, next_obs, done)
            
            # Update agent
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.update()
                episode_losses.append(loss)
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            # Prevent infinite games
            if episode_steps >= 100:
                break
        
        # Track metrics
        rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Log progress
        if episode % 100 == 0:
            mean_reward = np.mean(rewards[-100:])
            mean_length = np.mean(episode_lengths[-100:])
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            
            logger.info(f"Self-play episode {episode}/{num_episodes} | "
                       f"Reward: {mean_reward:.2f} | Length: {mean_length:.2f} | "
                       f"Loss: {mean_loss:.4f} | Buffer: {len(agent.replay_buffer)}")
        
        # Update opponent with current agent's policy
        if episode % update_opponent_interval == 0:
            logger.info(f"Updating opponent agent at episode {episode}")
            opponent_agent.q_network.load_state_dict(agent.q_network.state_dict())
            opponent_agent.target_network.load_state_dict(agent.target_network.state_dict())
        
        # Save checkpoint
        if episode % 500 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"self_play_{episode}")
            agent.save(checkpoint_path)
            logger.info(f"Saved self-play checkpoint to {checkpoint_path}")
        
        # Evaluate agent
        if episode % eval_interval == 0:
            # Evaluate against all rule-based agents
            eval_wins = 0
            eval_episodes = min(100, len(CURRICULUM_LEVELS) * 10)
            episodes_per_opponent = max(5, eval_episodes // len(CURRICULUM_LEVELS))
            
            for opponent_type in CURRICULUM_LEVELS:
                eval_env = LiarsDiceEnvWrapper(
                    num_players=num_players,
                    num_dice=num_dice,
                    dice_faces=dice_faces,
                    seed=seed,
                    rule_agent_types=[opponent_type]
                )
                
                for _ in range(episodes_per_opponent):
                    eval_obs = eval_env.reset()
                    eval_done = False
                    
                    while not eval_done:
                        valid_indices = eval_env.get_valid_actions()
                        valid_acts = [eval_env.action_mapping[idx] for idx in valid_indices]
                        
                        # Use greedy policy for evaluation
                        eval_action = agent.select_action(eval_obs, valid_acts, training=False)
                        
                        # Convert to index
                        eval_idx = None
                        for i, act in enumerate(valid_acts):
                            if agent._actions_equal(eval_action, act):
                                eval_idx = valid_indices[i]
                                break
                        
                        if eval_idx is None:
                            raise ValueError("Could not find action index")
                        
                        # Take step
                        eval_obs, eval_reward, eval_done, eval_info = eval_env.step(eval_idx)
                    
                    # Check if won
                    if eval_reward > 0:
                        eval_wins += 1
            
            # Calculate win rate
            win_rate = eval_wins / eval_episodes
            win_rates.append(win_rate)
            
            logger.info(f"Self-play evaluation at episode {episode}: win rate = {win_rate:.2f}")
    
    # Restore original agent epsilon
    agent.epsilon = original_epsilon
    
    # Save final self-play model
    final_path = os.path.join(checkpoint_dir, "final_self_play")
    agent.save(final_path)
    
    # Return results
    return {
        'rewards': rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'win_rates': win_rates,
        'final_win_rate': win_rates[-1] if win_rates else 0.0
    }