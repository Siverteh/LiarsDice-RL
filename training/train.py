"""
Training module for reinforcement learning agents in Liar's Dice.

This module contains model-agnostic functions for training agents that implement
the RLAgent interface, supporting various algorithms like DQN, PPO, etc.
"""

import os
import time
import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional, Callable
from tqdm import tqdm

from agents.base_agent import RLAgent
from .environment_wrapper import LiarsDiceEnvWrapper
from .utils import setup_logger, save_training_data
from agents.rule_agent import CURRICULUM_LEVELS


def train_episode(
    env: LiarsDiceEnvWrapper,
    agent: RLAgent,
    render: bool = False,
    reward_shaping: bool = False,
    evaluation: bool = False
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Train an agent for a single episode.
    
    Args:
        env: Environment wrapper
        agent: RL agent that implements the RLAgent interface
        render: Whether to render the environment
        reward_shaping: Whether to use reward shaping
        evaluation: Whether this is an evaluation episode
        
    Returns:
        Tuple of (total_reward, episode_length, episode_info)
    """
    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    loss_values = []
    
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
            action = agent.select_action(obs, valid_actions, training=not evaluation)
            
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
            
            # Add experience to agent if not in evaluation mode
            if not evaluation:
                agent.add_experience(obs, action, shaped_reward, next_obs, done)
                
                # Update agent (for on-policy algorithms like PPO, this may be a no-op)
                loss = agent.update()
                if loss > 0:
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
    except Exception as e:
        print(f"Error during episode: {e}")
        raise e
    
    episode_info['total_reward'] = total_reward
    episode_info['episode_steps'] = episode_steps
    episode_info['mean_loss'] = np.mean(loss_values) if loss_values else 0.0
    
    return total_reward, episode_steps, episode_info


def train_agent(
    env: LiarsDiceEnvWrapper,
    agent: RLAgent,
    num_episodes: int = 1000,
    log_interval: int = 10,
    save_interval: int = 100,
    eval_interval: int = 50,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    render_interval: Optional[int] = None,
    eval_episodes: int = 20,
    early_stopping: bool = False,
    win_rate_threshold: float = 0.9,
    patience: int = 3,
    callback: Optional[Callable[[int, Dict[str, Any], RLAgent], bool]] = None,
    reward_shaping: bool = False
) -> Dict[str, Any]:
    """
    Train an agent using a model-agnostic approach.
    
    Args:
        env: Environment wrapper
        agent: RL agent that implements the RLAgent interface
        num_episodes: Number of episodes to train for
        log_interval: Interval (in episodes) for logging
        save_interval: Interval (in episodes) for saving checkpoints
        eval_interval: Interval (in episodes) for evaluation
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        render_interval: Interval for rendering (if None, no rendering)
        eval_episodes: Number of episodes for evaluation
        early_stopping: Whether to enable early stopping based on win rate
        win_rate_threshold: Win rate threshold for early stopping
        patience: Number of consecutive evaluations above threshold to trigger early stopping
        callback: Optional callback function after each evaluation
                 Returns True if training should stop early
        reward_shaping: Whether to use reward shaping
        
    Returns:
        Dictionary with training results
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('train', os.path.join(log_dir, 'train.log'))
    logger.info(f"Starting training for {num_episodes} episodes with agent type: {agent.__class__.__name__}")
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
        reward, steps, info = train_episode(
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
                f"Loss: {mean_loss:.4f}"
            )
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_episode_{episode}")
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
            eval_reward = evaluate_agent(env, agent, eval_episodes)
            eval_rewards.append(eval_reward)
            
            # Calculate win rate for early stopping
            wins = 0
            for _ in range(30):  # Use 30 episodes to get a stable win rate estimate
                eval_reward, _, _ = train_episode(env, agent, evaluation=True)
                if eval_reward > 0:  # Win
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
                stop_early = callback(episode, current_data, agent)
                if stop_early:
                    logger.info(f"Training stopped early by callback at episode {episode}")
                    break
    
    logger.info("Training completed!")
    
    # Final evaluation
    final_eval_reward = evaluate_agent(env, agent, eval_episodes * 2)
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
        'trained_episodes': len(rewards)
    }
    
    return results


def evaluate_agent(
    env: LiarsDiceEnvWrapper,
    agent: RLAgent,
    num_episodes: int = 20
) -> float:
    """
    Evaluate the agent.
    
    Args:
        env: Environment wrapper
        agent: RL agent to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Mean reward across evaluation episodes
    """
    eval_rewards = []
    
    for _ in range(num_episodes):
        reward, _, _ = train_episode(env, agent, evaluation=True)
        eval_rewards.append(reward)
    
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

    # Add a penalty for repetitive actions
    if hasattr(shape_reward, "last_action") and action == shape_reward.last_action:
        reward -= 0.2  # Small penalty for repeating the same action
    
    # Store the current action for next comparison
    shape_reward.last_action = action
    
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
        
        # Add reward for bidding values the agent actually has
        player_dice_key = 'player_0_dice'  # Assuming agent is player 0
        if player_dice_key in info and action.get('value') is not None:
            bid_value = action['value']
            own_dice = info[player_dice_key]
            own_dice_count = sum(1 for d in own_dice if d == bid_value)
            
            # Reward for bidding values you actually have
            if own_dice_count > 0:
                reward += 0.15 * own_dice_count
            # Small penalty for bidding values you don't have
            else:
                reward -= 0.1
        
    elif action['type'] == 'challenge':
        # If the challenge was successful (reward is positive)
        if reward > 0:
            reward += 0.5  # Bonus for successful challenges
        # If the challenge failed (reward is negative)
        elif reward < 0:
            reward -= 0.1  # Extra penalty for failed challenges
    
    # Reward for keeping dice (survival)
    if 'dice_counts' in info:
        current_player = 0  # Assumes player 0 is the agent
        if info['dice_counts'][current_player] > 0:
            reward += 0.1  # Small reward for survival
    
    # Reward for outlasting opponents
    if dice_counts is not None and sum(dice_counts) < len(dice_counts) * dice_counts[0]:
        # The agent has more dice than the average player
        reward += 0.1
    
    return reward


def train_self_play(
    agent: RLAgent,
    num_episodes: int,
    num_players: int,
    num_dice: int,
    dice_faces: int,
    checkpoint_dir: str,
    log_dir: str,
    seed: Optional[int] = None,
    eval_interval: int = 200,
    update_opponent_interval: int = 500,
    reward_shaping: bool = False
):
    """
    Train the agent through self-play.
    
    This function is agent-agnostic and works with any agent that implements
    the RLAgent interface.
    
    Args:
        agent: RL agent to train
        num_episodes: Number of self-play episodes
        num_players: Number of players in the game
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        seed: Random seed for reproducibility
        eval_interval: Interval for evaluation
        update_opponent_interval: Interval for updating opponent agent
        reward_shaping: Whether to use reward shaping
        
    Returns:
        Dictionary with training results
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger('self_play', os.path.join(log_dir, 'self_play.log'))
    logger.info(f"Starting self-play training for {num_episodes} episodes")
    
    # Create opponent agent (must be same type as main agent)
    # Since this is just a clone, we create it the same way
    opponent_agent = agent.__class__(**agent.get_statistics())
    
    # Create a specialized self-play environment wrapper
    class SelfPlayEnv:
        def __init__(self):
            from environment.game import LiarsDiceGame
            from environment.state import ObservationEncoder
            
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
            
            # Add player dice info for reward shaping
            for player in range(self.game.num_players):
                player_dice = [int(d) for d in self.game.dice[player, :self.game.dice_counts[player]]]
                info[f'player_{player}_dice'] = player_dice
            
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
                
                # Add player dice info again
                for player in range(self.game.num_players):
                    player_dice = [int(d) for d in self.game.dice[player, :self.game.dice_counts[player]]]
                    info[f'player_{player}_dice'] = player_dice
            
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
    opponent_agent.set_action_mapping(agent.action_to_game_action)
    
    # Training stats
    rewards = []
    episode_lengths = []
    losses = []
    win_rates = []
    
    # Main training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Self-Play Training"):
        # Run a training episode
        reward, steps, info = train_episode(
            env, agent, reward_shaping=reward_shaping
        )
        
        # Track metrics
        rewards.append(reward)
        episode_lengths.append(steps)
        if 'mean_loss' in info and info['mean_loss'] > 0:
            losses.append(info['mean_loss'])
        
        # Log progress
        if episode % 100 == 0:
            mean_reward = np.mean(rewards[-100:])
            mean_length = np.mean(episode_lengths[-100:])
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            
            logger.info(f"Self-play episode {episode}/{num_episodes} | "
                       f"Reward: {mean_reward:.2f} | Length: {mean_length:.2f} | "
                       f"Loss: {mean_loss:.4f}")
        
        # Update opponent with current agent periodically
        if episode % update_opponent_interval == 0:
            logger.info(f"Updating opponent agent at episode {episode}")
            # Save agent to temporary location and load to opponent
            temp_dir = os.path.join(checkpoint_dir, 'temp_self_play')
            os.makedirs(temp_dir, exist_ok=True)
            agent.save(temp_dir)
            opponent_agent.load(temp_dir)
        
        # Save checkpoint
        if episode % 500 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"self_play_{episode}")
            agent.save(checkpoint_path)
            logger.info(f"Saved self-play checkpoint to {checkpoint_path}")
            
            # Save training data
            training_data = {
                'rewards': rewards,
                'episode_lengths': episode_lengths,
                'losses': losses,
                'win_rates': win_rates,
                'episode': episode,
                'agent_stats': agent.get_statistics()
            }
            save_training_data(training_data, os.path.join(log_dir, 'self_play_data.pkl'))
        
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
                
                # Set action mapping for the environment
                agent.set_action_mapping(eval_env.action_mapping)
                
                for _ in range(episodes_per_opponent):
                    # Use evaluation mode to avoid adding experience
                    reward, _, _ = train_episode(
                        eval_env, agent, evaluation=True
                    )
                    
                    # Check if won
                    if reward > 0:
                        eval_wins += 1
            
            # Calculate win rate
            win_rate = eval_wins / eval_episodes
            win_rates.append(win_rate)
            
            logger.info(f"Self-play evaluation at episode {episode}: win rate = {win_rate:.2f}")
    
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