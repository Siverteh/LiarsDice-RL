"""
Trainer module for Liar's Dice reinforcement learning.

This module provides the Trainer class for training RL agents
to play Liar's Dice.
"""

import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import torch
import random
from dataclasses import dataclass
import json
import logging

from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from agents.rule_agent import RandomAgent, ConservativeAgent, StrategicAgent, MildConservativeAgent, VeryMildConservativeAgent, MixedAgent
from .utils import plot_training_curves, save_metrics


@dataclass
class TrainingConfig:
    """Configuration class for training settings."""
    
    # Environment settings
    num_players: int = 2
    num_dice: int = 3
    dice_faces: int = 6
    
    # Training settings
    num_episodes: int = 10000
    eval_interval: int = 100
    num_eval_games: int = 50
    save_interval: int = 500
    
    # Agent settings
    learning_agent_type: str = "dqn"  # "dqn", "ppo", "a2c"
    opponent_type: str = "strategic"  # "random", "conservative", "strategic", "self"
    
    # Learning parameters
    learning_rate: float = 0.0005
    gamma: float = 0.99
    
    # DQN specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    
    # PPO specific
    clip_param: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # A2C specific
    lstm_layers: int = 1
    
    # Output settings
    output_dir: str = "models"
    experiment_name: str = None
    
    def __post_init__(self):
        """Initialize derived settings and create directories."""
        # Set experiment name if not provided
        if self.experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.learning_agent_type}_vs_{self.opponent_type}_{timestamp}"
        
        # Create output directory
        self.output_path = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(self.output_path, exist_ok=True)
    
    def save(self) -> None:
        """Save the configuration to a JSON file."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        
        with open(os.path.join(self.output_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load a configuration from a JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)


class Trainer:
    """
    Trainer class for Liar's Dice agents.
    
    This class handles:
    - Setting up the environment and agents
    - Running training episodes
    - Evaluating agent performance
    - Saving models and metrics
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.output_path, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Save configuration
        self.config.save()
        self.logger.info(f"Training configuration saved to {config.output_path}")
        
        # Initialize environment
        self.game = LiarsDiceGame(
            num_players=config.num_players,
            num_dice=config.num_dice,
            dice_faces=config.dice_faces
        )
        
        # Create observation encoder
        self.obs_encoder = ObservationEncoder(
            num_players=config.num_players,
            num_dice=config.num_dice,
            dice_faces=config.dice_faces
        )
        
        # Estimate action dimension
        self.action_dim = self._estimate_action_dim()
        
        # Create agents
        self.learning_agent = self._create_learning_agent()
        self.opponent = self._create_opponent()
        
        # Metrics for tracking progress
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'losses': []
        }
    
    def _estimate_action_dim(self) -> int:
        """
        Estimate the action dimension based on the game rules.
        
        Returns:
            Estimated action dimension
        """
        # Max number of bids + challenge action
        max_bids = self.config.num_players * self.config.num_dice * self.config.dice_faces
        return max_bids + 1
    
    def _create_learning_agent(self) -> BaseAgent:
        """
        Create the learning agent based on the configuration.
        
        Returns:
            The learning agent
        """
        # Create a sample observation to get the actual dimension
        sample_obs = self.game.reset()[0]  # Get observation for player 0
        encoded_obs = self.obs_encoder.encode(sample_obs)
        actual_input_dim = encoded_obs.shape[0]
        
        if self.config.learning_agent_type == "dqn":
            return DQNAgent(
                player_id=0,
                observation_encoder=self.obs_encoder,
                action_dim=self.action_dim,
                input_dim=actual_input_dim,  # Use actual dimension instead of calculated
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay=self.config.epsilon_decay
            )
        elif self.config.learning_agent_type == "ppo":
            return PPOAgent(
                player_id=0,
                observation_encoder=self.obs_encoder,
                action_dim=self.action_dim,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                clip_param=self.config.clip_param,
                value_coef=self.config.value_coef,
                entropy_coef=self.config.entropy_coef
            )
        elif self.config.learning_agent_type == "a2c":
            return A2CAgent(
                player_id=0,
                observation_encoder=self.obs_encoder,
                action_dim=self.action_dim,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                value_coef=self.config.value_coef,
                entropy_coef=self.config.entropy_coef,
                lstm_layers=self.config.lstm_layers
            )
        else:
            raise ValueError(f"Unknown agent type: {self.config.learning_agent_type}")
    
    def _create_opponent(self) -> BaseAgent:
        """
        Create the opponent agent based on the configuration.
        
        Returns:
            The opponent agent
        """
        if self.config.opponent_type == "random":
            return RandomAgent(player_id=1)
        elif self.config.opponent_type == "conservative":
            return ConservativeAgent(player_id=1)
        elif self.config.opponent_type == "mildconservative":
            return MildConservativeAgent(player_id=1, challenge_threshold=2.5)
        elif self.config.opponent_type == "verymildconservative":
            return VeryMildConservativeAgent(player_id=1)
        elif self.config.opponent_type == "mixed":
            return MixedAgent(player_id=1)
        elif self.config.opponent_type == "strategic":
            return StrategicAgent(player_id=1)
        elif self.config.opponent_type == "self":
            # For self-play, create another instance of the same agent
            if self.config.learning_agent_type == "dqn":
                return DQNAgent(
                    player_id=1,
                    observation_encoder=self.obs_encoder,
                    action_dim=self.action_dim,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    epsilon_start=self.config.epsilon_start,
                    epsilon_end=self.config.epsilon_end,
                    epsilon_decay=self.config.epsilon_decay
                )
            elif self.config.learning_agent_type == "ppo":
                return PPOAgent(
                    player_id=1,
                    observation_encoder=self.obs_encoder,
                    action_dim=self.action_dim,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    clip_param=self.config.clip_param,
                    value_coef=self.config.value_coef,
                    entropy_coef=self.config.entropy_coef
                )
            elif self.config.learning_agent_type == "a2c":
                return A2CAgent(
                    player_id=1,
                    observation_encoder=self.obs_encoder,
                    action_dim=self.action_dim,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    value_coef=self.config.value_coef,
                    entropy_coef=self.config.entropy_coef,
                    lstm_layers=self.config.lstm_layers
                )
        else:
            raise ValueError(f"Unknown opponent type: {self.config.opponent_type}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the agent for the specified number of episodes.
        
        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Starting training...")
        
        # Track starting time
        start_time = time.time()
        
        # Track best model
        best_win_rate = 0.0
        
        for episode in tqdm(range(self.config.num_episodes)):
            # Reset environment and agents
            observations = self.game.reset(seed=episode)
            
            if hasattr(self.learning_agent, 'reset'):
                self.learning_agent.reset()
            if hasattr(self.opponent, 'reset'):
                self.opponent.reset()
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Play one episode
            while not done:
                # Get current player
                current_player = self.game.current_player
                
                # Get valid actions
                valid_actions = self.game.get_valid_actions(current_player)
                
                # Select action based on current player
                if current_player == self.learning_agent.player_id:
                    action = self.learning_agent.act(observations[current_player], valid_actions)
                    agent = self.learning_agent
                else:
                    action = self.opponent.act(observations[current_player], valid_actions)
                    agent = self.opponent
                
                # Take action
                next_observations, rewards, done, info = self.game.step(action)
                
                # Update agent with the experience
                agent.update(
                    observations[current_player],
                    action,
                    rewards[current_player],
                    next_observations[current_player],
                    done,
                    valid_actions
                )
                
                # Track reward for the learning agent
                if current_player == self.learning_agent.player_id:
                    episode_reward += rewards[current_player]
                
                # Update observations
                observations = next_observations
                
                # Train agent
                loss = agent.train()
                if loss is not None and current_player == self.learning_agent.player_id:
                    self.metrics['losses'].append(loss)
                
                episode_length += 1
            
            # Track episode metrics
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            
            # Evaluate periodically
            if (episode + 1) % self.config.eval_interval == 0:
                win_rate = self._evaluate()
                self.metrics['win_rates'].append(win_rate)
                
                avg_reward = np.mean(self.metrics['episode_rewards'][-self.config.eval_interval:])
                avg_length = np.mean(self.metrics['episode_lengths'][-self.config.eval_interval:])
                
                self.logger.info(
                    f"Episode {episode+1}/{self.config.num_episodes}, "
                    f"Win Rate: {win_rate:.2f}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Avg Length: {avg_length:.2f}"
                )
                
                # Save best model when we get a new highest win rate
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self._save_model("best")
                    self.logger.info(f"New best model saved with win rate: {best_win_rate:.2f}")
                
                # Plot training curves
                plot_training_curves(
                    self.metrics,
                    os.path.join(self.config.output_path, 'training_curves.png')
                )
                
                # Save metrics
                save_metrics(
                    self.metrics,
                    os.path.join(self.config.output_path, 'metrics.json')
                )
            
            # Save model periodically
            if (episode + 1) % self.config.save_interval == 0:
                self._save_model(episode + 1)
        
        # Calculate training time
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_win_rate = self._evaluate(num_games=self.config.num_eval_games * 2)
        self.logger.info(f"Final Win Rate: {final_win_rate:.2f}")
        
        # Save final model
        self._save_model("final")
        
        # Check if final model is better than best model
        if final_win_rate > best_win_rate:
            best_win_rate = final_win_rate
            self._save_model("best")
            self.logger.info(f"Final model is best model with win rate: {best_win_rate:.2f}")
        
        self.logger.info(f"Best win rate achieved: {best_win_rate:.2f}")
        
        return self.metrics
    
    def _evaluate(self, num_games: Optional[int] = None) -> float:
        """
        Evaluate the agent against the opponent.
        
        Args:
            num_games: Number of games to play (default: config.num_eval_games)
            
        Returns:
            Win rate of the agent
        """
        if num_games is None:
            num_games = self.config.num_eval_games
        
        win_count = 0
        
        # Temporarily set exploration to minimum (for DQN)
        original_epsilon = None
        if hasattr(self.learning_agent, 'epsilon'):
            original_epsilon = self.learning_agent.epsilon
            self.learning_agent.epsilon = self.config.epsilon_end
        
        # Play evaluation games
        for game_idx in range(num_games):
            # Reset the game
            observations = self.game.reset(seed=10000 + game_idx)
            
            if hasattr(self.learning_agent, 'reset'):
                self.learning_agent.reset()
            if hasattr(self.opponent, 'reset'):
                self.opponent.reset()
            
            done = False
            
            # Play until game is done
            while not done:
                current_player = self.game.current_player
                valid_actions = self.game.get_valid_actions(current_player)
                
                if current_player == self.learning_agent.player_id:
                    with torch.no_grad():  # Disable gradient computation for evaluation
                        action = self.learning_agent.act(observations[current_player], valid_actions)
                else:
                    action = self.opponent.act(observations[current_player], valid_actions)
                
                observations, rewards, done, info = self.game.step(action)
            
            # Check winner
            dice_counts = self.game.dice_counts
            winner = np.argmax(dice_counts)
            
            if winner == self.learning_agent.player_id:
                win_count += 1
        
        # Restore exploration rate
        if original_epsilon is not None:
            self.learning_agent.epsilon = original_epsilon
        
        return win_count / num_games
    
    def _save_model(self, identifier: Any) -> None:
        """
        Save the learning agent's model.
        
        Args:
            identifier: Episode number or identifier for the filename
        """
        save_path = os.path.join(
            self.config.output_path,
            f"{self.config.learning_agent_type}_agent_{identifier}.pt"
        )
        
        self.learning_agent.save(save_path)
        self.logger.info(f"Model saved to {save_path}")
        
        # For self-play, also save the opponent if it's a learning agent
        if self.config.opponent_type == "self":
            opponent_path = os.path.join(
                self.config.output_path,
                f"{self.config.learning_agent_type}_opponent_{identifier}.pt"
            )
            
            self.opponent.save(opponent_path)
            self.logger.info(f"Opponent model saved to {opponent_path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a saved model for the learning agent.
        
        Args:
            path: Path to the saved model
        """
        self.learning_agent.load(path)
        self.logger.info(f"Model loaded from {path}")


def train_agent(config: Optional[TrainingConfig] = None) -> Trainer:
    """
    Train an agent with the specified configuration.
    
    Args:
        config: Training configuration (default: use default configuration)
        
    Returns:
        Trained Trainer object
    """
    if config is None:
        config = TrainingConfig()
    
    trainer = Trainer(config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Create configuration
    config = TrainingConfig(
        num_episodes=5000,
        eval_interval=100,
        save_interval=500,
        epsilon_decay=0.999,
        learning_rate=0.001,
        num_dice=2,
        learning_agent_type="dqn",
        opponent_type="random"
    )
    
    # Train agent
    trainer = train_agent(config)