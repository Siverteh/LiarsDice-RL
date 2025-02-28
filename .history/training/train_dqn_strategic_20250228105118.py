"""
Simple script to train a DQN agent against a strategic rule-based opponent.

Run this script to start training a DQN agent for Liar's Dice.
"""

import os
import numpy as np
import random
import torch
from training import Trainer, TrainingConfig

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    # Create training configuration
    config = TrainingConfig(
        # Environment settings
        num_players=2,
        num_dice=5,  # Start with fewer dice for faster learning
        
        # Training settings
        num_episodes=5000,
        eval_interval=100,
        num_eval_games=50,
        save_interval=500,
        
        # Agent settings
        learning_agent_type="dqn",
        opponent_type="strategic",
        
        # Learning parameters
        learning_rate=0.0005,
        gamma=0.99,
        
        # DQN specific
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        
        # Output settings
        output_dir="models"
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    metrics = trainer.train()
    
    print("Training complete!")
    print(f"Model saved to {config.output_path}")