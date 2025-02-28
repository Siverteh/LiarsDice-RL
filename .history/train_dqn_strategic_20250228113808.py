"""
Simple script to train a DQN agent against a strategic rule-based opponent.

Run this script to start training a DQN agent for Liar's Dice.
"""

import os
import numpy as np
import random
import torch
from training.trainer import Trainer, TrainingConfig

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
        num_dice=2, 
        
        # Training settings
        num_episodes=5000,
        eval_interval=100,
        num_eval_games=100,
        save_interval=500,
        
        # Agent settings
        learning_agent_type="dqn",
        opponent_type="random",
        
        # Learning parameters
        learning_rate=0.0001,
        gamma=0.99,
        
        # DQN specific
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        
        # Output settings
        output_dir="models",
        experiment_name="dqn_vs_random_2dice"  # Custom name here!
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    #trainer.load_model("models/dqn_vs_random_20250228_110526/dqn_agent_1500.pt")
    metrics = trainer.train()
    
    print("Training complete!")
    print(f"Model saved to {config.output_path}")