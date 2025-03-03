# LiarsDice-RL

A reinforcement learning framework for training AI agents to play the game of Liar's Dice.

## Overview

This repository implements various reinforcement learning approaches to master Liar's Dice, a classic dice game of bluffing and deception. The project includes DQN and PPO agent implementations, along with rule-based agents of varying skill levels for training and evaluation.

## Features

- **Multiple Agent Types**: 
  - DQN agent with prioritized experience replay, double DQN, and dueling architecture
  - PPO agent with adaptive entropy, GAE, and action masking
  - Various rule-based agents from random to near-optimal strategies

- **Comprehensive Training Approaches**:
  - Curriculum learning against progressively harder opponents
  - Self-play learning where agents train against themselves
  - Mixed training combining curriculum and self-play methods

- **Flexible Game Environment**:
  - Configurable number of players, dice, and faces
  - Reward shaping for better learning signals
  - Observation encoding optimized for neural networks

- **Evaluation Tools**:
  - Robust evaluation against rule-based agents
  - Performance visualization and tracking
  - Interactive gameplay to test trained agents

## Project Structure

The repository is organized into three main components:

- `agents/`: Implementations of various agent types (DQN, PPO, rule-based)
- `environment/`: Game logic, state encoding, and reward calculation
- `training/`: Training loops, evaluation functions, and utilities

Standalone scripts (`curriculum_learning.py`, `self_play_learning.py`, etc.) provide easy access to different training methodologies.

## Getting Started

Train an agent using curriculum learning:
```bash
python curriculum_learning.py --agent dqn --preset standard
