"""
Script to test a trained agent by playing against it interactively.

This script loads a trained agent and allows the user to play against it
in a terminal-based interface.
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.game import LiarsDiceGame
from environment.state import ObservationEncoder
from environment.play_game import Colors, print_banner, print_dice, clear_screen
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent


def estimate_action_dim(game):
    """Estimate action dimension based on game rules."""
    max_bids = game.num_players * game.num_dice * game.dice_faces
    return max_bids + 1


def load_agent(model_path, agent_type="dqn", player_id=1):
    """Load a trained agent from disk."""
    # Create game instance for encoder initialization
    game = LiarsDiceGame(num_players=2, num_dice=3)
    
    # Create observation encoder
    obs_encoder = ObservationEncoder(
        num_players=game.num_players,
        num_dice=game.num_dice,
        dice_faces=game.dice_faces
    )
    
    # Estimate action dimension
    action_dim = estimate_action_dim(game)
    
    # Create the appropriate agent type
    if agent_type == "dqn":
        agent = DQNAgent(
            player_id=player_id,
            observation_encoder=obs_encoder,
            action_dim=action_dim
        )
    elif agent_type == "ppo":
        agent = PPOAgent(
            player_id=player_id,
            observation_encoder=obs_encoder,
            action_dim=action_dim
        )
    elif agent_type == "a2c":
        agent = A2CAgent(
            player_id=player_id,
            observation_encoder=obs_encoder,
            action_dim=action_dim
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Load model weights
    agent.load(model_path)
    
    # Set to evaluation mode
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.05  # Minimal exploration for DQN
    
    return agent, game


def print_game_state(game, human_player_id):
    """Print the current game state for the human player."""
    clear_screen()
    print_banner()
    
    # Print round and current player info
    current_player = game.current_player
    print(f"\n{Colors.BOLD}Round: {game.round_num} | Current Player: {current_player}{Colors.RESET}")
    
    # Indicate whose turn it is
    if current_player == human_player_id:
        turn_indicator = f"{Colors.GREEN}Your turn!{Colors.RESET}"
    else:
        turn_indicator = f"{Colors.YELLOW}AI is thinking...{Colors.RESET}"
    print(turn_indicator)
    
    # Print dice counts for all players
    print(f"\n{Colors.BOLD}Dice Counts:{Colors.RESET}")
    for player in range(game.num_players):
        if player == current_player:
            player_indicator = f"{Colors.GREEN}▶{Colors.RESET}"
        else:
            player_indicator = " "
        
        if player == human_player_id:
            player_name = "You"
        else:
            player_name = "AI"
        
        print(f"{player_indicator} {player_name}: {game.dice_counts[player]} dice")
    
    # Print current bid
    print(f"\n{Colors.BOLD}Current Bid:{Colors.RESET}")
    if game.current_bid is None:
        print(f"{Colors.GRAY}No current bid{Colors.RESET}")
    else:
        quantity, value = game.current_bid
        previous_player = "You" if game.previous_player == human_player_id else "AI"
        print(f"{Colors.YELLOW}▶ {quantity} {value}'s (made by {previous_player}) ◀{Colors.RESET}")
    
    # Print the human player's dice
    print(f"\n{Colors.BOLD}Your Dice:{Colors.RESET}")
    dice_values = [int(d) for d in game.dice[human_player_id, :game.dice_counts[human_player_id]]]
    print_dice(dice_values)


def get_human_action(valid_actions):
    """Get an action from the human player."""
    if not valid_actions:
        return None
    
    # Group actions by type
    bid_actions = [a for a in valid_actions if a['type'] == 'bid']
    challenge_actions = [a for a in valid_actions if a['type'] == 'challenge']
    
    print(f"\n{Colors.BOLD}Choose your action:{Colors.RESET}")
    
    # Start with challenge option
    action_index = 0
    action_map = {}
    
    if challenge_actions:
        print(f"{Colors.RED}C. Challenge the previous bid{Colors.RESET}")
        action_map['c'] = challenge_actions[0]
    
    # Display bid options in columns
    if bid_actions:
        print(f"\n{Colors.BOLD}Bid Options:{Colors.RESET}")
        
        # Group bids by value
        values = sorted(set(a['value'] for a in bid_actions))
        value_groups = {}
        
        # First, collect all possible quantities for each value
        for value in values:
            value_groups[value] = []
            for action in bid_actions:
                if action['value'] == value:
                    value_groups[value].append((action['quantity'], action))
            
            # Sort by quantity
            value_groups[value].sort(key=lambda x: x[0])
        
        # Find the maximum number of quantities for any value
        max_rows = max(len(quantities) for quantities in value_groups.values())
        
        # Print column headers
        header = ""
        column_width = 20  # Width for each column
        for value in values:
            header_text = f"{Colors.CYAN}Value: {value}{Colors.RESET}"
            # Center the header text in the column
            padding = (column_width - len(header_text) + len(Colors.CYAN) + len(Colors.RESET)) // 2
            header += " " * padding + header_text + " " * (column_width - len(header_text) + len(Colors.CYAN) + len(Colors.RESET) - padding)
        print(header)
        
        # Print rows
        for row in range(max_rows):
            row_text = ""
            for value in values:
                if row < len(value_groups[value]):
                    quantity, action = value_groups[value][row]
                    action_index += 1
                    key = str(action_index)
                    action_map[key] = action
                    option_text = f"{key}. Bid {quantity} {value}'s"
                    # Center the option text in the column
                    padding = (column_width - len(option_text)) // 2
                    row_text += " " * padding + option_text + " " * (column_width - len(option_text) - padding)
                else:
                    row_text += " ".ljust(column_width)
            print(row_text)
    
    # Get user choice
    while True:
        choice = input(f"\n{Colors.BOLD}Enter your choice: {Colors.RESET}").lower()
        if choice in action_map:
            return action_map[choice]
        else:
            print(f"{Colors.RED}Invalid choice. Try again.{Colors.RESET}")


def play_against_ai(model_path, agent_type="dqn", human_player_id=0):
    """Play an interactive game against a trained agent."""
    # Load the trained agent
    agent, game = load_agent(model_path, agent_type, player_id=1 - human_player_id)
    
    # Reset the game
    observations = game.reset()
    done = False
    
    # Main game loop
    while not done:
        # Get current player
        current_player = game.current_player
        
        # Display game state
        print_game_state(game, human_player_id)
        
        # Get valid actions
        valid_actions = game.get_valid_actions(current_player)
        
        # Get action based on current player
        if current_player == human_player_id:
            # Human player's turn
            action = get_human_action(valid_actions)
        else:
            # AI's turn
            with torch.no_grad():
                action = agent.act(observations[current_player], valid_actions)
            
            # Display AI action
            if action['type'] == 'bid':
                print(f"\nAI bids {action['quantity']} {action['value']}'s")
            else:
                print(f"\nAI challenges your bid!")
            time.sleep(1)  # Pause to show AI action
        
        # Take action
        next_observations, rewards, done, info = game.step(action)
        
        # Show challenge results
        if action['type'] == 'challenge':
            clear_screen()
            print_banner()
            
            # Show all dice
            print(f"\n{Colors.BOLD}Challenge! All dice revealed:{Colors.RESET}")
            for p in range(game.num_players):
                player_name = "You" if p == human_player_id else "AI"
                dice_values = [int(d) for d in game.dice[p, :game.dice_counts[p]]]
                print(f"\n{player_name}'s dice:")
                print_dice(dice_values)
            
            # Get challenge results
            bid_quantity, bid_value = (0, 0) if game.current_bid is None else game.current_bid
            total_matching = 0
            for p in range(game.num_players):
                player_dice = game.dice[p, :game.dice_counts[p]]
                matching_dice = np.sum(player_dice == bid_value)
                total_matching += matching_dice
            
            # Determine challenge outcome
            bid_successful = total_matching >= bid_quantity
            
            # Display result
            if (current_player == human_player_id and not bid_successful) or \
               (current_player != human_player_id and bid_successful):
                # AI loses a die
                loser = "AI"
                result = f"{Colors.GREEN}Challenge successful!{Colors.RESET}"
            else:
                # Human loses a die
                loser = "You"
                result = f"{Colors.RED}Challenge failed!{Colors.RESET}"
            
            print(f"\n{result} {loser} lose a die.")
            print(f"Bid was: {bid_quantity} {bid_value}'s | Actual count: {total_matching} {bid_value}'s")
            input("Press Enter to continue...")
        
        # Update observations
        observations = next_observations
    
    # Game over, show final state
    clear_screen()
    print_banner()
    
    # Determine winner
    winner_id = np.argmax(game.dice_counts)
    winner = "You" if winner_id == human_player_id else "AI"
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Game Over! {winner} win!{Colors.RESET}")
    print(f"\nFinal dice counts:")
    for player in range(game.num_players):
        player_name = "You" if player == human_player_id else "AI"
        if player == winner_id:
            print(f"{Colors.GREEN}{player_name}: {game.dice_counts[player]} dice{Colors.RESET}")
        else:
            print(f"{player_name}: {game.dice_counts[player]} dice")
    
    # Ask to play again
    if input("\nPlay again? (y/n): ").lower() == 'y':
        play_against_ai(model_path, agent_type, human_player_id)


if __name__ == "__main__":
    print_banner()
    print(f"\n{Colors.BOLD}Test a trained Liar's Dice AI{Colors.RESET}")
    
    # Get model path from user
    model_dir = "models"
    if os.path.isdir(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if model_files:
            print("\nAvailable models:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            model_choice = int(input("\nChoose a model number: "))
            model_path = os.path.join(model_dir, model_files[model_choice - 1])
        else:
            model_path = input("\nEnter the path to the model file: ")
    else:
        model_path = input("\nEnter the path to the model file: ")
    
    # Determine agent type from filename
    if 'dqn' in model_path.lower():
        agent_type = 'dqn'
    elif 'ppo' in model_path.lower():
        agent_type = 'ppo'
    elif 'a2c' in model_path.lower():
        agent_type = 'a2c'
    else:
        agent_type = input("Enter agent type (dqn, ppo, a2c): ").lower()
    
    # Choose player order
    player_choice = input("\nDo you want to go first? (y/n): ").lower()
    human_player_id = 0 if player_choice == 'y' else 1
    
    # Start the game
    play_against_ai(model_path, agent_type, human_player_id)