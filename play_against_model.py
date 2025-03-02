"""
Interactive gameplay against a trained RL agent.

This script allows you to play Liar's Dice against a trained DQN or PPO agent.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple

from environment.game import LiarsDiceGame, GameState
from environment.state import ObservationEncoder
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from training.utils import get_action_mapping


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner(agent_type="RL"):
    """Print game banner."""
    banner = f"""
    {Colors.BOLD}{Colors.YELLOW}╔═══════════════════════════════════════════════════════╗
    ║                   LIAR'S DICE                         ║
    ║           Playing against a trained {agent_type} agent         ║
    ╚═══════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(banner)


def print_dice(dice_values: List[int]):
    """Print a visual representation of dice."""
    dice_faces = {
        1: [" ┌─────┐ ", 
            " │     │ ", 
            " │  ●  │ ", 
            " │     │ ", 
            " └─────┘ "],
        2: [" ┌─────┐ ", 
            " │ ●   │ ", 
            " │     │ ", 
            " │   ● │ ", 
            " └─────┘ "],
        3: [" ┌─────┐ ", 
            " │ ●   │ ", 
            " │  ●  │ ", 
            " │   ● │ ", 
            " └─────┘ "],
        4: [" ┌─────┐ ", 
            " │ ● ● │ ", 
            " │     │ ", 
            " │ ● ● │ ", 
            " └─────┘ "],
        5: [" ┌─────┐ ", 
            " │ ● ● │ ", 
            " │  ●  │ ", 
            " │ ● ● │ ", 
            " └─────┘ "],
        6: [" ┌─────┐ ", 
            " │ ● ● │ ", 
            " │ ● ● │ ", 
            " │ ● ● │ ", 
            " └─────┘ "]
    }
    
    # Print dice side by side
    if not dice_values:
        print(f"{Colors.GRAY}No dice remaining{Colors.RESET}")
        return
        
    for i in range(5):  # 5 lines per die
        line = ""
        for die in dice_values:
            if die > 0:
                line += dice_faces[die][i]
        print(line)


def print_game_state(game: LiarsDiceGame, human_player: int, show_all: bool = False):
    """Print the game state for the human player."""
    clear_screen()
    print_banner()
    
    # Print round and current player info
    current_player = game.current_player
    current_player_label = "Your turn" if current_player == human_player else "AI's turn"
    print(f"\n{Colors.BOLD}Round: {game.round_num} | Current Player: {current_player_label}{Colors.RESET}\n")
    
    # Print dice counts for all players
    print(f"{Colors.BOLD}Dice Counts:{Colors.RESET}")
    for player in range(game.num_players):
        if player == current_player:
            player_indicator = f"{Colors.GREEN}▶{Colors.RESET}"
        else:
            player_indicator = " "
        
        player_label = "You" if player == human_player else "AI"
        print(f"{player_indicator} {player_label}: {game.dice_counts[player]} dice")
    
    # Print current bid with enhanced visibility
    print(f"\n{Colors.BOLD}Current Bid:{Colors.RESET}")
    if game.current_bid is None:
        print(f"{Colors.GRAY}No current bid{Colors.RESET}")
    else:
        quantity, value = game.current_bid
        previous_player_label = "You" if game.previous_player == human_player else "AI"
        print(f"{Colors.BOLD}{Colors.YELLOW}▶ {quantity} {value}'s (made by {previous_player_label}) ◀{Colors.RESET}")
    
    # Print the human player's dice
    print(f"\n{Colors.BOLD}Your Dice:{Colors.RESET}")
    dice_values = [int(d) for d in game.dice[human_player, :game.dice_counts[human_player]]]
    print_dice(dice_values)
    
    # Show AI's dice if show_all is True (for debugging)
    if show_all:
        ai_player = 1 - human_player  # Assuming 2-player game
        print(f"\n{Colors.YELLOW}AI's Dice (Debug Mode):{Colors.RESET}")
        ai_dice = [int(d) for d in game.dice[ai_player, :game.dice_counts[ai_player]]]
        print_dice(ai_dice)
        dice_str = ", ".join(str(d) for d in ai_dice)
        print(f"AI's dice: [{dice_str}]")
    
    # Count total number of each value if debug mode
    if show_all and game.current_bid is not None:
        _, bid_value = game.current_bid
        # Count of each dice value
        value_counts = {}
        for v in range(1, 7):
            value_counts[v] = 0
            for player in range(game.num_players):
                player_dice = game.dice[player, :game.dice_counts[player]]
                count = np.sum(player_dice == v)
                value_counts[v] += count
        
        # Print counts of all values
        counts_str = " | ".join([f"{v}'s: {count}" for v, count in value_counts.items()])
        print(f"\n{Colors.CYAN}Dice counts: {counts_str}{Colors.RESET}")


def get_human_action(game: LiarsDiceGame, human_player: int) -> Dict[str, Any]:
    """Get action from the human player."""
    valid_actions = game.get_valid_actions(human_player)
    
    # If there's only one valid action, return it automatically
    if len(valid_actions) == 1:
        print(f"\n{Colors.YELLOW}Only one valid action: {valid_actions[0]}{Colors.RESET}")
        input("Press Enter to continue...")
        return valid_actions[0]
    
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
    
    # Display bid options
    if bid_actions:
        print(f"\n{Colors.BOLD}Bid Options:{Colors.RESET}")
        
        # Group bids by value
        values = sorted(set(a['value'] for a in bid_actions))
        value_groups = {}
        
        # Collect all possible quantities for each value
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


def play_against_model(
    model_path: str,
    agent_type: str = None,
    num_dice: int = 2,
    dice_faces: int = 6,
    human_goes_first: bool = True,
    show_ai_dice: bool = False,
    device: str = 'auto'
) -> None:
    """
    Play Liar's Dice against a trained RL model.
    
    Args:
        model_path: Path to the trained model
        agent_type: Type of agent ('dqn' or 'ppo')
        num_dice: Number of dice per player
        dice_faces: Number of faces on each die
        human_goes_first: Whether the human player goes first
        show_ai_dice: Whether to show the AI's dice (debug mode)
        device: Device to run the model on ('cpu', 'cuda', or 'auto')
    """
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if metadata exists
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded model metadata: {metadata}")
        
        # Use metadata num_dice and dice_faces if available
        if 'num_dice' in metadata:
            num_dice = metadata['num_dice']
        if 'dice_faces' in metadata:
            dice_faces = metadata['dice_faces']
        
        # Get agent type from metadata if not specified
        if agent_type is None and 'agent_type' in metadata:
            agent_type = metadata['agent_type'].lower()
    
    # If agent_type is still None, try to guess from file presence
    if agent_type is None:
        if os.path.exists(os.path.join(model_path, 'q_network.pth')):
            agent_type = 'dqn'
        elif os.path.exists(os.path.join(model_path, 'actor_critic.pth')):
            agent_type = 'ppo'
        else:
            raise ValueError("Could not determine agent type. Please specify --agent_type")
    
    print(f"Using agent type: {agent_type}")
    
    # Determine player IDs
    human_player = 0 if human_goes_first else 1
    ai_player = 1 - human_player
    
    # Create game instance
    game = LiarsDiceGame(
        num_players=2,
        num_dice=num_dice,
        dice_faces=dice_faces
    )
    
    # Create observation encoder
    encoder = ObservationEncoder(
        num_players=2,
        num_dice=num_dice,
        dice_faces=dice_faces
    )
    
    # Generate action mapping
    action_mapping = get_action_mapping(2, num_dice, dice_faces)
    
    # Load the model
    try:
        # First, create a temporary agent to get dimensions
        test_obs = game.reset()
        encoded_test = encoder.encode(test_obs[0])
        obs_dim = encoded_test.shape[0]
        action_dim = len(action_mapping)
        
        # Get network size from metadata if available
        network_size = metadata.get('network_size', [1024, 512, 256, 128, 64]) if metadata else [1024, 512, 256, 128, 64]
        
        # Create the appropriate agent based on agent_type
        if agent_type.lower() == 'dqn':
            agent = DQNAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=network_size if isinstance(network_size, list) else 
                        [1024, 512, 256, 128, 64] if network_size == 'very large' else 
                        [512, 256, 128, 64] if network_size == 'large' else 
                        [256, 128, 64] if network_size == 'medium' else 
                        [128, 64],
                device=device
            )
            # For DQN, we want a deterministic policy during play
            try_set_epsilon = getattr(agent, 'epsilon', None)
            if try_set_epsilon is not None:
                agent.epsilon = 0.0
                
        elif agent_type.lower() == 'ppo':
            agent = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=network_size if isinstance(network_size, list) else 
                        [1024, 512, 256, 128, 64] if network_size == 'very large' else 
                        [512, 256, 128, 64] if network_size == 'large' else 
                        [256, 128, 64] if network_size == 'medium' else 
                        [128, 64],
                device=device
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        # Load the agent
        agent.load(model_path)
        agent.set_action_mapping(action_mapping)
        
        print(f"Successfully loaded {agent_type.upper()} model from {model_path}")
        print(f"Playing with {num_dice} dice per player, {dice_faces} faces per die")
        print(f"You are {'going first' if human_goes_first else 'going second'}")
        input("Press Enter to start the game...")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Main game loop
    observations = game.reset()
    done = False
    rounds_played = 0
    
    # Variable to store the game state before final reset
    pre_step_dice = None
    pre_step_dice_counts = None
    
    while not done:
        current_player = game.current_player
        
        # Print game state
        print_game_state(game, human_player, show_all=show_ai_dice)
        
        if current_player == human_player:
            # Human turn
            action = get_human_action(game, human_player)
        else:
            # AI turn
            print(f"\n{Colors.CYAN}AI is thinking...{Colors.RESET}")
            time.sleep(1)  # Add a delay to make it seem like the AI is thinking
            
            # Encode observation
            obs = encoder.encode(observations[ai_player])
            valid_actions = game.get_valid_actions(ai_player)
            
            # Get action from the model
            action = agent.select_action(obs, valid_actions, training=False)
        
        # Execute action
        if action['type'] == 'challenge':
            # Store the current bid for challenge resolution display
            current_bid = game.current_bid
            challenger = current_player
            bidder = game.previous_player
            
            # Store the dice state before the step
            pre_step_dice = np.copy(game.dice)
            pre_step_dice_counts = np.copy(game.dice_counts)
            
            # Execute the action
            observations, rewards, done, info = game.step(action)
            
            # Print game state after challenge
            print_game_state(game, human_player, show_all=True)
            
            # Count the actual dice matching the bid from the pre-step state
            if current_bid is not None:
                bid_quantity, bid_value = current_bid
                total_matching = 0
                for p in range(game.num_players):
                    player_dice = pre_step_dice[p, :pre_step_dice_counts[p]]
                    matching_dice = np.sum(player_dice == bid_value)
                    total_matching += int(matching_dice)
                
                # Find the loser by checking whose dice count decreased
                loser = None
                for p in range(game.num_players):
                    if game.dice_counts[p] < pre_step_dice_counts[p]:
                        loser = p
                        break
                
                # If we couldn't determine the loser (shouldn't happen), use a fallback
                if loser is None:
                    print(f"{Colors.RED}Warning: Could not determine the loser. Using fallback logic.{Colors.RESET}")
                    loser = bidder  # Just a guess based on most common scenario
                
                # Set result message based on who actually lost in the game
                challenge_successful = (loser == bidder)
                
                if challenge_successful:
                    result = f"{Colors.GREEN}Challenge successful!{Colors.RESET}"
                    # The bid must have been incorrect according to the game's logic
                    bid_correct_text = f"The bid of {bid_quantity} {bid_value}'s was deemed incorrect by the game."
                else:
                    result = f"{Colors.RED}Challenge failed!{Colors.RESET}"
                    # The bid must have been correct according to the game's logic
                    bid_correct_text = f"The bid of {bid_quantity} {bid_value}'s was deemed correct by the game."
                    
                # Convert to player labels
                loser_label = "You" if loser == human_player else "AI"
                
                print(f"\n{result} {loser_label} loses a die.")
                print(f"Bid was: {bid_quantity} {bid_value}'s | Actual count: {total_matching} {bid_value}'s")
                
                # Explanation of game result
                if challenge_successful:
                    explanation = f"{bid_correct_text} The bidder loses a die."
                else:
                    explanation = f"{bid_correct_text} The challenger loses a die."
                
                print(f"{Colors.CYAN}{explanation}{Colors.RESET}")
                
                # Show dice when an extra layer of verification/interest
                print("\nDice before the challenge:")
                for p in range(game.num_players):
                    player_label = "You" if p == human_player else "AI"
                    dice_values = [int(d) for d in pre_step_dice[p, :pre_step_dice_counts[p]]]
                    print(f"{player_label}'s dice:")
                    print_dice(dice_values)
                
                input("\nPress Enter to continue...")
        else:
            # For normal bids, just execute the action
            observations, rewards, done, info = game.step(action)
            
            # If it's the AI's action, pause briefly
            if current_player == ai_player:
                print(f"\nAI bid: {action['quantity']} {action['value']}'s")
                time.sleep(1)
        
        rounds_played += 1
        
        # If the game is done, show the final state
        if done:
            # Store the final dice state
            pre_step_dice = np.copy(game.dice)
            pre_step_dice_counts = np.copy(game.dice_counts)
            
            # Print game state
            print_game_state(game, human_player, show_all=True)
            
            # Determine winner
            winner = np.argmax(game.dice_counts)
            winner_label = "You" if winner == human_player else "AI"
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}Game Over! {winner_label} win!{Colors.RESET}")
            print(f"\nFinal dice counts:")
            for player in range(game.num_players):
                player_label = "You" if player == human_player else "AI"
                if player == winner:
                    print(f"{Colors.GREEN}{player_label}: {game.dice_counts[player]} dice{Colors.RESET}")
                else:
                    print(f"{player_label}: {game.dice_counts[player]} dice")
            
            input("\nPress Enter to continue...")
    
    # Ask to play again
    if input("\nPlay again? (y/n): ").lower() == 'y':
        play_against_model(
            model_path=model_path,
            agent_type=agent_type,
            num_dice=num_dice,
            dice_faces=dice_faces,
            human_goes_first=not human_goes_first,  # Switch who goes first
            show_ai_dice=show_ai_dice,
            device=device
        )


def list_available_models(models_dir: str = 'results/models') -> List[str]:
    """List all available models in the models directory."""
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} not found.")
        return []
    
    models = []
    for item in os.listdir(models_dir):
        full_path = os.path.join(models_dir, item)
        if os.path.isdir(full_path):
            # Check if it's a supported model type (DQN or PPO)
            if (os.path.exists(os.path.join(full_path, 'q_network.pth')) or 
                os.path.exists(os.path.join(full_path, 'actor_critic.pth'))):
                models.append(item)
    
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Liar's Dice against a trained RL model")
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to the trained model. If not specified, will use the most recent model.')
    parser.add_argument('--agent_type', type=str, choices=['dqn', 'ppo'], default=None,
                        help='Type of RL agent. If not specified, will try to determine from model files.')
    parser.add_argument('--models_dir', type=str, default='results/models',
                        help='Directory containing trained models')
    parser.add_argument('--dice', type=int, default=3, help='Number of dice per player')
    parser.add_argument('--faces', type=int, default=6, help='Number of faces per die')
    parser.add_argument('--ai_first', action='store_true', help='Let the AI go first')
    parser.add_argument('--show_ai_dice', action='store_true', help='Show the AI\'s dice (debug mode)')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='Device to run the model on')
    parser.add_argument('--list_models', action='store_true', help='List all available models and exit')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = list_available_models(args.models_dir)
        if models:
            print(f"Available models ({len(models)}):")
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
        else:
            print("No models found.")
        sys.exit(0)
    
    # Determine which model to use
    model_path = args.model
    if model_path is None:
        # Find the most recent model
        models = list_available_models(args.models_dir)
        if not models:
            print("No models found. Please train a model first or specify a model path.")
            sys.exit(1)
        
        # Sort by timestamped filename (assuming format with timestamp)
        models.sort(reverse=True)
        model_path = os.path.join(args.models_dir, models[0])
        print(f"Using most recent model: {models[0]}")
    
    # Play the game
    play_against_model(
        model_path=model_path,
        agent_type=args.agent_type,
        num_dice=args.dice,
        dice_faces=args.faces,
        human_goes_first=not args.ai_first,
        show_ai_dice=args.show_ai_dice,
        device=args.device
    )