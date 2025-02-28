# Print column headers
        header = ""
        column_width = 20  # Width for each column
        for value in values:
            header_text = f"{Colors.CYAN}Value: {value}{Colors.RESET}"
            # Center the header text in the column
            padding = (column_width - len(header_text) + len(Colors.CYAN) + len(Colors.RESET)) // 2
            header += " " * padding + header_text + " " * (column_width - len(header_text) + len(Colors.CYAN) + len(Colors.RESET) - padding)
        print(header)
        
        # Print r"""
Interactive terminal-based Liar's Dice game for testing.

This script allows playing Liar's Dice in the terminal to test game mechanics.
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Optional

# Add the parent directory to the path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.game import LiarsDiceGame, GameState


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


def print_banner():
    """Print game banner."""
    banner = f"""
    {Colors.BOLD}{Colors.YELLOW}╔═══════════════════════════════════════════════════════╗
    ║                   LIAR'S DICE                         ║
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


def print_game_state(game: LiarsDiceGame, current_player: int, reveal_all: bool = False):
    """
    Print the current game state.
    
    Args:
        game: The game object
        current_player: The current player's ID
        reveal_all: Whether to reveal all dice (for debugging)
    """
    clear_screen()
    print_banner()
    
    # Print round and current player info
    print(f"\n{Colors.BOLD}Round: {game.round_num} | Current Player: {current_player}{Colors.RESET}\n")
    
    # Print dice counts for all players
    print(f"{Colors.BOLD}Dice Counts:{Colors.RESET}")
    for player in range(game.num_players):
        if player == current_player:
            player_indicator = f"{Colors.GREEN}▶{Colors.RESET}"
        else:
            player_indicator = " "
        
        print(f"{player_indicator} Player {player}: {game.dice_counts[player]} dice")
    
    # Print current bid with enhanced visibility
    print(f"\n{Colors.BOLD}Current Bid:{Colors.RESET}")
    if game.current_bid is None:
        print(f"{Colors.GRAY}No current bid{Colors.RESET}")
    else:
        quantity, value = game.current_bid
        print(f"{Colors.BOLD}{Colors.YELLOW}▶ {quantity} {value}'s (made by Player {game.previous_player}) ◀{Colors.RESET}")
    
    # Print the current player's dice
    print(f"\n{Colors.BOLD}Your Dice (Player {current_player}):{Colors.RESET}")
    dice_values = [d for d in game.dice[current_player, :game.dice_counts[current_player]]]
    print_dice(dice_values)
    
    # If reveal_all is True, show all players' dice (for debugging)
    if reveal_all:
        print(f"\n{Colors.YELLOW}All Players' Dice (Debug Mode):{Colors.RESET}")
        for player in range(game.num_players):
            dice_values = [int(d) for d in game.dice[player, :game.dice_counts[player]]]
            dice_str = ", ".join(str(d) for d in dice_values)
            print(f"Player {player}: [{dice_str}]")
    
    # Count total number of each value (if reveal_all is True)
    if reveal_all and game.current_bid is not None:
        _, bid_value = game.current_bid
        total_count = 0
        for player in range(game.num_players):
            player_dice = game.dice[player, :game.dice_counts[player]]
            count = np.sum(player_dice == bid_value)
            total_count += count
        print(f"\nTotal {bid_value}'s in play: {Colors.CYAN}{total_count}{Colors.RESET}")


def get_valid_action_choice(valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask the user to choose an action from valid actions.
    
    Args:
        valid_actions: List of valid actions
        
    Returns:
        The chosen action
    """
    if not valid_actions:
        return None
    
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
    
    # Display bid options in columns - one column per die value
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
            header += f"{Colors.CYAN}Value: {value}{Colors.RESET}".ljust(column_width)
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
                    row_text += option_text.ljust(column_width)
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


def play_interactive_game():
    """Run an interactive game of Liar's Dice."""
    # Get game setup from user
    print_banner()
    print(f"\n{Colors.BOLD}Game Setup{Colors.RESET}")
    
    num_players = int(input("Number of players (2-6): ") or "4")
    num_dice = int(input("Number of dice per player (1-5): ") or "3")
    
    debug_mode = input("Enable debug mode to see all dice? (y/n): ").lower() == 'y'
    seed = int(input("Random seed (leave blank for random): ") or "0")
    if seed == 0:
        seed = None
    
    # Initialize game
    game = LiarsDiceGame(
        num_players=num_players,
        num_dice=num_dice,
        seed=seed
    )
    
    # Main game loop
    observations = game.reset()
    done = False
    
    while not done:
        # Get current player
        current_player = game.current_player
        
        # Display game state
        print_game_state(game, current_player, reveal_all=debug_mode)
        
        # Get valid actions
        valid_actions = game.get_valid_actions(current_player)
        
        # User selects action
        action = get_valid_action_choice(valid_actions)
        
        # Perform action and get next state
        next_observations, rewards, done, info = game.step(action)
        
        # Show outcome
        if action['type'] == 'challenge':
            print_game_state(game, current_player, reveal_all=True)
            
            if rewards[current_player] > 0:
                result = f"{Colors.GREEN}Challenge successful!{Colors.RESET}"
                loser = game.previous_player
            else:
                result = f"{Colors.RED}Challenge failed!{Colors.RESET}"
                loser = current_player
                
            print(f"\n{result} Player {loser} loses a die.")
            input("Press Enter to continue...")
        
        # Update observations
        observations = next_observations
    
    # Game over, show final state
    print_game_state(game, -1, reveal_all=True)
    
    # Determine winner
    winner = np.argmax(game.dice_counts)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Game Over! Player {winner} wins!{Colors.RESET}")
    print(f"\nFinal dice counts:")
    for player in range(game.num_players):
        if player == winner:
            print(f"{Colors.GREEN}Player {player}: {game.dice_counts[player]} dice{Colors.RESET}")
        else:
            print(f"Player {player}: {game.dice_counts[player]} dice")
    
    # Ask to play again
    if input("\nPlay again? (y/n): ").lower() == 'y':
        play_interactive_game()


if __name__ == "__main__":
    try:
        play_interactive_game()
    except KeyboardInterrupt:
        print("\nGame terminated by user.")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")