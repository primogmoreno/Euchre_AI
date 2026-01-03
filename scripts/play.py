#!/usr/bin/env python3
"""
Play Euchre against the AI in the terminal.

Usage:
    python scripts/play.py
    python scripts/play.py --model checkpoints/final_model.pth
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from game import EuchreGame, GamePhase
from agents import RuleBasedAgent, NeuralAgent
from agents.base import BaseAgent


class HumanAgent(BaseAgent):
    """Agent that prompts for human input."""
    
    def __init__(self):
        super().__init__("Human")
    
    def select_action(self, observation: dict, legal_actions: list):
        """Prompt human for action selection."""
        print("\n" + "=" * 40)
        print(f"Your hand: {observation['my_hand']}")
        
        # Show turned card during calling phase
        if observation["turned_card"]:
            dealer = observation["dealer"]
            dealer_label = "You" if dealer == 0 else f"Player {dealer}"
            print(f"Turned card: {observation['turned_card']}  (Dealer: {dealer_label})")
        
        if observation["trump"]:
            print(f"Trump: {observation['trump'].name}")
        
        if observation["current_trick"]:
            print(f"Current trick: {observation['current_trick']}")
        
        print(f"Tricks won - You: {observation['tricks_won'][0]}, Them: {observation['tricks_won'][1]}")
        print(f"Score - You: {observation['score'][0]}, Them: {observation['score'][1]}")
        
        print("\nLegal actions:")
        for i, action in enumerate(legal_actions):
            print(f"  {i}: {action}")
        
        while True:
            try:
                choice = input("\nYour choice (number): ").strip()
                idx = int(choice)
                if 0 <= idx < len(legal_actions):
                    return legal_actions[idx]
                print(f"Please enter a number between 0 and {len(legal_actions) - 1}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nGame cancelled")
                sys.exit(0)

def play_game(human: HumanAgent, ai: BaseAgent):
    """Play a single game."""
    game = EuchreGame()
    observations = game.reset()
    
    # Human is player 0 (team 0), AI plays for all other positions
    # But we'll make AI control player 2 (human's partner) as well
    
    print("\n" + "=" * 50)
    print("NEW GAME")
    print("You are Player 0, partnered with Player 2 (AI)")
    print("Opponents are Players 1 and 3 (AI)")
    print("=" * 50)
    
    done = False
    while not done:
        player = game.state.current_player
        obs = observations[player]
        legal_actions = game.get_legal_actions()
        phase = game.state.phase
        
        if player == 0:
            # Human's turn
            action = human.select_action(obs, legal_actions)
        else:
            # AI's turn
            action = ai.select_action(obs, legal_actions)
            
            # Format output based on phase
            if phase == GamePhase.DISCARD:
                print(f"\nPlayer {player} discards a card (hidden)")
            elif phase == GamePhase.PLAYING:
                print(f"\nPlayer {player} plays: {action}")
            elif phase in (GamePhase.CALLING_ROUND_1, GamePhase.CALLING_ROUND_2):
                print(f"\nPlayer {player} calls: {action}")
        
        result = game.step(action)
        observations = result.observations
        done = result.done
        
        # Show trick winner
        if result.info.get("trick_winner") is not None:
            winner = result.info["trick_winner"]
            team = "Your team" if winner % 2 == 0 else "Opponents"
            print(f"\n{team} wins the trick!")
    
    # Game over
    final_score = game.state.score
    print("\n" + "=" * 50)
    print("GAME OVER")
    print(f"Final score - You: {final_score[0]}, Opponents: {final_score[1]}")
    
    if final_score[0] >= 10:
        print("YOU WIN!")
    else:
        print("You lose. Better luck next time!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Play Euchre against AI")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (uses rule-based AI if not provided)",
    )
    
    args = parser.parse_args()
    
    human = HumanAgent()
    
    if args.model and os.path.exists(args.model):
        print(f"Loading AI model from {args.model}")
        ai = NeuralAgent(model_path=args.model, greedy=True, name="Neural AI")
    else:
        print("Using rule-based AI opponent")
        ai = RuleBasedAgent(name="Rule AI")
    
    while True:
        play_game(human, ai)
        
        try:
            again = input("\nPlay again? (y/n): ").strip().lower()
            if again != "y":
                break
        except KeyboardInterrupt:
            break
    
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()