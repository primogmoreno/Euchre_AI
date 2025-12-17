#!/usr/bin/env python3
"""
Evaluate agents against each other.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model checkpoints/final_model.pth --games 500
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from agents import RandomAgent, RuleBasedAgent, NeuralAgent
from evaluation import Arena


def main():
    parser = argparse.ArgumentParser(description="Evaluate Euchre agents")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (optional)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of games per matchup (default: 200)",
    )
    
    args = parser.parse_args()
    
    arena = Arena()
    
    # Create agents
    random_agent = RandomAgent()
    rule_based = RuleBasedAgent()
    neural_agent = NeuralAgent()
    
    print("=" * 50)
    print("EUCHRE AI EVALUATION")
    print("=" * 50)
    
    # Random vs Rule-based (baseline comparison)
    print("\n--- Baseline: Random vs Rule-Based ---")
    result = arena.compare_agents(random_agent, rule_based, args.games)
    print(f"{result['agent_a']}: {result['agent_a_win_rate']:.1%}")
    print(f"{result['agent_b']}: {result['agent_b_win_rate']:.1%}")
    
    # If model provided, test neural agent
    if args.model and os.path.exists(args.model):
        print(f"\n--- Loading model: {args.model} ---")
        neural_agent = NeuralAgent(model_path=args.model, greedy=True)
        
        # Neural vs Random
        print("\n--- Neural vs Random ---")
        result = arena.compare_agents(neural_agent, random_agent, args.games)
        print(f"{result['agent_a']}: {result['agent_a_win_rate']:.1%}")
        print(f"{result['agent_b']}: {result['agent_b_win_rate']:.1%}")
        
        # Neural vs Rule-based
        print("\n--- Neural vs Rule-Based ---")
        result = arena.compare_agents(neural_agent, rule_based, args.games)
        print(f"{result['agent_a']}: {result['agent_a_win_rate']:.1%}")
        print(f"{result['agent_b']}: {result['agent_b_win_rate']:.1%}")
    
    elif args.model:
        print(f"\nWarning: Model not found at {args.model}")
    
    print("\n" + "=" * 50)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()