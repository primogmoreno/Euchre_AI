#!/usr/bin/env python3
"""
Train the Euchre AI model.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 50000
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from training import Trainer
import config


def main():
    parser = argparse.ArgumentParser(description="Train Euchre AI")
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.NUM_EPISODES,
        help=f"Number of episodes to train (default: {config.NUM_EPISODES})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=config.CHECKPOINT_DIR,
        help=f"Directory for checkpoints (default: {config.CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=config.LOG_DIR,
        help=f"Directory for TensorBoard logs (default: {config.LOG_DIR})",
    )
    
    args = parser.parse_args()
    
    trainer = Trainer(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    trainer.train(num_episodes=args.episodes)


if __name__ == "__main__":
    main()