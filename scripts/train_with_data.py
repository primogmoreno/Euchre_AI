#!/usr/bin/env python3
"""
Train the Euchre AI model WITH data collection enabled.

This variant logs decision data for later analysis.

Usage:
    python scripts/train_with_data.py
    python scripts/train_with_data.py --episodes 50000 --collect-plays
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from model.network import EuchreNetwork
from training.ppo import PPOLoss, compute_gae
from training.self_play import experiences_to_tensors
from data_collection.instrumented_runner import InstrumentedSelfPlayRunner


def train_with_collection(
    num_episodes: int = config.NUM_EPISODES,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    log_dir: str = config.LOG_DIR,
    data_dir: str = "data",
    collect_plays: bool = False,
):
    """
    Train with data collection enabled.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize network and optimizer
    network = EuchreNetwork()
    optimizer = optim.Adam(network.parameters(), lr=config.LEARNING_RATE)
    
    # PPO loss calculator
    ppo = PPOLoss()
    
    # Self-play runner WITH data collection
    runner = InstrumentedSelfPlayRunner(
        network, 
        data_dir=data_dir,
        collect_plays=collect_plays,
    )
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir)
    
    # Training state
    episode = 0
    total_steps = 0
    
    # Try to resume from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        episode = checkpoint["episode"]
        total_steps = checkpoint["total_steps"]
        print(f"Resumed from episode {episode}")
    
    print(f"Starting training for {num_episodes} episodes")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print(f"Data collection: {data_dir}")
    print(f"Collect plays: {collect_plays}")
    
    pbar = tqdm(initial=episode, total=num_episodes, desc="Training")
    
    try:
        while episode < num_episodes:
            # Collect experiences through self-play
            experiences, episode_infos = runner.run_episodes(
                num_episodes=config.BATCH_SIZE // 20
            )
            
            if not experiences:
                continue
            
            # Convert to tensors
            batch = experiences_to_tensors(experiences)
            
            # Compute advantages and returns
            advantages, returns = compute_gae(
                batch["rewards"],
                batch["values"],
                batch["dones"],
            )
            
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # PPO update epochs
            for _ in range(config.NUM_EPOCHS):
                loss, loss_info = ppo.compute(
                    network,
                    batch["states"],
                    batch["actions"],
                    batch["log_probs"],
                    advantages,
                    returns,
                    batch["action_masks"],
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(),
                    config.MAX_GRAD_NORM,
                )
                optimizer.step()
            
            # Update counters
            episode += len(episode_infos)
            total_steps += len(experiences)
            
            # Logging
            for key, value in loss_info.items():
                writer.add_scalar(f"loss/{key}", value, episode)
            
            if episode_infos:
                win_rate = sum(1 for e in episode_infos if e["winner"] == 0) / len(episode_infos)
                writer.add_scalar("episode/win_rate_team0", win_rate, episode)
            
            # Log data collection stats periodically
            if episode % 1000 == 0:
                stats = runner.get_collection_stats()
                for category, count in stats.items():
                    writer.add_scalar(f"data/{category}", count, episode)
            
            # Checkpointing
            if episode % config.CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": episode,
                    "total_steps": total_steps,
                }
                torch.save(checkpoint, checkpoint_path)
                torch.save(
                    network.state_dict(),
                    os.path.join(checkpoint_dir, f"model_ep{episode}.pth")
                )
                tqdm.write(f"Saved checkpoint at episode {episode}")
                
                # Print data collection stats
                stats = runner.get_collection_stats()
                tqdm.write(f"Data collected: {stats}")
            
            pbar.update(len(episode_infos))
    
    finally:
        # Always close the data logger
        runner.close()
        pbar.close()
        
        # Save final model
        torch.save(
            network.state_dict(),
            os.path.join(checkpoint_dir, "final_model.pth")
        )
        
        print("\nTraining complete!")
        print(f"Final data collection stats: {runner.get_collection_stats()}")


def main():
    parser = argparse.ArgumentParser(description="Train Euchre AI with data collection")
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
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for collected data (default: data)",
    )
    parser.add_argument(
        "--collect-plays",
        action="store_true",
        help="Also collect card play decisions (generates more data)",
    )
    
    args = parser.parse_args()
    
    train_with_collection(
        num_episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        data_dir=args.data_dir,
        collect_plays=args.collect_plays,
    )


if __name__ == "__main__":
    main()