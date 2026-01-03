"""
Main training loop for Euchre AI.
"""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from model.network import EuchreNetwork
from training.ppo import PPOLoss
from training.self_play import SelfPlayRunner, experiences_to_tensors, compute_gae_by_player


class Trainer:
    """
    Main trainer for Euchre AI using PPO and self-play.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = config.CHECKPOINT_DIR,
        log_dir: str = config.LOG_DIR,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize network and optimizer
        self.network = EuchreNetwork()
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.LEARNING_RATE,
        )
        
        # PPO loss calculator
        self.ppo = PPOLoss()
        
        # Self-play runner
        self.runner = SelfPlayRunner(self.network)
        
        # TensorBoard logging
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.episode = 0
        self.total_steps = 0
    
    def train(self, num_episodes: int = config.NUM_EPISODES):
        """
        Main training loop.
        """
        print(f"Starting training for {num_episodes} episodes")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.log_dir}")
        
        # Try to resume from checkpoint
        self._load_checkpoint()
        
        pbar = tqdm(
            initial=self.episode,
            total=num_episodes,
            desc="Training",
        )
        
        while self.episode < num_episodes:
            # Collect experiences through self-play
            experiences, episode_infos = self.runner.run_episodes(
                num_episodes=config.BATCH_SIZE // 20  # ~20 steps per game
            )
            
            if not experiences:
                continue
            
            # Convert to tensors
            batch = experiences_to_tensors(experiences)
            
            # Compute advantages and returns PER PLAYER (critical for multi-agent)
            advantages, returns = compute_gae_by_player(experiences)
            
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # PPO update epochs
            for _ in range(config.NUM_EPOCHS):
                loss, loss_info = self.ppo.compute(
                    self.network,
                    batch["states"],
                    batch["actions"],
                    batch["log_probs"],
                    advantages,
                    returns,
                    batch["action_masks"],
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    config.MAX_GRAD_NORM,
                )
                self.optimizer.step()
            
            # Update counters
            self.episode += len(episode_infos)
            self.total_steps += len(experiences)
            
            # Logging
            self._log_metrics(loss_info, episode_infos)
            
            # Checkpointing
            if self.episode % config.CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint()
            
            pbar.update(len(episode_infos))
        
        pbar.close()
        
        # Save final model
        self._save_checkpoint(final=True)
        print("Training complete!")
    
    def _log_metrics(self, loss_info: dict, episode_infos: list):
        """Log metrics to TensorBoard."""
        # Loss components
        for key, value in loss_info.items():
            self.writer.add_scalar(f"loss/{key}", value, self.episode)
        
        # Episode stats
        if episode_infos:
            win_rate = sum(1 for e in episode_infos if e["winner"] == 0) / len(episode_infos)
            avg_steps = sum(e["total_steps"] for e in episode_infos) / len(episode_infos)
            
            self.writer.add_scalar("episode/win_rate_team0", win_rate, self.episode)
            self.writer.add_scalar("episode/avg_steps", avg_steps, self.episode)
            self.writer.add_scalar("episode/total_steps", self.total_steps, self.episode)
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": self.episode,
            "total_steps": self.total_steps,
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, path)
        
        # Save versioned
        if final:
            path = os.path.join(self.checkpoint_dir, "final_model.pth")
        else:
            path = os.path.join(self.checkpoint_dir, f"model_ep{self.episode}.pth")
        torch.save(self.network.state_dict(), path)
        
        tqdm.write(f"Saved checkpoint at episode {self.episode}")
    
    def _load_checkpoint(self):
        """Load checkpoint if it exists."""
        path = os.path.join(self.checkpoint_dir, "latest.pth")
        
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.episode = checkpoint["episode"]
            self.total_steps = checkpoint["total_steps"]
            print(f"Resumed from episode {self.episode}")