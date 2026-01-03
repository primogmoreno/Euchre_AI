#!/usr/bin/env python3
"""
Train the Euchre AI model against a mix of opponents WITH data collection.

This version alternates between:
- Self-play (neural vs neural)
- Playing against rule-based opponents

This prevents the model from learning degenerate strategies that only
work against itself.

Usage:
    python scripts/train_mixed.py
    python scripts/train_mixed.py --episodes 100000
    python scripts/train_mixed.py --episodes 100000 --rule-based-ratio 0.7
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from model.network import EuchreNetwork
from model.encoding import encode_state, action_to_index, index_to_action, create_action_mask
from training.ppo import PPOLoss
from training.self_play import Experience, EpisodeBuffer, experiences_to_tensors, compute_gae_by_player
from game.engine import EuchreGame
from game.state import GamePhase
from game.cards import Card
from agents.rule_based import RuleBasedAgent
from data_collection.logger import DataLogger
from data_collection.collectors import (
    GoingAloneCollector,
    TrumpCallCollector,
    PassCollector,
    PlayCollector,
    card_to_str,
)


class MixedTrainingRunner:
    """
    Runs games with mixed opponents for more robust training.
    Also collects decision data for analysis.
    """
    
    def __init__(
        self, 
        network: EuchreNetwork, 
        rule_based_ratio: float = 0.5,
        data_dir: str = "data",
        collect_plays: bool = False,
    ):
        """
        Args:
            network: The neural network being trained
            rule_based_ratio: Fraction of games to play against rule-based (0.0 to 1.0)
            data_dir: Directory for collected data
            collect_plays: Whether to collect card play data
        """
        self.network = network
        self.game = EuchreGame()
        self.rule_based = RuleBasedAgent()
        self.rule_based_ratio = rule_based_ratio
        
        # Data collection
        self.logger = DataLogger(data_dir)
        self.going_alone_collector = GoingAloneCollector(self.logger)
        self.trump_call_collector = TrumpCallCollector(self.logger)
        self.pass_collector = PassCollector(self.logger)
        self.play_collector = PlayCollector(self.logger, only_interesting=True) if collect_plays else None
        
        self.episode_count = 0
    
    def run_episode(self) -> tuple[list[Experience], dict]:
        """
        Run a single game, collecting experiences for the neural network players.
        """
        buffer = EpisodeBuffer()
        observations = self.game.reset()
        
        self.episode_count += 1
        episode_id = self.episode_count
        
        # Decide if this game is against rule-based or self-play
        use_rule_based = random.random() < self.rule_based_ratio
        
        # Neural network is always team 0 (players 0 and 2)
        # Rule-based is team 1 (players 1 and 3) when active
        
        done = False
        total_steps = 0
        
        # Track decisions for data collection
        alone_this_round = False
        trump_called_this_round = False
        round_number = 0
        last_trick_count = 0
        
        while not done:
            player = self.game.state.current_player
            obs = observations[player]
            legal_actions = self.game.get_legal_actions()
            
            # Decide which agent plays this turn
            is_opponent = (player % 2 == 1)  # Players 1 and 3 are opponents
            
            if use_rule_based and is_opponent:
                # Rule-based opponent
                action = self.rule_based.select_action(obs, legal_actions)
                action_idx = action_to_index(action)
                log_prob = 0.0
                value = 0.0
                state = None
                action_mask = None
            else:
                # Neural network
                state = encode_state(obs)
                action_mask = create_action_mask(legal_actions)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
                
                with torch.no_grad():
                    action_idx, log_prob, _, value = self.network.get_action_and_value(
                        state_tensor, mask_tensor
                    )
                
                action_idx = action_idx.item()
                log_prob = log_prob.item()
                value = value.item()
                
                action = index_to_action(action_idx, legal_actions)
            
            # =================================================================
            # DATA COLLECTION - Log decisions before taking the action
            # =================================================================
            
            self._collect_decision(
                action=action,
                player=player,
                episode_id=episode_id,
                round_number=round_number,
            )
            
            # Update tracking flags
            if isinstance(action, str):
                if "alone" in action:
                    alone_this_round = True
                elif action.startswith("order_up") or action.startswith("call_"):
                    trump_called_this_round = True
            
            # Track kitty size before step
            kitty_size_before = len(self.game.state.kitty)
            tricks_before = sum(self.game.state.tricks_won)
            
            # Take step
            result = self.game.step(action)
            observations = result.observations
            done = result.done
            
            # =================================================================
            # Track buried card
            # =================================================================
            
            kitty_size_after = len(self.game.state.kitty)
            if kitty_size_before == 3 and kitty_size_after == 4:
                fourth_kitty_card = self.game.state.kitty[-1]
                buried_card_str = card_to_str(fourth_kitty_card)
                round_key = f"{episode_id}_{round_number}"
                
                if round_key in self.going_alone_collector.pending:
                    self.going_alone_collector.pending[round_key]["buried_card"] = buried_card_str
                if round_key in self.trump_call_collector.pending:
                    self.trump_call_collector.pending[round_key]["buried_card"] = buried_card_str
            
            # =================================================================
            # Track trick outcomes
            # =================================================================
            
            if result.info.get("trick_winner") is not None:
                trick_num = last_trick_count
                winner = result.info["trick_winner"]
                
                if self.play_collector:
                    self.play_collector.record_trick_outcome(
                        self.game.state, episode_id, trick_num, winner
                    )
                
                last_trick_count = sum(self.game.state.tricks_won)
            
            # =================================================================
            # Check if round ended
            # =================================================================
            
            round_just_ended = (
                tricks_before == 4 and 
                any(r != 0 for r in result.rewards)
            )
            
            if round_just_ended:
                round_key = f"{episode_id}_{round_number}"
                
                if alone_this_round:
                    self._record_going_alone_outcome(round_key)
                elif trump_called_this_round:
                    self._record_trump_call_outcome(round_key)
                
                alone_this_round = False
                trump_called_this_round = False
                round_number += 1
                last_trick_count = 0
            
            # Only store experiences for neural network players
            if not (use_rule_based and is_opponent):
                if state is None:
                    state = encode_state(obs)
                    action_mask = create_action_mask(legal_actions)
                
                exp = Experience(
                    state=state,
                    action=action_idx,
                    reward=result.rewards[player],
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    action_mask=action_mask,
                    player=player,
                )
                buffer.add(player, exp)
            
            total_steps += 1
        
        final_score = self.game.state.score
        winner = 0 if final_score[0] >= 10 else 1
        
        info = {
            "total_steps": total_steps,
            "final_score": final_score,
            "winner": winner,
            "vs_rule_based": use_rule_based,
            "neural_won": (winner == 0),
            "episode_id": episode_id,
        }
        
        return buffer.get_all(), info
    
    def _collect_decision(
        self,
        action,
        player: int,
        episode_id: int,
        round_number: int,
    ):
        """Route decision to appropriate collector."""
        state = self.game.state
        round_key = f"{episode_id}_{round_number}"
        
        if isinstance(action, str):
            if "alone" in action:
                self.going_alone_collector.record_decision(
                    state, player, action, round_key
                )
            elif action.startswith("order_up") or action.startswith("call_"):
                self.trump_call_collector.record_decision(
                    state, player, action, round_key
                )
            elif action == "pass":
                self.pass_collector.record_decision(
                    state, player, round_key
                )
        elif isinstance(action, Card) and state.phase == GamePhase.PLAYING:
            if self.play_collector:
                self.play_collector.record_decision(
                    state, player, action, round_key
                )
    
    def _record_going_alone_outcome(self, round_key: str):
        """Record going alone outcome."""
        if round_key in self.going_alone_collector.pending:
            record = self.going_alone_collector.pending.pop(round_key)
            
            caller = record["player"]
            caller_team = caller % 2
            
            score_before = record.get("score_before", [0, 0])
            current_score = list(self.game.state.score)
            
            team0_gained = current_score[0] - score_before[0]
            team1_gained = current_score[1] - score_before[1]
            
            if caller_team == 0:
                points_gained = team0_gained
                points_lost = team1_gained
            else:
                points_gained = team1_gained
                points_lost = team0_gained
            
            if points_gained == 4:
                tricks = 5
                success = True
                march = True
                was_euchred = False
            elif points_gained >= 1:
                tricks = 3
                success = True
                march = False
                was_euchred = False
            else:
                tricks = 2
                success = False
                march = False
                was_euchred = True
            
            record.update({
                "tricks_won": tricks,
                "tricks_lost": 5 - tricks,
                "points_earned": points_gained,
                "was_euchred": was_euchred,
                "success": success,
                "march": march,
            })
            
            self.logger.log_going_alone(record)
    
    def _record_trump_call_outcome(self, round_key: str):
        """Record trump call outcome."""
        if round_key in self.trump_call_collector.pending:
            record = self.trump_call_collector.pending.pop(round_key)
            
            caller = record["player"]
            caller_team = caller % 2
            
            score_before = record.get("score_before", [0, 0])
            current_score = list(self.game.state.score)
            
            team0_gained = current_score[0] - score_before[0]
            team1_gained = current_score[1] - score_before[1]
            
            if caller_team == 0:
                points_gained = team0_gained
                points_lost = team1_gained
            else:
                points_gained = team1_gained
                points_lost = team0_gained
            
            if points_gained == 2:
                tricks = 5
                success = True
                march = True
                was_euchred = False
            elif points_gained == 1:
                tricks = 3
                success = True
                march = False
                was_euchred = False
            else:
                tricks = 2
                success = False
                march = False
                was_euchred = True
            
            record.update({
                "tricks_won": tricks,
                "tricks_lost": 5 - tricks,
                "points_earned": points_gained,
                "was_euchred": was_euchred,
                "success": success,
                "march": march,
            })
            
            self.logger.log_trump_call(record)
    
    def run_episodes(self, num_episodes: int) -> tuple[list[Experience], list[dict]]:
        """Run multiple episodes."""
        all_experiences = []
        episode_infos = []
        
        for _ in range(num_episodes):
            experiences, info = self.run_episode()
            all_experiences.extend(experiences)
            episode_infos.append(info)
        
        return all_experiences, episode_infos
    
    def get_collection_stats(self) -> dict:
        """Return counts of collected data."""
        return self.logger.get_counts()
    
    def close(self):
        """Close data logger."""
        self.logger.close()


def train_mixed(
    num_episodes: int = config.NUM_EPISODES,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    log_dir: str = config.LOG_DIR,
    data_dir: str = "data",
    rule_based_ratio: float = 0.5,
    collect_plays: bool = False,
    finetune: bool = False,
):
    """
    Train with mixed opponents and data collection.
    
    Args:
        finetune: If True, use smaller learning rate to preserve pre-trained weights
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Choose learning rate based on mode
    if finetune:
        learning_rate = config.FINETUNE_LEARNING_RATE
        print(f"Fine-tuning mode: using smaller learning rate {learning_rate}")
    else:
        learning_rate = config.LEARNING_RATE
    
    # Initialize
    network = EuchreNetwork()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    ppo = PPOLoss()
    runner = MixedTrainingRunner(
        network, 
        rule_based_ratio=rule_based_ratio,
        data_dir=data_dir,
        collect_plays=collect_plays,
    )
    writer = SummaryWriter(log_dir)
    
    episode = 0
    total_steps = 0
    
    # Try to resume
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Handle both checkpoint formats:
        # 1. Full checkpoint: {"network": state_dict, "optimizer": ..., "episode": ...}
        # 2. Just weights: state_dict directly (from pretrain_imitation.py)
        if isinstance(checkpoint, dict) and "network" in checkpoint:
            # Full checkpoint format
            network.load_state_dict(checkpoint["network"])
            if not finetune and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            episode = checkpoint.get("episode", 0)
            total_steps = checkpoint.get("total_steps", 0)
        else:
            # Just weights (from imitation pre-training)
            network.load_state_dict(checkpoint)
            episode = 0
            total_steps = 0
        print(f"Resumed from episode {episode}")
    
    print(f"Training for {num_episodes} episodes")
    print(f"Rule-based opponent ratio: {rule_based_ratio:.0%}")
    print(f"Data collection: {data_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    pbar = tqdm(initial=episode, total=num_episodes, desc="Training")
    
    # Track win rates
    recent_vs_rb_wins = []
    recent_self_play_wins = []
    
    try:
        while episode < num_episodes:
            # Collect experiences
            experiences, episode_infos = runner.run_episodes(
                num_episodes=config.BATCH_SIZE // 20
            )
            
            if not experiences:
                continue
            
            # Track win rates
            for info in episode_infos:
                if info["vs_rule_based"]:
                    recent_vs_rb_wins.append(1 if info["neural_won"] else 0)
                    if len(recent_vs_rb_wins) > 100:
                        recent_vs_rb_wins.pop(0)
                else:
                    recent_self_play_wins.append(1 if info["neural_won"] else 0)
                    if len(recent_self_play_wins) > 100:
                        recent_self_play_wins.pop(0)
            
            # Convert to tensors
            batch = experiences_to_tensors(experiences)
            
            # Compute advantages per player
            advantages, returns = compute_gae_by_player(experiences)
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # PPO update
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
                torch.nn.utils.clip_grad_norm_(network.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
            
            episode += len(episode_infos)
            total_steps += len(experiences)
            
            # Logging
            for key, value in loss_info.items():
                writer.add_scalar(f"loss/{key}", value, episode)
            
            if recent_vs_rb_wins:
                rb_win_rate = sum(recent_vs_rb_wins) / len(recent_vs_rb_wins)
                writer.add_scalar("win_rate/vs_rule_based", rb_win_rate, episode)
            
            if recent_self_play_wins:
                sp_win_rate = sum(recent_self_play_wins) / len(recent_self_play_wins)
                writer.add_scalar("win_rate/self_play", sp_win_rate, episode)
            
            # Log data collection stats
            if episode % 1000 == 0:
                stats = runner.get_collection_stats()
                for category, count in stats.items():
                    writer.add_scalar(f"data/{category}", count, episode)
            
            # Checkpoint
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
                
                rb_wr = sum(recent_vs_rb_wins) / len(recent_vs_rb_wins) if recent_vs_rb_wins else 0
                stats = runner.get_collection_stats()
                tqdm.write(f"Episode {episode}: vs RuleBased={rb_wr:.1%}, Data={stats}")
            
            pbar.update(len(episode_infos))
    
    finally:
        runner.close()
        pbar.close()
        
        torch.save(network.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
        print("\nTraining complete!")
        print(f"Final data stats: {runner.get_collection_stats()}")


def main():
    parser = argparse.ArgumentParser(description="Train with mixed opponents")
    parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES)
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=str, default=config.LOG_DIR)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--rule-based-ratio",
        type=float,
        default=0.7,
        help="Fraction of games against rule-based (0.0-1.0, default 0.7)"
    )
    parser.add_argument(
        "--collect-plays",
        action="store_true",
        help="Also collect card play decisions"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Use smaller learning rate to preserve pre-trained weights"
    )
    
    args = parser.parse_args()
    
    train_mixed(
        num_episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        data_dir=args.data_dir,
        rule_based_ratio=args.rule_based_ratio,
        collect_plays=args.collect_plays,
        finetune=args.finetune,
    )


if __name__ == "__main__":
    main()