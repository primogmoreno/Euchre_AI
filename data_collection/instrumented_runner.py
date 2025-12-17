"""
Self-play runner with data collection instrumentation.

Drop-in replacement for self_play.py that also logs decision data.
"""

import torch
from typing import Optional

from game.engine import EuchreGame
from game.state import GamePhase
from game.cards import Card
from model.network import EuchreNetwork
from model.encoding import encode_state, action_to_index, index_to_action, create_action_mask
from training.self_play import Experience, EpisodeBuffer
from data_collection.logger import DataLogger
from data_collection.collectors import (
    GoingAloneCollector,
    TrumpCallCollector,
    PassCollector,
    PlayCollector,
)


class InstrumentedSelfPlayRunner:
    """
    Self-play runner that also collects decision data for analysis.
    
    Use this instead of SelfPlayRunner when you want to gather
    data about specific decision types.
    """
    
    def __init__(
        self,
        network: EuchreNetwork,
        data_dir: str = "data",
        collect_plays: bool = False,  # Can generate lots of data
    ):
        self.network = network
        self.game = EuchreGame()
        
        # Set up data collection
        self.logger = DataLogger(data_dir)
        self.going_alone_collector = GoingAloneCollector(self.logger)
        self.trump_call_collector = TrumpCallCollector(self.logger)
        self.pass_collector = PassCollector(self.logger)
        self.play_collector = PlayCollector(self.logger, only_interesting=True) if collect_plays else None
        
        self.episode_count = 0
    
    def run_episode(self) -> tuple[list[Experience], dict]:
        """
        Run a single game, collecting both training data and decision logs.
        """
        buffer = EpisodeBuffer()
        observations = self.game.reset()
        
        self.episode_count += 1
        episode_id = self.episode_count
        
        done = False
        total_steps = 0
        last_trick_count = 0
        
        # Track if someone went alone this episode
        alone_this_episode = False
        trump_called_this_episode = False
        
        while not done:
            player = self.game.state.current_player
            obs = observations[player]
            legal_actions = self.game.get_legal_actions()
            
            # Encode state
            state = encode_state(obs)
            action_mask = create_action_mask(legal_actions)
            
            # Get action from network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
            
            with torch.no_grad():
                action_idx, log_prob, _, value = self.network.get_action_and_value(
                    state_tensor, mask_tensor
                )
            
            action_idx = action_idx.item()
            log_prob = log_prob.item()
            value = value.item()
            
            # Convert action index to actual action
            action = index_to_action(action_idx, legal_actions)
            
            # =================================================================
            # DATA COLLECTION - Log decisions before taking the action
            # =================================================================
            
            self._collect_decision(
                action=action,
                player=player,
                episode_id=episode_id,
                alone_this_episode=alone_this_episode,
                trump_called_this_episode=trump_called_this_episode,
            )
            
            # Update tracking flags
            if isinstance(action, str):
                if "alone" in action:
                    alone_this_episode = True
                elif action.startswith("order_up") or action.startswith("call_"):
                    trump_called_this_episode = True
            
            # Take step
            result = self.game.step(action)
            observations = result.observations
            done = result.done
            
            # =================================================================
            # Log trick outcomes
            # =================================================================
            
            if result.info.get("trick_winner") is not None:
                trick_num = last_trick_count
                winner = result.info["trick_winner"]
                
                if self.play_collector:
                    self.play_collector.record_trick_outcome(
                        self.game.state, episode_id, trick_num, winner
                    )
                
                last_trick_count = sum(self.game.state.tricks_won)
            
            # Store experience for training
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
        
        # =================================================================
        # Log round outcomes for pending decisions
        # =================================================================
        
        if alone_this_episode:
            self.going_alone_collector.record_outcome(self.game.state, episode_id)
        if trump_called_this_episode and not alone_this_episode:
            self.trump_call_collector.record_outcome(self.game.state, episode_id)
        
        # Compute final info
        final_score = self.game.state.score
        winner = 0 if final_score[0] >= 10 else 1
        
        info = {
            "total_steps": total_steps,
            "final_score": final_score,
            "winner": winner,
            "episode_id": episode_id,
        }
        
        return buffer.get_all(), info
    
    def _collect_decision(
        self,
        action,
        player: int,
        episode_id: int,
        alone_this_episode: bool,
        trump_called_this_episode: bool,
    ):
        """Route decision to appropriate collector."""
        
        state = self.game.state
        
        if isinstance(action, str):
            # Calling phase actions
            if "alone" in action:
                self.going_alone_collector.record_decision(
                    state, player, action, episode_id
                )
            elif action.startswith("order_up") or action.startswith("call_"):
                self.trump_call_collector.record_decision(
                    state, player, action, episode_id
                )
            elif action == "pass":
                self.pass_collector.record_decision(
                    state, player, episode_id
                )
        
        elif isinstance(action, Card) and state.phase == GamePhase.PLAYING:
            # Card play
            if self.play_collector:
                self.play_collector.record_decision(
                    state, player, action, episode_id
                )
    
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