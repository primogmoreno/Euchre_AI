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
    card_to_str,
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
        
        # Track decisions for the CURRENT ROUND (reset each round)
        alone_this_round = False
        trump_called_this_round = False
        round_number = 0
        
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
                round_number=round_number,
            )
            
            # Update tracking flags for this round
            if isinstance(action, str):
                if "alone" in action:
                    alone_this_round = True
                elif action.startswith("order_up") or action.startswith("call_"):
                    trump_called_this_round = True
            
            # Track kitty size before step to detect when 4th card is added
            kitty_size_before = len(self.game.state.kitty)
            
            # Track tricks before step (to detect round end)
            tricks_before = sum(self.game.state.tricks_won)
            
            # Take step
            result = self.game.step(action)
            observations = result.observations
            done = result.done
            
            # =================================================================
            # Track the 4th kitty card (discard or turned-down card)
            # This can happen on a different step than the trump call:
            # - Order up: buried card is dealer's discard (DISCARD phase)
            # - Call (round 2): buried card was the turned-down card (already in kitty)
            # =================================================================
            
            kitty_size_after = len(self.game.state.kitty)
            if kitty_size_before == 3 and kitty_size_after == 4:
                # A 4th card was just added to kitty
                fourth_kitty_card = self.game.state.kitty[-1]
                buried_card_str = card_to_str(fourth_kitty_card)
                round_key = f"{episode_id}_{round_number}"
                
                # Update pending records with this info
                if round_key in self.going_alone_collector.pending:
                    self.going_alone_collector.pending[round_key]["buried_card"] = buried_card_str
                if round_key in self.trump_call_collector.pending:
                    self.trump_call_collector.pending[round_key]["buried_card"] = buried_card_str
            
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
            
            # =================================================================
            # CRITICAL: Check if round just ended (5 tricks were completed)
            # We need to record outcomes BEFORE _deal_round resets tricks_won
            # =================================================================
            
            # Detect round end: tricks went from 4->5 total, or rewards are non-zero
            round_just_ended = (
                tricks_before == 4 and 
                any(r != 0 for r in result.rewards)
            )
            
            if round_just_ended:
                # Record outcomes for this round using the CURRENT state
                # (before it gets reset by the next iteration)
                round_key = f"{episode_id}_{round_number}"
                
                if alone_this_round:
                    self._record_going_alone_outcome(round_key)
                elif trump_called_this_round:
                    self._record_trump_call_outcome(round_key)
                
                # Reset for next round
                alone_this_round = False
                trump_called_this_round = False
                round_number += 1
                last_trick_count = 0
            
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
        round_number: int,
    ):
        """Route decision to appropriate collector."""
        
        state = self.game.state
        # Use a unique key combining episode and round
        round_key = f"{episode_id}_{round_number}"
        
        if isinstance(action, str):
            # Calling phase actions
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
            # Card play
            if self.play_collector:
                self.play_collector.record_decision(
                    state, player, action, round_key
                )
    
    def _record_going_alone_outcome(self, round_key: str):
        """Record going alone outcome using current game state."""
        # The state still has the correct tricks_won from this round
        # because _deal_round hasn't been called yet on OUR side
        # (it's called inside engine.step, but we capture before next iteration)
        
        # Actually we need to get the tricks from the current state
        # But there's a problem - by now _deal_round may have been called
        
        # Let me check if we have pending record
        if round_key in self.going_alone_collector.pending:
            record = self.going_alone_collector.pending.pop(round_key)
            
            caller = record["player"]
            caller_team = caller % 2
            
            # We need to compute tricks from the rewards instead
            # Actually the state.tricks_won has been reset...
            
            # WORKAROUND: Look at score change to determine outcome
            # This is called right after rewards are given but before new round
            # We stored score_before in the record
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
            
            # Determine outcome from points
            if points_gained == 4:
                tricks = 5
                success = True
                march = True
                was_euchred = False
            elif points_gained == 2:
                # Could be march without alone, but we know it was alone
                tricks = 5
                success = True
                march = True
                was_euchred = False
            elif points_gained == 1:
                tricks = 3  # or 4, but we'll say 3+
                success = True
                march = False
                was_euchred = False
            elif points_lost == 2:
                tricks = 2  # or less
                success = False
                march = False
                was_euchred = True
            else:
                # Shouldn't happen
                tricks = 0
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
        """Record trump call outcome using current game state."""
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
            
            # Determine outcome from points (not going alone)
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
            elif points_lost == 2:
                tricks = 2
                success = False
                march = False
                was_euchred = True
            else:
                tricks = 0
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