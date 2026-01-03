"""
Self-play game runner for collecting training data.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from game.engine import EuchreGame
from model.network import EuchreNetwork
from model.encoding import encode_state, action_to_index, index_to_action, create_action_mask


@dataclass
class Experience:
    """Single step of experience."""
    state: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    action_mask: np.ndarray
    player: int


@dataclass 
class EpisodeBuffer:
    """Buffer for one episode's experiences, separated by player."""
    experiences: dict = field(default_factory=lambda: {0: [], 1: [], 2: [], 3: []})
    
    def add(self, player: int, exp: Experience):
        self.experiences[player].append(exp)
    
    def get_all(self) -> list[Experience]:
        """Get all experiences flattened."""
        all_exp = []
        for player_exp in self.experiences.values():
            all_exp.extend(player_exp)
        return all_exp
    
    def get_by_player(self, player: int) -> list[Experience]:
        """Get experiences for a specific player."""
        return self.experiences[player]
    
    def get_by_team(self, team: int) -> list[Experience]:
        """Get experiences for a team (0 or 1)."""
        if team == 0:
            return self.experiences[0] + self.experiences[2]
        else:
            return self.experiences[1] + self.experiences[3]
    
    def clear(self):
        self.experiences = {0: [], 1: [], 2: [], 3: []}


class SelfPlayRunner:
    """
    Runs self-play games to collect training data.
    
    All 4 players use the same network.
    """
    
    def __init__(self, network: EuchreNetwork):
        self.network = network
        self.game = EuchreGame()
    
    def run_episode(self) -> tuple[list[Experience], dict]:
        """
        Run a single game and collect experiences.
        
        Returns:
            experiences: List of Experience objects
            info: Episode statistics
        """
        buffer = EpisodeBuffer()
        observations = self.game.reset()
        
        done = False
        total_steps = 0
        
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
            
            # Take step
            result = self.game.step(action)
            observations = result.observations
            done = result.done
            
            # Store experience with the reward for THIS player
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
        
        # Compute final rewards for all players
        final_score = self.game.state.score
        winner = 0 if final_score[0] >= 10 else 1
        
        info = {
            "total_steps": total_steps,
            "final_score": final_score,
            "winner": winner,
        }
        
        return buffer.get_all(), info
    
    def run_episodes(self, num_episodes: int) -> tuple[list[Experience], list[dict]]:
        """
        Run multiple episodes.
        
        Returns:
            all_experiences: Combined experiences from all episodes
            episode_infos: List of info dicts from each episode
        """
        all_experiences = []
        episode_infos = []
        
        for _ in range(num_episodes):
            experiences, info = self.run_episode()
            all_experiences.extend(experiences)
            episode_infos.append(info)
        
        return all_experiences, episode_infos


def experiences_to_tensors(experiences: list[Experience]) -> dict:
    """
    Convert list of experiences to batched tensors.
    
    Returns dict with:
        states, actions, rewards, values, log_probs, dones, action_masks, players
    """
    return {
        "states": torch.FloatTensor(np.array([e.state for e in experiences])),
        "actions": torch.LongTensor([e.action for e in experiences]),
        "rewards": [e.reward for e in experiences],
        "values": [e.value for e in experiences],
        "log_probs": torch.FloatTensor([e.log_prob for e in experiences]),
        "dones": [e.done for e in experiences],
        "action_masks": torch.BoolTensor(np.array([e.action_mask for e in experiences])),
        "players": [e.player for e in experiences],
    }


def compute_gae_by_player(experiences: list[Experience], gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[list[float], list[float]]:
    """
    Compute GAE separately for each player's trajectory, then combine.
    
    This is critical because experiences are interleaved between 4 players,
    but each player's trajectory should be computed independently.
    """
    # Group experiences by player
    by_player = {0: [], 1: [], 2: [], 3: []}
    indices_by_player = {0: [], 1: [], 2: [], 3: []}
    
    for i, exp in enumerate(experiences):
        by_player[exp.player].append(exp)
        indices_by_player[exp.player].append(i)
    
    # Compute GAE for each player separately
    all_advantages = [0.0] * len(experiences)
    all_returns = [0.0] * len(experiences)
    
    for player in range(4):
        player_exps = by_player[player]
        player_indices = indices_by_player[player]
        
        if not player_exps:
            continue
        
        # Compute GAE for this player's trajectory
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(player_exps))):
            exp = player_exps[t]
            
            if exp.done:
                next_value = 0
                gae = 0
            
            delta = exp.reward + gamma * next_value - exp.value
            gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + exp.value)
            
            next_value = exp.value
        
        # Put back into original positions
        for idx, orig_idx in enumerate(player_indices):
            all_advantages[orig_idx] = advantages[idx]
            all_returns[orig_idx] = returns[idx]
    
    return all_advantages, all_returns