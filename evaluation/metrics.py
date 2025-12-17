"""
Metrics tracking for evaluation.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class WinRateTracker:
    """
    Track win rates over a sliding window.
    
    Useful for monitoring training progress.
    """
    
    window_size: int = 100
    wins: deque = field(default_factory=lambda: deque(maxlen=100))
    total_games: int = 0
    total_wins: int = 0
    
    def __post_init__(self):
        self.wins = deque(maxlen=self.window_size)
    
    def record(self, won: bool):
        """Record a game result."""
        self.wins.append(1 if won else 0)
        self.total_games += 1
        if won:
            self.total_wins += 1
    
    @property
    def recent_win_rate(self) -> float:
        """Win rate over recent window."""
        if not self.wins:
            return 0.0
        return sum(self.wins) / len(self.wins)
    
    @property
    def overall_win_rate(self) -> float:
        """Overall win rate."""
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games
    
    def reset(self):
        """Reset all statistics."""
        self.wins.clear()
        self.total_games = 0
        self.total_wins = 0


@dataclass
class EpisodeMetrics:
    """Metrics from a single episode."""
    winner: int
    final_score: tuple[int, int]
    num_rounds: int
    total_steps: int
    rewards: list[float]
    
    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "final_score": list(self.final_score),
            "num_rounds": self.num_rounds,
            "total_steps": self.total_steps,
            "total_reward": sum(self.rewards),
        }


class MetricsLogger:
    """
    Log metrics to file for later analysis.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entries = []
    
    def log(self, episode: int, metrics: dict):
        """Log metrics for an episode."""
        entry = {"episode": episode, **metrics}
        self.entries.append(entry)
    
    def save(self):
        """Save all entries to file."""
        with open(self.filepath, "w") as f:
            json.dump(self.entries, f, indent=2)
    
    def load(self):
        """Load entries from file."""
        try:
            with open(self.filepath, "r") as f:
                self.entries = json.load(f)
        except FileNotFoundError:
            self.entries = []