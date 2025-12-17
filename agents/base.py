"""
Base agent interface for Euchre players.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Abstract base class for Euchre agents.
    
    All agents must implement select_action(), which chooses
    an action given an observation and list of legal actions.
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
    
    @abstractmethod
    def select_action(self, observation: dict, legal_actions: list) -> Any:
        """
        Select an action given the current observation.
        
        Args:
            observation: Dict containing visible game state (from get_observation)
            legal_actions: List of valid actions (Cards or strings)
        
        Returns:
            The chosen action (must be in legal_actions)
        """
        pass
    
    def on_game_start(self) -> None:
        """Called at the start of a new game. Override if needed."""
        pass
    
    def on_game_end(self, final_score: list[int], winner: int) -> None:
        """Called at the end of a game. Override if needed."""
        pass
    
    def on_round_end(self, tricks_won: list[int], points: list[int]) -> None:
        """Called at the end of a round. Override if needed."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"