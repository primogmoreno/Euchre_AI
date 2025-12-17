"""
Random agent that selects uniformly from legal actions.

Useful as a baseline and for initial testing.
"""

import random
from typing import Any

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that plays randomly.
    
    Selects uniformly at random from legal actions.
    Useful as a weak baseline.
    """
    
    def __init__(self, name: str = "Random"):
        super().__init__(name)
    
    def select_action(self, observation: dict, legal_actions: list) -> Any:
        """Select a random legal action."""
        return random.choice(legal_actions)