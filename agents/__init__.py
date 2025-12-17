"""
Agent implementations for Euchre.
"""

from .base import BaseAgent
from .random_agent import RandomAgent
from .rule_based import RuleBasedAgent

# Lazy import for NeuralAgent to avoid requiring torch for basic usage
def __getattr__(name):
    if name == "NeuralAgent":
        from .neural_agent import NeuralAgent
        return NeuralAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BaseAgent", "RandomAgent", "RuleBasedAgent", "NeuralAgent"]