"""
Neural network agent using a trained neural network model.
"""

import torch
from typing import Any, Optional

from .base import BaseAgent
from model.network import EuchreNetwork
from model.encoding import encode_state, action_to_index, index_to_action, ALL_ACTIONS


class NeuralAgent(BaseAgent):
    """
    Agent that uses a trained neural network to select actions.
    Connected to a local file which stores the training weights for the model.
    
    Can operate in greedy mode (always pick best) or stochastic
    mode (sample from distribution) for exploration.

    Greedy - Picks highest probability weight from possible actions
        Ex. Order_up = 50%, so chose Order_up
    Stochastic - Takes the weights and randomly decides an action taking into account probs
        Ex. Pass: 45%
            Order_up: 50%
            Order_up_alone: 5%
    """
    
    def __init__(
        self,
        network: Optional[EuchreNetwork] = None,
        model_path: Optional[str] = None,
        greedy: bool = True,
        name: str = "Neural"
    ):
        """
        Initialize neural agent.
        
        Args:
            network: Pre-initialized network (optional)
            model_path: Path to saved model weights (optional)
            greedy: If True, always pick best action. If False, sample.
            name: Agent name
        """
        super().__init__(name)
        self.greedy = greedy
        
        if network is not None:
            self.network = network
        else:
            self.network = EuchreNetwork()
            
        if model_path is not None:
            self.load_model(model_path)
        
        self.network.eval()  # Set to evaluation mode
    
    def load_model(self, path: str) -> None:
        """Load model weights from file."""
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
        self.network.eval()
    
    def select_action(self, observation: dict, legal_actions: list) -> Any:
        """Select action using the neural network."""
        # Encode state
        state = encode_state(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get network output
        with torch.no_grad():
            logits, value = self.network(state_tensor)
        
        # Mask illegal actions
        mask = torch.full_like(logits, float("-inf"))
        for action in legal_actions:
            idx = action_to_index(action)
            mask[0, idx] = 0
        
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)
        
        if self.greedy:
            # Pick best action
            action_idx = torch.argmax(probs).item()
        else:
            # Sample from distribution
            action_idx = torch.multinomial(probs, 1).item()
        
        return index_to_action(action_idx, legal_actions)
    
    def get_action_probs(self, observation: dict, legal_actions: list) -> dict:
        """
        Get probability distribution over legal actions.
        
        Useful for analysis and debugging.
        """
        state = encode_state(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.network(state_tensor)
        
        mask = torch.full_like(logits, float("-inf"))
        for action in legal_actions:
            idx = action_to_index(action)
            mask[0, idx] = 0
        
        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=-1)
        
        return {
            action: probs[0, action_to_index(action)].item()
            for action in legal_actions
        }
    
    def get_value(self, observation: dict) -> float:
        """Get value estimate for current state."""
        state = encode_state(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            _, value = self.network(state_tensor)
        
        return value.item()