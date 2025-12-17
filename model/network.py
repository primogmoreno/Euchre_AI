"""
Neural network architecture for Euchre AI.
"""

import torch
import torch.nn as nn

import config


class EuchreNetwork(nn.Module):
    """
    Neural network for Euchre policy and value estimation.
    
    Architecture:
    - Shared feature extraction layers
    - Policy head: outputs action probabilities
    - Value head: estimates expected return
    """
    
    def __init__(
        self,
        state_size: int = config.STATE_SIZE,
        action_size: int = config.ACTION_SIZE,
        hidden_size: int = config.HIDDEN_SIZE,
    ):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, config.POLICY_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.POLICY_HIDDEN, action_size),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, config.VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.VALUE_HIDDEN, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Batch of encoded states, shape (batch, state_size)
        
        Returns:
            policy_logits: Raw action scores, shape (batch, action_size)
            value: State value estimate, shape (batch, 1)
        """
        features = self.shared(state)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Used during training for PPO.
        
        Args:
            state: Encoded state
            action_mask: Boolean mask of legal actions (True = legal)
        
        Returns:
            action: Sampled action index
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: Value estimate
        """
        logits, value = self.forward(state)
        
        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO loss calculation.
        
        Args:
            state: Batch of states
            action: Batch of actions taken
            action_mask: Boolean mask of legal actions
        
        Returns:
            log_prob: Log probabilities of the actions
            entropy: Policy entropies
            value: Value estimates
        """
        logits, value = self.forward(state)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value.squeeze(-1)