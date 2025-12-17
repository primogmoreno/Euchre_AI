"""
PPO (Proximal Policy Optimization) loss computation.
"""

import torch
import torch.nn as nn
import numpy as np

import config


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = config.GAMMA,
    gae_lambda: float = config.GAE_LAMBDA,
) -> tuple[list[float], list[float]]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: List of rewards received
        values: List of value estimates
        dones: List of episode termination flags
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
    
    Returns:
        advantages: GAE advantage estimates
        returns: Discounted returns (for value targets)
    """
    advantages = []
    returns = []
    
    gae = 0
    next_value = 0
    
    # Iterate backwards through the trajectory
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
            gae = 0
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        
        next_value = values[t]
    
    return advantages, returns


class PPOLoss:
    """
    PPO loss computation with clipping.
    """
    
    def __init__(
        self,
        clip_epsilon: float = config.CLIP_EPSILON,
        value_coef: float = config.VALUE_COEF,
        entropy_coef: float = config.ENTROPY_COEF,
    ):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def compute(
        self,
        network: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_masks: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute PPO loss.
        
        Args:
            network: The neural network
            states: Batch of states
            actions: Batch of actions taken
            old_log_probs: Log probabilities from when actions were taken
            advantages: Advantage estimates
            returns: Return targets for value function
            action_masks: Boolean masks for legal actions
        
        Returns:
            loss: Total loss to minimize
            info: Dict with loss components for logging
        """
        # Get current policy evaluations
        log_probs, entropy, values = network.evaluate_actions(
            states, actions, action_masks
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus (for exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        info = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": ((ratio - 1) - (ratio.log())).mean().item(),
        }
        
        return total_loss, info