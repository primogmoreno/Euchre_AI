"""
State encoding utilities for neural network input.

Converts game observations to fixed-size tensors.
"""

import numpy as np
from typing import Any

from game.cards import Card, Suit, ALL_CARDS
from game.state import GamePhase


# =============================================================================
# Action Space Definition
# =============================================================================

# All possible actions as a list
# 0-23: Play card (by card index)
# 24: Pass
# 25: Order up
# 26: Order up alone
# 27-30: Call suits (clubs, diamonds, hearts, spades) - offset by turned suit logic
# We'll use a flat action space and mask illegal actions

ALL_ACTIONS = (
    [f"card_{i}" for i in range(24)] +  # 0-23: cards
    ["pass", "order_up", "order_up_alone"] +  # 24-26
    ["call_clubs", "call_diamonds", "call_hearts", "call_spades"] +  # 27-30
    ["call_clubs_alone", "call_diamonds_alone", "call_hearts_alone", "call_spades_alone"]  # 31-34
)

ACTION_TO_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}


def action_to_index(action: Any) -> int:
    """
    Convert an action to its index in the action space.
    
    Args:
        action: Either a Card object or a string action
    
    Returns:
        Index in ALL_ACTIONS
    """
    if isinstance(action, Card):
        return action.to_index()
    elif isinstance(action, str):
        return ACTION_TO_IDX.get(action, 24)  # default to pass
    else:
        raise ValueError(f"Unknown action type: {type(action)}")


def index_to_action(index: int, legal_actions: list) -> Any:
    """
    Convert an action index back to the actual action.
    
    Args:
        index: Index in ALL_ACTIONS
        legal_actions: List of legal actions (to find matching Card objects)
    
    Returns:
        The corresponding action
    """
    if index < 24:
        # It's a card
        target_card = Card.from_index(index)
        for action in legal_actions:
            if isinstance(action, Card) and action == target_card:
                return action
        # Card not in legal actions (shouldn't happen with proper masking)
        return legal_actions[0]
    else:
        # It's a string action
        action_str = ALL_ACTIONS[index]
        if action_str in legal_actions:
            return action_str
        return legal_actions[0]


# =============================================================================
# State Encoding
# =============================================================================

def encode_state(observation: dict) -> np.ndarray:
    """
    Encode a game observation as a fixed-size numpy array.
    
    Encoding breakdown:
    - Hand: 24 binary (which cards I hold)
    - Played cards: 24 binary (which cards have been played)
    - Trump suit: 4 one-hot (or zeros if not called)
    - Turned card: 24 one-hot (for calling phase)
    - Current trick: 24 * 3 = 72 (up to 3 cards played this trick)
    - Tricks won: 2 float (normalized)
    - Score: 2 float (normalized)
    - Position info: 4 (my position, dealer, caller, phase)
    - Going alone: 1 binary
    
    Total: 24 + 24 + 4 + 24 + 72 + 2 + 2 + 4 + 1 = 157
    (We'll pad to 176 for nice round number)
    """
    encoding = []
    
    # 1. Cards in hand (24 binary)
    hand_vec = np.zeros(24)
    for card in observation["my_hand"]:
        hand_vec[card.to_index()] = 1
    encoding.append(hand_vec)
    
    # 2. Cards played this round (24 binary)
    played_vec = np.zeros(24)
    for player, card in observation["play_history"]:
        played_vec[card.to_index()] = 1
    encoding.append(played_vec)
    
    # 3. Trump suit (4 one-hot)
    trump_vec = np.zeros(4)
    if observation["trump"] is not None:
        trump_vec[observation["trump"].value] = 1
    encoding.append(trump_vec)
    
    # 4. Turned card (24 one-hot)
    turned_vec = np.zeros(24)
    if observation["turned_card"] is not None:
        turned_vec[observation["turned_card"].to_index()] = 1
    encoding.append(turned_vec)
    
    # 5. Current trick (24 * 3 = 72)
    trick_vec = np.zeros(72)
    for i, card in enumerate(observation["current_trick"][:3]):
        trick_vec[i * 24 + card.to_index()] = 1
    encoding.append(trick_vec)
    
    # 6. Tricks won (2 floats, normalized to 0-1)
    tricks_vec = np.array(observation["tricks_won"]) / 5.0
    encoding.append(tricks_vec)
    
    # 7. Score (2 floats, normalized to 0-1)
    score_vec = np.array(observation["score"]) / 10.0
    encoding.append(score_vec)
    
    # 8. Position info (4 floats)
    position_vec = np.array([
        observation["my_position"] / 3.0,
        observation["dealer"] / 3.0,
        (observation["caller"] if observation["caller"] is not None else -1) / 3.0,
        observation["phase"].value / 6.0,  # Normalize by max phase value
    ])
    encoding.append(position_vec)
    
    # 9. Going alone flag (1 binary)
    alone_vec = np.array([1.0 if observation["going_alone"] else 0.0])
    encoding.append(alone_vec)
    
    # 10. Lead player for current trick (4 one-hot)
    lead_vec = np.zeros(4)
    if observation["lead_player"] is not None:
        lead_vec[observation["lead_player"]] = 1
    encoding.append(lead_vec)
    
    # Concatenate all
    full_encoding = np.concatenate(encoding)
    
    # Pad to target size
    target_size = 176
    if len(full_encoding) < target_size:
        padding = np.zeros(target_size - len(full_encoding))
        full_encoding = np.concatenate([full_encoding, padding])
    
    return full_encoding.astype(np.float32)


def create_action_mask(legal_actions: list) -> np.ndarray:
    """
    Create a boolean mask for legal actions.
    
    Returns:
        Boolean array of shape (35,) where True = legal
    """
    mask = np.zeros(len(ALL_ACTIONS), dtype=bool)
    
    for action in legal_actions:
        idx = action_to_index(action)
        mask[idx] = True
    
    return mask