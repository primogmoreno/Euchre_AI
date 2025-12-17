"""
Game state representation for Euchre.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .cards import Card, Suit


class GamePhase(Enum):
    """Current phase of the game."""
    CALLING_ROUND_1 = auto()   # Deciding whether to order up the turned card
    CALLING_ROUND_2 = auto()   # Naming a different suit (dealer's turn to pick)
    DISCARD = auto()           # Dealer discarding after picking up
    PLAYING = auto()           # Trick-taking phase
    ROUND_OVER = auto()        # Round finished, scoring
    GAME_OVER = auto()         # Game finished


@dataclass
class GameState:
    """
    Complete state of a Euchre game.
    
    This contains ALL information, including hidden cards.
    For player observations (partial information), use get_observation().
    """
    
    # Hands for each player (index 0-3)
    hands: list[list[Card]] = field(default_factory=lambda: [[], [], [], []])
    
    # The kitty (remaining cards after deal, usually 3 + turned card)
    kitty: list[Card] = field(default_factory=list)
    
    # Card turned up for trump calling
    turned_card: Optional[Card] = None
    
    # Current trump suit (None until called)
    trump: Optional[Suit] = None
    
    # Game flow
    dealer: int = 0                    # Player index 0-3
    current_player: int = 0            # Whose turn it is
    phase: GamePhase = GamePhase.CALLING_ROUND_1
    
    # Trump calling info
    caller: Optional[int] = None       # Who called trump
    going_alone: bool = False          # Is caller going alone?
    
    # Current trick
    lead_player: Optional[int] = None  # Who led this trick
    current_trick: list[Card] = field(default_factory=list)
    
    # Round progress
    tricks_won: list[int] = field(default_factory=lambda: [0, 0])  # [team0, team1]
    
    # Overall game score
    score: list[int] = field(default_factory=lambda: [0, 0])  # [team0, team1]
    
    # History for this round (for observations)
    play_history: list[tuple[int, Card]] = field(default_factory=list)  # (player, card)
    
    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        return GameState(
            hands=[hand.copy() for hand in self.hands],
            kitty=self.kitty.copy(),
            turned_card=self.turned_card,
            trump=self.trump,
            dealer=self.dealer,
            current_player=self.current_player,
            phase=self.phase,
            caller=self.caller,
            going_alone=self.going_alone,
            lead_player=self.lead_player,
            current_trick=self.current_trick.copy(),
            tricks_won=self.tricks_won.copy(),
            score=self.score.copy(),
            play_history=self.play_history.copy(),
        )
    
    def get_team(self, player: int) -> int:
        """Return team index (0 or 1) for a player."""
        return player % 2
    
    def get_partner(self, player: int) -> int:
        """Return partner's player index."""
        return (player + 2) % 4
    
    def is_partner_sitting_out(self, player: int) -> bool:
        """Check if this player's partner is sitting out (going alone scenario)."""
        if not self.going_alone:
            return False
        partner = self.get_partner(player)
        return partner == self.get_partner(self.caller)