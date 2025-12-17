"""
Card, Suit, Rank, and Deck definitions for Euchre.
"""

from enum import Enum
from dataclasses import dataclass
import random


class Suit(Enum):
    """Card suits with values for indexing."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3
    
    @property
    def color(self) -> str:
        """Return 'black' or 'red'."""
        return "black" if self in (Suit.CLUBS, Suit.SPADES) else "red"
    
    @property
    def same_color_suit(self) -> "Suit":
        """Return the other suit of the same color (for left bower logic)."""
        mapping = {
            Suit.CLUBS: Suit.SPADES,
            Suit.SPADES: Suit.CLUBS,
            Suit.HEARTS: Suit.DIAMONDS,
            Suit.DIAMONDS: Suit.HEARTS,
        }
        return mapping[self]


class Rank(Enum):
    """Card ranks with values for basic ordering."""
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True)
class Card:
    """
    Immutable card representation.
    
    Using frozen=True makes cards hashable, so they can be used in sets.
    """
    rank: Rank
    suit: Suit
    
    def __repr__(self) -> str:
        rank_symbols = {
            Rank.NINE: "9", Rank.TEN: "10", Rank.JACK: "J",
            Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A"
        }
        suit_symbols = {
            Suit.CLUBS: "♣", Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥", Suit.SPADES: "♠"
        }
        return f"{rank_symbols[self.rank]}{suit_symbols[self.suit]}"
    
    def to_index(self) -> int:
        """Convert card to unique index 0-23 for neural network encoding."""
        return self.suit.value * 6 + (self.rank.value - 9)
    
    @classmethod
    def from_index(cls, index: int) -> "Card":
        """Create card from index 0-23."""
        suit = Suit(index // 6)
        rank = Rank((index % 6) + 9)
        return cls(rank=rank, suit=suit)


class Deck:
    """
    Euchre deck (24 cards: 9, 10, J, Q, K, A in each suit).
    """
    
    def __init__(self):
        self.cards: list[Card] = []
        self.reset()
    
    def reset(self) -> None:
        """Restore deck to full 24 cards."""
        self.cards = [
            Card(rank=rank, suit=suit)
            for suit in Suit
            for rank in Rank
        ]
    
    def shuffle(self) -> None:
        """Randomize card order."""
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int) -> list[Card]:
        """Deal cards from top of deck."""
        dealt = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt
    
    def __len__(self) -> int:
        return len(self.cards)


# Pre-computed list of all cards for convenience
ALL_CARDS = [Card.from_index(i) for i in range(24)]