"""
Euchre rules engine: trump logic, trick resolution, legal moves.
"""

from typing import Optional
from .cards import Card, Suit, Rank
from .state import GameState, GamePhase


class EuchreRules:
    """Static methods for Euchre game rules."""
    
    @staticmethod
    def get_effective_suit(card: Card, trump: Optional[Suit]) -> Suit:
        """
        Get the effective suit of a card considering trump.
        
        The left bower (jack of same color as trump) counts as trump.
        """
        if trump is None:
            return card.suit
            
        # Left bower: jack of same-color suit is trump
        if card.rank == Rank.JACK and card.suit == trump.same_color_suit:
            return trump
            
        return card.suit
    
    @staticmethod
    def get_card_strength(card: Card, trump: Optional[Suit], led_suit: Optional[Suit]) -> int:
        """
        Get the strength of a card for trick comparison.
        
        Returns a comparable integer (higher = stronger).
        
        Ordering:
        - Right bower (jack of trump): 1000
        - Left bower (jack of same color): 900
        - Other trump: 800 + rank value
        - Led suit: 100 + rank value
        - Off suit: 0 (can't win)
        """
        if trump is None:
            # No trump called yet
            if led_suit and card.suit == led_suit:
                return 100 + card.rank.value
            return card.rank.value
        
        effective_suit = EuchreRules.get_effective_suit(card, trump)
        
        # Right bower
        if card.rank == Rank.JACK and card.suit == trump:
            return 1000
        
        # Left bower
        if card.rank == Rank.JACK and card.suit == trump.same_color_suit:
            return 900
        
        # Other trump
        if effective_suit == trump:
            return 800 + card.rank.value
        
        # Led suit
        if led_suit and effective_suit == led_suit:
            return 100 + card.rank.value
        
        # Off suit (can't win trick)
        return 0
    
    @staticmethod
    def get_legal_plays(
        hand: list[Card],
        lead_card: Optional[Card],
        trump: Optional[Suit]
    ) -> list[Card]:
        """
        Get legal cards to play from a hand.
        
        Must follow led suit if possible (using effective suit for trump).
        """
        if not hand:
            return []
            
        if lead_card is None:
            # Leading the trick, can play anything
            return hand.copy()
        
        led_suit = EuchreRules.get_effective_suit(lead_card, trump)
        
        # Find cards that follow suit
        following = [
            card for card in hand
            if EuchreRules.get_effective_suit(card, trump) == led_suit
        ]
        
        # Must follow if possible, otherwise can play anything
        return following if following else hand.copy()
    
    @staticmethod
    def determine_trick_winner(
        trick: list[Card],
        lead_player: int,
        trump: Optional[Suit],
        trick_players: Optional[list[int]] = None
    ) -> int:
        """
        Determine which player won a trick.
        
        Args:
            trick: List of cards played in order
            lead_player: Player who led the trick
            trump: Trump suit (or None)
            trick_players: Optional list of player indices who played each card.
                          If not provided, assumes consecutive players from lead_player.
                          MUST be provided when going alone (3 players).
        
        Returns the player index (0-3) of the winner.
        """
        if not trick:
            raise ValueError("Cannot determine winner of empty trick")
        
        led_suit = EuchreRules.get_effective_suit(trick[0], trump)
        
        # If trick_players not provided, assume consecutive
        if trick_players is None:
            trick_players = [(lead_player + i) % 4 for i in range(len(trick))]
        
        best_player = trick_players[0]
        best_strength = EuchreRules.get_card_strength(trick[0], trump, led_suit)
        
        for i, card in enumerate(trick[1:], start=1):
            player = trick_players[i]
            strength = EuchreRules.get_card_strength(card, trump, led_suit)
            
            if strength > best_strength:
                best_strength = strength
                best_player = player
        
        return best_player
    
    @staticmethod
    def get_calling_options(
        state: GameState,
        player: int
    ) -> list[str]:
        """
        Get legal calling options for a player.
        
        Returns list of action strings like "pass", "order_up", "call_hearts", etc.
        """
        options = ["pass"]
        
        if state.phase == GamePhase.CALLING_ROUND_1:
            # Can order up the turned card
            options.append("order_up")
            options.append("order_up_alone")
            
        elif state.phase == GamePhase.CALLING_ROUND_2:
            # Can call any suit except the turned card's suit
            turned_suit = state.turned_card.suit if state.turned_card else None
            for suit in Suit:
                if suit != turned_suit:
                    options.append(f"call_{suit.name.lower()}")
                    options.append(f"call_{suit.name.lower()}_alone")
            
            # Dealer must call something (can't pass)
            if player == state.dealer:
                options.remove("pass")
        
        return options
    
    @staticmethod
    def calculate_round_score(
        tricks_won: list[int],
        caller: int,
        going_alone: bool
    ) -> tuple[int, int]:
        """
        Calculate points scored this round.
        
        Returns (team0_points, team1_points).
        """
        calling_team = caller % 2
        defending_team = 1 - calling_team
        
        caller_tricks = tricks_won[calling_team]
        defender_tricks = tricks_won[defending_team]
        
        team0_points = 0
        team1_points = 0
        
        if caller_tricks >= 3:
            # Calling team made it
            if caller_tricks == 5:
                # March (all 5 tricks)
                points = 4 if going_alone else 2
            else:
                # Made it (3-4 tricks)
                points = 1
            
            if calling_team == 0:
                team0_points = points
            else:
                team1_points = points
        else:
            # Euchred!
            points = 2
            if defending_team == 0:
                team0_points = points
            else:
                team1_points = points
        
        return team0_points, team1_points