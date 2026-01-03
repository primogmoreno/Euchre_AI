"""
Collectors for specific decision types.

Each collector knows how to extract relevant information
for its decision type from the game state.
"""

from typing import Optional
from game.state import GameState, GamePhase
from game.cards import Card, Suit


def card_to_str(card: Card) -> str:
    """Convert card to readable string like 'J♠'."""
    rank_symbols = {9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
    suit_symbols = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    return f"{rank_symbols[card.rank.value]}{suit_symbols[card.suit.value]}"


def hand_to_list(hand: list[Card]) -> list[str]:
    """Convert hand to list of readable strings."""
    return [card_to_str(c) for c in hand]


def suit_to_str(suit: Optional[Suit]) -> Optional[str]:
    """Convert suit to string."""
    if suit is None:
        return None
    return suit.name


class BaseCollector:
    """Base class for decision collectors."""
    
    def __init__(self, logger):
        self.logger = logger
        self.pending = {}  # Track decisions awaiting outcomes
    
    def _base_record(self, state: GameState, player: int, round_key) -> dict:
        """Create base record with common fields."""
        return {
            "episode": round_key,  # Keep as "episode" for backwards compatibility
            "player": player,
            "dealer": state.dealer,
            "score_before": state.score.copy(),
            "hand": hand_to_list(state.hands[player]),
            "all_hands": [hand_to_list(h) for h in state.hands],
            "kitty": hand_to_list(state.kitty),
            "turned_card": card_to_str(state.turned_card) if state.turned_card else None,
        }


class GoingAloneCollector(BaseCollector):
    """
    Collects data about going alone decisions.
    
    Records:
    - Hand composition
    - Trump suit
    - Whether ordering up or calling
    - Position relative to dealer
    - Calling round (1 or 2)
    - Outcome (tricks won, points earned, euchred?)
    """
    
    def record_decision(
        self,
        state: GameState,
        player: int,
        action: str,
        round_key,  # Can be int or string
    ):
        """Record when a player decides to go alone."""
        
        record = self._base_record(state, player, round_key)
        
        # Determine trump and calling round from action
        if "order_up" in action:
            trump = state.turned_card.suit if state.turned_card else None
            ordered_up = True
            calling_round = 1
            # When ordering up, the turned card goes to dealer
            # We don't know what dealer will discard yet - will be added later
            buried_card = None  # Will be filled in by instrumented_runner
        else:
            # Parse suit from action like "call_spades_alone"
            parts = action.split("_")
            trump = Suit[parts[1].upper()] if len(parts) > 1 else None
            ordered_up = False
            calling_round = 2
            # In round 2, the turned card was already buried (it's the 4th kitty card)
            # The kitty should already have 4 cards by the time round 2 happens
            buried_card = card_to_str(state.kitty[-1]) if len(state.kitty) >= 4 else None
        
        record.update({
            "trump": suit_to_str(trump),
            "ordered_up": ordered_up,
            "calling_round": calling_round,
            "buried_card": buried_card,
            "caller_position": player,
            "position_relative_to_dealer": (player - state.dealer) % 4,
        })
        
        # Store pending - will complete when round ends
        self.pending[round_key] = record
    
    def record_outcome(
        self,
        state: GameState,
        round_key,
    ):
        """Record the outcome after the round completes."""
        
        if round_key not in self.pending:
            return
        
        record = self.pending.pop(round_key)
        
        caller = record["player"]
        caller_team = caller % 2
        
        tricks = state.tricks_won[caller_team]
        opponent_tricks = state.tricks_won[1 - caller_team]
        
        # Calculate points earned
        if tricks == 5:
            points = 4  # March alone
        elif tricks >= 3:
            points = 1  # Made it (but alone should still get more usually)
        else:
            points = 0  # Euchred
            
        was_euchred = tricks < 3
        
        record.update({
            "tricks_won": tricks,
            "tricks_lost": opponent_tricks,
            "points_earned": points,
            "was_euchred": was_euchred,
            "success": tricks >= 3,
            "march": tricks == 5,
        })
        
        self.logger.log_going_alone(record)


class TrumpCallCollector(BaseCollector):
    """
    Collects data about trump calling decisions (not going alone).
    
    Records:
    - Hand composition
    - Trump suit chosen
    - Whether ordering up vs calling in round 2
    - Outcome
    """
    
    def record_decision(
        self,
        state: GameState,
        player: int,
        action: str,
        round_key,  # Can be int or string
    ):
        """Record when a player calls trump (without going alone)."""
        
        record = self._base_record(state, player, round_key)
        
        if "order_up" in action:
            trump = state.turned_card.suit if state.turned_card else None
            ordered_up = True
            round_num = 1
            # When ordering up, we don't know what dealer will discard yet
            buried_card = None  # Will be filled in by instrumented_runner
        else:
            parts = action.split("_")
            trump = Suit[parts[1].upper()] if len(parts) > 1 else None
            ordered_up = False
            round_num = 2
            # In round 2, the turned card was already buried
            buried_card = card_to_str(state.kitty[-1]) if len(state.kitty) >= 4 else None
        
        record.update({
            "trump": suit_to_str(trump),
            "ordered_up": ordered_up,
            "calling_round": round_num,
            "buried_card": buried_card,
            "caller_position": player,
            "position_relative_to_dealer": (player - state.dealer) % 4,
            "is_dealer": player == state.dealer,
            "is_dealer_partner": player == (state.dealer + 2) % 4,
        })
        
        self.pending[round_key] = record
    
    def record_outcome(
        self,
        state: GameState,
        round_key,
    ):
        """Record the outcome after the round completes."""
        
        if round_key not in self.pending:
            return
        
        record = self.pending.pop(round_key)
        
        caller = record["player"]
        caller_team = caller % 2
        
        tricks = state.tricks_won[caller_team]
        opponent_tricks = state.tricks_won[1 - caller_team]
        
        if tricks == 5:
            points = 2  # March
        elif tricks >= 3:
            points = 1  # Made it
        else:
            points = 0  # Euchred (opponents get 2)
        
        record.update({
            "tricks_won": tricks,
            "tricks_lost": opponent_tricks,
            "points_earned": points,
            "was_euchred": tricks < 3,
            "success": tricks >= 3,
            "march": tricks == 5,
        })
        
        self.logger.log_trump_call(record)


class PassCollector(BaseCollector):
    """
    Collects data about pass decisions.
    
    Useful for understanding when NOT to call.
    """
    
    def record_decision(
        self,
        state: GameState,
        player: int,
        round_key,  # Can be int or string
    ):
        """Record when a player passes."""
        
        record = self._base_record(state, player, round_key)
        
        record.update({
            "phase": state.phase.name,
            "calling_round": 1 if state.phase == GamePhase.CALLING_ROUND_1 else 2,
            "position_relative_to_dealer": (player - state.dealer) % 4,
            "turned_suit": suit_to_str(state.turned_card.suit) if state.turned_card else None,
        })
        
        # Count potential trump for each suit
        hand = state.hands[player]
        trump_counts = {}
        for suit in Suit:
            count = sum(1 for c in hand if c.suit == suit)
            # Add 1 if we have the left bower
            left_bower_suit = suit.same_color_suit
            if any(c.rank.value == 11 and c.suit == left_bower_suit for c in hand):
                count += 1
            trump_counts[suit.name] = count
        
        record["trump_counts_by_suit"] = trump_counts
        
        # Log immediately (no outcome tracking for passes)
        self.logger.log_pass(record)


class PlayCollector(BaseCollector):
    """
    Collects data about card play decisions.
    
    Can filter for interesting plays:
    - Leading decisions
    - Trump plays
    - Winning/losing trick plays
    """
    
    def __init__(self, logger, only_interesting: bool = True):
        super().__init__(logger)
        self.only_interesting = only_interesting
        self.trick_pending = {}
    
    def record_decision(
        self,
        state: GameState,
        player: int,
        card: Card,
        episode: int,
    ):
        """Record a card play."""
        
        is_lead = len(state.current_trick) == 0
        
        # Filter for interesting plays if enabled
        if self.only_interesting:
            is_trump = (
                state.trump and 
                (card.suit == state.trump or 
                 (card.rank.value == 11 and card.suit == state.trump.same_color_suit))
            )
            # Only log leads, trump plays, or when we'll track outcomes
            if not (is_lead or is_trump):
                return
        
        record = self._base_record(state, player, episode)
        
        record.update({
            "card_played": card_to_str(card),
            "is_lead": is_lead,
            "current_trick_before": hand_to_list(state.current_trick),
            "trump": suit_to_str(state.trump),
            "tricks_won_before": state.tricks_won.copy(),
            "caller": state.caller,
            "caller_team": state.caller % 2 if state.caller is not None else None,
            "player_team": player % 2,
            "is_caller_team": (state.caller is not None and player % 2 == state.caller % 2),
        })
        
        # Store to track trick outcome
        key = (episode, state.tricks_won[0] + state.tricks_won[1])
        self.trick_pending[key] = record
    
    def record_trick_outcome(
        self,
        state: GameState,
        episode: int,
        trick_num: int,
        winner: int,
    ):
        """Record who won the trick."""
        
        key = (episode, trick_num)
        if key not in self.trick_pending:
            return
        
        record = self.trick_pending.pop(key)
        
        player = record["player"]
        player_team = player % 2
        winner_team = winner % 2
        
        record.update({
            "trick_winner": winner,
            "won_trick": winner_team == player_team,
            "player_won": winner == player,
        })
        
        self.logger.log_play(record)