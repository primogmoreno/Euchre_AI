"""
Rule-based agent using common Euchre heuristics.

Provides a stronger baseline than random play.
"""

import random
from typing import Any, Optional

from .base import BaseAgent
from game.cards import Card, Suit, Rank
from game.state import GamePhase
from game.rules import EuchreRules


class RuleBasedAgent(BaseAgent):
    """
    Agent that uses simple Euchre heuristics.
    
    Calling strategy:
    - Order up / call with 2+ trump including a bower
    - Or with 3+ trump
    
    Playing strategy:
    - Lead trump to pull out opponent's trump
    - Play high to win, low to lose
    - Follow partner's lead
    """
    
    def __init__(self, name: str = "RuleBased"):
        super().__init__(name)
    
    def select_action(self, observation: dict, legal_actions: list) -> Any:
        """Select action using heuristics."""
        phase = observation["phase"]
        
        if phase in (GamePhase.CALLING_ROUND_1, GamePhase.CALLING_ROUND_2):
            return self._decide_call(observation, legal_actions)
        
        elif phase == GamePhase.DISCARD:
            return self._decide_discard(observation, legal_actions)
        
        elif phase == GamePhase.PLAYING:
            return self._decide_play(observation, legal_actions)
        
        return random.choice(legal_actions)
    
    def _count_trump(self, hand: list[Card], trump: Suit) -> int:
        """Count trump cards in hand."""
        count = 0
        for card in hand:
            if EuchreRules.get_effective_suit(card, trump) == trump:
                count += 1
        return count
    
    def _has_bower(self, hand: list[Card], trump: Suit) -> bool:
        """Check if hand has right or left bower."""
        for card in hand:
            if card.rank == Rank.JACK:
                if card.suit == trump or card.suit == trump.same_color_suit:
                    return True
        return False
    
    def _decide_call(self, observation: dict, legal_actions: list) -> str:
        """Decide whether to call trump."""
        hand = observation["my_hand"]
        phase = observation["phase"]
        my_position = observation["my_position"]
        dealer = observation["dealer"]
        
        if phase == GamePhase.CALLING_ROUND_1:
            # Consider ordering up the turned card
            trump_suit = observation["turned_card"].suit
            trump_count = self._count_trump(hand, trump_suit)
            has_bower = self._has_bower(hand, trump_suit)
            
            # If dealer is my partner, be more aggressive
            is_partner_dealer = (dealer == (my_position + 2) % 4)
            
            should_call = (
                (trump_count >= 3) or
                (trump_count >= 2 and has_bower) or
                (is_partner_dealer and trump_count >= 2)
            )
            
            if should_call and "order_up" in legal_actions:
                return "order_up"
            return "pass"
        
        else:  # CALLING_ROUND_2
            # Find best suit to call
            turned_suit = observation["turned_card"].suit if observation["turned_card"] else None
            
            best_suit = None
            best_count = 0
            
            for suit in Suit:
                if suit == turned_suit:
                    continue  # Can't call turned suit in round 2
                    
                count = self._count_trump(hand, suit)
                has_bower = self._has_bower(hand, suit)
                score = count + (1 if has_bower else 0)
                
                if score > best_count:
                    best_count = score
                    best_suit = suit
            
            # Call if we have decent trump, or if we're dealer (must call)
            is_dealer = (my_position == dealer)
            
            if best_suit and (best_count >= 3 or is_dealer):
                action = f"call_{best_suit.name.lower()}"
                if action in legal_actions:
                    return action
            
            return "pass" if "pass" in legal_actions else legal_actions[0]
    
    def _decide_discard(self, observation: dict, legal_actions: list[Card]) -> Card:
        """Decide which card to discard after picking up."""
        trump = observation["trump"]
        
        # Discard lowest non-trump, or lowest trump if all trump
        non_trump = [
            c for c in legal_actions
            if EuchreRules.get_effective_suit(c, trump) != trump
        ]
        
        if non_trump:
            # Discard lowest non-trump
            return min(non_trump, key=lambda c: c.rank.value)
        else:
            # All trump, discard lowest
            return min(legal_actions, key=lambda c: 
                EuchreRules.get_card_strength(c, trump, trump))
    
    def _decide_play(self, observation: dict, legal_actions: list[Card]) -> Card:
        """Decide which card to play."""
        trump = observation["trump"]
        current_trick = observation["current_trick"]
        lead_player = observation["lead_player"]
        my_position = observation["my_position"]
        
        if not current_trick:
            # We're leading
            return self._decide_lead(observation, legal_actions)
        
        # We're following
        led_card = current_trick[0]
        led_suit = EuchreRules.get_effective_suit(led_card, trump)
        
        # Determine current winning card/player
        winning_idx = 0
        winning_strength = EuchreRules.get_card_strength(current_trick[0], trump, led_suit)
        
        for i, card in enumerate(current_trick[1:], 1):
            strength = EuchreRules.get_card_strength(card, trump, led_suit)
            if strength > winning_strength:
                winning_strength = strength
                winning_idx = i
        
        winning_player = (lead_player + winning_idx) % 4
        partner_winning = (winning_player == (my_position + 2) % 4)
        
        if partner_winning:
            # Partner is winning, play low
            return min(legal_actions, key=lambda c: 
                EuchreRules.get_card_strength(c, trump, led_suit))
        else:
            # Try to win the trick
            winning_cards = [
                c for c in legal_actions
                if EuchreRules.get_card_strength(c, trump, led_suit) > winning_strength
            ]
            
            if winning_cards:
                # Play lowest winning card
                return min(winning_cards, key=lambda c:
                    EuchreRules.get_card_strength(c, trump, led_suit))
            else:
                # Can't win, play lowest
                return min(legal_actions, key=lambda c:
                    EuchreRules.get_card_strength(c, trump, led_suit))
    
    def _decide_lead(self, observation: dict, legal_actions: list[Card]) -> Card:
        """Decide which card to lead."""
        trump = observation["trump"]
        my_position = observation["my_position"]
        caller = observation["caller"]
        
        # If we called, lead trump to pull out theirs
        if caller is not None and (caller % 2) == (my_position % 2):
            trump_cards = [
                c for c in legal_actions
                if EuchreRules.get_effective_suit(c, trump) == trump
            ]
            if trump_cards:
                # Lead highest trump
                return max(trump_cards, key=lambda c:
                    EuchreRules.get_card_strength(c, trump, trump))
        
        # Otherwise lead highest off-suit ace, or just highest card
        aces = [c for c in legal_actions if c.rank == Rank.ACE 
                and EuchreRules.get_effective_suit(c, trump) != trump]
        
        if aces:
            return aces[0]
        
        # Lead highest card
        return max(legal_actions, key=lambda c:
            EuchreRules.get_card_strength(c, trump, trump))