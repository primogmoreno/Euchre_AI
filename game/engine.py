"""
Euchre game engine: main game loop and environment interface.
"""

from typing import Optional
from dataclasses import dataclass

from .cards import Card, Suit, Deck
from .state import GameState, GamePhase
from .rules import EuchreRules
import config


@dataclass
class StepResult:
    """Result of taking a step in the game."""
    observations: list[dict]    # One observation per player
    rewards: list[float]        # Reward for each player
    done: bool                  # Is the game over?
    info: dict                  # Additional info (for debugging)


class EuchreGame:
    """
    Euchre game environment.
    
    Provides a clean interface for RL training:
    - reset() starts a new game
    - step(action) applies an action and returns results
    - get_legal_actions() returns valid moves
    """
    
    def __init__(self):
        self.state: Optional[GameState] = None
        self.deck = Deck()
    
    def reset(self) -> list[dict]:
        """
        Start a new game.
        
        Returns observations for all 4 players.
        """
        self.state = GameState()
        self._deal_round()
        return [self.get_observation(p) for p in range(4)]
    
    def _deal_round(self) -> None:
        """Deal cards for a new round."""
        self.deck.reset()
        self.deck.shuffle()
        
        # Deal 5 cards to each player
        for i in range(4):
            player = (self.state.dealer + 1 + i) % 4
            self.state.hands[player] = self.deck.deal(5)
        
        # Remaining cards form kitty, turn one up
        self.state.kitty = self.deck.deal(3)
        self.state.turned_card = self.deck.deal(1)[0]
        
        # First to act is left of dealer
        self.state.current_player = (self.state.dealer + 1) % 4
        self.state.phase = GamePhase.CALLING_ROUND_1
        
        # Reset round state
        self.state.trump = None
        self.state.caller = None
        self.state.going_alone = False
        self.state.tricks_won = [0, 0]
        self.state.play_history = []
        self.state.current_trick = []
        self.state.lead_player = None
    
    def get_observation(self, player: int) -> dict:
        """
        Get the observation for a specific player.
        
        This is the partial information view - doesn't include other players' hands.
        """
        return {
            "my_hand": self.state.hands[player].copy(),
            "my_position": player,
            "dealer": self.state.dealer,
            "current_player": self.state.current_player,
            "phase": self.state.phase,
            "turned_card": self.state.turned_card,
            "trump": self.state.trump,
            "caller": self.state.caller,
            "going_alone": self.state.going_alone,
            "current_trick": self.state.current_trick.copy(),
            "lead_player": self.state.lead_player,
            "tricks_won": self.state.tricks_won.copy(),
            "score": self.state.score.copy(),
            "play_history": self.state.play_history.copy(),
        }
    
    def get_legal_actions(self) -> list:
        """
        Get legal actions for the current player.
        
        Returns a list of Card objects (for playing) or strings (for calling).
        """
        player = self.state.current_player
        phase = self.state.phase
        
        if phase in (GamePhase.CALLING_ROUND_1, GamePhase.CALLING_ROUND_2):
            return EuchreRules.get_calling_options(self.state, player)
        
        elif phase == GamePhase.DISCARD:
            # Dealer must discard one card (now has 6)
            return self.state.hands[player].copy()
        
        elif phase == GamePhase.PLAYING:
            lead_card = self.state.current_trick[0] if self.state.current_trick else None
            return EuchreRules.get_legal_plays(
                self.state.hands[player],
                lead_card,
                self.state.trump
            )
        
        return []
    
    def step(self, action) -> StepResult:
        """
        Apply an action and advance the game.
        
        Action can be:
        - A Card object (for playing or discarding)
        - A string like "pass", "order_up", "call_hearts", etc.
        
        Returns StepResult with observations, rewards, done flag, and info.
        """
        player = self.state.current_player
        phase = self.state.phase
        rewards = [0.0, 0.0, 0.0, 0.0]
        info = {"action": action, "player": player}
        
        if phase in (GamePhase.CALLING_ROUND_1, GamePhase.CALLING_ROUND_2):
            # Reward shaping: small immediate reward/penalty for calling decisions
            # This helps the model learn faster that calling with good hands is good
            if action != "pass" and "order_up" in action or action.startswith("call_"):
                # Count trump in hand to assess call quality
                hand = self.state.hands[player]
                
                # Determine what trump would be
                if "order_up" in action:
                    trump_suit = self.state.turned_card.suit if self.state.turned_card else None
                else:
                    # Parse from action like "call_hearts"
                    parts = action.split("_")
                    trump_suit = Suit[parts[1].upper()] if len(parts) > 1 else None
                
                if trump_suit:
                    trump_count = sum(1 for card in hand 
                                    if EuchreRules.get_effective_suit(card, trump_suit) == trump_suit)
                    
                    # Small reward for calling with 3+ trump, penalty for 0-1 trump
                    if trump_count >= 3:
                        rewards[player] += 0.1  # Good call
                    elif trump_count <= 1:
                        rewards[player] -= 0.05  # Risky call
            
            self._handle_calling(action)
        
        elif phase == GamePhase.DISCARD:
            self._handle_discard(action)
        
        elif phase == GamePhase.PLAYING:
            trick_complete = self._handle_play(action)
            
            if trick_complete:
                # Get the players who played in this trick from play_history
                num_cards = len(self.state.current_trick)
                recent_plays = self.state.play_history[-num_cards:]
                trick_players = [p for p, c in recent_plays]
                
                winner = EuchreRules.determine_trick_winner(
                    self.state.current_trick,
                    self.state.lead_player,
                    self.state.trump,
                    trick_players
                )
                winning_team = winner % 2
                self.state.tricks_won[winning_team] += 1
                
                info["trick_winner"] = winner
                
                # Check if round is over
                if sum(self.state.tricks_won) == 5:
                    rewards = self._end_round()
                else:
                    # Winner leads next trick
                    self.state.current_trick = []
                    self.state.lead_player = winner
                    self.state.current_player = winner
                    self._skip_sitting_out_player()
        
        done = self.state.phase == GamePhase.GAME_OVER
        observations = [self.get_observation(p) for p in range(4)]
        
        return StepResult(observations, rewards, done, info)
    
    def _handle_calling(self, action: str) -> None:
        """Handle a calling phase action."""
        player = self.state.current_player
        
        if action == "pass":
            # Move to next player
            next_player = (player + 1) % 4
            
            if self.state.phase == GamePhase.CALLING_ROUND_1:
                if next_player == self.state.dealer:
                    # Everyone passed, dealer's turn
                    self.state.current_player = next_player
                elif player == self.state.dealer:
                    # Dealer passed, move to round 2
                    # Turned card goes face-down into kitty
                    if self.state.turned_card:
                        self.state.kitty.append(self.state.turned_card)
                        self.state.turned_card = None
                    self.state.phase = GamePhase.CALLING_ROUND_2
                    self.state.current_player = (self.state.dealer + 1) % 4
                else:
                    self.state.current_player = next_player
            else:
                # Round 2 - dealer can't pass, handled elsewhere
                self.state.current_player = next_player
        
        elif action.startswith("order_up"):
            # Trump is the turned card's suit
            self.state.trump = self.state.turned_card.suit
            self.state.caller = player
            self.state.going_alone = "alone" in action
            
            # Dealer picks up the turned card
            self.state.hands[self.state.dealer].append(self.state.turned_card)
            self.state.turned_card = None
            
            # Dealer must discard
            self.state.phase = GamePhase.DISCARD
            self.state.current_player = self.state.dealer
        
        elif action.startswith("call_"):
            # Parse suit from action string
            parts = action.split("_")
            suit_name = parts[1].upper()
            self.state.trump = Suit[suit_name]
            self.state.caller = player
            self.state.going_alone = "alone" in action
            
            # Start playing
            self._start_playing()
    
    def _handle_discard(self, card: Card) -> None:
        """Handle dealer's discard after picking up."""
        self.state.hands[self.state.dealer].remove(card)
        self.state.kitty.append(card)
        self._start_playing()
    
    def _start_playing(self) -> None:
        """Transition to the playing phase."""
        self.state.phase = GamePhase.PLAYING
        self.state.current_trick = []
        self.state.lead_player = (self.state.dealer + 1) % 4
        self.state.current_player = self.state.lead_player
        self._skip_sitting_out_player()
    
    def _handle_play(self, card: Card) -> bool:
        """
        Handle playing a card.
        
        Returns True if the trick is complete.
        """
        player = self.state.current_player
        
        # Remove card from hand
        self.state.hands[player].remove(card)
        
        # Add to current trick
        self.state.current_trick.append(card)
        self.state.play_history.append((player, card))
        
        # Check if trick is complete
        expected_cards = 3 if self.state.going_alone else 4
        if len(self.state.current_trick) == expected_cards:
            return True
        
        # Move to next player
        self.state.current_player = (player + 1) % 4
        self._skip_sitting_out_player()
        
        return False
    
    def _skip_sitting_out_player(self) -> None:
        """Skip the partner of someone going alone."""
        if self.state.going_alone:
            partner_of_caller = (self.state.caller + 2) % 4
            if self.state.current_player == partner_of_caller:
                self.state.current_player = (self.state.current_player + 1) % 4
    
    def _end_round(self) -> list[float]:
        """
        End the round and calculate rewards.
        
        Returns rewards for each player.
        """
        team0_pts, team1_pts = EuchreRules.calculate_round_score(
            self.state.tricks_won,
            self.state.caller,
            self.state.going_alone
        )
        
        self.state.score[0] += team0_pts
        self.state.score[1] += team1_pts
        
        # Determine which team called
        caller_team = self.state.caller % 2 if self.state.caller is not None else None
        
        # Rewards: positive for winning team, negative for losing
        rewards = [0.0, 0.0, 0.0, 0.0]
        for p in range(4):
            team = p % 2
            if team == 0:
                base_reward = float(team0_pts - team1_pts)
            else:
                base_reward = float(team1_pts - team0_pts)
            
            rewards[p] = base_reward
            
            # BONUS: Reward defenders who successfully euchred the caller
            # This teaches that NOT calling with a bad hand is valuable
            # because you can euchre opponents who call badly
            if caller_team is not None and team != caller_team:
                if (caller_team == 0 and team1_pts > 0) or (caller_team == 1 and team0_pts > 0):
                    # We euchred them! Extra reward for good defense
                    rewards[p] += 0.5
            
            # PENALTY: If you were dealer and forced to call with garbage, small penalty
            # This encourages teammates to call earlier with good hands
            # rather than forcing dealer into bad spots
            if caller_team is not None and p == self.state.dealer:
                # Check if this was a forced call (dealer in round 2)
                # We can't easily detect this here, so skip for now
                pass
        
        # Check for game over
        if self.state.score[0] >= config.WINNING_SCORE:
            self.state.phase = GamePhase.GAME_OVER
            # Bonus for winning the game
            for p in [0, 2]:
                rewards[p] += 5.0
            for p in [1, 3]:
                rewards[p] -= 5.0
        elif self.state.score[1] >= config.WINNING_SCORE:
            self.state.phase = GamePhase.GAME_OVER
            for p in [0, 2]:
                rewards[p] -= 5.0
            for p in [1, 3]:
                rewards[p] += 5.0
        else:
            # Start next round
            self.state.phase = GamePhase.ROUND_OVER
            self.state.dealer = (self.state.dealer + 1) % 4
            self._deal_round()
        
        return rewards
    
    def copy(self) -> "EuchreGame":
        """Create a copy of the game (for MCTS simulations)."""
        new_game = EuchreGame()
        new_game.state = self.state.copy()
        return new_game