"""
Arena for evaluating agents against each other.
"""

from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm

from game.engine import EuchreGame
from agents.base import BaseAgent


@dataclass
class MatchResult:
    """Result of a single game."""
    winner: int  # Team index (0 or 1)
    final_score: tuple[int, int]
    total_tricks: int
    rounds_played: int


@dataclass
class TournamentResult:
    """Results of a tournament (multiple games)."""
    num_games: int
    team0_wins: int
    team1_wins: int
    avg_score_team0: float
    avg_score_team1: float
    avg_rounds: float
    
    @property
    def team0_win_rate(self) -> float:
        return self.team0_wins / self.num_games if self.num_games > 0 else 0
    
    @property
    def team1_win_rate(self) -> float:
        return self.team1_wins / self.num_games if self.num_games > 0 else 0


class Arena:
    """
    Arena for running games between agents.
    
    Team 0: Players 0 and 2
    Team 1: Players 1 and 3
    """
    
    def __init__(self):
        self.game = EuchreGame()
    
    def play_game(
        self,
        team0_agent: BaseAgent,
        team1_agent: BaseAgent,
    ) -> MatchResult:
        """
        Play a single game between two teams.
        
        Args:
            team0_agent: Agent for players 0 and 2
            team1_agent: Agent for players 1 and 3
        
        Returns:
            MatchResult with game outcome
        """
        observations = self.game.reset()
        
        # Notify agents of game start
        team0_agent.on_game_start()
        team1_agent.on_game_start()
        
        done = False
        rounds = 0
        total_tricks = 0
        
        while not done:
            player = self.game.state.current_player
            obs = observations[player]
            legal_actions = self.game.get_legal_actions()
            
            # Select agent based on team
            agent = team0_agent if player % 2 == 0 else team1_agent
            
            # Get action
            action = agent.select_action(obs, legal_actions)
            
            # Take step
            result = self.game.step(action)
            observations = result.observations
            done = result.done
            
            # Track rounds
            if result.info.get("trick_winner") is not None:
                total_tricks += 1
        
        final_score = tuple(self.game.state.score)
        winner = 0 if final_score[0] >= 10 else 1
        rounds = (total_tricks + 4) // 5  # Approximate
        
        # Notify agents of game end
        team0_agent.on_game_end(list(final_score), winner)
        team1_agent.on_game_end(list(final_score), winner)
        
        return MatchResult(
            winner=winner,
            final_score=final_score,
            total_tricks=total_tricks,
            rounds_played=rounds,
        )
    
    def run_tournament(
        self,
        team0_agent: BaseAgent,
        team1_agent: BaseAgent,
        num_games: int = 100,
        show_progress: bool = True,
    ) -> TournamentResult:
        """
        Run multiple games and aggregate results.
        
        Args:
            team0_agent: Agent for team 0
            team1_agent: Agent for team 1
            num_games: Number of games to play
            show_progress: Whether to show progress bar
        
        Returns:
            TournamentResult with aggregated statistics
        """
        results = []
        
        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc="Tournament")
        
        for _ in iterator:
            result = self.play_game(team0_agent, team1_agent)
            results.append(result)
        
        # Aggregate
        team0_wins = sum(1 for r in results if r.winner == 0)
        team1_wins = sum(1 for r in results if r.winner == 1)
        avg_score_team0 = sum(r.final_score[0] for r in results) / num_games
        avg_score_team1 = sum(r.final_score[1] for r in results) / num_games
        avg_rounds = sum(r.rounds_played for r in results) / num_games
        
        return TournamentResult(
            num_games=num_games,
            team0_wins=team0_wins,
            team1_wins=team1_wins,
            avg_score_team0=avg_score_team0,
            avg_score_team1=avg_score_team1,
            avg_rounds=avg_rounds,
        )
    
    def compare_agents(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_games: int = 200,
    ) -> dict:
        """
        Compare two agents by playing both sides.
        
        Plays num_games/2 with agent_a as team0, then num_games/2 with agent_b as team0.
        
        Returns dict with win rates from each agent's perspective.
        """
        half = num_games // 2
        
        # Agent A as team 0
        print(f"\n{agent_a.name} vs {agent_b.name} ({half} games)")
        result1 = self.run_tournament(agent_a, agent_b, half)
        
        # Agent B as team 0
        print(f"\n{agent_b.name} vs {agent_a.name} ({half} games)")
        result2 = self.run_tournament(agent_b, agent_a, half)
        
        # Combine (agent_a wins = team0 wins in result1 + team1 wins in result2)
        agent_a_wins = result1.team0_wins + result2.team1_wins
        agent_b_wins = result1.team1_wins + result2.team0_wins
        
        return {
            "agent_a": agent_a.name,
            "agent_b": agent_b.name,
            "agent_a_wins": agent_a_wins,
            "agent_b_wins": agent_b_wins,
            "agent_a_win_rate": agent_a_wins / num_games,
            "agent_b_win_rate": agent_b_wins / num_games,
            "total_games": num_games,
        }