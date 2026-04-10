"""
Flask web UI for playing Euchre against AI agents.
Run from project root: python ui/app.py
"""

import sys
import os
import uuid
import glob as glob_module
from dataclasses import dataclass, field

# Add project root to path so we can import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flask import Flask, session, jsonify, request, render_template

from game.engine import EuchreGame
from game.state import GamePhase
from game.cards import Card, Suit, Rank
from agents.neural_agent import NeuralAgent
from agents.rule_based import RuleBasedAgent
from agents.random_agent import RandomAgent
from model.network import EuchreNetwork

app = Flask(__name__)
app.secret_key = "euchre-ui-secret-key-2024"

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

game_sessions: dict[str, "GameSession"] = {}


@dataclass
class GameSession:
    game: EuchreGame
    agents: list  # index 0 = None (human), 1-3 = AI agents
    agent_names: list[str]
    event_log: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Available checkpoints / agent configs
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

CHECKPOINT_MAP = {
    "final_model":         ("final_model.pth",          "Final Model"),
    "latest":              ("latest.pth",                "Latest"),
    "pretrained":          ("pretrained_imitation.pth",  "Pretrained (Imitation)"),
    "ep210000":            ("model_ep210000.pth",         "Episode 210k"),
    "ep105000":            ("model_ep105000.pth",         "Episode 105k"),
    "ep90000":             ("model_ep90000.pth",          "Episode 90k"),
    "ep75000":             ("model_ep75000.pth",          "Episode 75k"),
    "ep60000":             ("model_ep60000.pth",          "Episode 60k"),
    "ep45000":             ("model_ep45000.pth",          "Episode 45k"),
    "ep30000":             ("model_ep30000.pth",          "Episode 30k"),
    "ep15000":             ("model_ep15000.pth",          "Episode 15k"),
}


def get_available_checkpoints() -> list[dict]:
    """Return checkpoints that actually exist on disk."""
    available = []
    for ckpt_id, (filename, label) in CHECKPOINT_MAP.items():
        path = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(path):
            available.append({"id": ckpt_id, "label": label})
    return available


def load_network(path: str) -> EuchreNetwork:
    """Load network weights, handling both full checkpoints and bare state_dicts."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = EuchreNetwork()
    if isinstance(checkpoint, dict) and "network" in checkpoint:
        net.load_state_dict(checkpoint["network"])
    else:
        net.load_state_dict(checkpoint)
    net.eval()
    return net


def build_agent(cfg: dict, player_idx: int):
    """Build an agent from a config dict sent by the frontend."""
    agent_type = cfg.get("type", "rule_based")

    if agent_type == "neural":
        ckpt_id = cfg.get("checkpoint", "final_model")
        filename, label = CHECKPOINT_MAP.get(ckpt_id, ("final_model.pth", "Neural"))
        path = os.path.join(CHECKPOINT_DIR, filename)
        if not os.path.exists(path):
            # Fall back to rule-based if file is missing
            return RuleBasedAgent(), f"Rule-Based (fallback)"
        net = load_network(path)
        return NeuralAgent(network=net, greedy=True, name=f"Neural ({label})"), f"Neural ({label})"

    elif agent_type == "rule_based":
        return RuleBasedAgent(), "Rule-Based"

    else:
        return RandomAgent(), "Random"


# ---------------------------------------------------------------------------
# Card / action serialization helpers
# ---------------------------------------------------------------------------

RANK_DISPLAY = {
    Rank.NINE: "9", Rank.TEN: "10", Rank.JACK: "J",
    Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A",
}

SUIT_SYMBOL = {
    Suit.CLUBS: "♣", Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥", Suit.SPADES: "♠",
}


def card_to_dict(card: Card) -> dict:
    rank_str = RANK_DISPLAY[card.rank]
    suit_str = card.suit.name       # e.g. "SPADES"
    symbol   = SUIT_SYMBOL[card.suit]
    return {
        "rank":    rank_str,
        "suit":    suit_str,
        "index":   card.to_index(),
        "id":      rank_str + suit_str[0],   # e.g. "JS"
        "display": rank_str + symbol,         # e.g. "J♠"
        "color":   card.suit.color,           # "red" or "black"
    }


CALLING_ACTION_DISPLAY = {
    "pass":              "Pass",
    "order_up":          "Order Up",
    "order_up_alone":    "Order Up Alone",
    "call_clubs":        "Call Clubs ♣",
    "call_clubs_alone":  "Call Clubs Alone ♣",
    "call_diamonds":     "Call Diamonds ♦",
    "call_diamonds_alone": "Call Diamonds Alone ♦",
    "call_hearts":       "Call Hearts ♥",
    "call_hearts_alone": "Call Hearts Alone ♥",
    "call_spades":       "Call Spades ♠",
    "call_spades_alone": "Call Spades Alone ♠",
}


def action_to_json(action) -> dict:
    """Convert a legal action (Card or string) to a JSON-serializable dict."""
    if isinstance(action, Card):
        d = card_to_dict(action)
        d["type"] = "card"
        return d
    else:
        return {
            "type":    "call",
            "action":  action,
            "display": CALLING_ACTION_DISPLAY.get(action, action),
        }


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------

def serialize_state(session_obj: GameSession) -> dict:
    game = session_obj.game
    obs  = game.get_observation(0)
    state = game.state

    phase = obs["phase"]
    legal = game.get_legal_actions() if phase not in (GamePhase.GAME_OVER,) else []

    winner = None
    if phase == GamePhase.GAME_OVER:
        winner = "Your team" if state.score[0] >= 10 else "Opponents"

    # Determine card counts for AI hands (5 cards each, minus tricks played)
    tricks_total = sum(state.tricks_won)
    going_alone  = state.going_alone
    caller       = state.caller

    ai_hand_sizes = {}
    for p in range(1, 4):
        if going_alone and caller is not None and (caller + 2) % 4 == p:
            ai_hand_sizes[p] = 0   # sitting out
        else:
            ai_hand_sizes[p] = max(0, 5 - tricks_total)

    return {
        "phase":        phase.name,
        "current_player": state.current_player,
        "my_hand":      [card_to_dict(c) for c in obs["my_hand"]],
        "turned_card":  card_to_dict(obs["turned_card"]) if obs["turned_card"] else None,
        "trump":        obs["trump"].name if obs["trump"] else None,
        "dealer":       obs["dealer"],
        "caller":       obs["caller"],
        "going_alone":  obs["going_alone"],
        "score":        obs["score"],
        "tricks_won":   obs["tricks_won"],
        "current_trick": [
            {"card": card_to_dict(c), "player": p}
            for p, c in zip(
                _trick_players(state),
                state.current_trick,
            )
        ],
        "lead_player":  obs["lead_player"],
        "legal_actions": [action_to_json(a) for a in legal],
        "events":       list(session_obj.event_log),
        "player_names": session_obj.agent_names,
        "ai_hand_sizes": ai_hand_sizes,
        "done":         phase == GamePhase.GAME_OVER,
        "winner":       winner,
    }


def _trick_players(state) -> list[int]:
    """Reconstruct which players played the cards in the current trick."""
    if not state.current_trick:
        return []
    num = len(state.current_trick)
    players = []
    lead = state.lead_player
    if lead is None:
        return list(range(num))
    p = lead
    for _ in range(num):
        players.append(p)
        p = (p + 1) % 4
        if state.going_alone and state.caller is not None:
            partner = (state.caller + 2) % 4
            if p == partner:
                p = (p + 1) % 4
    return players


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def _fmt_action(action, player_name: str) -> str:
    """Format an AI action as a human-readable event string."""
    if isinstance(action, Card):
        return f"{player_name} played {action}"
    return f"{player_name} {CALLING_ACTION_DISPLAY.get(action, action).lower()}"


def _log_action(session_obj: GameSession, player: int, action, result, prev_score: list) -> None:
    name = session_obj.agent_names[player]
    session_obj.event_log.append(_fmt_action(action, name))

    # Trick result
    if "trick_winner" in result.info:
        winner_p = result.info["trick_winner"]
        wname = session_obj.agent_names[winner_p]
        session_obj.event_log.append(f"{wname} won the trick.")

    # Round result (score changed)
    new_score = session_obj.game.state.score
    if new_score[0] != prev_score[0] or new_score[1] != prev_score[1]:
        t0_delta = new_score[0] - prev_score[0]
        t1_delta = new_score[1] - prev_score[1]
        if t0_delta > 0:
            pts_word = "point" if t0_delta == 1 else "points"
            session_obj.event_log.append(
                f"--- Round over: Your team scored {t0_delta} {pts_word}. Score: {new_score[0]}-{new_score[1]} ---"
            )
        elif t1_delta > 0:
            pts_word = "point" if t1_delta == 1 else "points"
            session_obj.event_log.append(
                f"--- Round over: Opponents scored {t1_delta} {pts_word}. Score: {new_score[0]}-{new_score[1]} ---"
            )


# ---------------------------------------------------------------------------
# Auto-advance loop
# ---------------------------------------------------------------------------

def advance_until_human(session_obj: GameSession) -> None:
    """Step AI agents until it's player 0's turn (or the game is over)."""
    game = session_obj.game
    for _ in range(200):   # safety cap
        if game.state.phase == GamePhase.GAME_OVER:
            break
        if game.state.current_player == 0:
            break
        cp    = game.state.current_player
        obs   = game.get_observation(cp)
        legal = game.get_legal_actions()
        action = session_obj.agents[cp].select_action(obs, legal)
        prev_score = game.state.score.copy()
        result = game.step(action)
        _log_action(session_obj, cp, action, result, prev_score)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def get_or_create_sid() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    return jsonify({
        "checkpoints": get_available_checkpoints(),
        "agents": [
            {"id": "rule_based", "label": "Rule-Based"},
            {"id": "random",     "label": "Random"},
        ],
    })


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(force=True)

    # Build the four agent slots (index 0 = human)
    agents = [None]
    agent_names = ["You"]
    for p in range(1, 4):
        cfg = data.get(f"player{p}", {"type": "rule_based"})
        agent, name = build_agent(cfg, p)
        agents.append(agent)
        agent_names.append(name)

    # Create game session
    game = EuchreGame()
    game.reset()
    sess = GameSession(game=game, agents=agents, agent_names=agent_names)

    sid = get_or_create_sid()
    game_sessions[sid] = sess

    # Advance if the human isn't the first to act
    advance_until_human(sess)

    return jsonify(serialize_state(sess))


@app.route("/api/state")
def api_state():
    sid = session.get("sid")
    if sid not in game_sessions:
        return jsonify({"error": "No active game. Start a new game first."}), 404
    return jsonify(serialize_state(game_sessions[sid]))


@app.route("/api/action", methods=["POST"])
def api_action():
    sid = session.get("sid")
    if sid not in game_sessions:
        return jsonify({"error": "No active game"}), 404

    sess = game_sessions[sid]
    game = sess.game
    data = request.get_json(force=True)

    # Clear event log for this turn so the frontend only sees fresh events
    sess.event_log = []

    # Reconstruct the action
    action_type = data.get("action")
    if action_type == "card":
        card_index = data.get("card_index")
        if card_index is None:
            return jsonify({"error": "Missing card_index"}), 400
        action = Card.from_index(card_index)
    else:
        action = action_type   # e.g. "pass", "order_up", "call_hearts"

    # Validate
    legal = game.get_legal_actions()
    if action not in legal:
        return jsonify({
            "error": "Illegal action",
            "legal_actions": [action_to_json(a) for a in legal],
        }), 400

    # Apply human action and log it
    prev_score = game.state.score.copy()
    result = game.step(action)

    # Log the human's action
    if isinstance(action, Card):
        sess.event_log.append(f"You played {action}")
    else:
        sess.event_log.append(f"You: {CALLING_ACTION_DISPLAY.get(action, action)}")

    if "trick_winner" in result.info:
        winner_p = result.info["trick_winner"]
        wname = sess.agent_names[winner_p]
        if winner_p == 0:
            sess.event_log.append("You won the trick!")
        else:
            sess.event_log.append(f"{wname} won the trick.")

    new_score = game.state.score
    if new_score[0] != prev_score[0] or new_score[1] != prev_score[1]:
        t0_delta = new_score[0] - prev_score[0]
        t1_delta = new_score[1] - prev_score[1]
        if t0_delta > 0:
            pts_word = "point" if t0_delta == 1 else "points"
            sess.event_log.append(
                f"--- Round over: Your team scored {t0_delta} {pts_word}. Score: {new_score[0]}-{new_score[1]} ---"
            )
        elif t1_delta > 0:
            pts_word = "point" if t1_delta == 1 else "points"
            sess.event_log.append(
                f"--- Round over: Opponents scored {t1_delta} {pts_word}. Score: {new_score[0]}-{new_score[1]} ---"
            )

    # Now auto-advance AI turns
    advance_until_human(sess)

    return jsonify(serialize_state(sess))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Euchre UI at http://localhost:5000")
    app.run(debug=True, port=5000)
