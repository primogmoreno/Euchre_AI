# Euchre AI - Project Overview

Quick reference for understanding each component of the project.

---

## Project Structure at a Glance

```
euchre_ai/
├── config.py              # All settings in one place
├── game/                  # The Euchre simulator
├── agents/                # Different player types
├── model/                 # Neural network definition
├── training/              # How the AI learns
├── evaluation/            # Testing and comparison
├── scripts/               # Command-line entry points
└── checkpoints/           # Saved model weights (created during training)
```

---

## config.py

**Purpose:** Central location for all adjustable settings.

| Setting | Default | What It Controls |
|---------|---------|------------------|
| `NUM_EPISODES` | 100,000 | How many games to play during training |
| `LEARNING_RATE` | 0.0003 | How big each weight adjustment is |
| `BATCH_SIZE` | 64 | Games processed before each learning update |
| `GAMMA` | 0.99 | How much future rewards matter vs immediate |
| `CLIP_EPSILON` | 0.2 | Max allowed policy change per update (PPO) |
| `CHECKPOINT_INTERVAL` | 5,000 | Save model every N episodes |
| `STATE_SIZE` | 176 | Size of encoded game state vector |
| `ACTION_SIZE` | 30 | Number of possible actions |

**When to edit:** Tuning training performance or changing network architecture.

---

## game/ - The Simulator

The environment where games are played. Knows all Euchre rules but nothing about AI.

### cards.py
**Purpose:** Define what cards are.

- `Suit` enum: CLUBS, DIAMONDS, HEARTS, SPADES
- `Rank` enum: NINE through ACE
- `Card` class: Combines suit + rank, can convert to/from index (0-23)
- `Deck` class: Shuffles and deals cards

**Key function:** `card.to_index()` converts a card to a number for the neural network.

### state.py
**Purpose:** Track everything about a game in progress.

- `GamePhase` enum: CALLING_ROUND_1, CALLING_ROUND_2, DISCARD, PLAYING, etc.
- `GameState` dataclass: Holds hands, trump, score, tricks, current player, etc.

**Key insight:** Contains ALL information, including hidden cards. Players only see filtered "observations."

### rules.py
**Purpose:** Euchre game logic—what's legal, who wins tricks.

| Function | What It Does |
|----------|--------------|
| `get_effective_suit()` | Handles left bower counting as trump |
| `get_card_strength()` | Ranks cards for trick comparison |
| `get_legal_plays()` | Returns cards you can legally play |
| `determine_trick_winner()` | Figures out who won a trick |
| `calculate_round_score()` | Computes points after a round |

**Key insight:** The left bower logic lives here. Jack of same color as trump becomes trump suit.

### engine.py
**Purpose:** Main game controller. The interface everything else uses.

| Method | What It Does |
|--------|--------------|
| `reset()` | Start new game, deal cards, return observations |
| `step(action)` | Apply an action, advance game, return results |
| `get_legal_actions()` | What can current player do right now? |
| `get_observation(player)` | What can this player see? (hides opponents' cards) |

**Key insight:** This is the "environment" in RL terms. Training loop calls `reset()` and `step()` repeatedly.

---

## agents/ - Player Implementations

Different strategies for playing Euchre. All share the same interface.

### base.py
**Purpose:** Define what an "agent" must do.

```python
class BaseAgent:
    def select_action(self, observation, legal_actions) -> action:
        # Given what I can see and what's legal, pick something
```

**Key insight:** Any agent can play against any other agent because they share this interface.

### random_agent.py
**Purpose:** Weakest baseline—picks randomly from legal actions.

**Win rate:** ~15-25% vs rule-based. If your neural net can't beat this, something is broken.

### rule_based.py
**Purpose:** Hand-coded Euchre heuristics. Stronger baseline.

Strategies implemented:
- Call trump with 2+ trump including a bower, or 3+ trump
- Lead trump when you called to pull out opponents' trump
- Play high to win tricks, low when partner is winning
- Lead off-suit aces when not calling

**Win rate:** ~75-85% vs random. Your neural net should eventually beat this.

### neural_agent.py
**Purpose:** Uses the trained neural network to pick actions.

| Method | What It Does |
|--------|--------------|
| `select_action()` | Encode state → network → mask illegal → pick action |
| `get_action_probs()` | See probability distribution (for debugging) |
| `get_value()` | Get network's evaluation of current position |

**Key setting:** `greedy=True` always picks best action. `greedy=False` samples from distribution (for exploration during training).

---

## model/ - Neural Network

The brain that learns to play.

### network.py
**Purpose:** Define the neural network architecture.

```
State (176 floats)
       ↓
   [256 neurons] ← shared layers (feature extraction)
       ↓
   [256 neurons]
       ↓
   ┌───┴───┐
   ↓       ↓
[Policy] [Value]
   ↓       ↓
30 action  1 number
probs     (how good is
          this state?)
```

**Key methods:**
- `forward()` - Basic pass through network
- `get_action_and_value()` - Sample action + get training info
- `evaluate_actions()` - Recompute probabilities for PPO loss

### encoding.py
**Purpose:** Convert game state to numbers the network understands.

**What gets encoded (176 total values):**
| Component | Size | Description |
|-----------|------|-------------|
| Hand | 24 | Binary: which cards do I hold? |
| Played cards | 24 | Binary: which cards have been played? |
| Trump suit | 4 | One-hot: which suit is trump? |
| Turned card | 24 | One-hot: what was turned up? |
| Current trick | 72 | Up to 3 cards played this trick |
| Tricks won | 2 | Normalized count per team |
| Score | 2 | Normalized game score |
| Position info | 4 | Player positions and phase |
| Going alone | 1 | Binary flag |
| Lead player | 4 | One-hot: who led this trick? |
| Padding | 15 | Round to 176 |

**Key functions:**
- `encode_state(observation)` → numpy array
- `action_to_index(action)` → integer
- `create_action_mask(legal_actions)` → boolean array

---

## training/ - Learning Process

How the network improves through self-play.

### self_play.py
**Purpose:** Generate training data by playing games.

```
Network plays all 4 positions against itself
        ↓
Each step records: (state, action, reward, value, log_prob)
        ↓
Game ends → experiences returned for learning
```

**Key class:** `SelfPlayRunner` - runs episodes and collects experiences.

**Why self-play:** The network learns by playing against itself. As it improves, its opponent (itself) also improves, pushing it to find better strategies.

### ppo.py
**Purpose:** The learning algorithm (Proximal Policy Optimization).

**Core idea:** Don't change too much at once.

```python
ratio = new_probability / old_probability
# Clip between 0.8 and 1.2 — can't change more than 20% per update
```

**Loss components:**
| Component | What It Measures |
|-----------|------------------|
| Policy loss | Were actions better/worse than expected? |
| Value loss | How wrong were value predictions? |
| Entropy bonus | Encourage exploration (don't get stuck) |

**Key function:** `compute_gae()` - Generalized Advantage Estimation. Figures out "was this action better or worse than average?"

### trainer.py
**Purpose:** Orchestrates the full training loop.

```
1. Load checkpoint if resuming
2. Loop:
   a. Run self-play games → collect experiences
   b. Compute advantages and returns
   c. Run PPO updates for NUM_EPOCHS
   d. Log metrics to TensorBoard
   e. Save checkpoint periodically
3. Save final model
```

**Key methods:**
- `train()` - Main loop
- `_save_checkpoint()` - Persist model to disk
- `_load_checkpoint()` - Resume interrupted training

---

## evaluation/ - Testing

Measure how good agents are.

### arena.py
**Purpose:** Run games between any two agents.

| Method | What It Does |
|--------|--------------|
| `play_game(team0, team1)` | Single game, returns winner and score |
| `run_tournament(team0, team1, n)` | Play n games, return stats |
| `compare_agents(a, b, n)` | Fair comparison (each agent plays both sides) |

**Why both sides:** Dealer position matters in Euchre. Playing 100 games each way removes this bias.

### metrics.py
**Purpose:** Track statistics over time.

- `WinRateTracker` - Rolling window of recent win rate
- `MetricsLogger` - Save results to JSON for analysis

---

## scripts/ - Entry Points

What you actually run.

### train.py
```bash
python scripts/train.py
python scripts/train.py --episodes 50000
```
Starts or resumes training. Progress shown with tqdm bar. Checkpoints saved to `checkpoints/`.

### evaluate.py
```bash
python scripts/evaluate.py
python scripts/evaluate.py --model checkpoints/final_model.pth --games 500
```
Tests agents against each other. Without `--model`, just runs random vs rule-based baseline.

### play.py
```bash
python scripts/play.py
python scripts/play.py --model checkpoints/final_model.pth
```
Interactive terminal game. You play as Player 0, AI controls other positions.

---

## checkpoints/ - Saved Models

Created during training. Contains:

| File | Contents | Use |
|------|----------|-----|
| `latest.pth` | Weights + optimizer + episode count | Resume training |
| `model_ep5000.pth` | Just weights at episode 5000 | Historical snapshot |
| `final_model.pth` | Just weights after training | Deploy/play against |

**File size:** ~1-2 MB each (just 180,000 float numbers)

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING                                 │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Game    │───▶│  Self    │───▶│   PPO    │───▶│  Network │  │
│  │  Engine  │    │  Play    │    │   Loss   │    │  Update  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │         │
│       └───────────────────────────────────────────────┘         │
│                    (repeat millions of times)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         PLAYING                                  │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  Game    │───▶│  Encode  │───▶│  Network │───▶ Action        │
│  │  State   │    │  State   │    │  Forward │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Common Tasks

| I want to... | Go to... |
|--------------|----------|
| Change training speed/quality | `config.py` |
| Fix a Euchre rule bug | `game/rules.py` |
| Change what the network sees | `model/encoding.py` |
| Make the network bigger/smaller | `model/network.py` |
| Add a new agent strategy | `agents/` (copy `rule_based.py` as template) |
| Change the reward signal | `game/engine.py` → `_end_round()` |
| See training progress | `tensorboard --logdir logs/` |