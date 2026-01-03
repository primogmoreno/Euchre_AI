"""
Configuration and hyperparameters for Euchre AI.
"""

# =============================================================================
# Game Constants
# =============================================================================

NUM_PLAYERS = 4
TEAM_0 = [0, 2]  # Players 0 and 2 are partners
TEAM_1 = [1, 3]  # Players 1 and 3 are partners
CARDS_PER_HAND = 5
TRICKS_PER_ROUND = 5

# Points for different outcomes
POINTS_MAKERS_3_4 = 1    # Calling team wins 3-4 tricks
POINTS_MAKERS_5 = 2      # Calling team wins all 5 (march)
POINTS_EUCHRE = 2        # Defending team wins 3+ tricks
POINTS_ALONE_5 = 4       # Caller goes alone and wins all 5
WINNING_SCORE = 10

# =============================================================================
# Neural Network
# =============================================================================

STATE_SIZE = 176         # Size of encoded state vector
ACTION_SIZE = 35         # Max possible actions (24 cards + 11 calling options)
HIDDEN_SIZE = 256        # Hidden layer size
POLICY_HIDDEN = 128      # Policy head hidden size
VALUE_HIDDEN = 128       # Value head hidden size

# =============================================================================
# Training Hyperparameters
# =============================================================================

LEARNING_RATE = 3e-4
FINETUNE_LEARNING_RATE = 1e-5  # Much smaller for fine-tuning pre-trained models
GAMMA = 0.99             # Discount factor
GAE_LAMBDA = 0.95        # GAE parameter
CLIP_EPSILON = 0.2       # PPO clipping
ENTROPY_COEF = 0.01      # Entropy bonus for exploration
VALUE_COEF = 0.5         # Value loss weight
MAX_GRAD_NORM = 0.5      # Gradient clipping

BATCH_SIZE = 64
NUM_EPOCHS = 4           # PPO epochs per update
NUM_EPISODES = 100000    # Total training episodes
CHECKPOINT_INTERVAL = 5000

# =============================================================================
# Paths
# =============================================================================

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"