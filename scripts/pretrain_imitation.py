#!/usr/bin/env python3
"""
Pre-train the neural network by imitating rule-based decisions.

This uses supervised learning (behavior cloning) to give the model
a good starting point before RL fine-tuning.

The idea:
1. Generate states from random game positions
2. Get rule-based agent's action for each state
3. Train neural network to predict those actions

Usage:
    python scripts/pretrain_imitation.py --episodes 50000
    python scripts/train_mixed.py --episodes 100000  # Then fine-tune with RL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

import config
from model.network import EuchreNetwork
from model.encoding import encode_state, action_to_index, create_action_mask
from game.engine import EuchreGame
from agents.rule_based import RuleBasedAgent


def collect_imitation_data(num_episodes: int, focus_on_calling: bool = True):
    """
    Collect state-action pairs from rule-based agent.
    
    Args:
        num_episodes: Number of games to play
        focus_on_calling: If True, oversample calling phase decisions
    
    Returns:
        states: List of encoded states
        actions: List of action indices
        masks: List of action masks
    """
    game = EuchreGame()
    rule_based = RuleBasedAgent()
    
    states = []
    actions = []
    masks = []
    
    calling_states = []
    calling_actions = []
    calling_masks = []
    
    for _ in tqdm(range(num_episodes), desc="Collecting data"):
        observations = game.reset()
        done = False
        
        while not done:
            player = game.state.current_player
            obs = observations[player]
            legal_actions = game.get_legal_actions()
            
            # Get rule-based decision
            action = rule_based.select_action(obs, legal_actions)
            
            # Encode state
            state = encode_state(obs)
            action_idx = action_to_index(action)
            mask = create_action_mask(legal_actions)
            
            # Store the experience
            is_calling = obs["phase"].name.startswith("CALLING")
            
            if is_calling:
                calling_states.append(state)
                calling_actions.append(action_idx)
                calling_masks.append(mask)
            else:
                states.append(state)
                actions.append(action_idx)
                masks.append(mask)
            
            # Take step
            result = game.step(action)
            observations = result.observations
            done = result.done
    
    # If focusing on calling, oversample calling decisions
    if focus_on_calling and calling_states:
        # Repeat calling data to match playing data size
        num_calling = len(calling_states)
        num_playing = len(states)
        
        if num_playing > num_calling:
            # Oversample calling decisions
            repeat_factor = max(1, num_playing // num_calling)
            calling_states = calling_states * repeat_factor
            calling_actions = calling_actions * repeat_factor
            calling_masks = calling_masks * repeat_factor
    
    # Combine
    all_states = states + calling_states
    all_actions = actions + calling_actions
    all_masks = masks + calling_masks
    
    # Shuffle
    combined = list(zip(all_states, all_actions, all_masks))
    random.shuffle(combined)
    all_states, all_actions, all_masks = zip(*combined)
    
    print(f"Collected {len(all_states)} total samples")
    print(f"  - Playing decisions: {len(states)}")
    print(f"  - Calling decisions: {len(calling_states)} (after oversampling)")
    
    return list(all_states), list(all_actions), list(all_masks)


def train_imitation(
    num_episodes: int = 50000,
    batch_size: int = 256,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
):
    """
    Train the neural network to imitate rule-based decisions.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Collect data
    print("Step 1: Collecting imitation data from rule-based agent...")
    states, actions, masks = collect_imitation_data(num_episodes, focus_on_calling=True)
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    masks_tensor = torch.BoolTensor(masks)
    
    # Create dataset and dataloader
    dataset = TensorDataset(states_tensor, actions_tensor, masks_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize network
    print("\nStep 2: Training neural network...")
    network = EuchreNetwork()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_states, batch_actions, batch_masks in tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            # Forward pass - get policy logits
            policy_logits, _ = network(batch_states)
            
            # Mask illegal actions
            policy_logits = policy_logits.masked_fill(~batch_masks, float('-inf'))
            
            # Compute loss
            loss = criterion(policy_logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track accuracy
            _, predicted = policy_logits.max(1)
            correct += (predicted == batch_actions).sum().item()
            total += batch_actions.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1%}")
    
    # Save the pre-trained model
    pretrain_path = os.path.join(checkpoint_dir, "pretrained_imitation.pth")
    torch.save(network.state_dict(), pretrain_path)
    print(f"\nPre-trained model saved to: {pretrain_path}")
    
    # Also save as latest.pth so train_mixed.py can resume from it
    checkpoint = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": 0,
        "total_steps": 0,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pth"))
    print(f"Checkpoint saved for RL fine-tuning")
    
    return network


def evaluate_imitation(network: EuchreNetwork, num_games: int = 500):
    """
    Quick evaluation of the imitation-trained model.
    """
    from agents.neural_agent import NeuralAgent
    
    game = EuchreGame()
    neural = NeuralAgent(network)
    rule_based = RuleBasedAgent()
    
    neural_wins = 0
    
    for _ in tqdm(range(num_games), desc="Evaluating"):
        observations = game.reset()
        done = False
        
        while not done:
            player = game.state.current_player
            obs = observations[player]
            legal_actions = game.get_legal_actions()
            
            # Neural is team 0, rule-based is team 1
            if player % 2 == 0:
                action = neural.select_action(obs, legal_actions)
            else:
                action = rule_based.select_action(obs, legal_actions)
            
            result = game.step(action)
            observations = result.observations
            done = result.done
        
        if game.state.score[0] >= 10:
            neural_wins += 1
    
    win_rate = neural_wins / num_games
    print(f"\nImitation model vs Rule-Based: {win_rate:.1%}")
    return win_rate


def main():
    parser = argparse.ArgumentParser(description="Pre-train by imitating rule-based")
    parser.add_argument("--episodes", type=int, default=50000,
                        help="Number of games to collect data from")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate after training")
    
    args = parser.parse_args()
    
    network = train_imitation(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    if args.evaluate:
        evaluate_imitation(network)
    
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Run RL fine-tuning:")
    print("   python scripts/train_mixed.py --episodes 100000")
    print("="*50)


if __name__ == "__main__":
    main()