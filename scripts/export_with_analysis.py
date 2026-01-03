#!/usr/bin/env python3
"""
Export collected Euchre data to CSV with hand strength analysis.

Adds calculated columns for:
- Caller's hand strength
- Partner's hand strength  
- Opponents' combined hand strength
- Trump count for each player

Usage:
    python scripts/export_with_analysis.py
    python scripts/export_with_analysis.py --data-dir data --output-dir exports
"""

import sys
import os
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from data_collection.logger import DataReader


# =============================================================================
# Card Strength Calculation (mirrors rules.py logic)
# =============================================================================

SUIT_SYMBOLS = {"♣": "CLUBS", "♦": "DIAMONDS", "♥": "HEARTS", "♠": "SPADES"}
SYMBOL_TO_SUIT = {v: k for k, v in SUIT_SYMBOLS.items()}

SAME_COLOR_SUIT = {
    "CLUBS": "SPADES",
    "SPADES": "CLUBS", 
    "HEARTS": "DIAMONDS",
    "DIAMONDS": "HEARTS",
}

RANK_VALUES = {
    "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}


def parse_card(card_str: str) -> tuple:
    """
    Parse card string like 'J♠' into (rank, suit).
    
    Returns: (rank_str, suit_name) e.g. ('J', 'SPADES')
    """
    # Handle card strings like "J♠", "10♥", "A♣"
    for symbol, suit_name in SUIT_SYMBOLS.items():
        if symbol in card_str:
            rank = card_str.replace(symbol, "")
            return (rank, suit_name)
    return (None, None)


def get_card_strength(card_str: str, trump: str) -> int:
    """
    Calculate card strength given trump suit.
    
    Uses same logic as rules.py:
    - Right bower (jack of trump): 1000
    - Left bower (jack of same color): 900
    - Other trump: 800 + rank value
    - Off suit ace: 14 (can take tricks if led)
    - Other off suit: rank value
    
    Args:
        card_str: Card like "J♠"
        trump: Trump suit name like "SPADES"
    
    Returns:
        Integer strength value
    """
    rank, suit = parse_card(card_str)
    
    if rank is None or suit is None:
        return 0
    
    rank_value = RANK_VALUES.get(rank, 0)
    
    # Right bower
    if rank == "J" and suit == trump:
        return 1000
    
    # Left bower
    if rank == "J" and suit == SAME_COLOR_SUIT.get(trump):
        return 900
    
    # Other trump
    # Need to check effective suit (left bower already handled above)
    if suit == trump:
        return 800 + rank_value
    
    # Off suit - still has some value for leading
    return rank_value


def get_trump_count(hand: list, trump: str) -> int:
    """
    Count trump cards in hand (including left bower).
    """
    count = 0
    for card_str in hand:
        rank, suit = parse_card(card_str)
        if suit == trump:
            count += 1
        # Left bower counts as trump
        elif rank == "J" and suit == SAME_COLOR_SUIT.get(trump):
            count += 1
    return count


def calculate_hand_strength(hand: list, trump: str) -> dict:
    """
    Calculate various strength metrics for a hand.
    
    Returns dict with:
    - total_strength: Sum of all card strengths
    - trump_count: Number of trump cards
    - has_right_bower: Boolean
    - has_left_bower: Boolean
    - trump_strength: Strength of just trump cards
    - off_suit_aces: Count of non-trump aces
    """
    if not hand or not trump:
        return {
            "total_strength": 0,
            "trump_count": 0,
            "has_right_bower": False,
            "has_left_bower": False,
            "trump_strength": 0,
            "off_suit_aces": 0,
        }
    
    total = 0
    trump_strength = 0
    trump_count = 0
    has_right = False
    has_left = False
    off_aces = 0
    
    for card_str in hand:
        rank, suit = parse_card(card_str)
        if rank is None:
            continue
            
        strength = get_card_strength(card_str, trump)
        total += strength
        
        # Check for bowers
        if rank == "J" and suit == trump:
            has_right = True
            trump_count += 1
            trump_strength += strength
        elif rank == "J" and suit == SAME_COLOR_SUIT.get(trump):
            has_left = True
            trump_count += 1
            trump_strength += strength
        elif suit == trump:
            trump_count += 1
            trump_strength += strength
        elif rank == "A":
            off_aces += 1
    
    return {
        "total_strength": total,
        "trump_count": trump_count,
        "has_right_bower": has_right,
        "has_left_bower": has_left,
        "trump_strength": trump_strength,
        "off_suit_aces": off_aces,
    }


def parse_hand_list(hand_data) -> list:
    """
    Parse a hand into a list of card strings.
    
    Handles both:
    - List: ['J♠', 'A♠', 'K♥']
    - String: 'J♠, A♠, K♥'
    """
    if not hand_data:
        return []
    
    # Already a list
    if isinstance(hand_data, list):
        return [str(card) for card in hand_data]
    
    # String format
    if isinstance(hand_data, str):
        return [card.strip() for card in hand_data.split(",") if card.strip()]
    
    return []


def parse_all_hands(all_hands_data) -> list:
    """
    Parse all_hands field into list of 4 hands.
    
    Handles both:
    - List of lists: [['J♠', 'A♠'], ['K♥', '10♦'], ...]
    - String representation: "['J♠', 'A♠'], ['K♥', '10♦'], ..."
    """
    if not all_hands_data:
        return [[], [], [], []]
    
    # Already a list of lists
    if isinstance(all_hands_data, list):
        if len(all_hands_data) == 4:
            # Convert any Card objects to strings
            result = []
            for hand in all_hands_data:
                if isinstance(hand, list):
                    result.append([str(card) for card in hand])
                else:
                    result.append([])
            return result
        return [[], [], [], []]
    
    # String format - try to parse
    if isinstance(all_hands_data, str):
        import ast
        try:
            hands = ast.literal_eval(all_hands_data.replace("'", "'").replace("'", "'"))
            if isinstance(hands, list) and len(hands) == 4:
                return hands
        except:
            pass
    
    return [[], [], [], []]


# =============================================================================
# Export Functions
# =============================================================================

def export_going_alone_with_analysis(reader: DataReader, output_path: str) -> int:
    """
    Export going alone data with calculated hand strengths.
    """
    records = list(reader.iter_records("going_alone"))
    
    if not records:
        print("  No going alone records found")
        return 0
    
    output_rows = []
    
    for record in records:
        trump = record.get("trump", "")
        player = record.get("player", 0)
        
        # Parse hands
        caller_hand = parse_hand_list(record.get("hand", ""))
        all_hands = record.get("all_hands", [[], [], [], []])
        
        # Ensure all_hands is a list of lists
        if isinstance(all_hands, str):
            all_hands = parse_all_hands(all_hands)
        
        # Calculate strengths for each position
        partner_idx = (player + 2) % 4
        opponent1_idx = (player + 1) % 4
        opponent2_idx = (player + 3) % 4
        
        caller_stats = calculate_hand_strength(caller_hand, trump)
        
        # Get other hands from all_hands
        partner_hand = all_hands[partner_idx] if len(all_hands) > partner_idx else []
        opponent1_hand = all_hands[opponent1_idx] if len(all_hands) > opponent1_idx else []
        opponent2_hand = all_hands[opponent2_idx] if len(all_hands) > opponent2_idx else []
        
        partner_stats = calculate_hand_strength(partner_hand, trump)
        opponent1_stats = calculate_hand_strength(opponent1_hand, trump)
        opponent2_stats = calculate_hand_strength(opponent2_hand, trump)
        
        # Combined opponent strength
        combined_opponent_strength = opponent1_stats["total_strength"] + opponent2_stats["total_strength"]
        combined_opponent_trump = opponent1_stats["trump_count"] + opponent2_stats["trump_count"]
        
        # Build output row
        row = {
            # Original fields
            "episode": record.get("episode"),
            "player": player,
            "trump": trump,
            "hand": record.get("hand", ""),
            "turned_card": record.get("turned_card", ""),
            "ordered_up": record.get("ordered_up", ""),
            "calling_round": record.get("calling_round", ""),
            "buried_card": record.get("buried_card", ""),
            "dealer": record.get("dealer"),
            "position_relative_to_dealer": record.get("position_relative_to_dealer"),
            
            # Outcome fields
            "success": record.get("success"),
            "was_euchred": record.get("was_euchred"),
            "march": record.get("march"),
            "tricks_won": record.get("tricks_won"),
            "tricks_lost": record.get("tricks_lost"),
            "points_earned": record.get("points_earned"),
            
            # Caller hand analysis
            "caller_total_strength": caller_stats["total_strength"],
            "caller_trump_count": caller_stats["trump_count"],
            "caller_trump_strength": caller_stats["trump_strength"],
            "caller_has_right_bower": caller_stats["has_right_bower"],
            "caller_has_left_bower": caller_stats["has_left_bower"],
            "caller_off_suit_aces": caller_stats["off_suit_aces"],
            
            # Partner hand analysis (sitting out when going alone, but interesting)
            "partner_total_strength": partner_stats["total_strength"],
            "partner_trump_count": partner_stats["trump_count"],
            "partner_has_right_bower": partner_stats["has_right_bower"],
            "partner_has_left_bower": partner_stats["has_left_bower"],
            
            # Opponent analysis
            "opponent1_total_strength": opponent1_stats["total_strength"],
            "opponent1_trump_count": opponent1_stats["trump_count"],
            "opponent2_total_strength": opponent2_stats["total_strength"],
            "opponent2_trump_count": opponent2_stats["trump_count"],
            "combined_opponent_strength": combined_opponent_strength,
            "combined_opponent_trump": combined_opponent_trump,
            
            # Comparative metrics
            "strength_advantage": caller_stats["total_strength"] - combined_opponent_strength,
            "trump_advantage": caller_stats["trump_count"] - combined_opponent_trump,
            
            # Raw hands for reference
            "partner_hand": ", ".join(partner_hand) if partner_hand else "",
            "opponent1_hand": ", ".join(opponent1_hand) if opponent1_hand else "",
            "opponent2_hand": ", ".join(opponent2_hand) if opponent2_hand else "",
            "kitty": record.get("kitty", ""),
        }
        
        output_rows.append(row)
    
    # Write CSV
    if output_rows:
        columns = list(output_rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(output_rows)
    
    return len(output_rows)


def export_trump_calls_with_analysis(reader: DataReader, output_path: str) -> int:
    """
    Export trump call data with calculated hand strengths.
    """
    records = list(reader.iter_records("trump_calls"))
    
    if not records:
        print("  No trump call records found")
        return 0
    
    output_rows = []
    
    for record in records:
        trump = record.get("trump", "")
        player = record.get("player", 0)
        
        # Parse hands
        caller_hand = parse_hand_list(record.get("hand", ""))
        all_hands = record.get("all_hands", [[], [], [], []])
        
        if isinstance(all_hands, str):
            all_hands = parse_all_hands(all_hands)
        
        # Calculate strengths
        partner_idx = (player + 2) % 4
        opponent1_idx = (player + 1) % 4
        opponent2_idx = (player + 3) % 4
        
        caller_stats = calculate_hand_strength(caller_hand, trump)
        
        partner_hand = all_hands[partner_idx] if len(all_hands) > partner_idx else []
        opponent1_hand = all_hands[opponent1_idx] if len(all_hands) > opponent1_idx else []
        opponent2_hand = all_hands[opponent2_idx] if len(all_hands) > opponent2_idx else []
        
        partner_stats = calculate_hand_strength(partner_hand, trump)
        opponent1_stats = calculate_hand_strength(opponent1_hand, trump)
        opponent2_stats = calculate_hand_strength(opponent2_hand, trump)
        
        # Team strengths
        team_strength = caller_stats["total_strength"] + partner_stats["total_strength"]
        team_trump = caller_stats["trump_count"] + partner_stats["trump_count"]
        opponent_strength = opponent1_stats["total_strength"] + opponent2_stats["total_strength"]
        opponent_trump = opponent1_stats["trump_count"] + opponent2_stats["trump_count"]
        
        row = {
            "episode": record.get("episode"),
            "player": player,
            "trump": trump,
            "hand": record.get("hand", ""),
            "turned_card": record.get("turned_card", ""),
            "ordered_up": record.get("ordered_up", ""),
            "calling_round": record.get("calling_round"),
            "buried_card": record.get("buried_card", ""),
            "dealer": record.get("dealer"),
            "position_relative_to_dealer": record.get("position_relative_to_dealer"),
            
            # Outcomes
            "success": record.get("success"),
            "was_euchred": record.get("was_euchred"),
            "march": record.get("march"),
            "tricks_won": record.get("tricks_won"),
            "points_earned": record.get("points_earned"),
            
            # Caller analysis
            "caller_total_strength": caller_stats["total_strength"],
            "caller_trump_count": caller_stats["trump_count"],
            "caller_has_right_bower": caller_stats["has_right_bower"],
            "caller_has_left_bower": caller_stats["has_left_bower"],
            "caller_off_suit_aces": caller_stats["off_suit_aces"],
            
            # Partner analysis
            "partner_total_strength": partner_stats["total_strength"],
            "partner_trump_count": partner_stats["trump_count"],
            "partner_has_right_bower": partner_stats["has_right_bower"],
            "partner_has_left_bower": partner_stats["has_left_bower"],
            
            # Team totals
            "team_total_strength": team_strength,
            "team_trump_count": team_trump,
            
            # Opponent analysis
            "opponent_total_strength": opponent_strength,
            "opponent_trump_count": opponent_trump,
            
            # Comparative
            "strength_advantage": team_strength - opponent_strength,
            "trump_advantage": team_trump - opponent_trump,
        }
        
        output_rows.append(row)
    
    if output_rows:
        columns = list(output_rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(output_rows)
    
    return len(output_rows)


def create_strength_reference_csv(output_path: str):
    """
    Create a reference CSV showing card strengths for each trump suit.
    """
    rows = []
    
    cards = ["9", "10", "J", "Q", "K", "A"]
    suits = ["CLUBS", "DIAMONDS", "HEARTS", "SPADES"]
    suit_symbols = {"CLUBS": "♣", "DIAMONDS": "♦", "HEARTS": "♥", "SPADES": "♠"}
    
    for suit in suits:
        for rank in cards:
            card_str = f"{rank}{suit_symbols[suit]}"
            row = {"card": card_str, "suit": suit, "rank": rank}
            
            # Calculate strength for each possible trump
            for trump in suits:
                strength = get_card_strength(card_str, trump)
                row[f"strength_if_{trump.lower()}_trump"] = strength
            
            rows.append(row)
    
    columns = ["card", "suit", "rank"] + [f"strength_if_{s.lower()}_trump" for s in suits]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Export Euchre data with hand strength analysis")
    parser.add_argument(
        "--data-dir",
        type=str, 
        default="data",
        help="Directory containing collected data (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Directory for CSV exports (default: exports)",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    reader = DataReader(args.data_dir)
    
    print("=" * 60)
    print("EXPORTING EUCHRE DATA WITH HAND STRENGTH ANALYSIS")
    print("=" * 60)
    print(f"Reading from: {args.data_dir}")
    print(f"Writing to: {args.output_dir}")
    
    # Export going alone with analysis
    print("\nExporting going alone data...")
    count = export_going_alone_with_analysis(
        reader, 
        os.path.join(args.output_dir, "going_alone_analyzed.csv")
    )
    print(f"  Exported {count} records to going_alone_analyzed.csv")
    
    # Export trump calls with analysis
    print("\nExporting trump call data...")
    count = export_trump_calls_with_analysis(
        reader,
        os.path.join(args.output_dir, "trump_calls_analyzed.csv")
    )
    print(f"  Exported {count} records to trump_calls_analyzed.csv")
    
    # Create reference sheet
    print("\nCreating card strength reference...")
    create_strength_reference_csv(
        os.path.join(args.output_dir, "card_strength_reference.csv")
    )
    print("  Created card_strength_reference.csv")
    
    print("\n" + "=" * 60)
    print("COLUMN DESCRIPTIONS")
    print("=" * 60)
    print("""
Hand Strength Columns:
  - total_strength: Sum of all card strengths in hand
  - trump_count: Number of trump cards (including left bower)
  - trump_strength: Strength of just the trump cards
  - has_right_bower: True if hand has jack of trump
  - has_left_bower: True if hand has jack of same color
  - off_suit_aces: Count of aces not in trump suit

Strength Values (same as rules.py):
  - Right bower (J of trump): 1000
  - Left bower (J of same color): 900  
  - Other trump: 800 + rank (809-814)
  - Off suit: rank value (9-14)

Comparative Columns:
  - strength_advantage: Your strength minus opponent strength
  - trump_advantage: Your trump count minus opponent trump count
    (Positive = advantage, Negative = disadvantage)
""")
    print("=" * 60)
    print("Export complete!")


if __name__ == "__main__":
    main()