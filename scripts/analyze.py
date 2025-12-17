#!/usr/bin/env python3
"""
Analyze collected Euchre decision data.

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --category going_alone
    python scripts/analyze.py --export results.json
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from data_collection import DataAnalyzer
from data_collection.logger import DataReader


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def format_percent(value: float) -> str:
    """Format a float as a percentage."""
    return f"{value:.1%}"


def format_float(value: float) -> str:
    """Format a float to 2 decimal places."""
    return f"{value:.2f}"


def analyze_going_alone(analyzer: DataAnalyzer):
    """Print going alone analysis."""
    print_section("GOING ALONE ANALYSIS")
    
    summary = analyzer.going_alone_summary()
    
    if summary.get("total_attempts", 0) == 0:
        print("\nNo going alone data collected yet.")
        print("Run training with data collection enabled to gather data.")
        return
    
    print(f"\nTotal attempts: {summary['total_attempts']}")
    print(f"Success rate:   {format_percent(summary['success_rate'])}")
    print(f"March rate:     {format_percent(summary['march_rate'])}")
    print(f"Euchre rate:    {format_percent(summary['euchre_rate'])}")
    print(f"Avg points:     {format_float(summary['avg_points'])}")
    
    # By position
    print("\n--- By Position ---")
    by_pos = analyzer.going_alone_by_position()
    for pos, data in by_pos.items():
        print(f"  {pos}: {format_percent(data['success_rate'])} success ({data['total']} samples)")
    
    # By trump count
    print("\n--- By Trump Count ---")
    by_trump = analyzer.going_alone_by_trump_count()
    for count, data in by_trump.items():
        print(f"  {count}: {format_percent(data['success_rate'])} success, "
              f"{format_percent(data['march_rate'])} march ({data['total']} samples)")
    
    # Best hands
    print("\n--- Best Hands (80%+ success, 10+ samples) ---")
    best = analyzer.going_alone_best_hands(min_success_rate=0.8, min_samples=10)
    for i, hand_data in enumerate(best[:10]):
        hand_str = ", ".join(hand_data["hand"])
        print(f"  {i+1}. [{hand_str}]")
        print(f"      {format_percent(hand_data['success_rate'])} success, "
              f"{format_percent(hand_data['march_rate'])} march ({hand_data['total']} samples)")


def analyze_trump_calls(analyzer: DataAnalyzer):
    """Print trump calling analysis."""
    print_section("TRUMP CALLING ANALYSIS")
    
    summary = analyzer.trump_call_summary()
    
    if summary.get("total_calls", 0) == 0:
        print("\nNo trump call data collected yet.")
        return
    
    print(f"\nTotal calls:  {summary['total_calls']}")
    print(f"Success rate: {format_percent(summary['success_rate'])}")
    print(f"March rate:   {format_percent(summary['march_rate'])}")
    print(f"Euchre rate:  {format_percent(summary['euchre_rate'])}")
    print(f"Avg points:   {format_float(summary['avg_points'])}")
    
    if "round_1_success_rate" in summary:
        print(f"\nRound 1 success: {format_percent(summary['round_1_success_rate'])}")
    if "round_2_success_rate" in summary:
        print(f"Round 2 success: {format_percent(summary['round_2_success_rate'])}")
    
    # By suit
    print("\n--- By Suit ---")
    by_suit = analyzer.trump_call_by_suit()
    for suit, data in sorted(by_suit.items()):
        print(f"  {suit}: {format_percent(data['success_rate'])} success ({data['total']} samples)")


def analyze_passes(analyzer: DataAnalyzer):
    """Print pass analysis."""
    print_section("PASS ANALYSIS")
    
    summary = analyzer.pass_analysis()
    
    if summary.get("total_passes", 0) == 0:
        print("\nNo pass data collected yet.")
        return
    
    print(f"\nTotal passes: {summary['total_passes']}")
    print(f"Round 1 passes: {summary['round_1_passes']}")
    print(f"Round 2 passes: {summary['round_2_passes']}")
    
    print("\n--- Max Trump Count When Passing ---")
    for count, num in summary.get("passes_by_max_trump_count", {}).items():
        print(f"  {count} trump: {num} passes")


def show_sample_records(data_dir: str, category: str, num: int = 5):
    """Show sample records from a category."""
    print_section(f"SAMPLE {category.upper()} RECORDS")
    
    reader = DataReader(data_dir)
    records = list(reader.iter_records(category))
    
    if not records:
        print(f"\nNo {category} records found.")
        return
    
    print(f"\nShowing {min(num, len(records))} of {len(records)} records:\n")
    
    for i, record in enumerate(records[:num]):
        print(f"--- Record {i+1} ---")
        # Remove internal fields for cleaner display
        display = {k: v for k, v in record.items() if not k.startswith("_")}
        print(json.dumps(display, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Euchre decision data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing collected data (default: data)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["going_alone", "trump_calls", "passes", "plays", "all"],
        default="all",
        help="Category to analyze (default: all)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export summary to JSON file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Show N sample records",
    )
    
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(args.data_dir)
    
    print("\n" + "=" * 60)
    print("  EUCHRE DECISION DATA ANALYSIS")
    print("=" * 60)
    
    if args.category in ["going_alone", "all"]:
        analyze_going_alone(analyzer)
    
    if args.category in ["trump_calls", "all"]:
        analyze_trump_calls(analyzer)
    
    if args.category in ["passes", "all"]:
        analyze_passes(analyzer)
    
    if args.samples > 0:
        if args.category != "all":
            show_sample_records(args.data_dir, args.category, args.samples)
        else:
            for cat in ["going_alone", "trump_calls", "passes"]:
                show_sample_records(args.data_dir, cat, args.samples)
    
    if args.export:
        print(f"\nExporting summary to {args.export}...")
        analyzer.export_summary(args.export)
        print("Done!")
    
    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()