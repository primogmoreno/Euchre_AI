"""
Analyzer for querying and summarizing collected data.
"""

import json
from collections import defaultdict
from typing import Callable, Optional
from .logger import DataReader


class DataAnalyzer:
    """
    Analyze collected Euchre decision data.
    
    Provides methods to filter, group, and compute statistics
    on the logged data.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.reader = DataReader(data_dir)
    
    # =========================================================================
    # Going Alone Analysis
    # =========================================================================
    
    def going_alone_summary(self) -> dict:
        """Get overall summary of going alone decisions."""
        
        total = 0
        successes = 0
        marches = 0
        euchres = 0
        total_points = 0
        
        for record in self.reader.iter_records("going_alone"):
            total += 1
            if record.get("success"):
                successes += 1
            if record.get("march"):
                marches += 1
            if record.get("was_euchred"):
                euchres += 1
            total_points += record.get("points_earned", 0)
        
        if total == 0:
            return {"total": 0, "message": "No going alone data collected yet"}
        
        return {
            "total_attempts": total,
            "successes": successes,
            "success_rate": successes / total,
            "marches": marches,
            "march_rate": marches / total,
            "euchres": euchres,
            "euchre_rate": euchres / total,
            "avg_points": total_points / total,
            "expected_value": total_points / total,  # vs 0 for not going alone
        }
    
    def going_alone_by_position(self) -> dict:
        """Analyze going alone success by position relative to dealer."""
        
        by_position = defaultdict(lambda: {"total": 0, "success": 0, "points": 0})
        
        for record in self.reader.iter_records("going_alone"):
            pos = record.get("position_relative_to_dealer", 0)
            by_position[pos]["total"] += 1
            if record.get("success"):
                by_position[pos]["success"] += 1
            by_position[pos]["points"] += record.get("points_earned", 0)
        
        result = {}
        position_names = {0: "dealer", 1: "left_of_dealer", 2: "across", 3: "right_of_dealer"}
        
        for pos, data in by_position.items():
            name = position_names.get(pos, f"position_{pos}")
            if data["total"] > 0:
                result[name] = {
                    "total": data["total"],
                    "success_rate": data["success"] / data["total"],
                    "avg_points": data["points"] / data["total"],
                }
        
        return result
    
    def going_alone_by_trump_count(self) -> dict:
        """Analyze success rate by number of trump cards held."""
        
        by_count = defaultdict(lambda: {"total": 0, "success": 0, "march": 0, "points": 0})
        
        for record in self.reader.iter_records("going_alone"):
            hand = record.get("hand", [])
            trump = record.get("trump")
            
            if not trump:
                continue
            
            # Count trump (simplified - doesn't handle left bower perfectly)
            trump_symbol = {"CLUBS": "♣", "DIAMONDS": "♦", "HEARTS": "♥", "SPADES": "♠"}.get(trump)
            trump_count = sum(1 for c in hand if trump_symbol and trump_symbol in c)
            
            # Check for left bower
            same_color = {"CLUBS": "♠", "SPADES": "♣", "HEARTS": "♦", "DIAMONDS": "♥"}.get(trump)
            if any(c.startswith("J") and same_color in c for c in hand):
                trump_count += 1
            
            by_count[trump_count]["total"] += 1
            if record.get("success"):
                by_count[trump_count]["success"] += 1
            if record.get("march"):
                by_count[trump_count]["march"] += 1
            by_count[trump_count]["points"] += record.get("points_earned", 0)
        
        result = {}
        for count in sorted(by_count.keys()):
            data = by_count[count]
            if data["total"] > 0:
                result[f"{count}_trump"] = {
                    "total": data["total"],
                    "success_rate": data["success"] / data["total"],
                    "march_rate": data["march"] / data["total"],
                    "avg_points": data["points"] / data["total"],
                }
        
        return result
    
    def going_alone_best_hands(self, min_success_rate: float = 0.8, min_samples: int = 10) -> list:
        """Find hand patterns with highest success rates."""
        
        # Group by sorted hand
        by_hand = defaultdict(lambda: {"total": 0, "success": 0, "march": 0})
        
        for record in self.reader.iter_records("going_alone"):
            hand = tuple(sorted(record.get("hand", [])))
            by_hand[hand]["total"] += 1
            if record.get("success"):
                by_hand[hand]["success"] += 1
            if record.get("march"):
                by_hand[hand]["march"] += 1
        
        results = []
        for hand, data in by_hand.items():
            if data["total"] >= min_samples:
                success_rate = data["success"] / data["total"]
                if success_rate >= min_success_rate:
                    results.append({
                        "hand": list(hand),
                        "total": data["total"],
                        "success_rate": success_rate,
                        "march_rate": data["march"] / data["total"],
                    })
        
        return sorted(results, key=lambda x: -x["success_rate"])
    
    # =========================================================================
    # Trump Calling Analysis
    # =========================================================================
    
    def trump_call_summary(self) -> dict:
        """Get overall summary of trump calling."""
        
        total = 0
        successes = 0
        marches = 0
        euchres = 0
        total_points = 0
        by_round = {1: {"total": 0, "success": 0}, 2: {"total": 0, "success": 0}}
        
        for record in self.reader.iter_records("trump_calls"):
            total += 1
            round_num = record.get("calling_round", 1)
            by_round[round_num]["total"] += 1
            
            if record.get("success"):
                successes += 1
                by_round[round_num]["success"] += 1
            if record.get("march"):
                marches += 1
            if record.get("was_euchred"):
                euchres += 1
            total_points += record.get("points_earned", 0)
        
        if total == 0:
            return {"total": 0, "message": "No trump call data collected yet"}
        
        result = {
            "total_calls": total,
            "success_rate": successes / total,
            "march_rate": marches / total,
            "euchre_rate": euchres / total,
            "avg_points": total_points / total,
        }
        
        for round_num, data in by_round.items():
            if data["total"] > 0:
                result[f"round_{round_num}_success_rate"] = data["success"] / data["total"]
        
        return result
    
    def trump_call_by_suit(self) -> dict:
        """Analyze trump calling success by suit."""
        
        by_suit = defaultdict(lambda: {"total": 0, "success": 0, "points": 0})
        
        for record in self.reader.iter_records("trump_calls"):
            suit = record.get("trump")
            if suit:
                by_suit[suit]["total"] += 1
                if record.get("success"):
                    by_suit[suit]["success"] += 1
                by_suit[suit]["points"] += record.get("points_earned", 0)
        
        result = {}
        for suit, data in by_suit.items():
            if data["total"] > 0:
                result[suit] = {
                    "total": data["total"],
                    "success_rate": data["success"] / data["total"],
                    "avg_points": data["points"] / data["total"],
                }
        
        return result
    
    # =========================================================================
    # Pass Analysis
    # =========================================================================
    
    def pass_analysis(self) -> dict:
        """Analyze passing patterns."""
        
        total = 0
        by_round = {1: 0, 2: 0}
        trump_counts_when_passing = defaultdict(list)
        
        for record in self.reader.iter_records("passes"):
            total += 1
            round_num = record.get("calling_round", 1)
            by_round[round_num] += 1
            
            # Track what trump counts people pass with
            trump_counts = record.get("trump_counts_by_suit", {})
            max_trump = max(trump_counts.values()) if trump_counts else 0
            trump_counts_when_passing[max_trump].append(1)
        
        if total == 0:
            return {"total": 0, "message": "No pass data collected yet"}
        
        result = {
            "total_passes": total,
            "round_1_passes": by_round[1],
            "round_2_passes": by_round[2],
            "passes_by_max_trump_count": {
                k: len(v) for k, v in sorted(trump_counts_when_passing.items())
            },
        }
        
        return result
    
    # =========================================================================
    # General Utilities
    # =========================================================================
    
    def custom_query(
        self,
        category: str,
        filter_fn: Optional[Callable] = None,
        group_by: Optional[str] = None,
    ) -> dict:
        """
        Run a custom query on collected data.
        
        Args:
            category: "going_alone", "trump_calls", "passes", or "plays"
            filter_fn: Optional function to filter records
            group_by: Optional field name to group results by
        
        Returns:
            Dict with counts and optionally grouped results
        """
        records = self.reader.iter_records(category)
        
        if filter_fn:
            records = (r for r in records if filter_fn(r))
        
        if group_by:
            groups = defaultdict(list)
            for record in records:
                key = record.get(group_by, "unknown")
                groups[key].append(record)
            
            return {
                "grouped_by": group_by,
                "groups": {k: len(v) for k, v in groups.items()},
                "total": sum(len(v) for v in groups.values()),
            }
        else:
            record_list = list(records)
            return {
                "total": len(record_list),
                "records": record_list[:100],  # Limit to first 100
            }
    
    def export_summary(self, filepath: str = "analysis_summary.json"):
        """Export a full analysis summary to JSON."""
        
        summary = {
            "going_alone": self.going_alone_summary(),
            "going_alone_by_position": self.going_alone_by_position(),
            "going_alone_by_trump_count": self.going_alone_by_trump_count(),
            "trump_calls": self.trump_call_summary(),
            "trump_calls_by_suit": self.trump_call_by_suit(),
            "passes": self.pass_analysis(),
        }
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary