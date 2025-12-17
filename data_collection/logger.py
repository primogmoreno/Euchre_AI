"""
Data logger for writing decision records to JSONL files.
"""

import os
import json
from datetime import datetime
from typing import Any


class DataLogger:
    """
    Writes decision records to JSONL files.
    
    JSONL format: one JSON object per line, easy to stream and filter.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.files = {}
        self.counts = {}
        
        # Create directory structure
        self.subdirs = [
            "going_alone",
            "trump_calls", 
            "passes",
            "plays",
        ]
        
        for subdir in self.subdirs:
            path = os.path.join(data_dir, subdir)
            os.makedirs(path, exist_ok=True)
    
    def _get_file(self, category: str):
        """Get or create file handle for a category."""
        if category not in self.files:
            filepath = os.path.join(
                self.data_dir, 
                category,
                f"{category}_log.jsonl"
            )
            self.files[category] = open(filepath, "a", encoding="utf-8")
            self.counts[category] = 0
        return self.files[category]
    
    def log(self, category: str, record: dict):
        """
        Log a record to the appropriate file.
        
        Args:
            category: One of "going_alone", "trump_calls", "passes", "plays"
            record: Dictionary of data to log
        """
        # Add metadata
        record["_logged_at"] = datetime.now().isoformat()
        
        f = self._get_file(category)
        f.write(json.dumps(record) + "\n")
        self.counts[category] += 1
        
        # Flush periodically to avoid data loss
        if self.counts[category] % 100 == 0:
            f.flush()
    
    def log_going_alone(self, record: dict):
        """Log a going alone decision."""
        self.log("going_alone", record)
    
    def log_trump_call(self, record: dict):
        """Log a trump calling decision."""
        self.log("trump_calls", record)
    
    def log_pass(self, record: dict):
        """Log a pass decision."""
        self.log("passes", record)
    
    def log_play(self, record: dict):
        """Log a card play decision."""
        self.log("plays", record)
    
    def get_counts(self) -> dict:
        """Return count of records logged per category."""
        return self.counts.copy()
    
    def close(self):
        """Close all file handles."""
        for f in self.files.values():
            f.close()
        self.files = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DataReader:
    """
    Read and iterate over logged data.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def iter_records(self, category: str):
        """
        Iterate over all records in a category.
        
        Yields dict objects one at a time (memory efficient).
        """
        filepath = os.path.join(
            self.data_dir,
            category,
            f"{category}_log.jsonl"
        )
        
        if not os.path.exists(filepath):
            return
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    def load_all(self, category: str) -> list[dict]:
        """Load all records from a category into memory."""
        return list(self.iter_records(category))
    
    def count(self, category: str) -> int:
        """Count records in a category without loading all into memory."""
        return sum(1 for _ in self.iter_records(category))
    
    def filter_records(self, category: str, predicate):
        """
        Iterate over records matching a predicate function.
        
        Example:
            reader.filter_records("going_alone", lambda r: r["points_earned"] == 4)
        """
        for record in self.iter_records(category):
            if predicate(record):
                yield record