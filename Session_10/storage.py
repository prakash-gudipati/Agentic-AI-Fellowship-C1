"""
Session 10 — Python Capstone: CLI Productivity Tool
Module: storage.py — JSON-backed persistent goal store

DEMONSTRATES:
  S5 (OOP)   — GoalStore class wraps all file access in one place
  S6 (File I/O + JSON) — open(), json.load(), json.dump(), with statement
  S3 (Error handling)  — try/except around every file operation
  S4 (Data structures) — goals stored as a list of dicts, queried with comprehensions
  S2 (Comprehensions)  — filtering and aggregation without raw loops
"""

import json
import os
from datetime import datetime, timedelta

from goal import Goal
from config import DATA_FILE, WEEKLY_SUMMARY_DAYS


class GoalStore:
    """
    Manages all reading, writing, and querying of goals.

    Think of this as a smart filing cabinet:
    - You hand it a Goal and it files it away (add_goal).
    - You ask for today's goals and it retrieves them (get_todays_goals).
    - Everything stays in sync with the JSON file on disk automatically.
    """

    def __init__(self, filepath: str = DATA_FILE):
        """
        Open the store. Creates the data folder and file if they don't exist.

        Args:
            filepath: Path to the JSON file. Defaults to DATA_FILE from config.
        """
        self.filepath = filepath
        self._ensure_data_folder_exists()
        self.goals = self._load_all_goals()

    # ── PRIVATE HELPERS ────────────────────────────────────────────────────

    def _ensure_data_folder_exists(self):
        """Create the data/ folder if it is missing. Safe to call repeatedly."""
        folder = os.path.dirname(self.filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

    def _load_all_goals(self) -> list:
        """
        Read all goals from the JSON file.

        Returns an empty list if:
          - The file does not exist yet (first run).
          - The file is empty or contains invalid JSON.
        """
        if not os.path.exists(self.filepath):
            return []

        # PROD PATTERN (S3): try/except wraps every file operation
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            # Rebuild each dict into a proper Goal object (S5 factory pattern)
            return [Goal.from_dict(item) for item in raw_data]
        except (json.JSONDecodeError, KeyError) as error:
            print(f"[WARNING] Could not load data file ({error}). Starting fresh.")
            return []

    # ── PUBLIC WRITE OPERATIONS ────────────────────────────────────────────

    def save_all_goals(self):
        """
        Write every goal in memory to the JSON file.

        Called automatically after every change — the file is always up to date.
        indent=2 keeps the JSON human-readable (important for debugging in production).
        """
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(
                    [goal.to_dict() for goal in self.goals],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except OSError as error:
            print(f"[ERROR] Could not save data: {error}")

    def add_goal(self, goal: Goal):
        """Add a new Goal and immediately persist to disk."""
        self.goals.append(goal)
        self.save_all_goals()

    def mark_goal_complete(self, goal_index: int, date: str) -> bool:
        """
        Mark the goal at position goal_index (within a single day) as complete.

        Args:
            goal_index: Zero-based index within that day's goal list.
            date:       Date string (YYYY-MM-DD).

        Returns:
            True if the goal was found and marked; False if index out of range.
        """
        day_goals = self.get_goals_for_date(date)
        if 0 <= goal_index < len(day_goals):
            day_goals[goal_index].mark_complete()
            self.save_all_goals()
            return True
        return False

    # ── PUBLIC READ OPERATIONS ─────────────────────────────────────────────

    def get_goals_for_date(self, date: str) -> list:
        """Return all goals whose date matches the given YYYY-MM-DD string."""
        # List comprehension (S2) — one line, readable, no temporary variables
        return [goal for goal in self.goals if goal.date == date]

    def get_todays_goals(self) -> list:
        """Return all goals for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_goals_for_date(today)

    def get_weekly_summary(self) -> dict:
        """
        Aggregate the last WEEKLY_SUMMARY_DAYS days into a summary dictionary.

        Returns:
            {
              "2025-01-07": {"total": 3, "complete": 2},
              "2025-01-06": {"total": 5, "complete": 5},
              ...
            }
        """
        summary = {}
        today = datetime.now()

        for days_back in range(WEEKLY_SUMMARY_DAYS):
            date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            goals_that_day = self.get_goals_for_date(date)
            summary[date] = {
                "total":    len(goals_that_day),
                "complete": sum(1 for g in goals_that_day if g.is_complete()),
            }
        return summary
