"""
Session 10 — Python Capstone: CLI Productivity Tool
Module: goal.py — Goal data model

DEMONSTRATES (S5 — OOP for AI Builders):
  - Class definition with __init__
  - Instance attributes (self.title, self.status, …)
  - Instance methods (mark_complete, is_complete, to_dict)
  - Class method (from_dict) — the factory pattern
  - Basic OOP scoped to AI builder needs

A Goal represents one thing a student commits to completing today.
It knows how to save itself to a dictionary (for JSON storage) and
how to rebuild itself from a dictionary (when loaded back from disk).
"""

from datetime import datetime
from config import STATUS_COMPLETE, STATUS_PENDING, PRIORITY_LEVELS


class Goal:
    """Represents a single daily focus goal."""

    def __init__(self, title: str, priority: str = "medium", date: str = None):
        """
        Create a new Goal.

        Args:
            title:    Plain-English description of what to accomplish.
            priority: One of "high", "medium", "low". Defaults to "medium".
            date:     Date string in YYYY-MM-DD format. Defaults to today.
        """
        self.title    = title
        # Guard against invalid priorities — fall back to "medium"
        self.priority = priority if priority in PRIORITY_LEVELS else "medium"
        self.date     = date or datetime.now().strftime("%Y-%m-%d")
        self.status   = STATUS_PENDING
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def mark_complete(self):
        """Mark this goal as done."""
        self.status = STATUS_COMPLETE

    def is_complete(self) -> bool:
        """Return True if this goal has been marked complete."""
        return self.status == STATUS_COMPLETE

    def to_dict(self) -> dict:
        """
        Convert this Goal to a plain dictionary so it can be saved as JSON.
        Every field that matters is included — nothing is lost on disk.
        """
        return {
            "title":      self.title,
            "priority":   self.priority,
            "date":       self.date,
            "status":     self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        """
        Rebuild a Goal object from a dictionary loaded from JSON.

        This is the factory pattern — one clean place to reconstruct objects
        from raw data, rather than scattering the logic around the codebase.
        """
        goal = cls(
            title    = data["title"],
            priority = data.get("priority", "medium"),
            date     = data.get("date"),
        )
        goal.status     = data.get("status", STATUS_PENDING)
        goal.created_at = data.get("created_at", "")
        return goal
