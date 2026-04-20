"""
Session 10 — Python Capstone: CLI Productivity Tool
Module: config.py — Named constants for the Daily Focus Tracker

PROD PATTERN (S4): Named constants at the top of the file.
Every value that might change lives here — not scattered through the code.
"""

# Maximum number of goals allowed per day
MAX_GOALS_PER_DAY = 10

# Path to the JSON file where all goals are stored (relative to project root)
DATA_FILE = "data/goals.json"

# Number of past days to include in the weekly summary display
WEEKLY_SUMMARY_DAYS = 7

# Public API endpoint — ZenQuotes, free tier, no key required
QUOTE_API_URL = "https://zenquotes.io/api/random"

# Allowed values for a goal's priority field
PRIORITY_LEVELS = ["high", "medium", "low"]

# String constants for goal status — avoids typos in comparisons
STATUS_COMPLETE = "complete"
STATUS_PENDING  = "pending"
