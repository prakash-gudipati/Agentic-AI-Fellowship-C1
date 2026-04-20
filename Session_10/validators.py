"""
Session 10 — Python Capstone: CLI Productivity Tool
Module: validators.py — Input validation functions

PROD PATTERN (S8): Validate ALL user input BEFORE processing it.
Every function returns a (is_valid: bool, error_message: str) tuple.
The caller checks is_valid before doing anything with the data.
This pattern prevents bad data from reaching the storage layer.

DEMONSTRATES:
  S3 (Functions + return values) — each function does exactly one job
  S8 (Input validation)          — check first, process second
  S4 (Named constants)           — PRIORITY_LEVELS and MAX_GOALS_PER_DAY from config
"""

from config import PRIORITY_LEVELS, MAX_GOALS_PER_DAY


def validate_goal_title(title: str) -> tuple:
    """
    Check that a goal title is non-empty and within the length limit.

    Returns:
        (True, "")          — if valid
        (False, "<reason>") — if invalid
    """
    if not title or not title.strip():
        return False, "Goal title cannot be empty."
    if len(title.strip()) > 120:
        return False, "Goal title must be 120 characters or fewer."
    return True, ""


def validate_priority(priority: str) -> tuple:
    """
    Check that the priority is one of the accepted levels.

    Returns:
        (True, "")          — if valid
        (False, "<reason>") — if invalid
    """
    if priority.lower() not in PRIORITY_LEVELS:
        levels_str = ", ".join(PRIORITY_LEVELS)
        return False, f"Priority must be one of: {levels_str}"
    return True, ""


def validate_goal_count(current_count: int) -> tuple:
    """
    Check whether the user can add another goal today without exceeding the limit.

    Returns:
        (True, "")          — if adding is allowed
        (False, "<reason>") — if the daily limit has been reached
    """
    if current_count >= MAX_GOALS_PER_DAY:
        return False, f"You have reached the daily limit of {MAX_GOALS_PER_DAY} goals."
    return True, ""


def get_valid_menu_choice(prompt: str, valid_choices: list) -> str:
    """
    Keep asking the user for input until they enter a valid choice.

    Args:
        prompt:        The text shown to the user.
        valid_choices: A list of acceptable string inputs (case-insensitive).

    Returns:
        The validated choice, stripped and lowercased.
    """
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in valid_choices:
            return user_input
        valid_str = ", ".join(valid_choices)
        print(f"  [ERROR] Please enter one of: {valid_str}")
