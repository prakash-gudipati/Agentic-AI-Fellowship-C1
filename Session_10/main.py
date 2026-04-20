"""
Session 10 — Python Capstone: CLI Productivity Tool
=======================================================
Project: Daily Focus Tracker
File:    main.py — entry point and main application loop

What it does:
  - Lets you add goals for today with a priority level
  - Marks goals as complete
  - Shows a weekly summary with a visual progress bar
  - Fetches a motivational quote from a real API
  - Saves all data to a JSON file so nothing is lost between runs

EVERY PHASE 1 PRODUCTION PATTERN IS ACTIVE IN THIS PROJECT:

  Pattern                         | Introduced | Where Used
  ──────────────────────────────────────────────────────────
  Variables, types, f-strings     | S1         | display.py, main.py
  Control flow, loops, comprehens | S2         | storage.py, display.py
  Single-responsibility functions | S3         | all modules
  try/except error handling       | S3         | storage.py, quote_api.py, main.py
  Named constants at top of file  | S4         | config.py
  Data structures (list + dict)   | S4         | storage.py, display.py
  OOP — classes + instance methods| S5         | goal.py, storage.py
  File I/O + JSON persistence     | S6         | storage.py
  Modular project structure       | S7         | separate .py files
  Input validation before process | S8         | validators.py
  Environment variables for keys  | S8         | quote_api.py, .env
  HTTP requests with error handling| S8        | quote_api.py
  Git-ready (README + .gitignore) | S9         | project root

Run: python main.py
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Load .env FIRST — quote_api.py reads ZENQUOTES_API_KEY from the environment
load_dotenv()

from goal import Goal
from storage import GoalStore
from quote_api import get_motivational_quote
from validators import (
    validate_goal_title,
    validate_priority,
    validate_goal_count,
    get_valid_menu_choice,
)
from display import (
    print_welcome,
    print_menu,
    display_goals,
    display_weekly_summary,
    display_quote,
    print_header,
    print_separator,
)
from config import PRIORITY_LEVELS


# ── ACTION HANDLERS ────────────────────────────────────────────────────────────

def add_goal(store: GoalStore):
    """Prompt for a new goal and save it to today's list."""
    print_header("ADD A GOAL FOR TODAY")

    today       = datetime.now().strftime("%Y-%m-%d")
    todays_goals = store.get_goals_for_date(today)

    # PROD PATTERN (S8): validate BEFORE accepting input
    count_valid, count_error = validate_goal_count(len(todays_goals))
    if not count_valid:
        print(f"  [ERROR] {count_error}")
        return

    # Keep asking until the user enters a non-empty title
    while True:
        title = input("  Goal title: ").strip()
        is_valid, error = validate_goal_title(title)
        if is_valid:
            break
        print(f"  [ERROR] {error}")

    # Keep asking until the user enters a valid priority
    priority_hint = " / ".join(PRIORITY_LEVELS)
    print(f"  Priority options: {priority_hint}")
    while True:
        raw = input("  Priority [default: medium]: ").strip().lower() or "medium"
        is_valid, error = validate_priority(raw)
        if is_valid:
            break
        print(f"  [ERROR] {error}")

    new_goal = Goal(title=title, priority=raw)
    store.add_goal(new_goal)
    print(f'\n  ✅ Added: "{new_goal.title}"  [{new_goal.priority} priority]')


def view_todays_goals(store: GoalStore):
    """Display all of today's goals with status and priority icons."""
    today = datetime.now().strftime("%Y-%m-%d")
    goals = store.get_todays_goals()

    print_header(f"TODAY'S GOALS — {today}")
    display_goals(goals)

    if goals:
        # Comprehension (S2) — count completed goals without a manual loop
        complete_count = sum(1 for goal in goals if goal.is_complete())
        print()
        print(f"  {complete_count} of {len(goals)} complete")


def mark_goal_complete(store: GoalStore):
    """Let the user pick one of today's pending goals to mark as done."""
    today = datetime.now().strftime("%Y-%m-%d")
    goals = store.get_todays_goals()

    print_header("MARK A GOAL COMPLETE")
    display_goals(goals)

    if not goals:
        return

    # Filter with a comprehension (S2) — only show work still to be done
    pending = [g for g in goals if not g.is_complete()]
    if not pending:
        print("\n  All of today's goals are already complete. Well done! 🎉")
        return

    # PROD PATTERN (S3): try/except around user input that could be non-numeric
    try:
        raw = input("\n  Enter goal number to mark complete: ").strip()
        goal_index = int(raw) - 1   # Convert 1-based display to 0-based list index

        success = store.mark_goal_complete(goal_index, today)
        if success:
            updated_goal = store.get_goals_for_date(today)[goal_index]
            print(f'\n  ✅ Marked complete: "{updated_goal.title}"')
        else:
            print("  [ERROR] That number is out of range. Please try again.")
    except ValueError:
        print("  [ERROR] Please enter a number.")


def show_weekly_summary(store: GoalStore):
    """Show a 7-day progress table."""
    summary = store.get_weekly_summary()
    display_weekly_summary(summary)


def show_motivational_quote():
    """Fetch and display a motivational quote from the API."""
    print("  Fetching your quote...")
    quote = get_motivational_quote()
    display_quote(quote)


# ── MAIN LOOP ──────────────────────────────────────────────────────────────────

def run():
    """
    Main application loop — keeps running until the user types 'q'.

    Uses a dictionary (S4) to map menu choices to functions.
    This is called the 'dispatch table' pattern — clean, extendable,
    no long if/elif chains.
    """
    print_welcome()

    store = GoalStore()   # Load existing goals from JSON on startup

    # Dispatch table — maps each menu key to its handler function
    actions = {
        "1": add_goal,
        "2": view_todays_goals,
        "3": mark_goal_complete,
        "4": show_weekly_summary,
        "5": show_motivational_quote,
    }

    while True:
        print_menu()
        choice = get_valid_menu_choice(
            "  Your choice: ",
            valid_choices=list(actions.keys()) + ["q"],
        )

        if choice == "q":
            print_separator()
            print("  See you tomorrow. Keep building! 🚀")
            print_separator()
            break

        handler = actions[choice]

        try:
            # Choice 5 (quote) doesn't need the store — all others do
            if choice == "5":
                handler()
            else:
                handler(store)
        except KeyboardInterrupt:
            print("\n\n  [INFO] Interrupted — returning to menu.")


if __name__ == "__main__":
    run()
