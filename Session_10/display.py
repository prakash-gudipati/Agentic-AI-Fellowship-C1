"""
Session 10 — Python Capstone: CLI Productivity Tool
Module: display.py — All terminal output formatting

DEMONSTRATES:
  S1 (f-strings, print)     — formatted output with f-strings throughout
  S2 (Loops + comprehensions) — building progress bars and goal lists
  S3 (Single-responsibility functions) — each function does one display job
  S4 (Dictionaries)         — PRIORITY_ICONS maps priority → visual symbol

PROD PATTERN: All display logic lives in one module.
When you want to change how something looks, you change one file — not ten.
"""

# Map each priority level to a coloured circle for quick visual scanning
PRIORITY_ICONS = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "🟢",
}

# Separator width in characters
SEPARATOR_WIDTH = 55


def print_separator(char: str = "─", width: int = SEPARATOR_WIDTH):
    """Print a full-width horizontal divider line."""
    print(char * width)


def print_header(title: str):
    """Print a section header surrounded by separator lines."""
    print_separator()
    print(f"  {title}")
    print_separator()


def print_welcome():
    """Print the application banner when the program starts."""
    print_separator("═")
    print("  📋  DAILY FOCUS TRACKER")
    print("  Track your goals. Build your momentum.")
    print_separator("═")


def print_menu():
    """Print the main navigation menu."""
    print()
    print("  What would you like to do?")
    print("  [1] Add a goal for today")
    print("  [2] View today's goals")
    print("  [3] Mark a goal complete")
    print("  [4] Weekly summary")
    print("  [5] Get a motivational quote")
    print("  [q] Quit")
    print()


def display_goals(goals: list, show_date: bool = False):
    """
    Print a numbered, annotated list of goals.

    Each line shows:
      - A completion tick or empty box
      - A colour-coded priority dot
      - The goal title
      - Optionally, the date

    Args:
        goals:     List of Goal objects to display.
        show_date: If True, append the goal date to each line.
    """
    if not goals:
        print("  No goals found.")
        return

    # Enumerate starts at 1 so the numbers match what the user types
    for index, goal in enumerate(goals, start=1):
        status_icon   = "✅" if goal.is_complete() else "⬜"
        priority_icon = PRIORITY_ICONS.get(goal.priority, "⬜")
        date_suffix   = f"  [{goal.date}]" if show_date else ""
        print(f"  {index}. {status_icon} {priority_icon}  {goal.title}{date_suffix}")


def display_weekly_summary(summary: dict):
    """
    Print a formatted 7-day progress table.

    Each row shows:
      - The date
      - Total goals set
      - Goals completed
      - A visual progress bar
      - A completion percentage

    Args:
        summary: Dict returned by GoalStore.get_weekly_summary()
                 Format: { "YYYY-MM-DD": {"total": int, "complete": int} }
    """
    print_header("WEEKLY SUMMARY — Last 7 Days")
    print(f"  {'DATE':<14} {'SET':>5} {'DONE':>5}  {'PROGRESS':<22}")
    print_separator()

    # Sort by date descending — most recent day at the top
    for date, stats in sorted(summary.items(), reverse=True):
        total    = stats["total"]
        complete = stats["complete"]

        # Build a 10-character progress bar using a list comprehension (S2)
        bar_filled = int(complete / total * 10) if total > 0 else 0
        bar = "".join(["█" if i < bar_filled else "░" for i in range(10)])

        percentage = f"{int(complete / total * 100)}%" if total > 0 else "—"
        print(f"  {date:<14} {total:>5} {complete:>5}  {bar}  {percentage}")

    print()


def display_quote(quote: dict):
    """
    Print a motivational quote with author attribution.

    Args:
        quote: Dict with keys "q" (text) and "a" (author).
    """
    print_header("YOUR MOTIVATIONAL QUOTE")
    print()
    print(f'  "{quote["q"]}"')
    print()
    print(f'  — {quote["a"]}')
    print()
