# Daily Focus Tracker

A command-line productivity tool that helps you set daily goals, track your progress, and stay motivated.

**Portfolio Project #1 — Agentic AI Builders Fellowship, Phase 1**

---

## What It Does

- Add up to 10 goals per day with a priority level (high / medium / low)
- Mark goals as complete throughout the day
- View a 7-day weekly summary with a visual progress bar
- Fetch a motivational quote from a live API
- All data persists between sessions in a local JSON file

---

## Skills Demonstrated

| Session | Concept | Where |
|---------|---------|-------|
| S1 | Variables, types, f-strings | `display.py`, `main.py` |
| S2 | Conditions, loops, comprehensions | `storage.py`, `display.py` |
| S3 | Functions, error handling | All modules |
| S4 | Data structures, named constants | `config.py`, `storage.py` |
| S5 | OOP — classes and methods | `goal.py`, `storage.py` |
| S6 | File I/O + JSON | `storage.py` |
| S7 | Modular project structure | Separate `.py` files |
| S8 | HTTP requests, env vars, validation | `quote_api.py`, `validators.py` |
| S9 | Git-ready (README + .gitignore) | This file |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/daily-focus-tracker.git
cd daily-focus-tracker

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (optional)
cp .env.example .env
# Edit .env if you have a ZenQuotes API key

# 5. Run the tracker
python main.py
```

---

## Project Structure

```
daily_focus_tracker/
├── main.py          ← Entry point and main application loop
├── goal.py          ← Goal data model (OOP)
├── storage.py       ← JSON-backed persistent store
├── validators.py    ← Input validation functions
├── quote_api.py     ← Motivational quote API client
├── display.py       ← All terminal output formatting
├── config.py        ← Named constants
├── .env.example     ← Environment variable template
├── .gitignore       ← Keeps secrets and data off GitHub
├── requirements.txt ← Python dependencies
└── data/            ← Created automatically on first run
    └── goals.json   ← Your goal data (not committed to git)
```

---

*Built with Python 3.10+ · Agentic AI Builders Fellowship · I Build. I Ship. I Teach.*
