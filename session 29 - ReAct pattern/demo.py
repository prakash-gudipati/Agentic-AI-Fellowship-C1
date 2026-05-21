"""
Session 29 — demo.py

The runnable entrypoint we walk through on screen share.

Three questions are baked in — each one exercises a different aspect of
the loop:

  Q1. Pure-calc question        → exercises the calculator tool only
                                  (1 tool call, single Thought → Action turn)
  Q2. Compound research + calc  → exercises web_search TWICE + calculator
                                  (the textbook ReAct trace)
  Q3. "Today" question          → exercises the datetime tool
                                  (the model has to recognise it needs a date)

Run it with:

    python demo.py            # runs all three
    python demo.py 2          # runs only Q2 (the compound one)

Production note: this file does almost nothing on purpose. The agent
orchestration is in agent.py. demo.py is just the wiring. In production,
your "wiring" file is the one your CI smoke-tests; keeping it boring is
the whole point.
"""

from __future__ import annotations

import sys

from agent import ReactAgent


DEMO_QUESTIONS = {
    1: "If a freelance contract pays 850 dollars and the platform takes 12 percent, "
       "how many dollars does the freelancer actually receive?",
    2: "What is the GDP of India in 2024 divided by its population in 2024? "
       "Give the answer in dollars per person, rounded to the nearest dollar.",
    3: "What is today's date in UTC, and how many days until 1 January next year?",
}


def main() -> None:
    args = sys.argv[1:]
    if args:
        question_ids = [int(a) for a in args]
    else:
        question_ids = list(DEMO_QUESTIONS.keys())

    agent = ReactAgent()

    for question_id in question_ids:
        question = DEMO_QUESTIONS[question_id]
        print("\n" + "=" * 78)
        print(f"DEMO QUESTION {question_id}")
        print("=" * 78)
        agent.run(question)


if __name__ == "__main__":
    main()
