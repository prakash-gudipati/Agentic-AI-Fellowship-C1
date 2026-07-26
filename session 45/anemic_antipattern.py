"""
Session 45 — what Claude Code writes WITHOUT a glossary (the anti-pattern).

Same problem, but the language is gone. Generic tech words, no invariants, a
'status' string anyone can set to anything, a UserManager that knows everything
and means nothing. This 'works' — and it is the wrong thing, built correctly.
"""


class UserManager:
    def __init__(self):
        self.data = {}            # what's in here? nobody knows from the name

    def process_data(self, user_id, items, info):
        # one method, four responsibilities, zero rules
        self.data[user_id] = {
            "items": items,
            "status": "new",       # free-text status — drifts instantly
            "info": info,
        }

    def update(self, user_id, status):
        self.data[user_id]["status"] = status   # no rule stops "placed" before "confirmed"
