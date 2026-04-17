# Git Command Reference
### Agentic AI Builders Fellowship — Session 09
**Phase 1: Python for AI · I Build. I Ship. I Teach.**

---

## SETUP (run once per project)

```bash
git init
# Tells Git to start watching this folder.
# Creates a hidden .git/ folder that stores all history.
# Run this inside your project root — never inside another Git repo.

git remote add origin https://github.com/your-username/your-repo.git
# Connects your local repo to a GitHub URL.
# "origin" is the standard name for your remote — don't change it.
# Copy the exact URL from your GitHub repo page.

git branch -M main
# Renames your default branch to "main".
# Modern standard — older Git versions used "master" by default.
# Run once after git init, before the first push.
```

---

## DAILY WORKFLOW (use in this order, every time)

```bash
git status
# Shows what has changed since your last commit.
# Green = staged (ready to commit).
# Red = modified or untracked (not yet staged).
# Run this before every commit — confirm exactly what you're saving.

git add .
# Stages ALL changed files in the current folder (except what .gitignore lists).
# The dot means "everything here" — Git figures out what changed.
# Alternative: git add filename.py  — to stage one specific file only.

git add -p
# Interactive staging — shows each change and asks: stage this? (y/n).
# Use when you have multiple changes but only want to commit some of them.
# More precise than git add . — production teams use this daily.

git commit -m "type: what changed — why it matters"
# Saves a snapshot of all staged files with your message.
# The message is permanent — it stays in the project history forever.
# PROD PATTERN: every message answers WHAT changed and WHY.
# Bad: "fix"   Good: "fix: crash when city name is empty — missing input validation"

git push
# Uploads your local commits to GitHub.
# After the first push (-u origin main), just "git push" is enough.
# Nothing reaches GitHub until you push — commits are local until then.

git pull
# Downloads the latest commits from GitHub to your machine.
# Use before starting work when collaborating with others.
# Prevents conflicts from building up between your local and remote versions.
```

---

## BRANCHING (parallel workbench)

```bash
git checkout -b dev
# Creates a new branch called "dev" AND switches to it in one command.
# -b means "create branch". Without -b, you just switch to an existing one.
# Your main branch is untouched while you work on dev.
# Naming conventions: dev, feature/add-logging, fix/empty-input-crash

git branch
# Lists all local branches.
# The asterisk (*) shows which branch you are currently on.
# Run this when you are not sure which branch you are on.

git checkout main
# Switches back to the main branch.
# Always switch to main BEFORE running git merge.
# If you have uncommitted changes, Git will warn you — commit or stash first.

git merge dev
# Brings the commits from "dev" into the current branch (main).
# If no conflicts: Git auto-merges and creates a merge commit.
# If conflicts: Git marks the conflict in the file — you resolve manually.
# After merging, your dev branch still exists — delete it if no longer needed.

git branch -d dev
# Deletes the dev branch after a successful merge.
# -d is safe — Git refuses to delete an unmerged branch.
# Use -D (capital) to force-delete even if unmerged (use with caution).
```

---

## INSPECTING HISTORY

```bash
git log
# Shows the full commit history — hash, author, date, and message.
# Most recent commit is at the top.
# Press q to exit the log viewer.

git log --oneline
# Compact view: one commit per line — hash + message only.
# Best for quickly scanning what changed over the last several commits.
# Example output:
#   f02b7e Add README with setup instructions
#   d7f190 Fix crash when city name contains spaces
#   c8e4d2 Add weather API integration

git log --oneline --graph
# Adds an ASCII diagram showing branch and merge history.
# Useful for understanding how branches connected over time.

git diff
# Shows line-by-line what has changed since your last commit (unstaged).
# Lines starting with + are additions. Lines starting with - are removals.
# Use before git add to confirm you know exactly what changed.

git show f02b7e
# Shows the full diff for a specific commit (replace with any hash).
# Use the 7-character short hash from git log --oneline.
# Useful for understanding exactly what a past commit changed.
```

---

## UNDOING THINGS

```bash
git restore filename.py
# Discards all unsaved changes to a file and restores the last committed version.
# DESTRUCTIVE — the discarded changes are gone permanently.
# Use when you want to throw away experimental changes on a single file.

git restore --staged filename.py
# Removes a file from the staging area (undoes git add) without changing the file.
# Use when you accidentally staged something you don't want in this commit.

git revert abc1234
# Creates a NEW commit that undoes the changes from a specific past commit.
# NON-DESTRUCTIVE — the original commit stays in history, a reversal is added.
# Safe to use on shared branches because it doesn't rewrite history.
# Preferred over git reset for any commits already pushed to GitHub.

git stash
# Temporarily saves all uncommitted changes and restores a clean working state.
# Use when you need to switch branches but aren't ready to commit.
git stash pop
# Brings the stashed changes back to your working directory.
```

---

## .GITIGNORE RULES (add these before your first commit)

```
venv/           # Virtual environment — rebuild from requirements.txt, never commit
.env            # API keys and secrets — NEVER goes to GitHub
__pycache__/    # Auto-generated bytecode — not part of your source code
*.pyc           # Compiled Python files — auto-generated, not needed by others
.DS_Store       # macOS system file — irrelevant outside macOS
*.log           # Log files — environment-specific, not code
.idea/          # PyCharm/JetBrains IDE settings — not needed by others
.vscode/        # VS Code workspace settings — optional, team-dependent
```

---

## PRODUCTION PATTERN: COMMIT MESSAGE FORMAT

```
type: what changed — why it matters
```

| Type | When to use |
|------|-------------|
| `add` | New feature, new file, new function |
| `fix` | Bug fix, crash fix, error handling added |
| `refactor` | Restructured existing code — no new behaviour |
| `docs` | README, comments, docstrings only |
| `chore` | Dependencies, config files, .gitignore |

**Examples:**
```bash
git commit -m "add: weather API call with Open-Meteo — provides live data for city queries"
git commit -m "fix: crash when city name is empty — added input validation before API call"
git commit -m "refactor: extract input validation to single function — removes 4 duplicates"
git commit -m "docs: add README with setup instructions and skills demonstrated"
git commit -m "chore: add venv/ and .env to .gitignore — prevents secrets from being committed"
```

---

## QUICK REFERENCE CARD

| Command | What it does | When to run |
|---------|-------------|-------------|
| `git init` | Start Git tracking | Once per project |
| `git status` | See what changed | Before every commit |
| `git add .` | Stage all changes | After making changes |
| `git commit -m '...'` | Save a snapshot | After staging |
| `git push` | Upload to GitHub | After committing |
| `git pull` | Download from GitHub | Before starting work |
| `git checkout -b name` | Create + switch branch | Start of new feature |
| `git checkout main` | Switch to main | Before merging |
| `git merge branch` | Bring branch into main | When feature is ready |
| `git log --oneline` | Read history | Any time |
| `git diff` | See line changes | Before git add |

---

*Agentic AI Builders Fellowship · Session 09 · Git Command Reference · v1.0 · April 2026*
