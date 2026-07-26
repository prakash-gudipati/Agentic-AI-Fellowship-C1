# PrepDeck — Engineering Handbook

This handbook is the source of truth for how the PrepDeck engineering team
works. Last updated April 2026.

## Team Structure

The engineering team has 14 full-time engineers as of April 2026:
- 9 engineers in Bengaluru
- 4 engineers in Austin
- 1 staff engineer (CTO) splits time across both offices

The team is organised into three squads:
1. **Platform** (4 engineers) — auth, billing, data pipelines, deployment
2. **AI Mentor** (5 engineers) — the agent that reviews code and runs mocks
3. **Frontend** (4 engineers) — web app, mobile app, in-product UX

The CTO reports to the CEO. Each squad has a tech lead who reports to the CTO.

## On-Call Rotation

PrepDeck runs a follow-the-sun on-call rotation. Each squad has its own
rotation. A primary engineer is on call for 7 days at a time. The secondary
engineer is the next person in the rotation and steps in when the primary is
in deep focus or asleep.

On-call shifts run from Monday 09:00 IST to the following Monday 09:00 IST.
The Austin engineers cover the same 7-day blocks but their primary shift hours
are 09:00 CST to 21:00 CST.

If a P0 incident fires outside the primary's working hours, the secondary is
paged. The on-call engineer commits to a 15-minute first-response SLA for P0,
and a 4-hour first-response SLA for P1.

## Code Review

Every change to main requires at least one approving review from someone outside
the author's squad. Two approvals are required if the change touches billing,
auth, or the AI mentor prompt templates.

The team uses GitHub. Pull requests must include a test plan, a screenshot or
log snippet for any user-facing change, and a one-line summary of the production
risk.

## Deployment Cadence

The Platform squad deploys to production daily at 14:00 IST. The AI Mentor
squad deploys twice a week — Tuesday and Thursday at 14:00 IST. The Frontend
squad deploys weekly on Wednesday at 14:00 IST.

All deploys go through a staging environment for at least 30 minutes before
promotion to production.
