# README Writing Rules
### Agentic AI Builders Fellowship — Session 09
**Phase 1: Python for AI · I Build. I Ship. I Teach.**

---

## WHY YOUR README MATTERS

A README is the first thing a hiring manager sees when they open your repository. Without one, they see a folder of files with no context. With a good one, they understand your project in 30 seconds — what it does, what skills it shows, and how to run it.

**Rule:** Every project you push to GitHub must have a README. No exceptions.

---

## THE 6-SECTION STRUCTURE

Every Phase 1 README must contain these sections in this order:

```
1. Project Title (H1)
2. What It Does
3. What It Demonstrates
4. How to Run
5. Example Output  (optional but recommended)
6. Built During
```

---

## SECTION-BY-SECTION RULES

### 1. Project Title

```markdown
# Weather Dashboard CLI
```

- Use a single `#` for the title — this is an H1 heading, the largest.
- Keep it short: Project Name + what kind of thing it is (CLI, web app, API, etc.).
- No version numbers, dates, or jargon in the title.
- Bad:  `# My Project v2 FINAL`
- Good: `# Recipe Finder CLI`

---

### 2. What It Does

```markdown
## What It Does

A command-line tool that fetches live weather data for any city using the
Open-Meteo API. Enter a city name and get current temperature, humidity,
and wind speed in a formatted report. Handles invalid city names gracefully
with a clear error message instead of crashing.
```

**Rules:**
- Write 2–4 sentences in plain English.
- Say what the user experiences, not what the code does internally.
- Mention the main input and the main output.
- Mention one notable behaviour (e.g., how it handles errors).
- Bad:  `"This program uses the requests library to call an API."`
- Good: `"Enter a city name. Get a live weather report in under 2 seconds."`

---

### 3. What It Demonstrates

```markdown
## What It Demonstrates

- REST API calls using the `requests` library
- Environment variable management with `python-dotenv` (API keys never in code)
- JSON response parsing and data extraction
- Input validation before every API call
- Error handling with `try/except` for HTTP failures and invalid inputs
- Professional project structure: virtual environment, `.env`, `requirements.txt`
```

**Rules:**
- Use a bullet list — one skill per line.
- Include 4–8 items. More than 8 becomes noise.
- Each bullet starts with the skill concept, not the library name.
  - Bad:  `- requests`
  - Good: `- REST API calls using the requests library`
- Include the library or tool name in backticks after the concept.
- This section is what a hiring manager scans to decide if your skills match a job.
- Update this list every time you add a new capability to the project.

---

### 4. How to Run

```markdown
## How to Run

**Requirements:** Python 3.10+

**1. Clone the repository**
```bash
git clone https://github.com/your-username/weather-dashboard-cli.git
cd weather-dashboard-cli
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your environment variables**
```bash
# Create a .env file in the project root
# Add your API key (get one free at your-api-provider.com):
API_KEY=your_api_key_here
```

**5. Run the program**
```bash
python main.py
```
```

**Rules:**
- Number every step — never use bullets in How to Run.
- Show the **exact command** on its own line inside a code block.
- Separate macOS/Linux from Windows commands where they differ.
- Never skip the virtual environment step — it is always Step 2.
- Always include the `.env` setup step if your project uses API keys.
- The reader should be able to copy-paste each step and have it work.
- Test these steps yourself in a fresh folder before submitting.

---

### 5. Example Output (optional but recommended)

```markdown
## Example Output

```
Enter a city name: Mumbai

--- Weather Report ---
City:        Mumbai, India
Temperature: 32°C
Humidity:    78%
Wind Speed:  14 km/h
Condition:   Partly Cloudy
----------------------
```
```

**Rules:**
- Show the actual terminal output — copy-paste from a real run.
- Include the user input line so the reader sees the full interaction.
- Use a plain code block (triple backticks, no language tag).
- Keep it short: one successful example is enough.
- Do not show error messages as the main example.

---

### 6. Built During

```markdown
## Built During

[Agentic AI Builders Fellowship](https://your-course-url.com) — Phase 1: Python for AI
Instructor: Prakash Gudipati · *I Build. I Ship. I Teach.*
```

**Rules:**
- Always include this section — it attributes the course and adds context.
- Link to the course or cohort page if one exists.
- Keep it to 1–2 lines.

---

## MARKDOWN FORMATTING RULES

```markdown
# H1 — Project title only. Use once.
## H2 — Section headings (What It Does, How to Run, etc.)
### H3 — Sub-headings inside a section

**bold text**   — for important words or labels
`inline code`   — for file names, library names, commands, keys
> blockquote    — for important notes or warnings (use sparingly)

- bullet item   — for lists (What It Demonstrates, etc.)
1. numbered item — for steps (How to Run)

```bash
code block      — for terminal commands and multi-line code
```              — always specify the language (bash, python, etc.)
```

**Key rules:**
- Put a blank line before every heading, bullet list, and code block — required for correct rendering.
- Use `backticks` around any library name, file name, command, or technical term the first time it appears inline.
- Never use ALL CAPS for emphasis in body text — use **bold** instead.
- One blank line between sections is enough — two blank lines is fine, three or more looks broken.

---

## COMMON MISTAKES

| Mistake | Why it fails | Fix |
|---------|-------------|-----|
| No README at all | Visitor sees a wall of files with no context | Always create README.md before pushing |
| README says "This project..." | Third person — sounds impersonal | Write directly: "Enter a city. Get a report." |
| How to Run has no virtual env step | Reader's pip install pollutes global Python | Always include `python -m venv venv` as Step 2 |
| Code block not closed | Everything after becomes a code block | Always close with three backticks on their own line |
| `.env` setup missing | Reader can't run project — API key undefined | Include the .env step with what variable name to set |
| Heading missing space after # | `#Title` does not render as H1 | Always: `# Title` (space after hash) |
| How to Run uses bullets | Steps need order — bullets have no order | Always use numbered list for steps |
| What It Demonstrates lists libraries only | "requests" tells nothing — skill tells everything | Write the skill, then the library: "API calls with requests" |
| README never updated after adding features | Stale README misleads reviewers | Update What It Demonstrates every time you add a capability |
| Copy-pasted another project's README | Hiring managers notice immediately | Write it fresh for every project — 15 minutes well spent |

---

## GOOD README CHECKLIST (run before submitting)

```
[ ] Project title is clear and specific (not "My Project")
[ ] What It Does: 2–4 sentences, plain English, mentions input + output
[ ] What It Demonstrates: 4–8 bullets, skill first then library
[ ] How to Run: numbered steps, all commands in code blocks, .env step present
[ ] All code blocks are opened AND closed with triple backticks
[ ] Every heading has a space after the # symbol
[ ] Blank line before every heading and every list
[ ] Tested: can a new person follow How to Run and get it working?
[ ] Built During section credits the course
[ ] No spelling mistakes in section headings
```

---

## TEMPLATE (copy this and fill in the blanks)

```markdown
# [Project Name]

## What It Does

[2–4 sentences. What does the user type in? What do they get back?
Mention one error-handling behaviour.]

## What It Demonstrates

- [Skill 1 — e.g., REST API calls using the `requests` library]
- [Skill 2 — e.g., Environment variable management with `python-dotenv`]
- [Skill 3 — e.g., JSON response parsing and data extraction]
- [Skill 4 — e.g., Input validation before every external call]
- [Skill 5 — e.g., Error handling with `try/except` for HTTP failures]
- [Add more as relevant to your project]

## How to Run

**Requirements:** Python 3.10+

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your environment variables**
Create a `.env` file in the project root with:
```
API_KEY=your_api_key_here
```

**5. Run the program**
```bash
python main.py
```

## Example Output

```
[Paste your real terminal output here]
```

## Built During

[Agentic AI Builders Fellowship](https://your-course-url.com) — Phase 1: Python for AI
Instructor: Prakash Gudipati · *I Build. I Ship. I Teach.*
```

---

*Agentic AI Builders Fellowship · Session 09 · README Writing Rules · v1.0 · April 2026*
