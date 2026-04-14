"""
Session 06 — File Handling + Data Formats (including JSON)
Demonstrates: CSV reading, JSON writing, the read → transform → write pipeline pattern
This is the pattern that connects every data source to every AI system in production.
"""

import csv
import json

# Named constants — change these here, not scattered through the code (Production Pattern from S4)
INPUT_CSV_FILE  = "students.csv"
OUTPUT_JSON_FILE = "students_processed.json"
PASSING_SCORE   = 60   # Anything below this score is flagged as a failure


def create_sample_csv():
    """Create a sample CSV file for the demo — so the demo is fully self-contained."""
    # In production, this file would come from a form, database export, or API response
    sample_records = [
        {"name": "Priya Sharma",   "score": "88", "subject": "Data Science"},
        {"name": "Alex Chen",      "score": "55", "subject": "Machine Learning"},
        {"name": "Ravi Kumar",     "score": "92", "subject": "Data Science"},
        {"name": "Sarah Johnson",  "score": "47", "subject": "AI Ethics"},
        {"name": "Diego Ramirez",  "score": "61", "subject": "Data Science"},
        {"name": "Ananya Nair",    "score": "83", "subject": "AI Ethics"},
        {"name": "James Park",     "score": "39", "subject": "Machine Learning"},
    ]
    with open(INPUT_CSV_FILE, "w", newline="", encoding="utf-8") as csv_file:
        # DictWriter uses column names — much safer than writing raw comma-separated values
        field_names = ["name", "score", "subject"]
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()     # Write the column name row first
        writer.writerows(sample_records)
    print(f"[Pipeline] Created: {INPUT_CSV_FILE} with {len(sample_records)} student records.")


def calculate_grade(numeric_score):
    """Convert a numeric score to a letter grade — same logic used in real grade systems."""
    if numeric_score >= 90:
        return "A"
    elif numeric_score >= 75:
        return "B"
    elif numeric_score >= 60:
        return "C"
    else:
        return "F"


def read_csv_records(file_path):
    """Read all records from a CSV file and return them as a list of dictionaries."""
    try:
        with open(file_path, "r", encoding="utf-8") as csv_file:
            # DictReader maps each row to a dict using the header row as keys
            reader = csv.DictReader(csv_file)
            all_records = list(reader)   # Convert the reader to a list so we can use it outside the `with` block
        print(f"[Pipeline] Read: {len(all_records)} records from {file_path}")
        return all_records
    except FileNotFoundError:
        print(f"[Pipeline] Error: {file_path} not found. Run create_sample_csv() first.")
        return []


def process_records(raw_records):
    """Transform raw CSV records: convert types, add grade, flag failures."""
    processed_records = []
    for raw_record in raw_records:
        # Scores come in as strings from CSV — convert to int before comparing
        numeric_score = int(raw_record["score"])
        processed_record = {
            "name":    raw_record["name"],
            "subject": raw_record["subject"],
            "score":   numeric_score,
            "grade":   calculate_grade(numeric_score),
            "passed":  numeric_score >= PASSING_SCORE,   # Boolean — True or False
        }
        processed_records.append(processed_record)

    failure_count = sum(1 for r in processed_records if not r["passed"])
    print(f"[Pipeline] Processed: {len(processed_records)} records. {failure_count} failure(s) flagged.")
    return processed_records


def write_json_output(records, file_path):
    """Write processed records to a JSON file — the output format AI pipelines expect."""
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            # indent=2 makes the file readable by humans — always use this for output files
            json.dump(records, json_file, indent=2)
        print(f"[Pipeline] Written: {len(records)} records to {file_path}")
    except IOError as file_error:
        print(f"[Pipeline] Write failed: {file_error}")


def show_json_sample(file_path):
    """Read back the JSON file and display the first two records — verify the output."""
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            loaded_records = json.load(json_file)
        print(f"\n[Pipeline] First 2 records from {file_path}:")
        for record in loaded_records[:2]:
            print(f"  {json.dumps(record, indent=2)}")
    except FileNotFoundError:
        print(f"[Pipeline] Cannot show sample: {file_path} not found.")


# ── PIPELINE EXECUTION ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("DATA PIPELINE: CSV → Process → JSON")
    print("=" * 55)

    # Step 1: Create sample data (in production, this file already exists)
    create_sample_csv()

    # Step 2: Read the raw CSV
    raw_data = read_csv_records(INPUT_CSV_FILE)

    # Step 3: Transform — add grades and pass/fail flags
    processed_data = process_records(raw_data)

    # Step 4: Write clean JSON output
    write_json_output(processed_data, OUTPUT_JSON_FILE)

    # Step 5: Verify — read back and display a sample
    show_json_sample(OUTPUT_JSON_FILE)

    print("\n[Pipeline] Complete. This pattern runs inside every AI data system.")
