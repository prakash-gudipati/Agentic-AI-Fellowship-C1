PASSING_SCORE   = 60   # Anything below this score is flagged as a failure

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