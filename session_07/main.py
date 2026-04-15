from utils.file_handler import read_csv_records, write_json_output
from utils.processor import process_records
import os
from tabulate import tabulate

DATA_DIR = "data"
INPUT_CSV_FILE = os.path.join(DATA_DIR, "students.csv")
OUTPUT_JSON_FILE = os.path.join(DATA_DIR, "stdents_processed.json")

def print_sumamry_table(processed_records):
    rows = [
        [r["name"], r["subject"], r["score"], r["grade"], "✓ Pass" if r["passed"] else "✗ Fail"]
        for r in processed_records
        ]
    headers = ["Name", "Subject", "Score", "Grade", "Result"]
    print("\n"+tabulate(rows, headers=headers, tablefmt="grid"))

def run_pipeline():
    print("=" * 50)
    print("Student Data Pipeline")
    print("="* 50)

    raw_data = read_csv_records(INPUT_CSV_FILE)

    processed_data = process_records(raw_data)

    print_sumamry_table(processed_data)

    write_json_output(processed_data, OUTPUT_JSON_FILE)

    print("[Main] Pipeline complete")


if __name__ == "__main__":
    run_pipeline()