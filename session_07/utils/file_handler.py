import csv
import json

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

def write_json_output(records, file_path):
    """Write processed records to a JSON file — the output format AI pipelines expect."""
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            # indent=2 makes the file readable by humans — always use this for output files
            json.dump(records, json_file, indent=2)
        print(f"[Pipeline] Written: {len(records)} records to {file_path}")
    except IOError as file_error:
        print(f"[Pipeline] Write failed: {file_error}")