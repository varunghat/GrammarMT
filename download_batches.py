from pathlib import Path
from openai import OpenAI
import time

# --- Load API key ---
with open("api_key.txt", "r") as f:
    client = OpenAI(api_key=f.read().strip())

# --- Paths ---
batch_log_path = Path("./outputs/batch_job_ids.txt")
output_dir = Path("./outputs/completions")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load all batch IDs ---
batch_jobs = []
with open(batch_log_path, "r", encoding="utf-8") as log:
    for line in log:
        if not line.startswith("#") and line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                batch_jobs.append((parts[0], parts[1]))  # (filename, batch_id)

print(f"Found {len(batch_jobs)} batches to check.\n")

for file_name, batch_id in batch_jobs:
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"{file_name} — Status: {batch.status}")

        if batch.status != "completed":
            print("Skipping (not completed yet)\n")
            continue

        # --- Download output file ---
        if batch.output_file_id:
            output_file_path = output_dir / f"{file_name}_output.jsonl"
            content = client.files.content(batch.output_file_id)
            with open(output_file_path, "wb") as f:
                f.write(content.read())
            print(f"Saved output to {output_file_path}")

        # --- Download error file (if any) ---
        if batch.error_file_id:
            error_file_path = output_dir / f"{file_name}_errors.jsonl"
            content = client.files.content(batch.error_file_id)
            with open(error_file_path, "wb") as f:
                f.write(content.read())
            print(f"Saved errors to {error_file_path}")

        print()

    except Exception as e:
        print(f"Error retrieving {batch_id}: {e}\n")
        time.sleep(2)

print("All available batch outputs downloaded.")
