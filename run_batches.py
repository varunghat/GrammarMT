import os
import time
from pathlib import Path
from openai import OpenAI

# --- Load API key ---
with open("api_key.txt", "r") as f:
    client = OpenAI(api_key=f.read().strip())

# --- Folder with batch .jsonl files ---
batch_folder = Path("./generation_prompts/kalamang")

# --- Output file to store batch IDs ---
batch_log_path = Path("./outputs/batch_job_ids.txt")
batch_log_path.parent.mkdir(parents=True, exist_ok=True)

# --- Batch settings ---
COMPLETION_WINDOW = "24h"
POLL_INTERVAL = 15 * 60  # 15 minutes
CONFIRM_SUBMIT = True

# --- Load already submitted batches (if resuming) ---
submitted_files = set()
if batch_log_path.exists():
    with open(batch_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                submitted_files.add(line.split("\t")[0])

# --- Helper function to wait for a batch to finish ---
def wait_for_completion(client, batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch.status
        print(f"Waiting {POLL_INTERVAL/60} minutes...")
        time.sleep(POLL_INTERVAL)

# --- Prepare log file header if new ---
if not batch_log_path.exists() or os.stat(batch_log_path).st_size == 0:
    with open(batch_log_path, "w", encoding="utf-8") as log:
        log.write("# Batch Jobs Created\n")

# --- Main loop ---
for file in sorted(os.listdir(batch_folder)):
    if not (file.endswith(".jsonl") and "_part" in file):
        continue

    if file in submitted_files:
        print(f"Skipping {file} (already submitted)")
        continue

    batch_path = batch_folder / file
    with open(batch_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    print(f"{file}: {line_count} requests")

    if not CONFIRM_SUBMIT:
        print("Dry run mode — skipping upload.")
        continue

    # --- Upload and create batch ---
    uploaded = client.files.create(file=open(batch_path, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
    )

    print(f"Submitted batch job {batch_job.id} for {file}")
    print(f"Initial status: {batch_job.status}\n")

    # --- Save to log ---
    with open(batch_log_path, "a", encoding="utf-8") as log:
        log.write(f"{file}\t{batch_job.id}\t{line_count} requests\n")

    # --- Wait until it finishes before submitting next one ---
    final_status = wait_for_completion(client, batch_job.id)
    print(f"Batch {batch_job.id} finished with status: {final_status}\n")

print("All batch jobs processed sequentially!")
print(f"Log saved at: {batch_log_path}")
