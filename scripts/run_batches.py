import os
import time
import csv
from pathlib import Path
from openai import OpenAI


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val


# --- Load API key ---
client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

# --- Folder with batch .jsonl files ---
filename = "sursilvan_romansh"
batch_folder = Path(f"./generation_prompts/{filename}")

# --- Log file to store batch statuses ---
batch_log_path = Path(f"./outputs/{filename}_batch_status.tsv")
batch_log_path.parent.mkdir(parents=True, exist_ok=True)

# --- Batch settings ---
COMPLETION_WINDOW = "24h"
POLL_INTERVAL = 10 * 60  # 10 minutes
CONFIRM_SUBMIT = True  # Set to True to actually submit batches
RETRY_FAILED = True  # Re-run failed batches automatically


# --- Helper functions ---
def read_log():
    """Read batch status log into a dict"""
    if not batch_log_path.exists():
        return {}
    with open(batch_log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["file"]: row for row in reader}


def append_to_log(file, batch_id, line_count, status):
    """Append new batch info to log"""
    file_exists = batch_log_path.exists()
    with open(batch_log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["file", "batch_id", "line_count", "status"])
        writer.writerow([file, batch_id, line_count, status])


def update_status(file, status):
    """Update batch status in log"""
    rows = read_log()
    if file not in rows:
        return
    rows[file]["status"] = status
    with open(batch_log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "batch_id", "line_count", "status"], delimiter="\t"
        )
        writer.writeheader()
        for r in rows.values():
            writer.writerow(r)


def wait_for_completion(client, batch_id):
    """Poll for completion status"""
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"🔍 Batch {batch_id} status: {batch.status}")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch.status
        print(f"⏳ Waiting {POLL_INTERVAL/60} minutes...")
        time.sleep(POLL_INTERVAL)


# --- Main process ---
log_data = read_log()

for file in sorted(os.listdir(batch_folder)):
    if not (file.endswith(".jsonl") and "_part" in file):
        continue

    batch_path = batch_folder / file
    with open(batch_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    # Check previous state
    if file in log_data:
        prev = log_data[file]
        status = prev["status"]
        print(f"📄 Found existing log for {file}: {status}")

        # Skip completed or already running batches
        if status == "completed":
            print(f"✅ Skipping {file}, already completed.\n")
            continue
        elif status in ("in_progress", "validating"):
            print(f"⏳ waiting for {file}, still in progress.\n")
            wait_for_completion(client, prev["batch_id"])
            continue
        elif status == "failed" and not RETRY_FAILED:
            print(f"❌ Skipping failed batch {file} (retry disabled).\n")
            continue
        elif status == "failed" and RETRY_FAILED:
            print(f"🔁 Retrying failed batch {file}...\n")

    print(f"📦 Submitting {file} ({line_count} requests)")

    if not CONFIRM_SUBMIT:
        print("⚠️ Dry run mode — skipping upload.")
        continue

    # --- Upload and create batch ---
    uploaded = client.files.create(file=open(batch_path, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
    )

    append_to_log(file, batch_job.id, line_count, "in_progress")

    print(f"🚀 Submitted batch job {batch_job.id} for {file}")
    print(f"🕒 Initial status: {batch_job.status}\n")

    # --- Wait until it finishes ---
    final_status = wait_for_completion(client, batch_job.id)
    update_status(file, final_status)

    print(f"✅ Batch {batch_job.id} finished with status: {final_status}\n")

print("🎉 All batch jobs processed (resumable mode).")
print(f"🧾 Log saved at: {batch_log_path}")
