#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import pandas as pd
from vertexai.tuning import sft
from google.cloud import aiplatform
from google.cloud import storage
import vertexai


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val


# --- Data preparation helpers ---

def clean_data(data):
    cleaned_data = []
    for entry in data:
        if entry is None:
            continue
        original = entry["source"]
        translated = entry["translation"]

        if "‘" in original and "’" in original:
            start = original.index("‘") + 1
            end = original.index("’")
            original = original[start:end].strip()

        if "‘" in translated and "’" in translated:
            start = translated.index("‘") + 1
            end = translated.index("’")
            translated = translated[start:end].strip()

        if translated.lower().startswith("translation:"):
            translated = translated[len("translation:"):].strip()
        if translated.lower().startswith("translation -"):
            translated = translated[len("translation -"):].strip()
        if translated.lower().startswith("translation in english:"):
            translated = translated[len("translation in english:"):].strip()

        quote_pairs = [("“", "”"), ("‘", "’"), ("“", "”"), ("'", "'")]
        for open_quote, close_quote in quote_pairs:
            if translated.startswith(open_quote) and translated.endswith(close_quote):
                translated = translated[1:-1].strip()
            if original.startswith(open_quote) and original.endswith(close_quote):
                original = original[1:-1].strip()

        cleaned_data.append({"source": original, "translation": translated})

    return cleaned_data


def get_gemini_finetune_json(train_data, test_data, base_filename, language_name):
    output_path = "../data/gemini_finetune_data"
    combined_output_path = os.path.join(output_path, language_name)
    os.makedirs(combined_output_path, exist_ok=True)
    output_filename = os.path.join(combined_output_path, base_filename + "_train.jsonl")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        for entry in train_data:
            json_line = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"Translate to english from {language_name}: " + entry["source"]}
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [{"text": entry["translation"]}],
                    },
                ]
            }
            outfile.write(json.dumps(json_line) + "\n")

    print(f"Saved {len(train_data)} training entries to {output_filename}")


# --- Per-language data prep pipeline ---

language_name = "kalamang"
file_dir = f"../data/generated_results_batched/{language_name}"

for filename in os.listdir(file_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(file_dir, filename)
        if file_path.endswith("test.jsonl"):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_filename = filename.split(".json")[0]
        cleaned_data = clean_data(data)
        get_gemini_finetune_json(cleaned_data, cleaned_data, base_filename, language_name)
        print(f"File: {base_filename}, Number of entries: {len(data)}")


# Create train set from original parallel sentences
parallel_sents_file = f"../data/parallel_sentences/{language_name}_train_set_cleaned.json"

if os.path.exists(parallel_sents_file):
    with open(parallel_sents_file, "r", encoding="utf-8") as file:
        parallel_data = json.load(file)
    train_set = parallel_data
    print(f"Creating train set with {len(train_set)} entries from parallel sentences.")
else:
    print(f"File {parallel_sents_file} does not exist.")

train_set_sents_only = [
    {"source": entry["source"], "translation": entry["translation"]}
    for entry in train_set
]

output_dir = os.path.join("../data/gemini_finetune_data", language_name)
output_filename = os.path.join(output_dir, "train_baseline.jsonl")
with open(output_filename, "w", encoding="utf-8") as outfile:
    for entry in train_set_sents_only:
        json_line = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Translate to english from {language_name}: " + entry["source"]}
                    ],
                },
                {"role": "model", "parts": [{"text": entry["translation"]}]},
            ]
        }
        outfile.write(json.dumps(json_line) + "\n")
print(f"Saved {len(train_set_sents_only)} training entries to {output_filename}")


# Create test set from original parallel sentences
parallel_sents_file = f"../data/parallel_sentences/{language_name}_test_set_cleaned.json"

if os.path.exists(parallel_sents_file):
    with open(parallel_sents_file, "r", encoding="utf-8") as file:
        parallel_data = json.load(file)
    test_set = parallel_data
    print(f"Creating test set with {len(test_set)} entries from parallel sentences.")
else:
    print(f"File {parallel_sents_file} does not exist.")

test_set_sents_only = [
    {"source": entry["source"], "translation": entry["translation"]}
    for entry in test_set
]

output_dir = os.path.join("../data/gemini_finetune_data", language_name)
output_filename = os.path.join(output_dir, "test.jsonl")
with open(output_filename, "w", encoding="utf-8") as outfile:
    for entry in test_set_sents_only:
        json_line = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Translate to english from {language_name}: " + entry["source"]}
                    ],
                },
                {"role": "model", "parts": [{"text": entry["translation"]}]},
            ]
        }
        outfile.write(json.dumps(json_line) + "\n")
print(f"Saved {len(test_set_sents_only)} testing entries to {output_filename}")


# Create test inference batch file (for Gemini batch prediction)
inference_output_filename = os.path.join(output_dir, "test_inference.jsonl")
with open(inference_output_filename, "w", encoding="utf-8") as outfile:
    for i, entry in enumerate(test_set_sents_only):
        json_line = {
            "key": f"{language_name}_sent_{i}",
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"Translate to english from {language_name}: " + entry["source"]}
                        ],
                    }
                ]
            },
        }
        outfile.write(json.dumps(json_line) + "\n")
print(f"Saved {len(test_set_sents_only)} inference entries to {inference_output_filename}")

baseline_prompt_path = os.path.join(output_dir, "test_baseline_inference.jsonl")
with open(baseline_prompt_path, "w", encoding="utf-8") as f:
    for i, entry in enumerate(test_set_sents_only):
        json_line = {
            "key": f"{language_name}_sent_{i}",
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    f"Translate to english from {language_name}. "
                                    "Provide the final sentence in a single yaml block at the end. "
                                    "Source: " + entry["source"]
                                )
                            }
                        ],
                    }
                ]
            },
        }
        f.write(json.dumps(json_line) + "\n")


# # SDK Fine-Tuning with Vertex AI

# Combine per-language "best" JSONL files into one file per language
for language in ["mandan", "kalamang", "sursilvan_romansh"]:
    best_path = f"../data/gemini_finetune_data/{language}/best/"
    combined_data = []
    for fname in os.listdir(best_path):
        if fname.endswith(".jsonl"):
            file_path = os.path.join(best_path, fname)
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    combined_data.append(json.loads(line))
    with open(
        f"../data/gemini_finetune_data/{language}_combined_best.jsonl", "w", encoding="utf-8"
    ) as outfile:
        for entry in combined_data:
            outfile.write(json.dumps(entry) + "\n")
    print(
        f"Combined {len(combined_data)} entries for {language} into "
        f"../data/gemini_finetune_data/{language}_combined_best.jsonl"
    )


# --- Vertex AI configuration ---
# Credentials are loaded from GOOGLE_APPLICATION_CREDENTIALS env var (Application Default Credentials)
_require_env("GOOGLE_APPLICATION_CREDENTIALS")

project_id = "project-128068bc-0c84-4cfe-944"
location = "europe-west1"
bucket_name = "thesis_finetune_bucket_1"

vertexai.init(project=project_id, location=location)
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)


def upload_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket.name}/{gcs_path}"


# Collect train/test files for the target language
language_name = "sursilvan_romansh"
data_folder = f"../data/gemini_finetune_data/{language_name}"

train_files = []
test_file = None
test_inference_file = None
train_baseline_file = None

for file in os.listdir(data_folder):
    if file.endswith("_train.jsonl"):
        train_files.append(os.path.join(data_folder, file))
    elif file.endswith("train_baseline.jsonl"):
        train_baseline_file = os.path.join(data_folder, file)
    elif file == "test.jsonl":
        test_file = os.path.join(data_folder, file)
    elif file == "test_inference.jsonl":
        test_inference_file = os.path.join(data_folder, file)

print(f"Training files: {len(train_files)}, test file: {test_file}")

# Use baseline train set
train_files = [train_baseline_file]

# Upload common test files to GCS
test_set_uri = upload_to_gcs(test_file, f"{language_name}/tuning/test.jsonl")
print(f"Uploaded test set to {test_set_uri}")

test_set_inference_uri = upload_to_gcs(
    test_inference_file, f"{language_name}/tuning/test_inference.jsonl"
)
print(f"Uploaded test inference file to {test_set_inference_uri}")


# --- Fine-tuning loop ---
MAX_CONCURRENT_JOBS = 1
TRACKING_FILE = f"{language_name}_tuning_experiments.csv"
TEST_SET_GCS_INFERENCE = test_inference_file

if os.path.exists(TRACKING_FILE):
    print(f"Tracking file {TRACKING_FILE} exists.")
else:
    pd.DataFrame(
        columns=["batch_name", "tuning_job_id", "tuned_model_resource", "batch_job_id", "status", "timestamp"]
    ).to_csv(TRACKING_FILE, index=False)


def get_active_jobs_count():
    jobs = sft.SupervisedTuningJob.list()
    active_states = ["JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"]
    return len([j for j in jobs if j.state.name in active_states])


for file in train_files:
    base_filename = file.split(".jsonl")[0]
    print(f"Processing file: {base_filename}")

    df_tracking = pd.read_csv(TRACKING_FILE)
    if not df_tracking[df_tracking["batch_name"] == base_filename].empty:
        if (
            df_tracking.loc[df_tracking["batch_name"] == base_filename, "status"].values[0]
            == "TUNING_DONE_BATCH_SUBMITTED"
        ):
            print(f"Skipping {base_filename}, already completed.")
            continue

    print(f"Starting tuning for {base_filename}...")

    while get_active_jobs_count() >= MAX_CONCURRENT_JOBS:
        print("Waiting for slot...")
        time.sleep(120)

    gcs_train_path = upload_to_gcs(file, f"{language_name}/tuning/{base_filename}.jsonl")

    tuning_job = sft.train(
        source_model="gemini-2.5-flash",
        train_dataset=gcs_train_path,
        validation_dataset=test_set_uri,
        tuned_model_display_name=f"tuned-{base_filename}",
        epochs=3,
        adapter_size=1,
    )

    print(f"Tuning {base_filename}... Job ID: {tuning_job.resource_name}")
    while not tuning_job.has_ended:
        time.sleep(120)
        tuning_job.refresh()
        print(f"Job running... State: {tuning_job.state.name}")

    tuning_job.refresh()
    if tuning_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Tuning job failed or cancelled. State: {tuning_job.state.name}")
    else:
        print("Tuning job succeeded. Preparing for batch prediction...")
        tuned_model_resource = tuning_job.tuned_model_name
        model = aiplatform.Model(tuned_model_resource)

        print("Waiting 3 minutes to ensure model is ready...")
        time.sleep(180)

        print(f"Submitting batch prediction job for {base_filename}...")
        try:
            batch_prediction_job = model.batch_predict(
                job_display_name=f"inference-eval-{base_filename}",
                gcs_source=test_set_inference_uri,
                gcs_destination_prefix=f"gs://{bucket_name}/{language_name}/inference_results/{base_filename}/",
                instances_format="jsonl",
                predictions_format="jsonl",
                sync=False,
            )
        except Exception as e:
            print(f"Batch prediction error: {e}")
            time.sleep(30)

        max_retries = 5
        for i in range(max_retries):
            try:
                job_id = batch_prediction_job.resource_name
                print(f"Batch job submitted: {job_id}")
                break
            except RuntimeError:
                if i == max_retries - 1:
                    raise
                print("Waiting for job ID to populate...")
                time.sleep(60)

        new_entry = {
            "batch_name": base_filename,
            "tuning_job_id": tuning_job.resource_name,
            "tuned_model_resource": tuned_model_resource,
            "batch_job_id": batch_prediction_job.resource_name,
            "status": "TUNING_DONE_BATCH_SUBMITTED",
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        try:
            df_current = pd.read_csv(TRACKING_FILE)
        except FileNotFoundError:
            df_current = pd.DataFrame()

        df_tracking = pd.concat([df_current, pd.DataFrame([new_entry])], ignore_index=True)
        df_tracking.to_csv(TRACKING_FILE, index=False)
        print("Tracking file updated.")


# --- Download inference results from GCS ---
from concurrent.futures import ThreadPoolExecutor


def download_blob(blob, bucket_name, prefix, local_destination):
    relative_path = blob.name[len(prefix):]
    if relative_path.startswith("/"):
        relative_path = relative_path[1:]
    relative_path = relative_path.split("/")[0].replace(":", "-")
    if not relative_path:
        return
    local_path = os.path.join(local_destination, relative_path) + ".jsonl"
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {blob.name} to {local_path}...")
    blob.download_to_filename(local_path)


def download_folder_parallel(bucket_name, source_folder, local_destination=".", workers=8):
    if not source_folder.endswith("/"):
        source_folder += "/"
    print("Listing blobs...")
    blobs = list(bucket.list_blobs(prefix=source_folder))
    print(f"Found {len(blobs)} files.")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_blob, blob, bucket_name, source_folder, local_destination)
            for blob in blobs
        ]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading file: {e}")


download_folder_parallel(
    bucket_name="thesis_finetune_bucket_1",
    source_folder=f"{language_name}/inference_results",
    local_destination=f"../data/gemini_finetune_inference_results/{language_name}",
)
