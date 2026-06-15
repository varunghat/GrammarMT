#!/usr/bin/env python
# coding: utf-8

"""
Batch wrapper: runs parallel_sentence_preprocess.py on every
*_parallel_sentences.json file in a directory.

Usage:
    python scripts/batch_clean_sentences.py
    python scripts/batch_clean_sentences.py --input-dir data/igt_classifier_results/
    python scripts/batch_clean_sentences.py --no-enrich
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch-clean all *_parallel_sentences.json files in a directory"
    )
    parser.add_argument(
        "--input-dir",
        default="data/igt_classifier_results",
        help="Directory containing *_parallel_sentences.json files (default: data/igt_classifier_results/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parallel_sentences",
        help="Directory to write cleaned/enriched files (default: data/parallel_sentences/)",
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Skip spacy enrichment (only clean)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process files even if the cleaned output already exists",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(input_dir.glob("*_parallel_sentences.json"))
    if not files:
        print(f"No *_parallel_sentences.json files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(files)} files to process.\n")

    preprocess = Path(__file__).parent / "parallel_sentence_preprocess.py"
    ok, skipped, failed = 0, 0, 0

    for i, path in enumerate(files, 1):
        stem = path.stem.replace("_parallel_sentences", "")
        cleaned_path = output_dir / f"{stem}_parallel_sentences_cleaned.json"

        if cleaned_path.exists() and not args.force:
            print(f"[{i}/{len(files)}] SKIP {path.name} (already cleaned, use --force to redo)")
            skipped += 1
            continue

        print(f"[{i}/{len(files)}] {path.name}")
        cmd = [
            sys.executable, str(preprocess),
            str(path),
            "--output-dir", str(output_dir),
        ]
        if args.no_enrich:
            cmd.append("--no-enrich")

        result = subprocess.run(cmd)
        if result.returncode == 0:
            ok += 1
        else:
            print(f"  ERROR — preprocess failed for {path.name}")
            failed += 1

    print(f"\nDone. {ok} processed, {skipped} skipped, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
