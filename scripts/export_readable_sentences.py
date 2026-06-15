#!/usr/bin/env python
# coding: utf-8

"""
Export a stripped-down, human-readable JSON from cleaned parallel sentence files.

Removes bboxes, page_num, spacy_info and keeps only:
    id, source, gloss, translation

Usage:
    # Single file
    python scripts/export_readable_sentences.py \
        data/parallel_sentences/bar12neverver_parallel_sentences_cleaned.json

    # Whole folder — writes one readable file per cleaned JSON
    python scripts/export_readable_sentences.py \
        --directory data/parallel_sentences/

    # Merge all files into one big readable file
    python scripts/export_readable_sentences.py \
        --directory data/parallel_sentences/ --merge
"""

import argparse
import json
import sys
from pathlib import Path


KEEP_KEYS = {"source", "gloss", "translation"}


def strip_sentence(sent: dict, idx: int) -> dict:
    return {
        "id": idx,
        "source": sent.get("source", ""),
        "gloss": sent.get("gloss", ""),
        "translation": sent.get("translation", ""),
    }


def load_sentences(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("sentences", [])
    return raw


def export_file(input_path: Path, output_path: Path):
    sentences = load_sentences(input_path)
    readable = [strip_sentence(s, i + 1) for i, s in enumerate(sentences)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(readable, f, ensure_ascii=False, indent=2)
    print(f"  {len(readable)} sentences → {output_path}")
    return readable


def main():
    parser = argparse.ArgumentParser(
        description="Export human-readable JSON (source / gloss / translation only)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "input",
        nargs="?",
        help="Single cleaned JSON file to export",
    )
    group.add_argument(
        "--directory", "-d",
        help="Folder of cleaned JSONs to export (processes all *_cleaned.json files)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/readable_sentences",
        help="Where to write the readable files (default: data/readable_sentences/)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Combine all files into one merged output file (only with --directory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        stem = input_path.stem.replace("_cleaned", "").replace("_parallel_sentences", "")
        output_path = output_dir / f"{stem}_readable.json"
        export_file(input_path, output_path)

    else:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"ERROR: directory not found: {directory}", file=sys.stderr)
            sys.exit(1)

        files = sorted(directory.glob("*_cleaned*.json"))
        if not files:
            print(f"No *_cleaned*.json files found in {directory}")
            sys.exit(0)

        print(f"Found {len(files)} file(s).\n")

        if args.merge:
            all_sentences = []
            for path in files:
                sentences = load_sentences(path)
                book = path.stem.replace("_parallel_sentences_cleaned", "").replace("_cleaned", "")
                for i, s in enumerate(sentences):
                    entry = strip_sentence(s, len(all_sentences) + i + 1)
                    entry["book"] = book
                    all_sentences.append(entry)

            output_path = output_dir / "all_sentences_readable.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_sentences, f, ensure_ascii=False, indent=2)
            print(f"{len(all_sentences)} total sentences → {output_path}")

        else:
            for path in files:
                stem = path.stem.replace("_parallel_sentences_cleaned", "").replace("_cleaned", "")
                output_path = output_dir / f"{stem}_readable.json"
                export_file(path, output_path)


if __name__ == "__main__":
    main()
