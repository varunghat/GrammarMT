#!/usr/bin/env python
# coding: utf-8

"""
Converts the raw CSV output from dictionary_extraction.py into a structured
JSON dictionary with spacy morphological analysis on each English definition.

Input:  data/extracted_dictionaries/{name}_extracted_word_list_columns.csv
Output: data/dictionary/{language}_dictionary_with_metadata.json

The output format is:
  {
    "headword": {
      "senses": [
        {
          "translation": "...",
          "cleaned_translation": "...",
          "grammatical_features": {...},
          "spacy_analysis": {
            "head_pos": "NOUN",
            "head_lemma": "...",
            ...
          }
        }
      ]
    },
    ...
  }

This JSON is consumed by sentence_generation.py for POS-filtered word selection.
"""

import csv
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
import spacy


def clean_gloss_text(gloss: str) -> str:
    gloss = re.sub(r"\s+", " ", gloss).strip()
    gloss = re.sub(r"\bk\.?\s*o\.?\b", "kind of", gloss, flags=re.IGNORECASE)
    gloss = re.sub(r"\bsp\.?\b", "species", gloss, flags=re.IGNORECASE)
    gloss = re.sub(r"\((1|2|3)?(sg|pl)?( poss| pron| refl)?\)", "", gloss, flags=re.IGNORECASE)
    gloss = re.sub(r"\b1sg\b", "first person singular", gloss)
    gloss = re.sub(r"\b2sg\b", "second person singular", gloss)
    gloss = re.sub(r"\([^a-zA-Z]*\)", "", gloss)
    gloss = gloss.strip(" .;:,-")
    return gloss


def extract_grammatical_info(gloss: str) -> dict:
    info = {}
    if "(sg" in gloss:
        info["Number"] = "Sing"
    if "(pl" in gloss:
        info["Number"] = "Plur"
    if "1sg" in gloss:
        info["Person"] = "1"
        info["Possessive"] = True
    if "2sg" in gloss:
        info["Person"] = "2"
    if "3sg" in gloss:
        info["Person"] = "3"
    if "incl" in gloss:
        info["Clusivity"] = "Inclusive"
    if "excl" in gloss:
        info["Clusivity"] = "Exclusive"
    return info


def load_csv_dictionary(csv_path: str) -> dict:
    """
    Parse the two-column CSV from dictionary_extraction.py.
    Columns: page, line_idx, continuation, col1 (LRL headword), col2 (English translation).
    Continuation rows are appended to the previous entry.
    """
    dictionary = {}
    last_word = None

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            page, line_idx, continuation, col1, col2 = (
                row[0], row[1], row[2].strip().lower(), row[3], row[4]
            )

            is_continuation = continuation in ("true", "1", "yes")

            headword = col1.strip() if col1 else ""
            translation = col2.strip() if col2 else ""

            if is_continuation and last_word and last_word in dictionary:
                # Append translation to last sense or extend headword
                if translation and dictionary[last_word]["senses"]:
                    dictionary[last_word]["senses"][-1]["translation"] += " " + translation
                continue

            if not headword or not translation:
                continue

            definitions = [d.strip() for d in translation.split(",") if d.strip()]
            if headword not in dictionary:
                dictionary[headword] = {
                    "senses": [{"translation": d, "page_num": int(page)} for d in definitions]
                }
            else:
                dictionary[headword]["senses"] += [
                    {"translation": d, "page_num": int(page)} for d in definitions
                ]
            last_word = headword

    return dictionary


def enhance_with_spacy(dictionary: dict) -> dict:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
        return dictionary

    for entry, data in tqdm(dictionary.items(), desc="Enhancing with spacy"):
        for sense in data["senses"]:
            translation = sense.get("translation", "")
            if not translation or not translation.strip():
                continue

            gloss_features = extract_grammatical_info(translation)
            cleaned = clean_gloss_text(translation)
            sense["cleaned_translation"] = cleaned
            sense["grammatical_features"] = gloss_features

            if not cleaned.strip():
                continue

            doc = nlp(cleaned)
            if len(doc) == 0:
                continue

            # Find the root token
            root_candidates = [t for t in doc if t.head == t]
            head = root_candidates[0] if root_candidates else doc[0]

            sense["spacy_analysis"] = {
                "is_multiword": len(doc) > 1,
                "pos_sequence": [t.pos_ for t in doc],
                "tag_sequence": [t.tag_ for t in doc],
                "morph_sequence": [t.morph.to_dict() for t in doc],
                "lemmas": [t.lemma_ for t in doc],
                "text": [t.text for t in doc],
                "head_index": head.i,
                "head_text": head.text,
                "head_pos": head.pos_,
                "head_lemma": head.lemma_,
                "dependencies": [
                    {"token": t.text, "dep": t.dep_, "head": t.head.text} for t in doc
                ],
            }

    return dictionary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhance dictionary CSV with spacy morphological analysis"
    )
    parser.add_argument(
        "csv_path",
        help="Path to CSV from dictionary_extraction.py (e.g. data/extracted_dictionaries/kalamang_extracted_word_list_columns.csv)",
    )
    parser.add_argument(
        "language_name",
        help="Language identifier (e.g. kalamang, mandan, sursilvan_romansh)",
    )
    args = parser.parse_args()

    print(f"Loading dictionary from {args.csv_path}...")
    dictionary = load_csv_dictionary(args.csv_path)
    print(f"Loaded {len(dictionary)} dictionary entries.")

    print("Running spacy enhancement...")
    enhanced = enhance_with_spacy(dictionary)

    output_dir = Path("data/dictionary")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.language_name}_dictionary_with_metadata.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(enhanced)} entries to {output_path}")
