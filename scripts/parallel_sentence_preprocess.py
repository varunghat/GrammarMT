#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import spacy
from tqdm import tqdm


def enrich_parallel_sentences(parallel_sents):
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        print("Warning: en_core_web_sm not found, skipping enrichment. Run: python -m spacy download en_core_web_sm")
        return parallel_sents

    for sent in tqdm(parallel_sents, desc="Enriching with spacy"):
        doc = nlp(sent["translation"])
        res, tense, number, genders = [], None, None, []
        plural_details, genders_details = None, []

        for tok in doc:
            morph = tok.morph.to_dict()
            res.append(
                {
                    "word": tok.text,
                    "lemma": tok.lemma_,
                    "upos": tok.pos_,
                    "tag": tok.tag_,
                    "morph": morph,
                }
            )
            if not number and morph.get("Number"):
                number = morph["Number"]
            elif morph.get("Number") == "Plur":
                number = "Plur"
                plural_details = {"word": tok.text, "morph": morph}
            if not tense and morph.get("Tense"):
                tense = morph["Tense"]
            if morph.get("Gender"):
                genders.append(morph["Gender"])
                genders_details.append({"word": tok.text, "morph": morph})

        sent["spacy_info"] = {
            "res": res,
            "tense": tense,
            "number": number,
            "plural_details": plural_details,
            "genders": genders,
            "genders_details": genders_details,
        }

    return parallel_sents


def clean_parallel_sentences(parallel_sentences):
    parallel_sentences_cleaned = deepcopy(parallel_sentences)
    for i, sent in enumerate(parallel_sentences_cleaned):
        sent["source"] = sent["source"].strip().replace("\n", " ")
        sent["gloss"] = sent.get("gloss", "").strip().replace("\n", " ")
        sent["translation"] = sent["translation"].strip().replace("\n", " ")

        # Remove (n) from source (where n is a number or a letter) only at the start
        sent["source"] = re.sub(r'^\s*\([a-zA-Z0-9]+\)\s*', ' ', sent["source"]).strip()

        # Remove any single letter with . after it in source (e.g., "A. ", "b. ", "A: ", etc.)
        sent["source"] = re.sub(r'\b[a-zA-Z][\.:]\s*', ' ', sent["source"]).strip()

        # Get the text within the largest pair of same quotes in translation
        quote_patterns = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
        max_len = 0
        matched = False
        current_match = ""
        for open_q, close_q in quote_patterns:
            first_index = sent["translation"].find(open_q)
            last_index = sent["translation"].rfind(close_q)
            if first_index != -1 and last_index != -1 and last_index > first_index:
                current_len = last_index - first_index
                if current_len > max_len:
                    max_len = current_len
                    current_match = sent["translation"][first_index + 1:last_index].strip()
                    matched = True
        if matched:
            sent["translation"] = current_match
        if not matched:
            sent["translation"] = sent["translation"].strip()
            sent["translation"] = re.sub(r'^[“”‘’"\']+', '', sent["translation"])
            sent["translation"] = re.sub(r'[“”‘’"\']+$', '', sent["translation"]).strip()

        # Remove extra spaces
        sent["source"] = re.sub(r'\s+', ' ', sent["source"])
        sent["gloss"] = re.sub(r'\s+', ' ', sent["gloss"])
        sent["translation"] = re.sub(r'\s+', ' ', sent["translation"])

        # Remove citation patterns in translation ONLY AT THE END
        sent["translation"] = re.sub(r'\s*\[[^\]]*\]\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s*\(see [^\)]*\)\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s*\([^\)]*et al\)\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s*\([^\)]*202[0-9][^\)]*\)\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s+', ' ', sent["translation"]).strip()

    return parallel_sentences_cleaned


def plot_corpus_stats(data_dir: Path, images_dir: Path):
    languages = ["kalamang", "tuatschin", "mandan"]
    clean_names = ["Kalamang", "Tuatschin", "Mandan"]
    colors = ['dimgray', 'gray', 'lightgray']

    sentence_lengths_dist = []
    word_lengths_dist = []
    present_names = []

    for language_name, display_name in zip(languages, clean_names):
        candidate = data_dir / f"{language_name}_parallel_sentences_cleaned_enriched.json"
        if not candidate.exists() and language_name == "tuatschin":
            candidate = data_dir / "sursilvan_romansh_parallel_sentences_cleaned_enriched.json"
        if not candidate.exists():
            print(f"Skipping {language_name}, file not found.")
            continue

        with open(candidate, encoding="utf-8") as f:
            sents = json.load(f)
        print(f"Loaded {len(sents)} sentences for {language_name}.")

        words_source = [len(s["source"].split()) for s in sents]
        word_lengths_source = [len(w) for s in sents for w in s["source"].split()]
        sentence_lengths_dist.append(words_source)
        word_lengths_dist.append(word_lengths_source)
        present_names.append(display_name)

    if not sentence_lengths_dist:
        print("No enriched files found, skipping corpus statistics plot.")
        return

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), dpi=300)
    for ax, data, ylabel in zip(
        axes,
        [sentence_lengths_dist, word_lengths_dist],
        ['Words per Sentence', 'Characters per Word'],
    ):
        bp = ax.boxplot(
            data, patch_artist=True, labels=present_names, widths=0.5,
            showfliers=True, flierprops=dict(marker='o', markersize=2, alpha=0.3),
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=0.8), whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Language', fontsize=9)
        ax.yaxis.grid(True, linestyle=':', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / "corpus_statistics.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(images_dir / "corpus_statistics.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {images_dir}/corpus_statistics.pdf")


def plot_dictionary_stats(dict_dir: Path, images_dir: Path):
    language_map = {
        "kalamang": "Kalamang",
        "sursilvan_romansh": "Tuatschin",
        "mandan": "Mandan",
    }
    languages = list(language_map.keys())
    colors = ['dimgray', 'gray', 'lightgray']

    all_pos_counts = {}
    entry_counts = {}

    for language_name, display_name in language_map.items():
        path = dict_dir / f"{language_name}_dictionary_with_metadata.json"
        if not path.exists():
            print(f"Skipping {language_name} dictionary, file not found.")
            continue
        with open(path, encoding="utf-8") as f:
            dictionary = json.load(f)

        lang_pos_counts = Counter()
        for headword, item in dictionary.items():
            for sense in item.get("senses", []):
                spacy_data = sense.get("spacy_analysis", {})
                pos = spacy_data.get("head_pos", "UNKNOWN") if isinstance(spacy_data, dict) else "UNKNOWN"
                lang_pos_counts[pos] += 1

        all_pos_counts[language_name] = lang_pos_counts
        entry_counts[language_name] = len(dictionary)
        print(f"{display_name}: {len(dictionary)} entries, top POS: {lang_pos_counts.most_common(6)}")

    if not all_pos_counts:
        print("No dictionary files found, skipping dictionary POS plot.")
        return

    global_counter = Counter()
    for counts in all_pos_counts.values():
        global_counter.update(counts)
    top_pos_tags = [tag for tag, _ in global_counter.most_common(7) if tag != "UNKNOWN"][:6]

    plt.rcParams.update({'legend.fontsize': 9})
    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
    x = np.arange(len(top_pos_tags))
    width = 0.25

    for i, lang in enumerate(languages):
        if lang not in all_pos_counts:
            continue
        counts = [all_pos_counts[lang].get(tag, 0) for tag in top_pos_tags]
        total = sum(all_pos_counts[lang].values())
        percentages = [(c / total * 100) if total > 0 else 0 for c in counts]
        ax.bar(
            x + (i - 1) * width, percentages, width,
            label=f"{language_map[lang]} (n={entry_counts[lang]:,})",
            color=colors[i], edgecolor='black', linewidth=0.6, alpha=0.85,
        )

    ax.set_ylabel('Percentage of Entries (%)', fontsize=10)
    ax.set_xlabel('Part of Speech', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(top_pos_tags, fontsize=9)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.4, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / "dictionary_pos_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(images_dir / "dictionary_pos_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {images_dir}/dictionary_pos_distribution.pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Clean and enrich parallel sentences extracted by the IGT classifier"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to the input JSON (e.g. data/igt_classifier_results/kalamang_parallel_sentences.json)",
    )
    parser.add_argument(
        "--language", "-l",
        help="Language name used for output filenames (inferred from input filename if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parallel_sentences",
        help="Directory to write cleaned/enriched JSON (default: data/parallel_sentences/)",
    )
    parser.add_argument(
        "--no-enrich", action="store_true",
        help="Skip spacy enrichment (only clean)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Generate corpus statistics and dictionary POS plots (no input file needed)",
    )
    args = parser.parse_args()

    if args.stats:
        base = Path(__file__).parent.parent
        plot_corpus_stats(base / "data" / "parallel_sentences", base / "images")
        plot_dictionary_stats(base / "data" / "dictionary", base / "images")
        return

    if not args.input:
        parser.error("input is required unless --stats is used")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    language = args.language or input_path.stem.split("_parallel_sentences")[0].split("_")[0]
    print(f"Language: {language}")

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Classifier output is {"book": ..., "sentences": [...], ...}; plain files are bare lists
    if isinstance(raw, dict):
        parallel_sentences = raw.get("sentences", [])
    else:
        parallel_sentences = raw
    print(f"Loaded {len(parallel_sentences)} sentences.")

    cleaned = clean_parallel_sentences(parallel_sentences)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / f"{language}_parallel_sentences_cleaned.json"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)
    print(f"Saved cleaned data to {cleaned_path}")

    if not args.no_enrich:
        enriched = enrich_parallel_sentences(cleaned)
        enriched_path = out_dir / f"{language}_parallel_sentences_cleaned_enriched.json"
        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=4)
        print(f"Saved enriched data to {enriched_path}")


if __name__ == "__main__":
    main()
