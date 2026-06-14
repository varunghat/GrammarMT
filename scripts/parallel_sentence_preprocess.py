#!/usr/bin/env python
# coding: utf-8

import json
import re
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import spacy
from tqdm import tqdm


language = "mandan"

train_or_test = "train"

with open(f"../data/parallel_sentences/{language}_{train_or_test}_set.json", "r", encoding="utf-8") as f:
    parallel_sentences = json.load(f)

print(len(parallel_sentences))
print(parallel_sentences[0])


def enrich_parallel_sentences(parallel_sents):
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return parallel_sents  # skip if model missing

    for sent in tqdm(parallel_sents):
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
        sent["gloss"] = sent["gloss"].strip().replace("\n", " ")
        sent["translation"] = sent["translation"].strip().replace("\n", " ")

        # Remove (n) from source (where n is a number or a letter) only at the start
        sent["source"] = re.sub(r'^\s*\([a-zA-Z0-9]+\)\s*', ' ', sent["source"]).strip()

        # Remove any single letter with . after it in source (e.g., "A. ", "b. ", "A: ", etc.)
        sent["source"] = re.sub(r'\b[a-zA-Z][\.:]\s*', ' ', sent["source"]).strip()

        # Get the text within the largest pair of same quotes in translation
        # Make sure to get the largest match if nested quotes exist
        quote_patterns = [("'", "'"), ('"', '"'), ('‘', '’'), ('“', '”')]
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
                    print(f"Matched quotes: {open_q}{close_q} in sentence {i} -> {current_match}")
        if matched:
            sent["translation"] = current_match
        if not matched:
            sent["translation"] = sent["translation"].strip()
            # Remove leading and trailing quotes if no matched pairs found
            sent["translation"] = re.sub(r'^[“”‘’"\']+', '', sent["translation"])
            sent["translation"] = re.sub(r'[“”‘’"\']+$', '', sent["translation"]).strip()

        # Remove extra spaces
        sent["source"] = re.sub(r'\s+', ' ', sent["source"])
        sent["gloss"] = re.sub(r'\s+', ' ', sent["gloss"])
        sent["translation"] = re.sub(r'\s+', ' ', sent["translation"])

        # Remove citation patterns in translation like [1], (see Smith 2020), etc. ONLY AT THE END
        sent["translation"] = re.sub(r'\s*\[[^\]]*\]\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s*\(see [^\)]*\)\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s*\([^\)]*et al\)\s*$', ' ', sent["translation"]).strip()
        sent["translation"] = re.sub(r'\s*\([^\)]*202[0-9][^\)]*\)\s*$', ' ', sent["translation"]).strip()
        # Remove extra spaces again after citation removal
        sent["translation"] = re.sub(r'\s+', ' ', sent["translation"]).strip()

    return parallel_sentences_cleaned


cleaned_parallel_sentences = clean_parallel_sentences(parallel_sentences)


for row, row_2 in zip(cleaned_parallel_sentences, parallel_sentences):
    len1 = len(row["translation"])
    len2 = len(row_2["translation"])
    print(len2 - len1)
    print(row["translation"], "\n", row_2["translation"], "\n")


# Store cleaned data
output_file = f"../data/parallel_sentences/{language}_{train_or_test}_set_cleaned.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_parallel_sentences, f, ensure_ascii=False, indent=4)

print(f"Saved cleaned data to {output_file}")


# Enrich with spacy morphological info (adds spacy_info to each sentence)
enriched_parallel_sentences = enrich_parallel_sentences(cleaned_parallel_sentences)

enriched_output_file = f"../data/parallel_sentences/{language}_{train_or_test}_set_cleaned_enriched.json"
with open(enriched_output_file, "w", encoding="utf-8") as f:
    json.dump(enriched_parallel_sentences, f, ensure_ascii=False, indent=4)

print(f"Saved enriched data to {enriched_output_file}")


# --- Corpus Statistics ---

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5

colors = ['dimgray', 'gray', 'lightgray']

languages = ["kalamang", "tuatschin", "mandan"]
clean_names = ["Kalamang", "Tuatschin", "Mandan"]

sentence_lengths_dist = []
word_lengths_dist = []

for language_name in languages:
    filename = f"../data/parallel_sentences/{language_name}_parallel_sentences_cleaned_enriched.json"
    if language_name == "tuatschin":
        try:
            open(filename)
        except FileNotFoundError:
            filename = "../data/parallel_sentences/sursilvan_romansh_parallel_sentences_cleaned_enriched.json"

    try:
        with open(filename, "r", encoding="utf-8") as f:
            enriched_parallel_sentences = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {language_name}, file not found.")
        continue

    print(f"Loaded {len(enriched_parallel_sentences)} sentences for {language_name}.")

    words_source = [len(sent["source"].split()) for sent in enriched_parallel_sentences]
    word_lengths_source = [len(word) for sent in enriched_parallel_sentences
                          for word in sent["source"].split()]

    sentence_lengths_dist.append(words_source)
    word_lengths_dist.append(word_lengths_source)

    print(f"{language_name} - Mean Sent Len: {np.mean(words_source):.2f}, "
          f"Mean Word Len: {np.mean(word_lengths_source):.2f}")

if sentence_lengths_dist:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), dpi=300)

    box1 = axes[0].boxplot(sentence_lengths_dist,
                           patch_artist=True,
                           labels=clean_names,
                           widths=0.5,
                           showfliers=True,
                           flierprops=dict(marker='o', markersize=2, alpha=0.3),
                           medianprops=dict(color="black", linewidth=1.2),
                           boxprops=dict(linewidth=0.8),
                           whiskerprops=dict(linewidth=0.8),
                           capprops=dict(linewidth=0.8))

    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    axes[0].set_ylabel('Words per Sentence', fontsize=9)
    axes[0].set_xlabel('Language', fontsize=9)
    axes[0].yaxis.grid(True, linestyle=':', alpha=0.4)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    box2 = axes[1].boxplot(word_lengths_dist,
                           patch_artist=True,
                           labels=clean_names,
                           widths=0.5,
                           showfliers=True,
                           flierprops=dict(marker='o', markersize=2, alpha=0.3),
                           medianprops=dict(color="black", linewidth=1.2),
                           boxprops=dict(linewidth=0.8),
                           whiskerprops=dict(linewidth=0.8),
                           capprops=dict(linewidth=0.8))

    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    axes[1].set_ylabel('Characters per Word', fontsize=9)
    axes[1].set_xlabel('Language', fontsize=9)
    axes[1].yaxis.grid(True, linestyle=':', alpha=0.4)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("images/corpus_statistics.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("images/corpus_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: images/corpus_statistics.pdf")


# --- Dictionary POS Statistics ---

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5

language_map = {
    "kalamang": "Kalamang",
    "sursilvan_romansh": "Tuatschin",
    "mandan": "Mandan"
}
languages = ["kalamang", "sursilvan_romansh", "mandan"]
colors = ['dimgray', 'gray', 'lightgray']

all_pos_counts = {}
entry_counts = {}

for language_name in languages:
    filename = f"../data/dictionary/{language_name}_dictionary_with_metadata.json"

    try:
        with open(filename, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {language_name}, file not found.")
        continue

    display_name = language_map[language_name]
    print(f"--- {display_name.upper()} ---")
    print(f"Entries: {len(dictionary)}")
    entry_counts[language_name] = len(dictionary)

    lang_pos_counts = Counter()
    senses_per_entry = []

    for headword, item in dictionary.items():
        senses = item.get("senses", [])
        num_senses = len(senses)
        senses_per_entry.append(num_senses)

        for sense in senses:
            spacy_data = sense.get("spacy_analysis", {})
            if spacy_data and isinstance(spacy_data, dict):
                pos = spacy_data.get("head_pos", "UNKNOWN")
            else:
                pos = "UNKNOWN"

            lang_pos_counts[pos] += 1

    all_pos_counts[language_name] = lang_pos_counts

    total_senses = sum(senses_per_entry)
    avg_senses = total_senses / len(dictionary) if dictionary else 0
    print(f"Total Senses: {total_senses}")
    print(f"Avg Senses/Entry: {avg_senses:.2f}")
    print("Top 6 POS:", lang_pos_counts.most_common(6))
    print("")

global_counter = Counter()
for counts in all_pos_counts.values():
    global_counter.update(counts)
top_pos_tags = [tag for tag, _ in global_counter.most_common(7) if tag != "UNKNOWN"][:6]

fig, ax = plt.subplots(figsize=(7, 3), dpi=300)

x = np.arange(len(top_pos_tags))
width = 0.25

for i, lang in enumerate(languages):
    if lang not in all_pos_counts:
        continue

    counts = [all_pos_counts[lang].get(tag, 0) for tag in top_pos_tags]
    total = sum(all_pos_counts[lang].values())
    percentages = [(c / total * 100) if total > 0 else 0 for c in counts]

    offset = (i - 1) * width
    display_name = language_map[lang]

    label = f"{display_name} (n={entry_counts[lang]:,})"

    rects = ax.bar(x + offset, percentages, width,
                   label=label,
                   color=colors[i],
                   edgecolor='black',
                   linewidth=0.6,
                   alpha=0.85)

ax.set_ylabel('Percentage of Entries (%)', fontsize=10)
ax.set_xlabel('Part of Speech', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(top_pos_tags, fontsize=9)
ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False, fontsize=8)
ax.grid(axis='y', linestyle=':', alpha=0.4, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max([max([(all_pos_counts[lang].get(tag, 0) / sum(all_pos_counts[lang].values()) * 100)
                         for tag in top_pos_tags]) for lang in languages if lang in all_pos_counts]) * 1.1)

plt.tight_layout()
plt.savefig("images/dictionary_pos_distribution.pdf", dpi=300, bbox_inches='tight')
plt.savefig("images/dictionary_pos_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print("Graph saved as images/dictionary_pos_distribution.pdf")
