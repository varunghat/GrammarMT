import pymupdf
import json
import os
import re

import collections

import spacy

import typer

from langdetect import detect_langs
from typing import Tuple
from tqdm import tqdm

app = typer.Typer(
    name="pdf_parser",
    help="PDF Parser to extract text, font styles, and sizes from PDF documents",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def collect_consecutive_lines_across_blocks(
    start_block_index, start_line_index, blocks, target_font=None
) -> Tuple[int, int, str, str]:
    """
    Collects consecutive lines of text across blocks in a PDF document.
    """
    collected_text = []
    block_index = start_block_index
    line_index = start_line_index

    if start_block_index >= len(blocks):
        return block_index, line_index, "", target_font

    while block_index < len(blocks):
        lines = blocks[block_index].get("lines", [])
        while line_index < len(lines):
            spans = lines[line_index].get("spans", [])
            if not spans or len(spans) != 1:
                return block_index, line_index, " ".join(collected_text), target_font

            span = spans[0]
            if target_font is None:
                target_font = span["font"]

            if span["font"] != target_font:
                return block_index, line_index, " ".join(collected_text), target_font

            collected_text.append(span["text"].strip())
            line_index += 1

        block_index += 1
        line_index = 0

    return block_index, line_index, " ".join(collected_text), target_font


def build_line_table(pdf_file_path, y_tolerance=2.0, line_tolerance=2.0):
    all_lines = []

    try:
        print("Opening PDF file...")
        doc = pymupdf.open(pdf_file_path)
        print("PDF file opened successfully.")
        print("Number of pages in PDF file:", doc.page_count)
    except Exception as e:
        print("Error occurred while opening PDF file:", e)
        return all_lines

    for page_num, page in tqdm(
        enumerate(doc), total=doc.page_count, desc="Processing pages"
    ):
        blocks = page.get_text("dict")["blocks"]
        for block_idx, block in enumerate(blocks):
            for line_idx, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Sort spans by y (top to bottom) and then x (left to right) with tolerance
                # This is to handle cases where spans are not in reading order
                for span in spans:
                    x0, y0, _, _ = span["bbox"]
                    span["_y_round"] = round(y0 / y_tolerance) * y_tolerance
                    span["_x"] = x0
                spans_sorted = sorted(spans, key=lambda s: (s["_y_round"], s["_x"]))
                # spans = spans_sorted
                # Merge spans into one line string
                merged_text = " ".join(
                    span["text"].strip()
                    for span in spans_sorted
                    if span["text"].strip()
                )
                if not merged_text:
                    continue

                # Choose a "dominant" font/size for the line
                sizes = [span["size"] for span in spans_sorted]
                fonts = [span["font"] for span in spans_sorted]

                # Representative = most common size & font
                rep_size = max(set(sizes), key=sizes.count)
                rep_font = max(set(fonts), key=fonts.count)

                # Position = from first span
                x, y = spans_sorted[0]["bbox"][:2]

                all_lines.append(
                    {
                        "page": page_num + 1,
                        "block": block_idx,
                        "line": line_idx,
                        "text": merged_text,
                        "size": rep_size,
                        "font": rep_font,
                        "x": x,
                        "y": y,
                    }
                )

    # Sort all lines by page, y (top to bottom), x (left to right)
    #

    for line in all_lines:
        line["_y_round"] = round(line["y"] / line_tolerance) * line_tolerance
    all_lines.sort(key=lambda l: (l["page"], l["_y_round"], l["x"]))

    return all_lines


def analyze_fonts(
    all_lines,
    min_heading_occ_count=5,
    min_heading_total_char_length=50,
    main_body_tolerance=0.5,
):

    size_occurrences = collections.Counter(line["size"] for line in all_lines)
    # Get lengths of text above main body size for headings total
    length_by_size = collections.Counter()
    for line in all_lines:
        length_by_size[line["size"]] += len(line["text"])

    main_body_size = max(length_by_size, key=length_by_size.get)
    print("Determined main body font size:", main_body_size)

    headings = []
    for size, total_length in length_by_size.most_common():
        if (
            size > main_body_size + main_body_tolerance
            and total_length >= min_heading_total_char_length
            and size_occurrences[size] >= min_heading_occ_count
        ):
            print(
                f"  Possible heading size: {size} (total text length: {total_length} chars, occurrences: {size_occurrences[size]})"
            )
            headings.append((size, total_length, size_occurrences[size]))
    headings_sorted = [size for size, _, _ in sorted(headings, reverse=True)]

    return main_body_size, headings_sorted


def build_sections(all_lines, main_body_size, headings_sizes, main_body_tolerance=0.5):
    sections = []
    current = None

    for line in all_lines:
        sz, txt, font = line["size"], line["text"], line["font"]
        if sz in headings_sizes:
            if current:
                sections.append(current)
            current = {
                "heading": txt,
                "heading_level": headings_sizes.index(sz) + 1,
                "text": "",
                "font_size": sz,
                "font_style": font,
                "page": line["page"],
            }
        elif current and abs(sz - main_body_size) <= main_body_tolerance:
            if current["text"]:
                current["text"] += " "
            current["text"] += txt

    if current:
        sections.append(current)

    return sections


def clean_sections(sections):

    for i, section in enumerate(sections):
        if section is None:
            continue

        # merge consecutive empty same-level headings
        if (
            i < len(sections) - 1
            and section["heading_level"] == sections[i + 1]["heading_level"]
            and section["text"].strip() == ""
        ):
            while (
                i < len(sections) - 1
                and sections[i + 1]
                and section["heading_level"] == sections[i + 1]["heading_level"]
                and sections[i + 1]["text"].strip() == ""
            ):

                section["heading"] += " " + sections[i + 1]["heading"]
                sections[i + 1] = None
                i += 1

            # only copy text if the *next* section actually has body text
            if (
                i < len(sections) - 1
                and sections[i + 1]
                and sections[i + 1]["text"].strip() != ""
            ):
                section["text"] = sections[i + 1]["text"]
                section["heading"] += " " + sections[i + 1]["heading"]
                sections[i + 1] = None

    sections_filtered = [s for s in sections if s is not None]
    return sections_filtered


def classify_text_igt_english(text: str) -> str:
    if not text or not text.strip():
        return "reject"

    _SPLIT_RE = re.compile(r"[\s.\-=:\/]+")
    IGT_ABBRS = {
        "sg",
        "pl",
        "du",
        "pst",
        "prs",
        "fut",
        "ipfv",
        "pfv",
        "prog",
        "hab",
        "nom",
        "acc",
        "erg",
        "gen",
        "dat",
        "loc",
        "all",
        "abl",
        "ins",
        "com",
        "ben",
        "top",
        "foc",
        "caus",
        "pass",
        "mid",
        "refl",
        "recip",
        "aux",
        "cop",
        "neg",
        "cond",
        "imp",
        "subj",
        "ind",
    }

    tokens = [t for t in _SPLIT_RE.split(text) if t]
    total = len(tokens)
    if total == 0:
        return "reject"

    gloss_hits = sum(
        2
        for t in tokens
        if t.lower() in IGT_ABBRS or re.fullmatch(r"[123](sg|pl|du)", t.lower())
    )
    normalized_gloss = gloss_hits / total

    cleaned = re.sub(r"\[.*?\]", "", text)
    english_score = 0.0
    try:
        probs = detect_langs(cleaned)
        if probs and probs[0].lang == "en":
            english_score = probs[0].prob
    except Exception:
        pass

    if normalized_gloss > 0.10:
        return "IGT"
    if len(cleaned) >= 20 and english_score >= 0.70:
        return "English"
    if len(cleaned) >= 40 and english_score >= 0.55:
        return "English"
    return "unknown"


def extract_parallel_sentences(all_lines):
    parallel_sents = []
    idx, N = 0, len(all_lines)
    MARK_RE = re.compile(r"\(\d+\)")
    while idx < N:
        line = all_lines[idx]
        if MARK_RE.fullmatch(line["text"]):
            col_x = None
            rows, j = [], idx + 1
            while j < N:
                ln = all_lines[j]
                if MARK_RE.fullmatch(ln["text"]):
                    break
                if col_x is None:
                    col_x = ln["x"]
                if (col_x - ln["x"]) > 3:
                    break
                rows.append((ln["y"], ln["text"]))
                j += 1

            rows.sort(key=lambda t: t[0])
            grouped, cur_y, buf = [], None, []
            for y, t in rows:
                if cur_y is None or abs(y - cur_y) <= 2:
                    cur_y = y if cur_y is None else (cur_y + y) / 2
                    buf.append(t)
                else:
                    grouped.append(" ".join(buf))
                    cur_y, buf = y, [t]
            if buf:
                grouped.append(" ".join(buf))

            igt, eng, other = [], [], []
            for k, g in enumerate(grouped):
                res = classify_text_igt_english(g)
                if res == "IGT":
                    igt.append(g)
                elif res == "English":
                    eng.extend(grouped[k:])
                    break
                elif res != "reject":
                    other.append(g)

            if eng and other:
                parallel_sents.append(
                    {
                        "page": line["page"],
                        "text": " ".join(other),
                        "IGT": " ".join(igt),
                        "English": " ".join(eng),
                    }
                )
            idx = j
        else:
            idx += 1
    return parallel_sents


def enrich_parallel_sentences(parallel_sents):
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return parallel_sents  # skip if model missing

    for sent in tqdm(parallel_sents):
        doc = nlp(sent["English"])
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
        sent["sentence"] = sent.pop("text")
        sent["translated_sentence"] = sent.pop("English")

    return parallel_sents


@app.command()
def parse_pdf(
    pdf_file_path: str = typer.Argument(None, help="Path to the PDF file to be parsed"),
    max_heading_number: int = typer.Option(
        4,
        "--max-heading-number",
        "-mhn",
        help="Maximum number of heading levels to consider",
    ),
    min_heading_occ_count: int = typer.Option(
        5,
        "--min-heading",
        "-mh",
        help="Minimum occurrences for a font size to be considered a heading",
    ),
    min_heading_total_char_length: int = typer.Option(
        50,
        "--min-heading-total-char-length",
        "-mhtcl",
        help="Minimum total character length for a font size to be considered a heading",
    ),
    main_body_tolerance: float = typer.Option(
        0.5,
        "--main-body-tolerance",
        "-mbt",
        help="Tolerance for main body font size matching",
    ),
):
    """
    Parses a PDF file and extracts text, font styles, and sizes.
    """
    # Open the PDF file
    if pdf_file_path is None:
        raise ValueError("Please provide a valid PDF file path.")

    pdf_file = os.path.abspath(pdf_file_path)
    pdf_file_name = os.path.splitext(os.path.basename(pdf_file))[0]
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")
    if not pdf_file.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF: {pdf_file}")

    print(f"Parsing PDF file: {pdf_file}")
    print("DEBUG: Building line table...")
    all_lines = build_line_table(pdf_file)
    print(f"DEBUG: Extracted {len(all_lines)} lines from PDF.")

    print("DEBUG: Analyzing fonts to determine main body and headings...")
    main_body, headings = analyze_fonts(
        all_lines,
        min_heading_occ_count,
        min_heading_total_char_length,
        main_body_tolerance,
    )
    print(f"DEBUG: Main body font size: {main_body}, Heading sizes: {headings}")
    print(f"DEBUG: Limiting to top {max_heading_number} heading sizes...")
    headings = headings[:max_heading_number]
    print(f"DEBUG: Using heading sizes: {headings}")

    print("DEBUG: Building sections from lines...")
    sections = build_sections(all_lines, main_body, headings, max_heading_number)
    print(f"DEBUG: Built {len(sections)} sections.")

    print("DEBUG: Cleaning sections...")
    sections_cleaned = clean_sections(sections)
    print(f"DEBUG: Cleaned sections count: {len(sections_cleaned)}")

    print("DEBUG: Extracting parallel sentences...")
    parallel_sentences = extract_parallel_sentences(all_lines)
    print(f"DEBUG: Extracted {len(parallel_sentences)} parallel sentences.")

    print("DEBUG: Enriching parallel sentences with POS and Unimorph info...")
    parallel_sentences_enriched = enrich_parallel_sentences(parallel_sentences)
    print(f"DEBUG: Enriched {len(parallel_sentences_enriched)} parallel sentences.")

    print("Storing section text to JSON file...")
    with open(
        f"parsed_grammar_json/{pdf_file_name}_sections.json", "w", encoding="utf-8"
    ) as f:
        json.dump(sections_cleaned, f, ensure_ascii=False, indent=4)

    print(
        f"Section text stored successfully to parsed_grammar_json/{pdf_file_name}_sections.json"
    )

    # store parallel sentences
    print("Storing parallel sentences to JSON file...")
    with open(
        f"parallel_sents/{pdf_file_name}_parallel_sentences.json", "w", encoding="utf-8"
    ) as f:
        json.dump(parallel_sentences, f, ensure_ascii=False, indent=4)

    print(
        f"Parallel sentences stored successfully to parallel_sents/{pdf_file_name}_parallel_sentences.json"
    )

    with open(
        f"parallel_sents/{pdf_file_name}_parallel_sentences_with_pos_and_unimorph.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(parallel_sentences_enriched, f, ensure_ascii=False, indent=4)
    print(
        f"Parallel sentences with POS and Unimorph info stored successfully to parallel_sents/{pdf_file_name}_parallel_sentences_with_pos_and_unimorph.json"
    )
    print("PDF parsing completed.")


if __name__ == "__main__":
    app()
