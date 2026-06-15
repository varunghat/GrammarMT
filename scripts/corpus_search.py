#!/usr/bin/env python
# coding: utf-8

"""
Streamlit corpus search tool.

Run:
    streamlit run scripts/corpus_search.py
"""

import json
from pathlib import Path

import streamlit as st
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Corpus Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "data" / "parallel_sentences"
SEARCH_FIELDS = ["source", "gloss", "translation"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading corpus…")
def load_corpus() -> dict[str, list[dict]]:
    """Return {book_name: [sentence, ...]} for every cleaned JSON in DATA_DIR."""
    corpus: dict[str, list[dict]] = {}
    for path in sorted(DATA_DIR.glob("*_cleaned*.json")):
        book = path.stem.replace("_parallel_sentences_cleaned", "").replace("_cleaned", "")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        sentences = raw.get("sentences", raw) if isinstance(raw, dict) else raw
        corpus[book] = sentences
    return corpus


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def exact_match(text: str, query: str) -> bool:
    return query.lower() in text.lower()


def fuzzy_score(text: str, query: str) -> int:
    """Return the best rapidfuzz partial_ratio score (0–100)."""
    return fuzz.partial_ratio(query.lower(), text.lower())


def search(
    sentences: list[dict],
    query: str,
    fields: list[str],
    fuzzy: bool,
    threshold: int,
    book_name: str,
) -> list[dict]:
    results = []
    for idx, sent in enumerate(sentences):
        best_score = 0
        matched_fields = []

        for field in fields:
            text = sent.get(field, "") or ""
            if fuzzy:
                score = fuzzy_score(text, query)
            else:
                score = 100 if exact_match(text, query) else 0

            if score >= threshold:
                best_score = max(best_score, score)
                matched_fields.append(field)

        if matched_fields:
            results.append(
                {
                    "book": book_name,
                    "idx": idx + 1,
                    "score": best_score,
                    "matched_in": ", ".join(matched_fields),
                    "source": sent.get("source", ""),
                    "gloss": sent.get("gloss", ""),
                    "translation": sent.get("translation", ""),
                    "page": sent.get("page_num", ""),
                }
            )
    return results


def highlight(text: str, query: str) -> str:
    """Wrap all case-insensitive occurrences of query in a bold, underlined, coloured span."""
    if not query or not text:
        return text
    import re
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(
        lambda m: (
            f'<span style="font-weight:bold;text-decoration:underline;color:#e67e22">'
            f'{m.group()}</span>'
        ),
        text,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
corpus = load_corpus()

if not corpus:
    st.error(f"No cleaned sentence files found in `{DATA_DIR}`. Run `batch_clean_sentences.py` first.")
    st.stop()

st.title("Parallel Sentence Corpus Search")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Search options")

    book_options = ["All books"] + sorted(corpus.keys())
    selected_book = st.selectbox("Book", book_options)

    query = st.text_input("Search query", placeholder="e.g. fish")

    field_options = st.multiselect(
        "Search in",
        options=SEARCH_FIELDS,
        default=SEARCH_FIELDS,
        format_func=str.capitalize,
    )

    fuzzy = st.toggle("Fuzzy matching", value=True)

    threshold = 80
    if fuzzy:
        threshold = st.slider(
            "Fuzzy threshold",
            min_value=50,
            max_value=100,
            value=80,
            step=5,
            help="100 = exact substring match, lower values allow more variation",
        )

    max_results = st.select_slider(
        "Max results",
        options=[25, 50, 100, 250, 500, 1000],
        value=100,
    )

# --- Run search ---
if not query.strip():
    total = sum(len(v) for v in corpus.values())
    st.info(
        f"Corpus loaded: **{len(corpus)} books**, **{total:,} sentences**. "
        "Enter a query in the sidebar to search."
    )
    st.stop()

if not field_options:
    st.warning("Select at least one field to search in.")
    st.stop()

with st.spinner("Searching…"):
    if selected_book == "All books":
        books_to_search = corpus.items()
    else:
        books_to_search = [(selected_book, corpus[selected_book])]

    all_results = []
    for book_name, sentences in books_to_search:
        all_results.extend(
            search(sentences, query.strip(), field_options, fuzzy, threshold, book_name)
        )

# Sort by score descending, then book + position
all_results.sort(key=lambda r: (-r["score"], r["book"], r["idx"]))
truncated = len(all_results) > max_results
displayed = all_results[:max_results]

# --- Results header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(
        f"{len(all_results):,} result{'s' if len(all_results) != 1 else ''} "
        f"for **\"{query}\"**"
        + (f" (showing first {max_results})" if truncated else "")
    )
with col2:
    sort_by = st.selectbox(
        "Sort by",
        ["Relevance", "Book", "Page"],
        label_visibility="collapsed",
    )
    if sort_by == "Book":
        displayed.sort(key=lambda r: (r["book"], r["idx"]))
    elif sort_by == "Page":
        displayed.sort(key=lambda r: (r["book"], r["page"] or 0))

# --- Results ---
if not displayed:
    st.warning("No matches found. Try lowering the fuzzy threshold or broadening the field selection.")
    st.stop()

for r in displayed:
    score_color = "#2ecc71" if r["score"] >= 90 else "#f39c12" if r["score"] >= 70 else "#e74c3c"

    with st.container(border=True):
        meta_col, score_col = st.columns([6, 1])
        with meta_col:
            book_label = f"📖 **{r['book']}**"
            page_label = f"  ·  p. {r['page']}" if r["page"] else ""
            match_label = f"  ·  matched in *{r['matched_in']}*"
            st.markdown(book_label + page_label + match_label)
        with score_col:
            st.markdown(
                f'<div style="text-align:right;color:{score_color};font-weight:bold">'
                f'{r["score"]}%</div>',
                unsafe_allow_html=True,
            )

        rows = [
            ("Source",      "#3498db", r["source"]),
            ("Gloss",       "#27ae60", r["gloss"]),
            ("Translation", "#e74c3c", r["translation"]),
        ]
        for label, color, text in rows:
            if text:
                hl = highlight(text, query)
                st.markdown(
                    f'<span style="color:{color};font-size:0.8em;font-weight:600">{label}</span>'
                    f'&nbsp;&nbsp;{hl}',
                    unsafe_allow_html=True,
                )
