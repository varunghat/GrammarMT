#!/usr/bin/env python
# coding: utf-8
"""
Batch IGT classifier for all grammar books in grammar_books/comprehensive_grammar_library/.

Uses the trained BERT+structural-features LineClassifier (bert-base-multilingual-cased)
to label every line in each PDF as SOURCE / GLOSS / TRANSLATION / OTHER, then assembles
parallel sentences from consecutive IGT blocks.

Pipeline per book:
  1. Extract visual lines from all pages (font-aware span merging)
  2. Build ±3-line context window and 8 structural features per line
  3. Run the trained LineClassifier (inference only, no GPT)
  4. Apply neighbourhood-based post-processing heuristics
  5. Assemble contiguous SOURCE/GLOSS/TRANSLATION runs into parallel sentences
  6. Save {book_stem}_predicted_roles.json and {book_stem}_parallel_sentences.json
  7. Render annotated PDF with coloured bounding boxes (SOURCE=blue, GLOSS=green, TRANSLATION=red)

Resumable: skips books whose output already exists (unless --force is passed).

Run from project root:
    python notebooks/lsp_igt_classifier.py
    python notebooks/lsp_igt_classifier.py --books japhug.pdf cayuga.pdf
    python notebooks/lsp_igt_classifier.py --force
    python notebooks/lsp_igt_classifier.py --no-render   # skip PDF rendering
"""

import argparse
import json
import re
import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
GRAMMAR_BOOKS_DIR = BASE_DIR / "grammar_books" / "comprehensive_grammar_library"
MODEL_DIR = BASE_DIR / "models" / "combined_bert_parallel_line_classifier_final"
OUT_DIR = BASE_DIR / "data" / "igt_classifier_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Label mapping  (sorted(df["role"].unique()) at training time)
# ---------------------------------------------------------------------------
LABELS = ["GLOSS", "OTHER", "SOURCE", "TRANSLATION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------
BERT_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128
BATCH_SIZE = 32
OTHER_CONF_THRESHOLD = 0.95  # if OTHER confidence < this, try second-best prediction


# ---------------------------------------------------------------------------
# LineClassifier (must match architecture used during training)
# ---------------------------------------------------------------------------
class LineClassifier(nn.Module):
    def __init__(self, bert_name, num_labels, feat_dim, dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden + feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        len,
        num_tokens,
        ascii_ratio,
        non_ascii,
        digit_ratio,
        has_gloss_markers,
        token_length_mean,
        token_length_std,
        labels=None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        feats = torch.stack(
            [
                len,
                num_tokens,
                ascii_ratio,
                non_ascii,
                digit_ratio,
                has_gloss_markers.float(),
                token_length_mean,
                token_length_std,
            ],
            dim=1,
        )
        logits = self.classifier(torch.cat([pooled, feats], dim=1))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Visual line extraction (identical to parallel_sentence_extractor.py)
# ---------------------------------------------------------------------------
def extract_visual_lines(page):
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sb = span["bbox"]
                spans.append(
                    {
                        "text": span["text"],
                        "x0": sb[0],
                        "y0": sb[1],
                        "x1": sb[2],
                        "y1": sb[3],
                        "font": span.get("font"),
                        "size": span.get("size"),
                    }
                )

    spans = sorted(spans, key=lambda s: s["y0"])
    if not spans:
        return None

    heights = [s["y1"] - s["y0"] for s in spans]
    median_h = float(np.median(heights)) if heights else 10.0
    Y_TOL = median_h * 0.4

    visual_lines, current_line = [], [spans[0]]
    for prev, curr in zip(spans, spans[1:]):
        if abs(curr["y0"] - prev["y0"]) < Y_TOL:
            current_line.append(curr)
        else:
            visual_lines.append(current_line)
            current_line = [curr]
    visual_lines.append(current_line)

    merged = []
    for line_spans in visual_lines:
        sorted_spans = sorted(line_spans, key=lambda s: s["x0"])
        merged_text = " ".join(s["text"] for s in sorted_spans if s["text"].strip())
        font_sizes: dict = {}
        for s in sorted_spans:
            fk = (s["font"], s["size"])
            font_sizes[fk] = font_sizes.get(fk, 0) + len(s["text"])
        common_font_size = (
            max(font_sizes, key=font_sizes.get)[1] if font_sizes else None
        )
        merged.append(
            {
                "text": merged_text,
                "x0": sorted_spans[0]["x0"],
                "y0": float(np.mean([s["y0"] for s in sorted_spans])),
                "x1": sorted_spans[-1]["x1"],
                "y1": max(s["y1"] for s in sorted_spans),
                "common_font_size": common_font_size,
            }
        )
    return merged


# ---------------------------------------------------------------------------
# Context window and structural features (identical to training)
# ---------------------------------------------------------------------------
def _structural_features(line: str) -> dict:
    tokens = line.split()
    ascii_chars = sum(c.isascii() for c in line)
    non_ascii = len(line) - ascii_chars
    return {
        "len": float(len(line)),
        "num_tokens": float(len(tokens)),
        "ascii_ratio": ascii_chars / max(1, len(line)),
        "non_ascii": float(non_ascii),
        "digit_ratio": sum(c.isdigit() for c in line) / max(1, len(line)),
        "has_gloss_markers": float(bool(re.search(r"[=~\-]", line))),
        "token_length_mean": (
            float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        ),
        "token_length_std": float(np.std([len(t) for t in tokens])) if tokens else 0.0,
    }


def build_dataframe(all_merged_lines: list[dict]) -> pd.DataFrame:
    rows = []
    for page_entry in all_merged_lines:
        page_num = page_entry["page_num"]
        for line_no, line in enumerate(page_entry["lines"]):
            rows.append(
                {
                    "page": page_num,
                    "line_no": line_no,
                    "text": line["text"],
                    "x0": line["x0"],
                    "y0": line["y0"],
                    "x1": line["x1"],
                    "y1": line["y1"],
                    "width": line["x1"] - line["x0"],
                    "height": line["y1"] - line["y0"],
                }
            )
    df = pd.DataFrame(rows)

    # Context windows within the same page
    texts = df["text"].tolist()
    pages = df["page"].tolist()
    for col, offset in [
        ("prev_3", -3),
        ("prev_2", -2),
        ("prev", -1),
        ("next", 1),
        ("next_2", 2),
        ("next_3", 3),
    ]:
        context = []
        for i in range(len(df)):
            j = i + offset
            if 0 <= j < len(df) and pages[j] == pages[i]:
                context.append(texts[j])
            else:
                context.append("")
        df[col] = context

    df["context_text"] = (
        "PREV_3] "
        + df["prev_3"]
        + "[PREV_2] "
        + df["prev_2"]
        + " [PREV] "
        + df["prev"]
        + " [CURR] "
        + df["text"]
        + " [NEXT] "
        + df["next"]
        + " [NEXT_2] "
        + df["next_2"]
        + " [NEXT_3] "
        + df["next_3"]
    )

    feat_df = pd.DataFrame(df["text"].apply(_structural_features).tolist())
    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return df


# ---------------------------------------------------------------------------
# BERT inference
# ---------------------------------------------------------------------------
def load_model(device: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = LineClassifier(BERT_NAME, num_labels=len(LABELS), feat_dim=8, dropout=0.4)
    state_dict = load_file(str(MODEL_DIR / "model.safetensors"))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, tokenizer


_FEAT_COLS = [
    "len",
    "num_tokens",
    "ascii_ratio",
    "non_ascii",
    "digit_ratio",
    "has_gloss_markers",
    "token_length_mean",
    "token_length_std",
]


def run_inference(df: pd.DataFrame, model, tokenizer, device: str) -> pd.DataFrame:
    all_preds, all_probs, all_second = [], [], []

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="    inference", leave=False):
        batch_df = df.iloc[start : start + BATCH_SIZE]
        enc = tokenizer(
            list(batch_df["context_text"]),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        feats = torch.tensor(
            batch_df[_FEAT_COLS].values.astype(float), dtype=torch.float
        ).to(device)

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                len=feats[:, 0],
                num_tokens=feats[:, 1],
                ascii_ratio=feats[:, 2],
                non_ascii=feats[:, 3],
                digit_ratio=feats[:, 4],
                has_gloss_markers=feats[:, 5],
                token_length_mean=feats[:, 6],
                token_length_std=feats[:, 7],
            )

        prob = torch.softmax(out["logits"], dim=-1)
        top2_probs, top2_classes = torch.topk(prob, 2, dim=-1)
        all_preds.extend(top2_classes[:, 0].cpu().tolist())
        all_probs.extend(top2_probs[:, 0].cpu().tolist())
        all_second.extend(top2_classes[:, 1].cpu().tolist())

    df = df.copy()
    df["pred_id"] = all_preds
    df["pred_conf"] = all_probs
    df["second_id"] = all_second
    df["predicted_role"] = [ID2LABEL[p] for p in all_preds]
    df["second_role"] = [ID2LABEL[p] for p in all_second]
    return df


# ---------------------------------------------------------------------------
# Post-processing: role assignment with heuristics
# (ported directly from parallel_sentence_extractor.py cells In[80]-In[80])
# ---------------------------------------------------------------------------
def apply_heuristics(df: pd.DataFrame) -> list[dict]:
    """
    Apply neighbourhood correction rules to predicted roles.
    Returns a list of dicts: {page_num, line_no, role, bbox, text}
    """
    # Build per-page role arrays for fast neighbour lookup
    lines_by_page: dict[int, list] = collections.defaultdict(list)
    for _, row in df.iterrows():
        # For OTHER predicted with low confidence, try second-best non-OTHER
        role = row["predicted_role"]
        conf = float(row["pred_conf"])
        if role == "OTHER" and conf < OTHER_CONF_THRESHOLD:
            second = row["second_role"]
            if second != "OTHER":
                role = second
        lines_by_page[row["page"]].append(
            {
                "page_num": int(row["page"]),
                "line_no": int(row["line_no"]),
                "text": row["text"],
                "role": role,
                "pred_conf": conf,
                "bbox": (
                    float(row["x0"]),
                    float(row["y0"]),
                    float(row["x1"]),
                    float(row["y1"]),
                ),
            }
        )

    final = []
    for page_num in sorted(lines_by_page):
        page_lines = lines_by_page[page_num]
        n = len(page_lines)
        # Keep original predictions for neighbour lookups (matches notebook behaviour:
        # corrections are never written back into the source array, only to output)
        orig = [l["role"] for l in page_lines]
        corrected = list(orig)

        def r(i):
            return orig[i] if 0 <= i < n else None

        for i in range(n):
            role = orig[i]
            prev  = r(i - 1)
            prev2 = r(i - 2)
            nxt   = r(i + 1)
            nxt2  = r(i + 2)

            # Isolated SOURCE with no IGT neighbours → OTHER
            if role == "SOURCE" and nxt in ["OTHER", "SOURCE"] and prev == "OTHER":
                if nxt2 in ["OTHER", None]:
                    role = "OTHER"

            # Line before a SOURCE+GLOSS pair should itself be SOURCE
            if role != "TRANSLATION" and nxt == "SOURCE" and nxt2 == "GLOSS":
                role = "SOURCE"

            # Missing GLOSS: SOURCE … TRANSLATION pattern
            if (
                role != "GLOSS"
                and prev == "SOURCE"
                and nxt == "TRANSLATION"
                and prev2 in ["OTHER", None]
            ):
                role = "GLOSS"

            # Isolated TRANSLATION (no IGT context)
            if (
                role == "TRANSLATION"
                and prev in ["OTHER", None]
                and nxt in ["OTHER", None]
                and prev2 in ["OTHER", None]
                and nxt2 in ["OTHER", None]
            ):
                role = "OTHER"

            # TRANSLATION between SOURCE lines → GLOSS
            if role == "TRANSLATION" and prev == "SOURCE" and nxt == "TRANSLATION":
                role = "GLOSS"

            # Double SOURCE: second one is GLOSS
            if (
                role == "SOURCE"
                and prev == "SOURCE"
                and prev2 == "GLOSS"
                and nxt in ["TRANSLATION", "SOURCE"]
            ):
                role = "GLOSS"

            # Triple GLOSS: middle is SOURCE
            if role == "GLOSS" and prev == "GLOSS" and nxt == "GLOSS":
                role = "SOURCE"

            # GLOSS before SOURCE is actually SOURCE
            if (
                role == "GLOSS"
                and prev in ["OTHER", None]
                and nxt == "GLOSS"
                and nxt2 == "SOURCE"
            ):
                role = "SOURCE"

            # Triple SOURCE: second is GLOSS
            if (
                role == "SOURCE"
                and prev == "SOURCE"
                and nxt == "SOURCE"
                and nxt2 == "GLOSS"
            ):
                role = "GLOSS"

            # Recover OTHER between IGT lines
            if role == "OTHER":
                if nxt == "GLOSS" and nxt2 in ["TRANSLATION", "SOURCE"]:
                    role = "SOURCE"
                elif prev == "SOURCE" and nxt == "TRANSLATION":
                    role = "GLOSS"
                elif prev == "GLOSS" and prev2 == "SOURCE":
                    role = "TRANSLATION"
                elif prev == "SOURCE" and nxt == "SOURCE":
                    role = "GLOSS"
                elif prev == "TRANSLATION" and nxt == "TRANSLATION":
                    role = "TRANSLATION"

            # Pure number / punctuation lines are never IGT
            if re.match(r"^[\d\W_]+$", page_lines[i]["text"].strip()):
                role = "OTHER"

            corrected[i] = role
            entry = dict(page_lines[i])
            entry["role"] = role
            final.append(entry)

    return final


# ---------------------------------------------------------------------------
# Render annotated PDF
# ---------------------------------------------------------------------------
_ROLE_COLORS = {
    "SOURCE": (0.0, 0.0, 1.0),  # blue
    "GLOSS": (0.0, 0.7, 0.0),  # green
    "TRANSLATION": (0.9, 0.0, 0.0),  # red
}
_LEGEND_LABELS = [
    ("SOURCE", (0.0, 0.0, 1.0)),
    ("GLOSS", (0.0, 0.7, 0.0)),
    ("TRANSLATION", (0.9, 0.0, 0.0)),
]


def render_annotated_pdf(
    pdf_path: Path, final_roles: list[dict], out_path: Path
) -> None:
    """Draw coloured bounding boxes on a copy of the PDF and save it."""
    # Group roles by page for fast lookup
    by_page: dict[int, list] = collections.defaultdict(list)
    for entry in final_roles:
        if entry["role"] in _ROLE_COLORS:
            by_page[entry["page_num"]].append(entry)

    doc = pymupdf.open(str(pdf_path))
    for page in doc:
        page_num = page.number + 1
        entries = by_page.get(page_num, [])
        if not entries:
            continue

        for entry in entries:
            color = _ROLE_COLORS[entry["role"]]
            bbox = pymupdf.Rect(entry["bbox"])
            # Border thickness scales with confidence (0.4 – 2.0 pt)
            conf = entry.get("pred_conf", 1.0)
            width = 0.4 + 1.6 * max(0.0, min(1.0, conf))
            page.draw_rect(bbox, color=color, width=width)

        # Insert a small per-page legend in the top-right corner
        _draw_legend(page)

    doc.save(str(out_path))
    doc.close()
    print(f"    Annotated PDF → {out_path.name}")


def _draw_legend(page) -> None:
    """Draw a small colour legend in the top-right corner of a page."""
    rect = page.rect
    x = rect.width - 90
    y = 8.0
    for label, color in _LEGEND_LABELS:
        swatch = pymupdf.Rect(x, y, x + 10, y + 7)
        page.draw_rect(swatch, color=color, fill=color, width=0)
        page.insert_text(
            pymupdf.Point(x + 13, y + 6.5),
            label,
            fontsize=6,
            color=(0, 0, 0),
        )
        y += 10


# ---------------------------------------------------------------------------
# Assemble parallel sentences from IGT role sequence
# ---------------------------------------------------------------------------
def assemble_parallel_sentences(final_roles: list[dict]) -> list[dict]:
    parallel_blocks: list[list[dict]] = []
    current_block: list[dict] = []

    for row in final_roles:
        if row["role"] in ("SOURCE", "GLOSS", "TRANSLATION"):
            current_block.append(row)
        else:
            if current_block:
                _split_and_add(current_block, parallel_blocks)
                current_block = []
    if current_block:
        _split_and_add(current_block, parallel_blocks)

    sentences = []
    unclean = 0
    for block in parallel_blocks:
        source_lines = [r["text"] for r in block if r["role"] == "SOURCE"]
        gloss_lines = [r["text"] for r in block if r["role"] == "GLOSS"]
        trans_lines = [r["text"] for r in block if r["role"] == "TRANSLATION"]
        if not source_lines or not trans_lines:
            unclean += 1
            continue
        sentences.append(
            {
                "source": " ".join(source_lines),
                "gloss": " ".join(gloss_lines),
                "translation": " ".join(trans_lines),
                "page_num": block[0]["page_num"],
                "bboxes": [r["bbox"] for r in block],
            }
        )

    print(
        f"    Parallel sentences: {len(sentences)}  (dropped {unclean} unclean blocks)"
    )
    return sentences


def _split_and_add(block: list[dict], out: list) -> None:
    """Split a continuous IGT run at each TRANSLATION boundary."""
    trans_indices = [i for i, r in enumerate(block) if r["role"] == "TRANSLATION"]
    source_indices = [i for i, r in enumerate(block) if r["role"] == "SOURCE"]
    if not trans_indices or not source_indices:
        out.append(block)
        return

    # Group consecutive translation indices
    splits: list[list[int]] = []
    prev = -2
    for idx in trans_indices:
        if idx != prev + 1:
            splits.append([idx])
        else:
            splits[-1].append(idx)
        prev = idx
    split_at = [s[-1] for s in splits]

    if len(split_at) < 2:
        out.append(block)
        return

    start = 0
    for end in split_at:
        out.append(block[start : end + 1])
        start = end + 1
    if start < len(block):
        out.append(block[start:])


# ---------------------------------------------------------------------------
# Per-book processing
# ---------------------------------------------------------------------------
def process_book(
    pdf_path: Path, model, tokenizer, device: str, force: bool, render: bool
) -> None:
    roles_file = OUT_DIR / f"{pdf_path.stem}_predicted_roles.json"
    sents_file = OUT_DIR / f"{pdf_path.stem}_parallel_sentences.json"
    annot_file = OUT_DIR / f"{pdf_path.stem}_annotated.pdf"

    if sents_file.exists() and not force:
        data = json.loads(sents_file.read_text(encoding="utf-8"))
        if data.get("status") == "complete":
            # Still render if the annotated PDF is missing
            if render and not annot_file.exists():
                roles = json.loads(roles_file.read_text(encoding="utf-8"))
                render_annotated_pdf(pdf_path, roles, annot_file)
            else:
                print(
                    f"  [skip] {pdf_path.name} — {data['num_sentences']} sentences already extracted"
                )
            return

    print(f"  Extracting visual lines …")
    doc = pymupdf.open(str(pdf_path))
    all_merged_lines = []
    for page in doc:
        lines = extract_visual_lines(page)
        if lines:
            all_merged_lines.append({"page_num": page.number + 1, "lines": lines})
    doc.close()
    print(f"    Pages: {len(all_merged_lines)}")

    df = build_dataframe(all_merged_lines)
    print(f"    Lines: {len(df)}")

    df = run_inference(df, model, tokenizer, device)

    final_roles = apply_heuristics(df)

    # Save per-line role predictions
    roles_file.write_text(
        json.dumps(final_roles, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sentences = assemble_parallel_sentences(final_roles)

    result = {
        "book": pdf_path.name,
        "num_sentences": len(sentences),
        "sentences": sentences,
        "status": "complete",
    }
    sents_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    Saved → {sents_file.name}  ({len(sentences)} sentences)")

    if render:
        render_annotated_pdf(pdf_path, final_roles, annot_file)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(pdf_paths: list[Path]) -> None:
    print("\n=== Summary ===")
    print(f"{'Book':<50} {'Sentences':>10}")
    print("-" * 62)
    total = 0
    for pdf_path in sorted(pdf_paths):
        f = OUT_DIR / f"{pdf_path.stem}_parallel_sentences.json"
        if not f.exists():
            print(f"{pdf_path.name:<50} {'ERROR':>10}")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        n = data.get("num_sentences", 0)
        total += n
        print(f"{pdf_path.name:<50} {n:>10}")
    print("-" * 62)
    print(f"{'TOTAL':<50} {total:>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch BERT IGT classifier for grammar books"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-process even if output already exists"
    )
    parser.add_argument(
        "--books",
        nargs="+",
        metavar="BOOK.pdf",
        help="Process only these specific books (filenames only)",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Skip writing annotated PDFs"
    )
    args = parser.parse_args()

    if args.books:
        pdf_paths = [GRAMMAR_BOOKS_DIR / b for b in args.books]
        missing = [p for p in pdf_paths if not p.exists()]
        if missing:
            print("ERROR — not found:", *[str(m) for m in missing])
            sys.exit(1)
    else:
        pdf_paths = sorted(GRAMMAR_BOOKS_DIR.glob("*.pdf"))
        print(f"Found {len(pdf_paths)} PDFs in {GRAMMAR_BOOKS_DIR.name}/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_DIR.name} on {device} …")
    model, tokenizer = load_model(device)
    print("Model loaded.\n")

    for i, pdf_path in enumerate(pdf_paths):
        print(f"[{i+1}/{len(pdf_paths)}] {pdf_path.name}")
        try:
            process_book(
                pdf_path,
                model,
                tokenizer,
                device,
                force=args.force,
                render=not args.no_render,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback

            traceback.print_exc()

    print_summary(pdf_paths)
    print(f"\nResults → {OUT_DIR}")


if __name__ == "__main__":
    main()
