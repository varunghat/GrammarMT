#!/usr/bin/env python
# coding: utf-8

import pymupdf


def extract_visual_lines(page):

    spans = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sb = span["bbox"]
                spans.append({
                    "text": span["text"],
                    "x0": sb[0],
                    "y0": sb[1],
                    "x1": sb[2],
                    "y1": sb[3],
                    "font": span.get("font"),
                    "size": span.get("size"),
                    "block": block,  # optional
                })

    spans = sorted(spans, key=lambda s: s["y0"])
    if len(spans) == 0:
        return None

    # Dynamic tolerance based on median height
    heights = [s["y1"] - s["y0"] for s in spans]
    median_h = np.median(heights) if heights else 10
    Y_TOL = median_h * 0.4

    visual_lines = []
    current_line = [spans[0]]

    for prev, curr in zip(spans, spans[1:]):
        if abs(curr["y0"] - prev["y0"]) < Y_TOL:
            current_line.append(curr)
        else:
            visual_lines.append(current_line)

            current_line = [curr]

    visual_lines.append(current_line)

    # Merge and sort spans within each line
    merged_lines = []
    for line_spans in visual_lines:
        sorted_spans = sorted(line_spans, key=lambda s: s["x0"])
        merged_text = " ".join([s["text"] for s in sorted_spans if s["text"].strip()])
        # Get most common font size and style by length weighting
        font_sizes = {}
        font_styles = {}
        for s in sorted_spans:
            length = len(s["text"])
            font_key = (s["font"], s["size"])
            font_sizes[font_key] = font_sizes.get(font_key, 0) + length
            font_styles[s["font"]] = font_styles.get(s["font"], 0) + length

        if font_sizes:
            common_font_size = max(font_sizes, key=font_sizes.get)
        else:
            common_font_size = (None, None)

        if font_styles:
            common_font_style = max(font_styles, key=font_styles.get)
        else:
            common_font_style = None

        merged_lines.append({
            "text": merged_text,
            "spans": sorted_spans,
            "x0": sorted_spans[0]["x0"],
            "y0": np.mean([s["y0"] for s in sorted_spans]),
            "x1": sorted_spans[-1]["x1"],
            "y1": max(s["y1"] for s in sorted_spans),
            "x0_list": [s["x0"] for s in sorted_spans],
            "y0_list": [s["y0"] for s in sorted_spans],
            "x1_list": [s["x1"] for s in sorted_spans],
            "y1_list": [s["y1"] for s in sorted_spans],
            "span_count": len(sorted_spans),
            "common_font_size": common_font_size[1],
            "common_font_style": common_font_style,
        })

    return merged_lines


parent_folder = "../Grammar_books"
pdf_file = "kalamang.pdf"
input_file_path = f"{parent_folder}/{pdf_file}"
output_folder = "../data/extracted_dictionaries"
output_file_name = f"{output_folder}/{pdf_file.replace('.pdf', '')}_extracted_word_list_columns.csv"
pageno_wordlist_start = 0
number_of_columns = 2


doc_pymupdf = pymupdf.open(input_file_path)

print("PDF opened with pymupdf successfully with page count:", doc_pymupdf.page_count)


from collections import Counter
import numpy as np


column_data = []
column_data_csv = []

VERBOSE = False

for page in doc_pymupdf:

    if page.number + 1 < pageno_wordlist_start:
        continue
    if VERBOSE: print(f"Processing page {page.number + 1}...")
    lines = extract_visual_lines(page)
    x0_counter = Counter()
    y0_counter = Counter()
    if lines is None:
        if VERBOSE: print("No lines found on this page.")
        continue
    for line in lines:
        spans = line["spans"]
        spans_x0_list = line["x0_list"]
        spans_y0_list = line["y0_list"]
        for span in spans:
            x0 = span["x0"]
            y0 = span["y0"]
            x0_counter[round(x0, 2)] += 1
            y0_counter[round(y0, 2)] += 1

    if VERBOSE: print("Most common x0 positions:", x0_counter.most_common(10))
    likely_columns = [x for x, count in x0_counter.most_common(number_of_columns)]
    likely_columns = sorted(likely_columns)
    if VERBOSE: print("Likely column x0 positions:", likely_columns)

    sort_y0_counts = sorted(y0_counter.items())
    sort_y0_counts = [y0 for y0, count in sort_y0_counts]
    y0_differences = [j - i for i, j in zip(sort_y0_counts[:-1], sort_y0_counts[1:])]
    # Find common y0 differences (line heights) to find normal line spacing VS continuation
    y0_diff_counter = Counter()
    for diff in y0_differences:
        y0_diff_counter[round(diff, 2)] += 1

    # Combine the values with a 0.5 tolerance
    combined_y0_diff_counter = Counter()
    for y0_diff, count in y0_diff_counter.items():
        combined = False
        for combined_y0_diff in combined_y0_diff_counter:
            if abs(y0_diff - combined_y0_diff) < 0.5:
                combined_y0_diff_counter[combined_y0_diff] += count
                combined = True
                break
        if not combined:
            combined_y0_diff_counter[y0_diff] = count
    # Pick top two, max for normal line height, second max for continuation lines
    common_y0_diffs = [y0_diff for y0_diff, count in combined_y0_diff_counter.most_common(2)]
    if len(common_y0_diffs) < 2:
        common_y0_diffs.append(common_y0_diffs[0] * 0.5)  # Fallback
    normal_line_height, continuation_line_height = max(common_y0_diffs), min(common_y0_diffs)
    if VERBOSE: print(f"Normal line height: {normal_line_height}, Continuation line height: {continuation_line_height}")

    column_text = []
    likely_rows_in_columns = []
    for line_idx, line in enumerate(lines):
        spans = line["spans"]
        spans_x0_list = line["x0_list"]

        count_spans_in_columns = 0
        column_text_row = {x: None for x in likely_columns}
        for i, (span, x0) in enumerate(zip(spans, spans_x0_list)):
            assigned = False
            for j, col_x0 in enumerate(likely_columns):
                next_col_x0 = None if j == len(likely_columns) - 1 else likely_columns[j + 1]
                x1 = span["x1"]
                if abs(x0 - col_x0) < 2.0:
                    if next_col_x0 is None or x1 < next_col_x0 - 1.0:  # Tolerance of 2.0 units
                        column_text_row[col_x0] = span["text"]
                        assigned = True
                        count_spans_in_columns += 1
                        break

            if assigned:
                continue
        column_text.append({"line_idx": line_idx, "columns": column_text_row, "y": line["y0"]})

        if count_spans_in_columns > number_of_columns - 1:
            likely_rows_in_columns.append(line_idx)

    for row in column_text:
        line_idx = row["line_idx"]
        y0 = row["y"]
        if line_idx == 0:
            continue
        prev_y0 = column_text[line_idx - 1]["y"]
        y0_diff = y0 - prev_y0
        # If line is a continuation line, mark it as such so we can append it to the previous row
        if abs(y0_diff - continuation_line_height) < 0.5:
            row["continuation"] = True
        else:
            row["continuation"] = False

    # NAIVE APPROACH FOR NOW TODO: FIX THIS
    if len(likely_rows_in_columns) == 0:
        if VERBOSE: print("No likely rows found for columns on this page.")
        continue
    likely_start_idx, likely_end_idx = likely_rows_in_columns[0], likely_rows_in_columns[-1] + 1

    page_column_data = []
    for i, row in enumerate(column_text):
        line_idx = row["line_idx"]
        column_text = row["columns"]
        continuation = row.get("continuation", False)

        if likely_start_idx <= line_idx <= likely_end_idx:
            if VERBOSE: print(f"\nLine {line_idx}:")
            if VERBOSE: print(column_text)
            page_column_data.append({"line_idx": line_idx, "columns": column_text, "continuation": continuation})
            column_text_list = list(column_text.values())

            column_text_list = [ct.strip() if ct is not None else "" for ct in column_text_list]
            column_text_list = [None if ct == "" else ct for ct in column_text_list]

            column_data_csv.append([page.number + 1, line_idx, continuation] + column_text_list)
    column_data.append({"page": page.number + 1, "data": page_column_data})


for page in column_data:

    number_of_rows = len(page['data'])
    average_length = np.mean([len(str(d['columns'])) for d in page['data']])
    number_of_discontinuous_rows = sum(1 for d in page['data'] if not d.get('continuation', False))
    if page['page'] > 471:
        print(f"\n---Page {page['page']} :")
        print(f"---  Number of rows: {number_of_rows}, Average row length: {average_length:.2f}, Discontinuous rows: {number_of_discontinuous_rows}")
    if number_of_rows > 25 and average_length < 50 and number_of_discontinuous_rows < 4:
        print(f"\nPage Selected {page['page']} column data:")
        print(f"  Number of rows: {number_of_rows}, Average row length: {average_length:.2f}, Discontinuous rows: {number_of_discontinuous_rows}")
