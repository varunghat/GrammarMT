import pymupdf
import json
import os
import re
from typing import List, Dict, Tuple
import typer
from langdetect import detect_langs

import spacy
import unimorph

app = typer.Typer(name="pdf_parser", help="PDF Parser to extract text, font styles, and sizes from PDF documents",pretty_exceptions_enable=False)



def classify_text_igt_english(text):
    if text is None or len(text.strip()) == 0:
        return "reject"
    IGT_ABBRS = {
        "1","2","3","SG","PL","DU","PST","PRS","FUT","IPFV","PFV","PROG","HAB",
        "NOM","ACC","ERG","GEN","DAT","LOC","ALL","ABL","INS","COM","BEN","TOP","FOC",
    "CAUS","PASS","MID","REFL","RECIP","AUX","COP","NEG","COND","IMP","SUBJ","IND"
    }

    IGT_ABBRS = {abbr.lower() for abbr in IGT_ABBRS}

    

    gloss_like_score = 0
    normalized_english_score = 0
    words = text.split()
    total_words = len(words)
    #print(words)
    for word in words:

        subwords = word.split(".")
        subwords = [sw.split("-") for sw in subwords]
        subwords = [item for sublist in subwords for item in sublist]
        for sw in subwords:
            if sw in IGT_ABBRS:
                gloss_like_score += 2
                #print(sw)
    
    try:
        # clean text (remove any text within brackets)
        res = re.sub(r'\[.*?\]', '', text)
        res = detect_langs(res)    
        if res and res[0].lang == 'en':
            normalized_english_score = res[0].prob
    except Exception as e:
        #print(f"Error detecting language: {e}")
        #print("Text causing error:", text)
        return "reject"

    normalized_gloss_score = gloss_like_score / total_words

    #print(f"Text: {text}, Gloss score: {normalized_gloss_score}, English score: {normalized_english_score}, Total words: {total_words}")

    if normalized_gloss_score > 0.1:
        return "IGT"
    elif normalized_english_score > 0.8:
        return "English"
    else:
        return "unknown"

def collect_consecutive_lines_across_blocks(start_block_index, start_line_index, blocks, target_font=None) -> Tuple[int, int, str, str]:
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


def build_line_table(pdf_file_path):
    all_lines = []
    text_sizes = []
    size_font_tuples = []

   
    try:
        print("Opening PDF file...")
        doc = pymupdf.open(pdf_file_path)
        print("PDF file opened successfully.")
        print("Number of pages in PDF file:", doc.page_count)
    except Exception as e:
        print("Error occurred while opening PDF file:", e)
        return all_lines, text_sizes, size_font_tuples

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block_idx, block in enumerate(blocks):
            for line_idx, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Merge spans into one line string
                merged_text = " ".join(span["text"].strip() for span in spans if span["text"].strip())
                if not merged_text:
                    continue

                # Choose a "dominant" font/size for the line
                sizes = [span["size"] for span in spans]
                fonts = [span["font"] for span in spans]

                # Representative = most common size & font
                rep_size = max(set(sizes), key=sizes.count)
                rep_font = max(set(fonts), key=fonts.count)

                # Position = from first span
                x, y = spans[0]["bbox"][:2]

                all_lines.append({
                    "page": page_num + 1,
                    "block": block_idx,
                    "line": line_idx,
                    "text": merged_text,
                    "size": rep_size,
                    "font": rep_font,
                    "x": x,
                    "y": y,
                })

                text_sizes.append(rep_size)
                size_font_tuples.append((rep_size, rep_font))


    return all_lines, text_sizes, size_font_tuples

@app.command()
def parse_pdf(pdf_file_path: str):
    """
    Parses a PDF file and extracts text, font styles, and sizes.
    """
    # Open the PDF file 
    
    pdf_file = os.path.abspath(pdf_file_path)
    pdf_file_name = os.path.splitext(os.path.basename(pdf_file))[0]
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")
    if not pdf_file.lower().endswith('.pdf'):
        raise ValueError(f"File is not a PDF: {pdf_file}")
    try:
        print("Opening PDF file...")
        doc = pymupdf.open(pdf_file)
        print("PDF file opened successfully.")
        print("Number of pages in PDF file:", doc.page_count)
    except Exception as e:
        print("Error occurred while opening PDF file:", e)


    # First pass: Collect all text sizes and fonts
    text_sizes = []
    size_font_tuples = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            lines = block.get("lines", [])
            for line_index,line in enumerate(lines):
                spans = line.get("spans", [])

                for span_index,span in enumerate(spans):
                    text = span["text"].strip()
                    size = span["size"]
                    font = span["font"]
                    if not text:
                        continue
                    text_sizes.append(size)
                    size_font_tuples.append((size, font))

                
    text_sizes_set = set(text_sizes)
    print("Unique text sizes found:", len(text_sizes_set))
    #print("Text sizes:", sorted(text_sizes_set))
    print("Total text sizes collected:", len(text_sizes))

    counter = {}
    for size in text_sizes_set:
        counter[size] = text_sizes.count(size)
                
    # Size font tuples counter
    size_font_counter = {}
    for size, font in size_font_tuples:
        if (size, font) not in size_font_counter:
            size_font_counter[(size, font)] = 0
        size_font_counter[(size, font)] += 1

        

    # Get max count
    max_count = max(counter.values())
    max_size_font = max(size_font_counter.items(), key=lambda x: x[1])[0]
    print(f"Max count: {max_count}, Size: {max_size_font[0]}, Font: {max_size_font[1]}")
    print("---")

    # Collect all consecutive lines with the same font across blocks
    collected_text_sizes = []
    sizes_text_lengths = {}
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block_index, block in enumerate(blocks):
            lines = block.get("lines", [])
            for line_index, line in enumerate(lines):
                _, _, collected_text, target_font = collect_consecutive_lines_across_blocks(
                    block_index, line_index, blocks
                )
                if collected_text:
                    for span in line.get("spans", []):
                        collected_text_sizes.append(span["size"])
                        if sizes_text_lengths.get(span["size"]) is None:
                            sizes_text_lengths[span["size"]] = len(collected_text)
                        else:
                            sizes_text_lengths[span["size"]] += len(collected_text)
                       
    #print("Collected text sizes across blocks:", collected_text_sizes)
    # No of blocks
    print("Number of blocks:", len(collected_text_sizes))

    sizes_text_lengths_sorted = sorted(sizes_text_lengths.items(), key=lambda x: x[1], reverse=True)
    
    # Take font size with max content and consider it as main body font size. Now find the font size which appears 
    main_body_font_size = sizes_text_lengths_sorted[0][0]
    print("Main body font size:", main_body_font_size)

    # Near body fonts
    near_body_fonts = [size for size in sizes_text_lengths_sorted if abs(size[0] - main_body_font_size) <= 1]
    print("Near body font sizes:", near_body_fonts)
    minimum_near_body_font_size = min(near_body_fonts, key=lambda x: x[1])
    maximum_near_body_font_size = max(near_body_fonts, key=lambda x: x[1])
    print("Maximum near body font size:", maximum_near_body_font_size)
    print("Minimum near body font size:", minimum_near_body_font_size)

    # Smaller than minimum near body font size
    smaller_than_near_body = [size for size in sizes_text_lengths_sorted if size[0] < minimum_near_body_font_size[0]]
    print("Smaller than minimum near body font sizes:", smaller_than_near_body)

    # Larger than maximum near body font size
    larger_than_near_body = [size for size in sizes_text_lengths_sorted if size[0] > maximum_near_body_font_size[0]]
    print("Larger than maximum near body font sizes:", larger_than_near_body)
               
    # Take larger than maximum near body font size as headings, get number of lines from counter variable before calculated

    # TODO: These thresholds are arbitrary, adjust them later
    length_threshold = 30
    lines_threshold = 5

    headings_filtered = []

    for size,length in larger_than_near_body:
        no_lines = counter.get(size, -1)
        if length > length_threshold and no_lines > lines_threshold:
            headings_filtered.append((size, length, no_lines))

    # sort by size and mark as heading, subheading and sub-subheading
    headings_sorted = sorted(headings_filtered, key=lambda x: x[0], reverse=True)
    
    HEADING_THRESHOLD = 4 # 4 levels of headings
    headings_sorted = headings_sorted[:HEADING_THRESHOLD]
    print("Headings found (size, total text length, no of lines):", headings_sorted)

    # Extract only sizes
    headings_sizes = [size for size, length, no_lines in headings_sorted]


    # Parse through and collect sections based on headings
    section_text = []  # List of sections
    current_section = None
    parallel_sentences = []

    parallel_temp_storage = []
    parallel_y_pos = None
    ilg_temp_storage = []
    all_lines = []

    text_buffer = []  # Buffer to collect text for the current section

    parallel_flag = False

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block_index, block in enumerate(blocks):
            lines = block.get("lines", [])
            for line_index, line in enumerate(lines):
                spans = line.get("spans", [])
                for span in spans:
                    text = span["text"].strip()
                    size = span["size"]
                    font = span["font"]
                    # y position
                    x_pos,y_pos = span["bbox"][:2]
                    if not text:
                        continue

                    all_lines.append({
                        "text": text,
                        "size": size,
                        "font": font,
                        "page": page_num + 1,
                        "x_position": x_pos,
                        "y_position": y_pos
                    })

                    #print(text, size, font)

                    

                    if size in headings_sizes:
                        heading_index = headings_sizes.index(size) + 1
                        # store the current section if it exists
                        if current_section is not None:
                            section_text.append(current_section)
                        current_section = {
                            "heading": text,
                            "text": "",
                            "header_number": heading_index,
                            "font_style": font,
                            "font_size": size,
                            "page": page_num + 1,
                        }
                        #print(f"Heading found: {text}, Font: {font}, H{heading_index}")
                    elif current_section is not None and size == main_body_font_size:
                        # Check for parallel sentences (starts with a paranthesis with number inside)
                        # TODO: HARDCODED TEMPORARY FIX, GENERALIZE LATER
                        if font == max_size_font[1]:
                            if current_section["text"]:
                                current_section["text"] += " "
                            current_section["text"] += text



    if current_section is not None:
        section_text.append(current_section)

    # Quick cleaning 
    for i, section in enumerate(section_text):
        if section is None:
            continue
        # Check if two headings are consecutive with no text in between, if so, merge them
        #print(section)
        if i < len(section_text) - 1 and section["header_number"] == section_text[i + 1]["header_number"] and section["text"].strip() == "":
            while(i < len(section_text) - 1 and section["header_number"] == section_text[i + 1]["header_number"]) and section_text[i+1]["text"].strip() == "":
                
                section["heading"] += " " + section_text[i + 1]["heading"]
                print(i+1,section_text[i+1]["heading"])
                
                section_text[i+1] = None
                i+=1
            section["text"] = section_text[i+1]["text"]
            section_text[i+1] = None
            
            
            
            

    section_text = [section for section in section_text if section is not None]

    
    # print how many sections were found
    print(f"Total sections found: {len(section_text)}")


    ################################################
    # Parallel sentence extraction
    ################################################
    parallel_sentences = []

    idx = 0
    while idx < len(all_lines):
        line = all_lines[idx]
        text = line["text"]
        page_number = line["page"]
        if re.fullmatch(r'\(\d+\)', text):
            #print(f"Parallel sentence found at page {page_number}: {text}")
            top_x = None
            bottom_y = None
            new_line = all_lines[idx+1]
            new_text = new_line["text"]
            new_x_pos = new_line["x_position"]
            top_x = new_x_pos

            #print(top_x)
            temp = []
            translation_idx = None
            counter = 0
            while idx < len(all_lines)-1:
                idx+=1
                new_line = all_lines[idx]
                new_text = new_line["text"]
                new_x_pos,new_y_pos = new_line["x_position"],new_line["y_position"]
                if(top_x - new_x_pos > 2):
                    break
                    temp.append((new_text,new_x_pos,new_y_pos))
                temp.append((new_text,new_x_pos,new_y_pos))
            #print(temp)
            # Sort temp by y position
            temp_sorted = sorted(temp, key=lambda x: x[2])
            
            # join all text in temp_sorted which are in same y position
            joined_text = []
            current_y = None
            current_line = []
            for t in temp_sorted:
                if current_y is None:
                    current_y = t[2]
                    current_line.append(t[0])
                elif abs(t[2] - current_y) < 2:
                    current_line.append(t[0])
                else:
                    joined_text.append(" ".join(current_line))
                    current_y = t[2]
                    current_line = [t[0]]

            if current_line:
                joined_text.append(" ".join(current_line))

            # Classify each line in the joined text as IGT, english or other based on some heuristics
            igt = []
            english = []
            other = []

            for i,line in enumerate(joined_text):
                result = classify_text_igt_english(line)
                if result == "IGT":
                    igt.append(line)
                elif result == "English":
                    english.append(line)
                    # For now, fix later TODO
                    # append rest of lines to english as well
                    english.extend(joined_text[i+1:])
                    break
                elif result == "reject":
                    continue
                else:
                    other.append(line)

            igt = " ".join(igt)
            english = " ".join(english)
            other = " ".join(other)
            
            #print("text:", other)
            #print("IGT:", igt)
            #print("English:", english)
            #print("---")
            #check lengths of the texts, if too long, it's a false flag
            if len(igt) > 1000 or len(english) > 1000 or len(other) > 1000:
                #print("False flag, skipping...")
                continue

            # Clean parallel sentences
            if (len(other) < 1 or len(english) < 1 ):
                #print("Too short, skipping...")
                continue
            parallel_sentences.append({
                "page": page_number,
                "text": other,
                "IGT": igt,
                "English": english,
            })
        else:
            idx+=1




    print("Storing section text to JSON file...")
    with open(f"parsed_grammar_json/{pdf_file_name}_sections.json", "w", encoding="utf-8") as f:
        json.dump(section_text, f, ensure_ascii=False, indent=4)

    print(f"Section text stored successfully to parsed_grammar_json/{pdf_file_name}_sections.json")

    # store parallel sentences
    print("Storing parallel sentences to JSON file...")
    with open(f"parallel_sents/{pdf_file_name}_parallel_sentences.json", "w", encoding="utf-8") as f:
        json.dump(parallel_sentences, f, ensure_ascii=False, indent=4)

    print(f"Parallel sentences stored successfully to parallel_sents/{pdf_file_name}_parallel_sentences.json")


    # Preprocess parallel sentences


    nlp = spacy.load("en_core_web_sm")

    for sentence in parallel_sentences:
        translated_sentence = sentence["English"]

        doc = nlp(translated_sentence)
        res = []
        tense = None
        genders = []
        number = None
        plural_details = None  # To store details if plural is detected
        genders_details = []
        for tok in doc:
            morph = tok.morph.to_dict()
            res.append({
                "word": tok.text,
                "lemma": tok.lemma_,
                "upos": tok.pos_,
                "tag": tok.tag_,
                "morph": morph,
            })

            if not number and morph.get("Number", None):
                number = morph.get("Number", None)
            elif morph.get("Number", None) == "Plur":
                number = "Plur"
                plural_details = {
                    "word": tok.text, 
                    "morph": morph
                }

            if not tense and morph.get("Tense", None):
                tense = morph.get("Tense", None)

            if morph.get("Gender", None):
                genders.append(morph.get("Gender", None))
                genders_details.append({
                    "word": tok.text, 
                    "morph": morph})

            
            
            #if morph.get("")


            
        #print(f"Sentence: {translated_sentence} \n Tense: {tense}, Number: {number}, Genders: {genders}")
        #print(f"Sentence: {translated_sentence} \n Plural_details: {plural_details}, gender_details: {genders_details}")
        

            
        sentence["spacy_info"] = {"res": res, "tense": tense, "number": number, "plural_details": plural_details, "genders": genders,"genders_details":genders_details}



    # Change the key from "text" to "sentence" and "English" to "translated_sentence"
    for sentence in parallel_sentences:
        sentence["sentence"] = sentence.pop("text")
        sentence["translated_sentence"] = sentence.pop("English")



    with open(f"parallel_sents/{pdf_file_name}_parallel_sentences_with_pos_and_unimorph.json", "w", encoding="utf-8") as f:
        json.dump(parallel_sentences, f, ensure_ascii=False, indent=4)
    print(f"Parallel sentences with POS and Unimorph info stored successfully to parallel_sents/{pdf_file_name}_parallel_sentences_with_pos_and_unimorph.json")

    ##### 
    # Word list extraction
    #####

    print("DEBUG: Extracting word list...\n\n\n\n")
    for page_num, page in enumerate(doc):
        if page_num<476:
            continue
        blocks = page.get_text("dict")["blocks"]
        for block_index, block in enumerate(blocks):
            lines = block.get("lines", [])
            for line_index, line in enumerate(lines):
                spans = line.get("spans", [])
                for span in spans:
                    text = span["text"].strip()
                    size = span["size"]
                    font = span["font"]
                    if not text:
                        continue
                    #print(text, size, font)


if __name__ == "__main__":
    app()


