import streamlit as st
import json
import os
from pathlib import Path
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("Thesis viewer")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_pdf:
    # Get the uploaded PDF's filename (without extension)
    pdf_filename = Path(uploaded_pdf.name)
    base_name = pdf_filename.stem

    # Construct paths based on the uploaded PDF's name
    data_dir = Path("C:/Users/Varun/master_thesis")
    # pdf_filename = data_dir / pdf_filename
    parsed_grammar_filename = data_dir / "parsed_grammar_json"/ f"{base_name}_sections.json"
    json_filename = data_dir / "classified_json" / f"{base_name}_sections_classified.json"
    parallel_sents_filename = data_dir / "parallel_sents" / f"{base_name}_parallel_sentences_with_pos_and_unimorph.json"
    gpt_section_tagging_filename = parsed_grammar_filename
    gpt_extracted_rules_filename = data_dir / f"{base_name}_gpt_extracted_rules_direct.json"

    print("Looking for JSON file at:", json_filename)
    print("Looking for parsed grammar JSON file at:", parsed_grammar_filename)
    print("Looking for parallel sentences JSON file at:", parallel_sents_filename)
    #print("Looking for GPT section tagging file at:", gpt_section_tagging_filename)
    print("Looking for GPT extracted rules file at:", gpt_extracted_rules_filename) 

    # TODO: Change how this is handled
    filtered_mapping_filename = Path("C:/Users/Varun/master_thesis/filtered_mapping.json")
    

    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Navigation bar using Streamlit's radio in the sidebar
            st.sidebar.title("Navigation")
            page = st.sidebar.radio("Go to", ["Sections", "Parallel Sentences", "Dictionary"])

            parallel_sentences = None
            if os.path.exists(parallel_sents_filename):
                with open(parallel_sents_filename, "r", encoding="utf-8") as f:
                    parallel_sentences = json.load(f)

            gpt_extracted_rules = None
            if os.path.exists(gpt_extracted_rules_filename):
                with open(gpt_extracted_rules_filename, "r", encoding="utf-8") as f:
                    gpt_extracted_rules = json.load(f)

            # Read the filtered mapping if available
            filtered_mapping = {}
            if os.path.exists(filtered_mapping_filename):
                with open(filtered_mapping_filename, "r", encoding="utf-8") as f:
                    filtered_mapping = json.load(f)


            # Main page logic based on navigation
            if page == "Sections":
                st.header(f"Extracted Sections from {pdf_filename}")
                num_sections = len(data)
                st.write(f"Total sections extracted: {num_sections}")
                # The rest of the section/page display logic will remain here (already present below)
                # Option to display by section or by page
                display_mode = st.radio(
                    "Display mode",
                    ("Each section separately", "Sections by page")
                )

                # Read the gpt tagging of each section if available
                gpt_section_tagging = {}
                if os.path.exists(gpt_section_tagging_filename):
                    with open(gpt_section_tagging_filename, "r", encoding="utf-8") as f:
                        gpt_section_tagging = json.load(f)

                else:
                    st.warning("GPT tagging file not found. Sections will not have tags.")


                # Add GPT tagging and book_tag to each section
                for section in data:
                    section_header = section.get('heading', '')
                    # Try to find the matching GPT-tagged section by heading
                    gpt_tag = next(
                        (item for item in gpt_section_tagging if item.get('heading', '') == section_header),
                        None
                    )
    
                    if gpt_tag:
                        section['book_tag'] = gpt_tag.get('book_tag', 'Unknown')
                        section['grammar_tags'] = gpt_tag.get('grammar_tags', [])
                    else:
                        section['book_tag'] = 'Unknown'
                        section['grammar_tags'] = []

                # Prepare data for page-wise display
                if display_mode == "Sections by page":
                    # Group sections by page
                   
                    page_dict = defaultdict(list)
                    for section in data:
                        page = section.get('page', 'Unknown')
                        page_dict[page].append(section)
                    pages = sorted(page_dict.keys())
                    if "page_idx" not in st.session_state:
                        st.session_state.page_idx = 0

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        if st.button("Previous Page", disabled=st.session_state.page_idx == 0):
                            st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
                    with col3:
                        if st.button("Next Page", disabled=st.session_state.page_idx == len(pages) - 1):
                            st.session_state.page_idx = min(len(pages) - 1, st.session_state.page_idx + 1)

                    # Option to enter page number directly
                    with col2:
                        page_input = st.number_input(
                            "Go to page number",
                            min_value=1,
                            max_value=len(pages),
                            value=st.session_state.page_idx + 1,
                            step=1,
                            key="page_number_input"
                        )
                        if page_input != st.session_state.page_idx + 1:
                            st.session_state.page_idx = int(page_input) - 1

                    current_page = pages[st.session_state.page_idx]
                    st.subheader(f"Page {current_page} ({st.session_state.page_idx + 1} of {len(pages)})")
                    for section in page_dict[current_page]:
                        st.header(section.get('heading', '') + f" ({section.get('type', 'Unknown')})")
                        st.markdown(section.get('text', ''))
                        # Display the tag below the section in green
                        st.markdown(
                            f"Tag: <span style='color: green;'>{section.get('book_tag', 'Unknown')}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("### Grammar Tags")
                        grammar_tags = section.get('grammar_tags', [])
                        if grammar_tags:
                            st.markdown(", ".join(f"<span style='color: lightblue;'>{tag}</span>" for tag in grammar_tags), unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color: red;'>No grammar tags found</span>", unsafe_allow_html=True)
                        st.markdown("---")

                        # Display the rule extracted by GPT if available
                        if gpt_extracted_rules:
                            # Find the rules for the current section by matching section index (page-wise)
                            # First, get the section's index in the full data list
                            section_idx = data.index(section)
                            index = filtered_mapping.get(str(section_idx), None)
                            if index is not None:
                                print("Index found in filtered mapping:", index)
                                # Get the GPT response for this section
                                gpt_response_section_paragraphed = gpt_extracted_rules[index] if index < len(gpt_extracted_rules) else None
                            
                                # Combine the list of lists into one list
                                if isinstance(gpt_response_section_paragraphed, list):
                                    gpt_response_section = [
                                        item
                                        for sublist in gpt_response_section_paragraphed if isinstance(sublist, list)
                                        for item in sublist
                                    ]
                                else:
                                    gpt_response_section = None
                           

                                if gpt_response_section:
                                    st.markdown("### GPT Extracted Rules")
                                    for idx, gpt_item in enumerate(gpt_response_section):
                                        st.markdown(f"**{idx + 1}. Sentence Index:** {gpt_item.get('sentence_index', 'Unknown')}")
                                        #st.markdown(f"**Sentence:** {gpt_item.get('sentence', 'No sentence provided')}")
                                        st.markdown(f"**Description:** {gpt_item.get('description', 'No description')}")
                                        st.markdown(f"**Condition:** {gpt_item.get('condition', 'None')}")
                                        st.markdown(f"**Action:** {gpt_item.get('action', 'None')}")
                                        st.markdown(f"**FST Rule:** {gpt_item.get('fst_rule', 'None')}")
                                        st.markdown(f"**Usage:** {gpt_item.get('usage', 'None')}")
                                        st.markdown(f"**Unimorph:** {gpt_item.get('unimorph', 'None')}")
                                        st.markdown("---")
                                else:
                                    st.warning("No GPT extracted rule found for this section.")

                    

                else:  # Each section separately
                    if "section_idx" not in st.session_state:
                        st.session_state.section_idx = 0

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        if st.button("Previous", disabled=st.session_state.section_idx == 0):
                            st.session_state.section_idx = max(0, st.session_state.section_idx - 1)
                    with col3:
                        if st.button("Next", disabled=st.session_state.section_idx == num_sections - 1):
                            st.session_state.section_idx = min(num_sections - 1, st.session_state.section_idx + 1)

                    # Option to enter section number directly
                    with col2:
                        section_input = st.number_input(
                            "Go to section number",
                            min_value=1,
                            max_value=num_sections,
                            value=st.session_state.section_idx + 1,
                            step=1,
                            key="section_number_input"
                        )
                        if section_input != st.session_state.section_idx + 1:
                            st.session_state.section_idx = int(section_input) - 1

                    section = data[st.session_state.section_idx]
                    st.subheader(f"Section {st.session_state.section_idx + 1} of {num_sections} (page {section.get('page', 'Unknown')})")
                    st.header(section.get('heading', '') + f" ({section.get('type', 'Unknown')})")
                    st.markdown(section.get('text', ''))
                    # Display the tag below the section in green
                    st.markdown(
                        f"Tag: <span style='color: lightgreen;'>{section.get('book_tag', 'Unknown')}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("### Grammar Tags")
                    grammar_tags = section.get('grammar_tags', [])
                    if grammar_tags:
                        st.markdown(", ".join(f"<span style='color: lightblue;'>{tag}</span>" for tag in grammar_tags), unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: red;'>No grammar tags found</span>", unsafe_allow_html=True)
                    st.markdown("---")

                    # Display the rule extracted by GPT if available
                    if gpt_extracted_rules:
                        #print(len(gpt_extracted_rules))
                        #print(len(data))
                        index = filtered_mapping.get(str(st.session_state.section_idx), None)
                        print("Section index:", st.session_state.section_idx)
                        if index is not None:
                            print(f"Index found for section_idx: {st.session_state.section_idx} in filtered mapping:", index)
                            gpt_response_section_paragraphed = gpt_extracted_rules[index] if index < len(gpt_extracted_rules) else None
                            #print(gpt_response_section)
                            # Combine the list of lists into one list
                            #print(gpt_response_section_paragraphed)
                            #print(index)
                            if isinstance(gpt_response_section_paragraphed, list):
                                gpt_response_section = [
                                    item
                                    for sublist in gpt_response_section_paragraphed if isinstance(sublist, list)
                                    for item in sublist
                                ]
                            
                            if gpt_response_section:
                                st.markdown("### GPT Extracted Rules")

                                for idx, gpt_item in enumerate(gpt_response_section):
                                    st.markdown(f"**{idx + 1}. Sentence Index:** {gpt_item.get('sentence_index', 'Unknown')}")
                                    #st.markdown(f"**Sentence:** {gpt_item.get('sentence', 'No sentence provided')}")
                                    st.markdown(f"**Description:** {gpt_item.get('description', 'No description')}")
                                    st.markdown(f"**Condition:** {gpt_item.get('condition', 'None')}")
                                    st.markdown(f"**Action:** {gpt_item.get('action', 'None')}")
                                    st.markdown(f"**FST Rule:** {gpt_item.get('fst_rule', 'None')}")
                                    st.markdown(f"**Usage:** {gpt_item.get('usage', 'None')}")
                                    st.markdown(f"**Unimorph:** {gpt_item.get('unimorph', 'None')}")
                                    st.markdown("---")
                            else:
                                st.warning("No GPT extracted rule found for this section.")
                        

                    
                    # Display index of headers and sections
                    st.markdown("### Index")
                    index_lines = []
                    # Split the index into 3 columns for better readability
                    num_columns = 3
                    index_lines = [[] for _ in range(num_columns)]
                    for idx, section in enumerate(data):
                        heading = section.get('heading', '')
                        section_type = section.get('type', '')
                        page = section.get('page', 'Unknown')
                        if section_type.lower() == "header":
                            line = f"**{idx + 1}. {heading}** &nbsp;&nbsp; _(Page {page})_"
                        elif section_type.lower() == "subheader":
                            line = f"&emsp;{idx + 1}. *{heading}* _(Page {page})_"
                        else:
                            line = f"&emsp;&emsp;{idx + 1}. {heading} _(Page {page})_"
                        index_lines[idx % num_columns].append(line)

                    cols = st.columns(num_columns)
                    for col, lines in zip(cols, index_lines):
                        with col:
                            st.markdown("\n\n".join(lines), unsafe_allow_html=True)

            elif page == "Parallel Sentences":
                st.header("Parallel Sentences")
                st.markdown("Number of parallel sentences: " + str(len(parallel_sentences) if parallel_sentences else 0))
                if parallel_sentences:
                    col1, col2,col3 = st.columns(3)
                    with col1:
                        min_words = st.number_input(
                            "Minimum number of words in transliteration",
                            min_value=1,
                            max_value=100,
                            value=1,
                            step=1,
                            key="min_words_input"
                        )
                    with col2:
                        max_words = st.number_input(
                            "Maximum number of words in transliteration",
                            min_value=min_words,
                            max_value=100,
                            value=10,
                            step=1,
                            key="max_words_input"
                        )
                    with col3:
                        add_info_checkbox = st.checkbox(
                            "Add POS and Affix Info",
                            value=True,
                            key="add_info_checkbox"
                        )

                    filtered_sentences = [
                        row for row in parallel_sentences
                        if min_words <= len(row.get('sentence', '').split()) <= max_words
                    ]

                    st.markdown(f"Filtered sentences: {len(filtered_sentences)}")

                    for idx, data_row in enumerate(filtered_sentences, 1):
                        transliteration = data_row.get('sentence', '')
                        translation = data_row.get('translated_sentence', '')
                        

                        st.markdown(f"**{idx}.** {transliteration} --- {translation}")
                        #additional_info = data_row.get('pos_unimorph_info', None)
                        if add_info_checkbox:
                            spacy_info = data_row.get('spacy_info', None)
                            if spacy_info:
                                print(spacy_info)
                                tense = spacy_info.get('tense', None)
                                number = spacy_info.get('number', None)
                                plural_details = spacy_info.get('plural_details', None)
                                genders = spacy_info.get('genders', None)
                                genders_details = spacy_info.get('genders_details', None)
                                details = []
                                if genders:
                                    details.append(f"Genders: {set(genders)}")
                                if genders_details:
                                    details.append(f"Genders_details: {genders_details}")
                                if plural_details:
                                    details.append(f"Plural_details: {plural_details}")
                                details_str = ", ".join(details)
                                st.markdown(
                                    f"**{tense}**, **{number}**" + (", " + details_str if details_str else ""),
                                    unsafe_allow_html=True
                                )
                            
                                

                        
                else:
                    st.info("No parallel sentences found.")
            elif page == "Dictionary":
                st.header("Dictionary")
                # Try to load Kannada-English dictionary from file if present
                kan_eng_dict_filename = Path("kan_to_eng_dict.json")
                dictionary = []

                if os.path.exists(kan_eng_dict_filename):
                    with open(kan_eng_dict_filename, "r", encoding="utf-8") as f:
                        dictionary = json.load(f)

                search_query = st.text_input("Search English or Kannada word")
                filtered_dict = [
                    {"Kannada": k, "English": v, "POS": ""}
                    for k, v in dictionary.items()
                    if search_query.lower() in k.lower() or search_query.lower() in v.lower()
                ] if search_query else [
                    {"Kannada": k, "English": v, "POS": ""} for k, v in dictionary.items()
                ]

                st.write(f"Total entries: {len(filtered_dict)}")
                if filtered_dict:
                    st.dataframe(
                        filtered_dict,
                        use_container_width=True
                    )
                else:
                    st.info("No dictionary entries found for your search.")

        
    else:
        st.error(f"JSON file '{json_filename}' not found in the app directory.")
else:
    st.info("Please upload a PDF file to view its extracted sections.")