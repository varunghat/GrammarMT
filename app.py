import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Thesis viewer")

uploaded_pdf = True

if uploaded_pdf:
    pdf_filename = Path("C:/Users/Varun/master_thesis/manual_on_modern_kannada.pdf")
    json_filename = Path("C:/Users/Varun/master_thesis/manual_on_modern_kannada.json")
    parallel_sents_filename = Path("C:/Users/Varun/master_thesis/parallel_sentences_cleaned.json")

    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Navigation bar using Streamlit's radio in the sidebar
            st.sidebar.title("Navigation")
            page = st.sidebar.radio("Go to", ["Sections", "Parallel Sentences"])

            parallel_sentences = None
            if os.path.exists(parallel_sents_filename):
                with open(parallel_sents_filename, "r", encoding="utf-8") as f:
                    parallel_sentences = json.load(f)

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


                # Prepare data for page-wise display
                if display_mode == "Sections by page":
                    # Group sections by page
                    from collections import defaultdict
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
                    col1, col2 = st.columns(2)
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

                    filtered_sentences = [
                        row for row in parallel_sentences
                        if min_words <= len(row.get('transliteration', '').split()) <= max_words
                    ]

                    st.markdown(f"Filtered sentences: {len(filtered_sentences)}")

                    for idx, data_row in enumerate(filtered_sentences, 1):
                        transliteration = data_row.get('transliteration', '')
                        translation = data_row.get('english', '')

                        st.markdown(f"**{idx}.** {transliteration} --- {translation}")
                else:
                    st.info("No parallel sentences found.")


        
    else:
        st.error(f"JSON file '{json_filename}' not found in the app directory.")
else:
    st.info("Please upload a PDF file to view its extracted sections.")