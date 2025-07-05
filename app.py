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

    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        st.header(f"Extracted Sections from {pdf_filename}")
        num_sections = len(data)
        st.write(f"Total sections extracted: {num_sections}")

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

            section = data[st.session_state.section_idx]
            st.subheader(f"Section {st.session_state.section_idx + 1} of {num_sections} (page {section.get('page', 'Unknown')})")
            st.header(section.get('heading', '') + f" ({section.get('type', 'Unknown')})")
            st.markdown(section.get('text', ''))

    else:
        st.error(f"JSON file '{json_filename}' not found in the app directory.")
else:
    st.info("Please upload a PDF file to view its extracted sections.")