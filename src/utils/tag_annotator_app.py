import streamlit as st
import json
import os

DATA_PATH = "manual_on_modern_kannada.json"
SAVE_PATH = "manual_on_modern_kannada_tagged.json"

# Load existing or fresh
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)
else:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)
    # Add empty tag fields
    for sec in sections:
        sec["book_tag"] = None
        sec["grammar_tags"] = []

TOPIC_OPTIONS = [
    "Grammar", "Structure", "Publishing",
    "History", "Culture", "Exercises", "Misc"
]

GRAMMAR_TAGS = [
    "Noun", "Verb", "Adjective", "Adverb", "Pronoun",
    "Preposition", "Postposition", "Conjunction", "Interjection",
    "Past", "Present", "Future",
    "Singular", "Plural",
    "1st Person", "2nd Person", "3rd Person",
    "Gender",
    "Nominative", "Accusative", "Dative", "Genitive", "Locative", "Instrumental"
]

st.title("Grammar Book Section Tagger")

# Sidebar for navigation
idx = st.sidebar.number_input("Section Index", min_value=0, max_value=len(sections)-1, value=0, step=1)

section = sections[idx]
st.markdown(f"### Section {idx}")
st.markdown(f"**Header**: {section.get('header', '')}")
st.markdown("**Section Text:**")
st.markdown(section.get("text", ""), unsafe_allow_html=False)


st.markdown("### 📘 Book Tag")

# Display buttons horizontally
cols = st.columns(len(TOPIC_OPTIONS))
book_tag = section.get("book_tag", None)

for i, option in enumerate(TOPIC_OPTIONS):
    if cols[i].button(option, use_container_width=True):
        section["book_tag"] = option
        #st.experimental_rerun()  # refresh to update state

st.markdown("### ✏️ Grammar Tags")
selected_grammar_tags = section.get("grammar_tags", [])

cols = st.columns(4)  # 4 buttons per row
for i, tag in enumerate(GRAMMAR_TAGS):
    col = cols[i % 4]
    is_selected = tag in selected_grammar_tags
    button_label = f"✅ {tag}" if is_selected else tag
    if col.button(button_label, key=f"gram_{i}"):
        if is_selected:
            selected_grammar_tags.remove(tag)
        else:
            selected_grammar_tags.append(tag)
        section["grammar_tags"] = selected_grammar_tags
        #st.experimental_rerun()

# Save
if st.button("💾 Save Tag"):
    section["book_tag"] = book_tag
    section["grammar_tags"] = selected_grammar_tags
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    st.success(f"Saved section {idx}.")

# Progress
tagged = sum(1 for s in sections if s["book_tag"])
st.sidebar.markdown(f"✅ Tagged: {tagged} / {len(sections)}")

