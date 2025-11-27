import streamlit as st
import fitz
import pandas as pd
import json
from io import BytesIO

##############################################
# SETTINGS
##############################################

PDF_PATH = "grammar_books/sursilvan_romansh.pdf"  # change to Romansh book
PREDICTIONS_CSV = "sursilvan_romansh_line_roles_predictions.csv"
OUTPUT_JSON = "sursilvan_romansh_corrected_labels.json"

CONFIDENCE_THRESHOLD = 0.60  # show only low-confidence lines


##############################################
# LOAD DATA
##############################################


@st.cache_resource
def load_pdf(path):
    return fitz.open(path)


@st.cache_data
def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    return df


doc = load_pdf(PDF_PATH)
df = load_predictions(PREDICTIONS_CSV)

# Filter low-confidence examples
df_low = df[df["prediction_confidence"] < CONFIDENCE_THRESHOLD].copy()

if "annotations" not in st.session_state:
    st.session_state.annotations = {}

##############################################
# UI TITLE
##############################################

st.title("📘 Active Learning Annotation Tool")
st.write("Review low-confidence lines and assign correct roles.")

##############################################
# SELECT SAMPLE
##############################################

sample_ids = list(df_low.index)
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0


def next_sample():
    st.session_state.sample_idx = min(
        st.session_state.sample_idx + 1, len(sample_ids) - 1
    )


def prev_sample():
    st.session_state.sample_idx = max(st.session_state.sample_idx - 1, 0)


st.write("### Navigate samples")
col1, col2 = st.columns(2)
with col1:
    st.button("⬅️ Previous", on_click=prev_sample)
with col2:
    st.button("➡️ Next", on_click=next_sample)

sample = df_low.iloc[st.session_state.sample_idx]
st.write(f"### Sample {st.session_state.sample_idx+1} / {len(sample_ids)}")


##############################################
# DISPLAY PAGE PREVIEW WITH HIGHLIGHT
##############################################


def render_page_with_highlight(doc, page_num, bbox):
    page = doc[page_num]
    rect = fitz.Rect(*bbox)

    # draw highlight on copy of page
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = BytesIO(pix.tobytes("png"))

    # Could later add highlight overlay using PyMuPDF draw
    return img


page_number = int(sample.page) - 1  # zero-based
bbox = (sample.x0, sample.y0, sample.x1, sample.y1)

st.write(f"### Page {page_number+1}")


##############################################
# SHOW LINE + MODEL INFO
##############################################

st.write("### Line Information")

st.markdown(
    f"""
**Text:**  
`{sample.text}`

**Model Prediction:** `{sample.predicted_role}`  
**Confidence:** `{sample.prediction_confidence:.3f}`
"""
)


##############################################
# ROLE CORRECTION
##############################################

roles = ["SOURCE", "GLOSS", "TRANSLATION", "OTHER"]

selected_role = st.radio(
    "Correct Role:",
    roles,
    index=roles.index(sample.predicted_role) if sample.predicted_role in roles else 3,
)

##############################################
# SAVE ANNOTATION
##############################################

if st.button("💾 Save Annotation"):
    st.session_state.annotations[int(sample.name)] = {
        "page": int(sample.page),
        "line_no": int(sample.line_no),
        "text": sample.text,
        "predicted_role": sample.predicted_role,
        "correct_role": selected_role,
    }
    st.success("Saved!")

### RENDER

img_data = render_page_with_highlight(doc, page_number, bbox)
st.image(img_data, caption="Page Preview")


##############################################
# EXPORT ALL ANNOTATIONS
##############################################

if st.button("📤 Export JSON"):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(st.session_state.annotations, f, indent=2, ensure_ascii=False)
    st.success(f"Saved to {OUTPUT_JSON}")
