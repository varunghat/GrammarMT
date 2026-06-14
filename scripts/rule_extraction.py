import json
import os
import time
import yaml
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path
from tqdm import tqdm
import typer
import spacy


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val


app = typer.Typer(
    name="rule_extraction",
    help="Extract grammatical rules from tagged sections using Gemini batch API",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def load_nlp():
    try:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return spacy.load(
            "en_core_web_sm", disable=["tagger", "ner", "lemmatizer", "attribute_ruler"]
        )


def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def sentences(nlp, text):
    return [s.text.strip() for s in nlp(text).sents] if text else []


def split_by_similarity(text, nlp, model, target_words=200, min_gap=1):
    sents = sentences(nlp, text)
    if len(sents) <= 1:
        return [text]

    emb = model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    sims = [util.cos_sim(emb[i], emb[i + 1]).item() for i in range(len(sents) - 1)]

    n_words = len(text.split())
    n_splits = max(0, n_words // target_words)
    if n_splits == 0:
        return [text]

    cand = sorted(range(len(sims)), key=lambda i: sims[i])
    picks = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in picks):
            picks.append(i + 1)
            if len(picks) == n_splits:
                break
    picks = sorted(picks)

    paras, start = [], 0
    for b in picks:
        paras.append(" ".join(sents[start:b]))
        start = b
    if start < len(sents):
        paras.append(" ".join(sents[start:]))

    return [p for p in paras if p.strip()]


def parse_yaml_blocks(s):
    s = (s or "").strip().replace("```yaml", "").replace("```", "")
    try:
        docs = list(yaml.safe_load_all(s))
    except yaml.YAMLError:
        return []
    out = []
    for d in docs:
        if d is None:
            continue
        if isinstance(d, dict):
            out.append(d)
        elif isinstance(d, list):
            out.extend([x for x in d if isinstance(x, dict)])
    return out


BASE_PROMPT = """
You are an expert descriptive linguist and data structurer.

Your core task is to **extract morphological and syntactic rules** from a grammar book paragraph for a low-resource language.
These rules will be used by a downstream AI agent to synthesize valid sentences.

## Input Context
You will receive a paragraph describing a specific grammatical phenomenon.

## Selection Criteria
Extract a rule IF AND ONLY IF:
1. It involves an **overt surface change** (a suffix, prefix, clitic, particle, or mutation).
2. It is **productive** (applies generally to a class of words, not just one specific exception).
3. It has a clear **grammatical function** (Case, Tense, Mood, Aspect, Agreement).

Do NOT extract:
- Purely phonological rules (e.g., "vowels lengthen before nasals").
- Typological trivia (e.g., "The language has SVO order" - unless it describes a specific marker).
- Examples or footnotes.

## Output Schema (YAML)
For each rule found, output a YAML block. The YAML must strictly follow this structure:

category: surface_rule
description: <ACTIONABLE summary: State the POS, morpheme, and grammatical function, e.g., 'Applies -an to NOUN to mark 1SG.POSS'.>
target_pos: <The specific Part-of-Speech this attaches to: NOUN, VERB, ADJ, PRON>
affix_type: <Choose one: SUFFIX, PREFIX, CLITIC, PARTICLE, CIRCUMFIX, REDUPLICATION>
morpheme: <The actual string literal, e.g., "-at", "ko=", "X~X">
application_string: <Formulaic application of the morpheme. Use 'STEM' for the base word. E.g., 'STEM + "-an"' for a suffix, or '"un-" + STEM' for a prefix.>
unimorph_feature: <The general feature type, e.g., "CASE", "TENSE", "AGR">
unimorph_value: <The specific value for the feature, e.g., "ACC", "PST", "3SG">
context_dependency: <Briefly describe the environment or agreement required, e.g., "Must agree with a 3SG subject", "Applies to consonant-final stems only". Output "N/A" if no specific dependency is mentioned.>
semantic_trigger: <A concise instruction on WHEN to apply this. E.g., "Use when the noun is the direct object of the sentence.">

---

## Output Handling (CRITICAL)

### Rule Found
If one or more rules are found, output the corresponding YAML blocks as defined above.

### No Rule Found
If the paragraph contains **NO** information that meets the **Selection Criteria**, you MUST output a single, fixed YAML block with the following structure:
category: N/A
description: "No applicable rule found."
DO NOT output an empty YAML block or any other format.

## Vocabulary Guidelines

- **unimorph_feature**: Use general tags like **CASE**, **TENSE**, **AGR** (Agreement), **NUM** (Number).
- **unimorph_value**: Use standard tags like **ACC** (Accusative), **GEN** (Genitive), **PST** (Past), **PL** (Plural), **3SG** (3rd Singular).
- **semantic_trigger**: Write this for a non-linguist AI. Focus on the mapping from English meaning to LRL form.

## Example

Input: "The clitic =at marks the object of a transitive verb on nouns."

Output:
category: surface_rule
description: "Applies the clitic =at to NOUN to mark the Accusative Case (ACC)."
target_pos: NOUN
affix_type: CLITIC
morpheme: "=at"
application_string: "STEM + '=at'"
unimorph_feature: CASE
unimorph_value: ACC
context_dependency: "N/A"
semantic_trigger: "Use when the noun is the direct object of the sentence."

## Task
Process the following paragraph:
{input_paragraph}
"""

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@app.command()
def process_file(
    filename: str = typer.Argument(None, help="Path to the input tagged JSON file"),
    target_words: int = typer.Option(500, help="Target words per split paragraph"),
    min_gap: int = typer.Option(1, help="Minimum sentence gap between splits"),
    dry_run: bool = typer.Option(False, help="Build batch file only, skip API submission"),
):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    language_name = Path(filename).stem.split("_")[0]
    print(f"Language: {language_name}, sections loaded: {len(data)}")

    # Filter out sections already marked filtered by section_tagger
    all_sections_filtered = []
    for section in data:
        text = section.get("text", None)
        section_id = section.get("id", None)
        if not text or section_id is None:
            continue
        if section.get("filtered") == True:
            continue
        all_sections_filtered.append(section)
    print(f"Sections after filtering: {len(all_sections_filtered)}")

    nlp = load_nlp()
    embed_model = load_embedder()

    # Split sections into smaller paragraphs by semantic similarity
    sections_split = []
    for section in tqdm(all_sections_filtered, desc="Splitting sections"):
        text = section.get("text", "")
        heading = section.get("heading", "")
        heading_level = section.get("heading_level", -1)
        section_id = section.get("id", None)
        if not text or section_id is None:
            continue
        paras = split_by_similarity(text, nlp, embed_model, target_words, min_gap)
        sections_split.append({
            "section_id": section_id,
            "heading": heading,
            "heading_level": heading_level,
            "paragraphs": paras,
        })

    print(f"Sections after splitting: {len(sections_split)}")
    total_paragraphs = sum(len(s["paragraphs"]) for s in sections_split)
    print(f"Total paragraphs: {total_paragraphs}")

    Path("data/sections_split").mkdir(parents=True, exist_ok=True)
    split_output = Path(f"data/sections_split/{language_name}_sections_classified_split.json")
    with open(split_output, "w", encoding="utf-8") as f:
        json.dump(sections_split, f, ensure_ascii=False, indent=4)
    print(f"Split sections saved to {split_output}")

    # Build Gemini batch requests
    batch_prompts = []
    for section in sections_split:
        section_id = section["section_id"]
        for p_idx, paragraph in enumerate(section["paragraphs"]):
            prompt = BASE_PROMPT.format(input_paragraph=paragraph)
            batch_prompts.append((section_id, p_idx, prompt))

    Path("scratch").mkdir(exist_ok=True)
    batch_jsonl_path = f"scratch/{language_name}_gemini_batch_requests.jsonl"
    batch_file = []
    for section_id, p_idx, prompt in batch_prompts:
        unique_id = f"{section_id}_{p_idx}"
        batch_file.append({
            "key": unique_id,
            "request": {
                "contents": [{"parts": [{"text": prompt}]}]
            },
        })

    with open(batch_jsonl_path, "w", encoding="utf-8") as f:
        for item in batch_file:
            f.write(json.dumps(item) + "\n")
    print(f"Batch file written: {batch_jsonl_path} ({len(batch_file)} requests)")

    if dry_run:
        print("Dry run — skipping Gemini batch submission.")
        return

    client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))

    # Upload and submit batch job
    uploaded_file = client.files.upload(
        file=batch_jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"{language_name}_gemini_batch_requests",
            mime_type="jsonl",
        ),
    )
    print(f"Uploaded file: {uploaded_file.name}")

    batch_job = client.batches.create(
        model="gemini-2.5-flash",
        src=uploaded_file.name,
        config={"display_name": f"{language_name}_rule_extraction_batch_job"},
    )
    print(f"Batch job created: {batch_job.name}")

    # Poll until done
    batch_job = client.batches.get(name=batch_job.name)
    while batch_job.state.name not in COMPLETED_STATES:
        print(f"Status: {batch_job.state.name} — waiting 5 min...")
        time.sleep(300)
        batch_job = client.batches.get(name=batch_job.name)
    print(f"Job finished: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Error: {batch_job.error}")
        raise typer.Exit(code=1)

    # Download results
    result_file_name = batch_job.dest.file_name
    file_content = client.files.download(file=result_file_name)
    raw_results_path = f"scratch/{language_name}_gemini_batch_results.txt"
    with open(raw_results_path, "wb") as f:
        f.write(file_content)
    print(f"Raw results saved to {raw_results_path}")

    # Parse results
    gemini_extracted_rules = []
    for res_line in file_content.decode("utf-8").splitlines():
        res = json.loads(res_line)
        key = res["key"]
        section_id, p_idx = key.split("_")
        text = res["response"]["candidates"][0]["content"]["parts"][0]["text"]
        gemini_extracted_rules.append({
            "section_id": section_id,
            "paragraph_index": int(p_idx),
            "extracted_rules": text,
        })

    # Parse YAML from each response and filter no-rule outputs
    final_extracted_rules = []
    for item in gemini_extracted_rules:
        response = item["extracted_rules"]
        section_id = item["section_id"]
        paragraph_index = item["paragraph_index"]

        cleaned = response.strip().replace("```yaml", "").replace("```", "")
        try:
            docs = list(yaml.safe_load_all(cleaned))
        except yaml.YAMLError as e:
            print(f"YAML parse error for section {section_id}, para {paragraph_index}: {e}")
            docs = []

        parsed_list = []
        for rule_dict in docs:
            if isinstance(rule_dict, dict):
                rule_dict["section_id"] = section_id
                rule_dict["paragraph_id"] = paragraph_index
                parsed_list.append(rule_dict)
            elif isinstance(rule_dict, list):
                for d in rule_dict:
                    if isinstance(d, dict):
                        d["section_id"] = section_id
                        d["paragraph_id"] = paragraph_index
                        parsed_list.append(d)
        final_extracted_rules.append(parsed_list)

    # Remove no-rule responses
    cleaned_rules = []
    for section in final_extracted_rules:
        valid_rules = []
        for rule in section:
            description = rule.get("description", "").lower()
            if (
                "no applicable rule found" in description
                or "no valid rules" in description
                or not description.strip()
            ):
                continue
            valid_rules.append(rule)
        if valid_rules:
            cleaned_rules.append(valid_rules)

    print(f"Extracted rule sets after cleaning: {len(cleaned_rules)}")

    Path("data/extracted_rules").mkdir(parents=True, exist_ok=True)
    output_path = f"data/extracted_rules/{language_name}_extracted_rules.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_rules, f, ensure_ascii=False, indent=4)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    app()
