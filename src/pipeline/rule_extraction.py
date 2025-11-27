import json
from sentence_transformers import SentenceTransformer, util
import torch  # for checking cuda availability
from pathlib import Path
from tqdm import tqdm
import typer
import spacy
from openai import OpenAI
import yaml
from time import sleep


app = typer.Typer(
    name="rule_extraction",
    help="Extract grammatical rules from tagged sections",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def load_nlp():
    try:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        # fallback
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

    # compute desired splits from length
    n_words = len(text.split())
    n_splits = max(0, n_words // target_words)
    if n_splits == 0:
        return [text]

    cand = sorted(range(len(sims)), key=lambda i: sims[i])  # lowest first
    picks = []
    for i in cand:
        # ensure we don't pick boundaries too close to each other
        if all(abs(i - j) >= min_gap for j in picks):
            picks.append(i + 1)  # boundary after sentence i
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


def filter_sections(data, tags_to_filter, nlp):
    kept, mapping = [], {}
    for idx, sec in enumerate(data):
        tags = sec.get("sorted_tags", [])
        hits = [t for t, s in tags if t in tags_to_filter and s >= tags_to_filter[t]]
        if hits or not any(t not in tags_to_filter for t, _ in tags):
            sec["filtered"] = {
                "status": True,
                "reason": f"hits={hits}" if hits else "only filtered tags",
            }
            continue
        sec["filtered"] = {"status": False, "reason": None}
        sec["sentences"] = sentences(nlp, sec.get("text", ""))
        mapping[idx] = len(kept)
        kept.append(sec)
    return kept, mapping


def call_llm(client, model, prompt, paragraph, retries=3):
    for a in range(retries):
        try:
            r = client.responses.create(
                model=model, input=prompt.format(input_paragraph=paragraph)
            )
            return getattr(r, "output_text", None) or r.output[0].content[0].text
        except Exception:
            if a == retries - 1:
                raise
            sleep(2 * (a + 1))


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


@app.command()
def process_file(
    filename: str = typer.Argument(None, help="Path to the input 'tagged' JSON file"),
    target_words: int = typer.Option(
        200, help="Target number of words per split section"
    ),
    min_gap: int = typer.Option(1, help="Minimum gap between splits (in sentences)"),
):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    print(len(data))

    # Get language name from filename
    language_name = Path(filename).stem.split("_")[0]

    nlp = load_nlp()

    model = load_embedder()

    ###
    sentences_data = []
    filtered_out_data = []

    # Tags to filter out with the threshold
    tags_to_filter = {
        "Publishing information": 1.0,
        "Structure": 2.0,
        "Culture": 2.0,
        "History": 2.0,
        "Phonetics": 2.0,
    }

    filtered_idx = 0
    section_to_filtered_mapping = {}

    for idx, section in enumerate(data):
        section_id = section.get("id", None)
        sorted_tags = section["sorted_tags"]
        sorted_tag_names = [tag[0] for tag in sorted_tags]
        tag_word_counts = section["tag_word_counts"]
        tag_counts_heading = section["tag_counts_heading"]

        # Track filtering reason
        filter_reason = None

        # Filter out sections with high scores in filtered tags
        if any(
            tag[0] in tags_to_filter and tag[1] >= tags_to_filter[tag[0]]
            for tag in sorted_tags
        ):
            filter_reason = f"High score in filtered tag: {[tag[0] for tag in sorted_tags if tag[0] in tags_to_filter and tag[1] >= tags_to_filter[tag[0]]]}"
            section["filtered"] = {"status": True, "reason": filter_reason}
            filtered_out_data.append(section)
            continue

        # Check if there are any other tags that are not in tags_to_filter
        other_tags = [tag for tag in sorted_tag_names if tag not in tags_to_filter]
        if not other_tags:
            filter_reason = "No tags other than filtered tags"
            section["filtered"] = {"status": True, "reason": filter_reason}
            filtered_out_data.append(section)
            continue

        text = section["text"]

        # Use spacy to split the text into sentences
        sentences = [sent.text.strip() for sent in nlp(text).sents]

        # Create sentence data structure with filtering info
        section_data = {
            "section_id": section_id,
            "text": text,
            "sentences": sentences,
            "sorted_tags": sorted_tags,
            "tagged_words": tag_word_counts,
            "tagged_words_heading": tag_counts_heading,
            "filtered": {"status": False, "reason": None},
        }

        sentences_data.append(section_data)
        section_to_filtered_mapping[idx] = filtered_idx
        filtered_idx += 1

        section["filtered"] = {"status": False, "reason": None}
        section["sentences"] = sentences

    print(f"Total sections: {len(data)}")
    print(f"Filtered out sections: {len(filtered_out_data)}")
    print(f"Remaining sections: {len(sentences_data)}")

    # Save the filtered mapping to a JSON file
    Path("scratch").mkdir(exist_ok=True)
    output_file = Path(f"scratch/{Path(filename).stem}_filtered_mapping.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(section_to_filtered_mapping, f, ensure_ascii=False, indent=4)
    print(f"Filtered mapping saved to {output_file}")

    all_sections_filtered = []
    for section in data:
        text = section.get("text", None)
        section_id = section.get("id", None)
        if not text or section_id is None:
            continue
        filtered = section.get("filtered", None)
        if filtered["status"] == True:
            continue
        all_sections_filtered.append({"section_id": section_id, "text": text})

    print(f"Total sections after filtering: {len(all_sections_filtered)}")

    # Split sections into smaller chunks based on similarity
    sections_split = []
    for section in all_sections_filtered:
        text = section.get("text", "")
        section_id = section.get("section_id", None)
        if not text or section_id is None:
            continue
        paras = split_by_similarity(text, nlp, model, target_words, min_gap)
        sections_split.append({"section_id": section_id, "paragraphs": paras})
    print(f"Total sections after splitting: {len(sections_split)}")

    ####################################################################################
    # Use gpt to extract rules from each paragraph using the prompt

    base_prompt = f"""You are an expert in computational linguistics.

    You will be given a paragraph from a descriptive grammar of the language: {{language_name}}.
    Your task is to extract every grammatical rule explicitly stated or clearly implied by examples in that paragraph.
    Only extract what the paragraph supports, do not invent or generalize beyond it.
    If no rule is described, output a single block stating that.

    Each rule must be expressed clearly and concisely in structured form.
    Output YAML only (no prose, no code fences). 
    One YAML block per rule, separated by `---`.

    For each rule, include the following fields:

    - description: A plain-language summary of the rule
    - condition: When or where the rule applies (context, environment, or restriction)
    - action: What the rule does (the grammatical or morphological change, or syntactic effect)
    - confidence: One of [High, Medium, Low] depending on how explicitly the rule is stated
    - category: One of [Morphology, Syntax, Phonology, Semantics, Pragmatics, Orthography, Other]
    - linguistic_level: word | phrase | clause | discourse
    - usage: One of [attested, prescriptive, archaic, dialectal, colloquial, formal]
    - unimorph: Optional UniMorph feature string (e.g., "V;PST;3;SG") or empty if not relevant
    - grammar_tags: [list of tags like "Plural", "Past Tense", "Agreement", "Negation"]
    - notes: Optional clarifications useful for later application

    If no grammatical rules are found, output:

    description: "No grammatical rules are described in the paragraph."
    condition: ""
    action: ""
    examples: []
    confidence: ""
    category: ""
    linguistic_level: ""
    usage: ""
    unimorph: ""
    grammar_tags: []
    notes: ""

    Paragraph: {{input_paragraph}}

    """

    with open("openai_api_key.txt", "r") as f:
        openai_api_key = f.read().strip()

    client = OpenAI(api_key=openai_api_key)

    gpt_extracted_rules_direct = []
    rule_section_map = []
    # Set limit for API calls
    LIMIT = -1
    # Iterate through the sections
    print("Total sections to process:", len(sections_split))
    api_model = "gpt-4o-mini"
    dry_run = False  # Set to True to skip actual API calls
    for section in tqdm(sections_split):
        section_id = section.get("section_id", None)
        paragraphs = section.get("paragraphs", None)
        print(
            f"Processing section ID: {section_id} with {len(paragraphs) if paragraphs else 0} paragraphs"
        )
        if not paragraphs or section_id is None:
            continue

        if LIMIT == 0:
            print("API call limit reached, stopping further requests.")
            break

        LIMIT -= 1
        # Process each section
        temp = []
        map_temp = []

        for p_idx, paragraph in enumerate(paragraphs):
            # paragraph["paragraph_id"] = p_idx
            if dry_run:
                temp.append("dry_run_response")
                continue
            response = client.responses.create(
                model=api_model,
                input=base_prompt.format(
                    input_paragraph=paragraph, language_name=language_name
                ),
            )

            temp.append(response.output[0].content[0].text)
            map_temp.append(p_idx)
        gpt_extracted_rules_direct.append(temp)
        rule_section_map.append({"section_id": section_id, "paragraph_map": map_temp})

    # Store the filtered sections as well for reference
    with open(
        f"extracted_rules/{Path(filename).stem}_split_sections.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(sections_split, f, ensure_ascii=False, indent=4)
    print(
        f"Saved split sections to extracted_rules/{Path(filename).stem}_split_sections.json"
    )
    if dry_run:
        print("Dry run complete, no parsing performed.")
        return
    with open(
        f"scratch/{Path(filename).stem}_gpt_extracted_rules_direct.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(gpt_extracted_rules_direct, f, ensure_ascii=False, indent=4)
    print(
        f"Saved raw GPT responses to scratch/{Path(filename).stem}_gpt_extracted_rules_direct.json"
    )

    with open(
        f"scratch/{Path(filename).stem}_rule_section_map.json", "w", encoding="utf-8"
    ) as f:
        json.dump(rule_section_map, f, ensure_ascii=False, indent=4)
    print(
        f"Saved rule-section map to scratch/{Path(filename).stem}_rule_section_map.json"
    )

    for i, section in enumerate(gpt_extracted_rules_direct):
        if section is None:
            continue
        row = rule_section_map[i]
        section_id = row["section_id"]
        paragraph_map = row["paragraph_map"]

        for j, response in enumerate(section):
            if response is None:
                continue

            # print(response)

            # Remove code fences and trim whitespace
            cleaned_response = (
                response.strip().replace("```yaml", "").replace("```", "")
            )

            # Try to parse all YAML documents in the string
            try:
                docs = list(yaml.safe_load_all(cleaned_response))
            except yaml.YAMLError as e:
                print(f"Error parsing YAML for section {i}, paragraph {j}: {e}")
                print("Offending text:\n", cleaned_response[:500], "...\n")
                docs = []

            # Normalize the output: ensure it's always a list of dicts
            parsed_list = []
            for rule_dict in docs:
                if isinstance(rule_dict, dict):
                    rule_dict["section_id"] = section_id
                    rule_dict["paragraph_id"] = paragraph_map[j]
                    parsed_list.append(rule_dict)
                elif isinstance(rule_dict, list):
                    # handle the case where the model returned a list of dicts
                    for d in rule_dict:
                        if isinstance(d, dict):
                            d["section_id"] = section_id
                            d["paragraph_id"] = paragraph_map[j]
                            parsed_list.append(d)

            gpt_extracted_rules_direct[i][j] = parsed_list

    # Store the responses in a JSON file
    Path("extracted_rules").mkdir(exist_ok=True)
    with open(
        f"extracted_rules/{Path(filename).stem}_gpt_extracted_rules_direct_parsed.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(gpt_extracted_rules_direct, f, ensure_ascii=False, indent=4)

    print(
        "Extraction complete. Results saved to",
        f"extracted_rules/{Path(filename).stem}_gpt_extracted_rules_direct_parsed.json",
    )


if __name__ == "__main__":
    app()
