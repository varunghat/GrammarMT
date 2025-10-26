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
        section_id = section.get("id",None)
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
            "id": section_id,
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

    all_sections = []
    for section in data:
        text = section.get("text", "")
        if not text:
            continue
        filtered = section.get("filtered", None)
        if filtered["status"] == True:
            continue
        all_sections.append(text)

    print(f"Total sections after filtering: {len(all_sections)}")

    # Split sections into smaller chunks based on similarity
    sections_split = []
    for section in all_sections:
        paras = split_by_similarity(section, nlp, model, target_words, min_gap)
        sections_split.append(paras)
    print(f"Total sections after splitting: {len(sections_split)}")

    ####################################################################################
    # Use gpt to extract rules from each paragraph using the prompt

    base_prompt = f"""You are an expert in computational linguistics.

    You will be given a paragraph from a descriptive grammar of the Kannada language.

    Your task is to extract all explicit or implicit grammatical rules described in the paragraph. Each rule should be expressed in both plain language and formal terms.
    If the rule cannot be expressed in a formal rewrite rule, leave the fst_rule field empty. If the paragraph does not describe any rules, return a description stating that no rules are described.
    Do not make up rules or provide explanations beyond the scope of the paragraph. Do not deduce rules from the sentence grammar, only extract rules that are explicitly stated in the paragraph.

    In addition, represent relevant morphological features using the UniMorph standard.

    The UniMorph format is a standardized, language-independent way of representing morphological features using uppercase tags separated by semicolons. 
    For example, a verb in past tense, third person singular is annotated as: V;PST;3;SG

    Follow these conventions for UniMorph:
    - Verbs (V): Annotate with Tense (PST, PRS, FUT), Person (1, 2, 3), Number (SG, PL), Mood (IND, SBJV, IMP), Aspect (IPFV, PFV), Voice (ACT, PASS), and Polarity (POS, NEG)
    - Nouns (N): Annotate with Number (SG, PL), Case (NOM, ACC, DAT, etc.), Gender (MASC, FEM, NEUT), Possession (POSS) where applicable
    - Adjectives (ADJ): Annotate with Degree (POS, CMP, SPRL), and if applicable, Number, Case, Gender
    - Adverbs (ADV): Annotate with Degree (POS, CMP, SPRL), and if applicable, Polarity (POS, NEG)
    - Pronouns (PRON): Annotate with Person (1, 2, 3), Number (SG, PL), Case (NOM, ACC, DAT, etc
    - Use POS tags like V (verb), N (noun), ADJ (adjective), ADV (adverb), PRON (pronoun), DET (determiner), etc.

    For each rule, output the following fields in YAML:

    - description: A simple plain-language summary of what the rule does
    - condition: When or where the rule applies
    - action: What transformation is made
    - fst_rule: A formal rewrite rule (use the format: X → Y / L ___ R).
    Include the relevant UniMorph feature(s) in brackets at the end of the rule, using the format:
    
    stem → stem+SUFFIX [UNIMORPH] / L ___ R

    For example:
    stem → stem+gaLu [N;PL] / ___ #

    Leave the field blank if the transformation cannot be expressed with a finite-state rewrite rule.

    - usage: Whether the rule is used in practice
    - unimorph: A UniMorph feature string (e.g., V;PST;3;SG)
    - grammar_tags: A list of grammar tags that apply to the rule (e.g., "Gender", "Plural", "Past Tense", "Third Person", "Singular", "Indicative Mood", "Active Voice", etc.) 

    If multiple rules are described in the paragraph, output multiple YAML blocks — one per rule.

    Do not include any explanation. Output only the YAML and nothing else.

    Paragraph: {{input_paragraph}}
    """

    with open("api_key.txt", "r") as f:
        openai_api_key = f.read().strip()

    client = OpenAI(api_key=openai_api_key)

    gpt_extracted_rules_direct = []
    rule_section_map = []
    # Set limit for API calls
    LIMIT = -1
    # Iterate through the sections
    print("Total sections to process:", len(sections_split))
    api_model = "gpt-4o-mini"
    for section in tqdm(sections_split):
        section_id = section.get("id", None)
        if LIMIT == 0:
            break

        LIMIT -= 1
        # Process each section
        temp = []
        for paragraph in section:
            response = client.responses.create(
                model=api_model, input=base_prompt.format(input_paragraph=paragraph)
            )

            temp.append(response.output[0].content[0].text)
        gpt_extracted_rules_direct.append(temp)
        rule_section_map.append(section_id)

    with open("scratch/gpt_extracted_rules_direct.json", "w", encoding="utf-8") as f:
        json.dump(gpt_extracted_rules_direct, f, ensure_ascii=False, indent=4)

    # clean the data
    for i, section in enumerate(gpt_extracted_rules_direct):
        if section is None:
            continue
        section_id = rule_section_map[i]
        for j, response in enumerate(section):
            if response is None:
                continue
            print(response)
            # Remove code block markers and extra spaces
            cleaned_response = (
                response.strip().replace("```yaml", "").replace("```", "")
            )
            # Parse the cleaned YAML response
            try:
                parsed_response = yaml.safe_load(cleaned_response)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML for section {i}, paragraph {j}: {e}")
                print(parsed_response)
                parsed_response = None
            # Update the response in the list
            if parsed_response is not None:
                # Ensure the cleaned response is a dictionary
                if isinstance(parsed_response, dict):
                    parsed_response = [parsed_response]
                elif not isinstance(parsed_response, list):
                    print(
                        f"Unexpected format for section {i}, paragraph {j}: {parsed_response}"
                    )
                    parsed_response = []
                for response in parsed_response:
                    response["section_id"] = section_id
                    
            gpt_extracted_rules_direct[i][j] = parsed_response
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
