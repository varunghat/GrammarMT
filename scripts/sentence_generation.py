import json
import os
from pathlib import Path
import typer
import random
from openai import OpenAI
from google import genai
from tqdm import tqdm
import re


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val


app = typer.Typer(
    name="sentence_generation",
    help="A CLI tool for working with linguistic rules and sentence generation.",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def select_model_client(model_provider="gemini"):

    print(f"Model provider = {model_provider}")

    if model_provider == "openai":
        client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    elif model_provider == "gemini":
        client = genai.Client(api_key=_require_env("GEMINI_API_KEY"))
    else:
        raise ValueError("Unsupported model_provider: choose 'openai' or 'gemini'")

    return client


# Rule selection function
def select_rules(
    all_rules,
    gender=None,
    number=None,
    case=None,
    tense=None,
    granularity="rule",
    sections=None,
):
    def has(desc, kw):
        return kw in desc.lower()

    selected = []

    def add(pred):
        selected.extend([r for r in all_rules if pred(r)])

    if gender == "Masc":
        add(lambda r: has(r["description"], "masculine"))
    if gender == "Fem":
        add(lambda r: has(r["description"], "feminine"))
    if number == "Plur":
        add(lambda r: has(r["description"], "plural"))
    if case == "Nom":
        add(lambda r: has(r["description"], "nominative"))
    if case == "Acc":
        add(lambda r: has(r["description"], "accusative"))
    if case == "Dat":
        add(lambda r: has(r["description"], "dative"))
    if case == "Gen":
        add(lambda r: has(r["description"], "genitive"))
    if case == "Loc":
        add(lambda r: has(r["description"], "locative"))
    if case == "Abl":
        add(lambda r: has(r["description"], "ablative"))
    if case == "Voc":
        add(lambda r: has(r["description"], "vocative"))
    if tense == "Past":
        add(lambda r: has(r["description"], "past"))
    if tense == "Pres":
        add(lambda r: has(r["description"], "present"))
    if tense == "Fut":
        add(lambda r: has(r["description"], "future"))

    # dedupe & cap
    unique = {}
    for r in selected:
        unique.setdefault(r["description"], r)
    capped = list(unique.values())[:50]  # keep context small

    if granularity == "rule":
        return (
            "\n".join(f"- {r['description']}" for r in capped)
            or "No specific rules apply."
        )
    elif granularity == "section":
        if sections is None:
            raise ValueError("Sections data must be provided for section granularity.")
        # Get section id of each rule
        section_ids = set(
            r.get("section_id") for r in capped if r.get("section_id") is not None
        )
        # print(f"Selected section IDs: {section_ids}")
        # Get all sections corresponding to these ids
        selected_sections = [s for s in sections if s["id"] in section_ids]
        # print(f"Selected sections: {selected_sections}")
        return (
            "\n".join(f"- {s['heading']}: {s['text']}" for s in selected_sections)
            or "No specific sections apply."
        )
    elif granularity == "book":
        if sections is None:
            raise ValueError("Sections data must be provided for book granularity.")
        # Combine all text from sections
        return (
            "\n".join(f"- {s['heading']}: {s['text']}" for s in sections)
            or "No specific book-level rules apply."
        )


def load_rules(rules_filename):
    with open(rules_filename, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    all_rules = []
    for section in rules_data:
        if section is not None:  # Check if section is not None
            for paragraph in section:
                if paragraph is not None:  # Check if paragraph is not None
                    all_rules.extend(paragraph)

    rules_with_fst = [
        rule
        for rule in all_rules
        if "fst_rule" in rule and rule["fst_rule"] is not None
    ]

    rules_with_unimorph = [
        rule
        for rule in all_rules
        if "unimorph" in rule and rule["unimorph"] is not None
    ]
    return all_rules, rules_with_fst, rules_with_unimorph


def load_dictionary(dict_path, reverse=False):
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
            if reverse:
                # reverse the dictionary to map from language to English
                dictionary = {v.lower(): k.lower() for k, v in dictionary.items()}
            return dictionary
    except FileNotFoundError:
        print(
            f"Dictionary file not found at {dict_path}. Please provide a valid dictionary file."
        )
        return {}


def load_dictionary_with_meta(dict_path):
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
            return dictionary
    except FileNotFoundError:
        print(
            f"Dictionary file not found at {dict_path}. Please provide a valid dictionary file."
        )
        return {}


def load_sections(sections_filename):
    try:
        with open(sections_filename, "r", encoding="utf-8") as f:
            sections = json.load(f)
            return sections
    except FileNotFoundError:
        print(
            f"Sections file not found at {sections_filename}. Please provide a valid sections file."
        )
        return []


def filter_dict_by_pos(dictionary_with_metadata, target_pos):
    """
    Filters a dictionary to include only words that are predominantly of the specified parts of speech (POS
    categories).
    Args:
        dictionary_with_metadata (dict): A dictionary where keys are words and values are metadata including senses
        target_pos (list[str]): List of POS categories to filter by (e.g. ["NOUN", "VERB"]).
    Returns:
        dict: Filtered dictionary containing only words that are predominantly of the specified POS categories.
    """
    filtered_dict = {}

    for word, word_metadata in dictionary_with_metadata.items():
        senses = word_metadata.get("senses", [])
        if not senses:
            continue

        pos_like = False
        count = 0
        for sense in senses:
            spacy_analysis = sense.get("spacy_analysis")
            if not spacy_analysis:
                continue

            pos = spacy_analysis.get("head_pos")
            if pos in target_pos:
                count += 1

        if count / len(senses) >= 0.5:  # at least half the senses are POS-like
            pos_like = True

        if pos_like:
            filtered_dict[word] = dictionary_with_metadata.get(word, word_metadata)

    return filtered_dict


def find_lrl_words_by_pos(sentence_text, pos_dicts, target_pos):
    """
    Finds which *target-language words* in a sentence appear in your filtered POS dictionaries.

    Args:
        sentence_text (str): Sentence in the target language (LRL).
        pos_dicts (dict): Mapping POS → dictionary of LRL words (keys).
        target_pos (list[str]): POS categories to match (e.g. ["NOUN", "VERB"]).

    Returns:
        dict: {pos_tag: [matched_words]} of words found in the sentence.
    """
    # Normalize the sentence (lowercase, basic punctuation removal)
    sentence_clean = re.sub(r"[^\w\s'=*-]", " ", sentence_text.lower())
    matches_by_pos = {}

    for pos_tag in target_pos:
        pos_words = set(pos_dicts.get(pos_tag, {}).keys())
        matched = set()

        # Prioritize longer multiword entries first
        for word in sorted(pos_words, key=len, reverse=True):
            # Use regex to match exact word boundaries (handles = and * prefixes too)
            pattern = rf"(?<!\S){re.escape(word.lower())}(?!\S)"
            if re.search(pattern, sentence_clean):
                matched.add(word)

        if matched:
            matches_by_pos[pos_tag] = sorted(matched, key=len, reverse=True)

    return matches_by_pos


def get_random_words_from_dict(pos_dicts, pos_to_replace, no_of_random_nouns):
    # Get a random pos word from the dictionary of selected pos
    random_words_of_selected_pos = {}
    for pos in pos_to_replace:
        pos_dict = pos_dicts.get(pos, {})
        random_words = random.sample(
            list(pos_dict.keys()), min(no_of_random_nouns, len(pos_dict))
        )
        for word in random_words:
            random_words_of_selected_pos[word] = pos_dict[word]

    return random_words_of_selected_pos


def get_lrl_words_meta(lrl_words_by_pos, spacy_info, dictionary, sentence):
    if lrl_words_by_pos == {}:
        print("No words found in the sentence to replace.")
        gender = None
        plural = None
        case = None
        morph = None

    else:
        print(f"Words to be replaced: {lrl_words_by_pos}")
        translation_words_to_be_replaced = {}
        for pos_tag, words in lrl_words_by_pos.items():
            for word in words:
                word = word.strip().lower()
                word_translation = dictionary.get(word, "No translation available")
                translation_words_to_be_replaced[word] = word_translation

        if translation_words_to_be_replaced == {}:
            print(f"Warning: No translation found for words {list(lrl_words_by_pos)}")
        else:
            # Clean translation, split by comma and / , remove bracket
            translation_words_to_be_replaced = {
                word: translation.split(",")[0].split("/")[0].split("(")[0].strip()
                for word, translation in translation_words_to_be_replaced.items()
            }
        print(f"Translation: {translation_words_to_be_replaced}")

        # Check if each selected word's translation is present in the target sentence
        # TODO: Fix naive search, this can result in words infixed in others.
        words_present = []
        for word, word_translation in translation_words_to_be_replaced.items():
            if word_translation in sentence["translated_sentence"]:
                print(f"{word_translation} ({word}) present in sentence")
                words_present.append(word)
        # Randomly select one word to be replaced from present, or else from all
        if words_present != []:
            word_to_be_replaced = random.choice(words_present)
        else:
            word_to_be_replaced = random.choice(
                list(translation_words_to_be_replaced.keys())
            )
        translation_words_to_be_replaced = translation_words_to_be_replaced[
            word_to_be_replaced
        ]
        print(
            f"Selected word to be replaced: {word_to_be_replaced} - {translation_words_to_be_replaced}"
        )
        # Get spacy POS mapping for the sentence
        word_stem_info = None
        for word_info in spacy_info:
            if word_info["lemma"] == translation_words_to_be_replaced:
                word_stem_info = word_info
                break

        if word_stem_info:
            print(f"Found word stem info: {word_stem_info}")
            morph = word_stem_info.get("morph", None)
            if morph:
                plural = morph.get("Number", None)
                if plural == "Plur":
                    print(f"The word '{word_to_be_replaced}' is plural.")
                else:
                    print(f"The word '{word_to_be_replaced}' is singular.")
                case = morph.get("Case", None)
                if case:
                    print(f"The word '{word_to_be_replaced}' is in the case: {case}.")
                gender = morph.get("Gender", None)
                if gender:
                    print(f"The word '{word_to_be_replaced}' is of gender {gender}.")
        else:
            print(f"No word stem info found for: {translation_words_to_be_replaced}")
            morph = None
            plural = None
            case = None
            gender = None

        # print(f"Morphology Info: {morphology_info}")
    return gender, plural, case


@app.command()
def generate_sentences(
    filename: str = typer.Argument(None, help="Path to the input pdf file"),
    sentence_limit: int = typer.Option(
        30, help="Number of sentences to use as seed to generate new sentences"
    ),
    no_of_random_nouns: int = typer.Option(
        2, help="Number of random nouns to use for generation"
    ),
    output_dir: str = typer.Option(
        "generated_sentences", help="Directory to save the generated sentences"
    ),
    filter_rules: bool = typer.Option(True, help="Whether to filter the rules or not"),
    pos_to_replace: str = typer.Option(
        "NOUN,PROPN",
        help="Parts of speech to replace in the sentences. E.g., NOUN, VERB, ADJ, ADV.",
    ),
    model_provider: str = typer.Option(
        "gemini", help="Model provider to use: 'gemini' or 'openai'"
    ),
    granularity: str = typer.Option(
        "rule",
        help="Granularity level of grammar info to use: 'rule','section' or 'book'",
    ),
    create_batches: bool = typer.Option(
        False, help="Whether to create batches for generation or not"
    ),
):
    """
    Generate sentences based on linguistic rules and a dictionary.
    Args:
        filename (str): Path to the input file (used to derive rules and sentences files).
        sentence_limit (int): Number of sentences to use as seed for generation.
        no_of_random_nouns (int): Number of random nouns to select from the dictionary.
        output_dir (str): Directory to save the generated sentences.
        filter_rules (bool): Whether to filter the rules based on POS or not.
        pos_to_replace (list): List of parts of speech to replace in the sentences.
    Returns:
        None
    """

    print("Starting sentence generation process...")
    print(f"Using {sentence_limit} seed sentences")
    print(f"Using {no_of_random_nouns} random nouns")
    print(f"Using POS tags: {pos_to_replace}")
    print(f"Using granularity level: {granularity}")
    #############################
    # LOAD DATA
    #############################
    client = select_model_client(model_provider=model_provider)

    pos_to_replace = [pos.strip() for pos in pos_to_replace.split(",")]

    base_filename = Path(filename).stem

    rules_filename = (
        Path("extracted_rules")
        / f"{base_filename}_sections_classified_gpt_extracted_rules_direct_parsed.json"
    )
    all_rules, rules_with_fst, rules_with_unimorph = load_rules(rules_filename)

    # Flattened list of all rules
    print(len(all_rules))
    print(len(rules_with_fst), len(rules_with_unimorph), len(all_rules))

    # Load sections TODO: Move to function
    sections_filename = (
        Path("classified_json") / f"{base_filename}_sections_classified.json"
    )
    sections = load_sections(sections_filename)
    print(f"Loaded {len(sections)} sections.")

    # Filter out rules into differnet categories (add tags?)
    for rule in rules_with_unimorph:
        try:
            split_unimorph = rule["unimorph"].split(";")
            # print(split_unimorph)
        except:
            print(rule["unimorph"])

    # Use the dynamically constructed parallel sentences filename

    parallel_sentences_file = (
        Path("parallel_sents")
        / f"{base_filename}_parallel_sentences_with_pos_and_unimorph.json"
    )

    with open(parallel_sentences_file, "r", encoding="utf-8") as f:
        parallel_sentences_with_pos_and_unimorph = json.load(f)

    # Load dictionary file based on base_filename if available, else use default path
    dict_path = Path("dictionary") / f"{base_filename}_dictionary.json"
    dictionary = load_dictionary(dict_path)
    dictionary_with_metadata_path = (
        Path("dictionary") / f"{base_filename}_dictionary_with_metadata.json"
    )
    dictionary_with_metadata = load_dictionary_with_meta(dictionary_with_metadata_path)

    # TODO: Make this filtering for each POS
    nouns_only_dict = filter_dict_by_pos(
        dictionary_with_metadata, target_pos=["NOUN", "PROPN"]
    )
    verbs_only_dict = filter_dict_by_pos(dictionary_with_metadata, target_pos=["VERB"])
    adjectives_only_dict = filter_dict_by_pos(
        dictionary_with_metadata, target_pos=["ADJ"]
    )
    adverbs_only_dict = filter_dict_by_pos(dictionary_with_metadata, target_pos=["ADV"])
    pos_dicts = {
        "NOUN": nouns_only_dict,
        "VERB": verbs_only_dict,
        "ADJ": adjectives_only_dict,
        "ADV": adverbs_only_dict,
    }

    #########################
    # BASE PROMPT
    #########################
    base_prompt = f"""You are an expert in linguistics and you are tasked with generating sentences in a language called {{lang}}.
    You will be given a set of rules that describe how to form sentences in {{lang}}. 
    You will also be given a sentence in {{lang}}, it's english translation, and the POS and Unimorph information for each word in the sentence.
    You will use these rules, the sentence, the translation, the POS and Unimorph information as well as a new {{pos}} to generate a new sentence in {{lang}}.
    Do not blindly replace the {{pos}} in the sentence, instead use the rules to generate a grammatically correct sentence that follows the rules provided.
    The rules provided are relavant to the {{pos}} in the sentence and the {{pos}} you will replace it with.
    Think step by step about how to select and apply the rules to generate the new sentence. 
    
    You will generate a sentence that is grammatically correct and follows the rules provided. 

    Here are the rules:
    {{rules_data}}
    Here is the sentence in {{lang}} (transliterated to English):
    {{sentence_text}}
    Here is the translation of the sentence in English:
    {{translated_sentence}}
    Here is the spacy metadata information for each word in the sentence:
    {{pos_unimorph_info}}

    Replace the {{pos}} in the sentence with the following {{pos}}:
    {{word}} - {{word_translation}}

    Provide the generated sentence in {{lang}} (transliterated to English) as well as the translation in English at the end of the reasoning in a new line with quotes.
    """

    #####################

    random_words_of_selected_pos = get_random_words_from_dict(
        pos_dicts, pos_to_replace, no_of_random_nouns
    )
    for word, pos_info in random_words_of_selected_pos.items():
        print(f"Random {pos_to_replace} word: {word} - {pos_info}")

    #########################################
    # Main generation loop
    #########################################

    generated_sentences = []

    LIMIT = min(max(sentence_limit, 0), len(parallel_sentences_with_pos_and_unimorph))
    print(
        f"Using {LIMIT} sentences for generation. out of {len(parallel_sentences_with_pos_and_unimorph)} available."
    )

    all_prompts = []

    for sentence in tqdm(parallel_sentences_with_pos_and_unimorph[:LIMIT]):

        # print(random_nouns)

        print("Selected sentence for generation:")
        print(sentence["sentence"], "\n", sentence["translated_sentence"], "\n")
        if LIMIT <= 0:
            break
        LIMIT -= 1
        sentence_text = sentence["sentence"]
        translated_sentence = sentence["translated_sentence"]
        spacy_info_all = sentence["spacy_info"]
        spacy_info = spacy_info_all.get("res", None)
        tense = spacy_info_all.get("tense", None)
        number = spacy_info_all.get("number", None)
        genders = spacy_info_all.get("genders", None)
        print("Tense: ", tense)
        print(f"Number: {number}")
        print(f"Genders: {genders}")

        # Find LRL words by POS in the sentence
        lrl_words_by_pos = find_lrl_words_by_pos(
            sentence_text, pos_dicts, target_pos=pos_to_replace
        )
        # print(f"LRL words by POS in the sentence: {lrl_words_by_pos}")

        gender, plural, case = get_lrl_words_meta(
            lrl_words_by_pos, spacy_info, dictionary, sentence
        )

        ################
        # RULE SELECTION
        ################

        print("\nRule selection\n")

        if filter_rules:
            rule_text = select_rules(
                all_rules,
                gender=gender,
                number=number,
                case=case,
                tense=tense,
                granularity=granularity,
                sections=sections,
            )
        else:
            rule_text = "\n".join("- " + rule["description"] for rule in all_rules)

        # print(rule_text)
        # print("\n")
        ###############
        ###############
        ###############
        # ruleset_text = "\n".join("- " + rule["description"] for rule in ruleset)

        pos_info_text = ""
        # print(pos_info_text)

        for word in random_words_of_selected_pos.keys():
            word = word.strip()
            word_translation = (
                random_words_of_selected_pos[word]
                .get("senses", [{}])[0]
                .get("translation", "No translation available")
            )
            print(f"Generating sentence by replacing '{word}' ({word_translation})")
            if word_translation == "No translation available":
                print(f"Warning: No translation found for word '{word}'")

            prompt = base_prompt.format(
                rules_data=rule_text,
                sentence_text=sentence_text,
                translated_sentence=translated_sentence,
                pos_unimorph_info=pos_info_text,
                lang=base_filename.split("_")[0],  # Extract language from base_filename
                word=word,
                word_translation=word_translation,
                pos=", ".join(pos_to_replace),
            )
            # print(prompt)

            if create_batches:
                print("Creating batch prompt, skipping generation...")
                all_prompts.append(prompt)
                continue

            if model_provider == "gemini":
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                generated_text = response.text.strip()
            elif model_provider == "openai":
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    # temperature=0.7
                )
                generated_text = response.choices[0].message.content.strip()
            print(f"Generated Text: {generated_text}")
            # remove multiple newlines
            generated_text = "\n".join(
                line.strip() for line in generated_text.split("\n") if line.strip()
            )
            # Split generated text into sentence and translation
            if "\n" in generated_text:
                try:
                    gen_sentence, translation_generation = generated_text.split("\n", 1)
                    print(
                        f"Gen Sentence: {gen_sentence}"
                        f"\nTranslation: {translation_generation}"
                    )

                except:
                    gen_sentence = generated_text
                    translation_generation = "No translation generated"

            generated_sentences.append(
                {
                    "original_sentence": sentence_text,
                    "translated_sentence": translated_sentence,
                    "pos_word_replaced": word,
                    "generated_text": gen_sentence,
                    "generated_text_translation": translation_generation,
                    "ruleset_text": ruleset_text,
                }
            )

    if create_batches:
        # Create output path
        batch_output_path = (
            Path(output_dir)
            / f"{base_filename}_{''.join(pos_to_replace)}_{granularity}_generation_prompts.jsonl"
        )
        batch_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build Gemini batch JSONL (one request per prompt)
        batch_tag = f"{base_filename}_{''.join(pos_to_replace)}_{granularity}"
        with open(batch_output_path, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(all_prompts):
                batch_request = {
                    "key": f"{batch_tag}_{i:05d}",
                    "request": {
                        "contents": [{"parts": [{"text": prompt.strip()}]}]
                    },
                }
                f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")

        print(f"Batch input file saved to {batch_output_path}")
        return
    # Save generated sentences to a file
    output_path = Path(output_dir) / f"{base_filename}_generated_sentences.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generated_sentences, f, ensure_ascii=False, indent=4)

    print(f"Generated sentences saved to {output_path}")


if __name__ == "__main__":
    app()
