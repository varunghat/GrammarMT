import json
from pathlib import Path
import typer
import random
from openai import OpenAI
from tqdm import tqdm


with open("api_key.txt", "r") as f:
    openai_api_key = f.read().strip()

client = OpenAI(api_key=openai_api_key)

app = typer.Typer(
    name="sentence_generation",
    help="A CLI tool for working with linguistic rules and sentence generation.",
    pretty_exceptions_enable=False,
    add_completion=False,
)


@app.command()
def generate_sentences(
    filename: str = typer.Argument(None, help="Path to the input pdf file"),
    sentence_limit: int = typer.Option(
        30, help="Number of sentences to use as seed to generate new sentences"
    ),
    no_of_random_nouns: int = typer.Option(
        2, help="Number of random nouns to use for generation"
    ),
):
    """
    Generate sentences based on linguistic rules and a dictionary."""
    base_filename = Path(filename).stem

    rules_filename = (
        Path("extracted_rules")
        / f"{base_filename}_sections_classified_gpt_extracted_rules_direct_parsed.json"
    )
    with open(rules_filename, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    parallel_sentences_file = (
        Path("parallel_sents")
        / f"{base_filename}_parallel_sentences_with_pos_and_unimorph.json"
    )
    with open(rules_filename, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    all_rules = []
    for section in rules_data:
        if section is not None:  # Check if section is not None
            for paragraph in section:
                if paragraph is not None:  # Check if paragraph is not None
                    all_rules.extend(paragraph)

    # Flattened list of all rules
    print(len(all_rules))

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

    print(len(rules_with_fst), len(rules_with_unimorph), len(all_rules))

    # Filter out rules into differnet categories (add tags?)
    for rule in rules_with_unimorph:
        try:
            split_unimorph = rule["unimorph"].split(";")
            # print(split_unimorph)
        except:
            print(rule["unimorph"])

    # Use the dynamically constructed parallel sentences filename
    with open(parallel_sentences_file, "r", encoding="utf-8") as f:
        parallel_sentences_with_pos_and_unimorph = json.load(f)

    # Load dictionary file based on base_filename if available, else use default path
    dict_path = Path("dictionary") / f"{base_filename}_dictionary_cleaned.json"

    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    # Find all rules with N in unimorph
    noun_rules = []

    for rule in all_rules:
        if "unimorph" in rule and rule["unimorph"] is not None:
            if type(rule["unimorph"]) == list:
                continue
            split_unimorph = rule["unimorph"].split(";")
            if "N" in split_unimorph:
                noun_rules.append(rule)
    len(noun_rules)

    filtered_noun_rules = []
    for rule in noun_rules:
        if "means" not in rule["description"] and rule["fst_rule"]:
            filtered_noun_rules.append(rule)

    len(filtered_noun_rules), filtered_noun_rules

    base_prompt = f"""You are an expert in linguistics and you are tasked with generating sentences in a language called {{lang}}.
    You will be given a set of rules that describe how to form sentences in {{lang}}. 
    You will also be given a sentence in {{lang}}, it's english translation, and the POS and Unimorph information for each word in the sentence.
    You will use these rules, the sentence, the translation, the POS and Unimorph information as well as a set of nouns to generate a new sentence in {{lang}}.
    Do not blindly replace the noun in the sentence, instead use the rules to generate a grammatically correct sentence that follows the rules provided. 
    The rules provided are relavant to the noun in the sentence and the noun you will replace it with. 

    You will generate a sentence that is grammatically correct and follows the rules provided. 

    Here are the rules:
    {{rules_data}}
    Here is the sentence in {{lang}} (transliterated to English):
    {{sentence_text}}
    Here is the translation of the sentence in English:
    {{translated_sentence}}
    Here is the POS and Unimorph information for each word in the sentence:
    {{pos_unimorph_info}}

    Replace the noun in the sentence with the following noun:
    {{noun}} - {{noun_translation}}

    Provide the generated sentence in {{lang}} (transliterated to English) as well as the translation in English.
    Only provide the generated sentence and the translation, do not provide any additional information.
    """

    # Select n random nouns from the nouns_pos_mapping

    random_nouns = random.sample(list(dictionary.items()), no_of_random_nouns)
    random_nouns = {noun: dictionary[noun] for noun, _ in random_nouns}

    print("Randomly selected nouns:", random_nouns)

    #####################

    generated_sentences = []

    # pronouns = ["avanu", "ivanu", "avaḷu","ivaḷu", "nīnu","nīvu", "nānu", "nāvu", "avaru", "ivaru","adu","idu"]
    # TODO: Fix hardcoded pronouns

    LIMIT = sentence_limit

    for sentence in tqdm(parallel_sentences_with_pos_and_unimorph[:LIMIT]):

        # print(random_nouns)
        #
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

        ########
        # Get the noun in the sentence by string matching with list of nouns
        # sentence_text = sentence["sentence"]
        # print(sentence_text)
        noun_replace = []
        for noun in dictionary.keys():
            noun = noun.strip()
            if noun in sentence_text.lower():
                noun_replace.append(noun)

        # for pronoun in pronouns:
        # if pronoun in sentence_text.lower():
        #     noun_replace.append(pronoun)
        # print(f"Found pronoun '{pronoun}' in sentence '{sentence_text}'")
        # else:
        # print(f"Warning: Noun '{noun}' not found in sentence '{sentence_text}'")
        # print(f"Sentence: {sentence_text}")
        # print(f"Noun Replace: {noun_replace}")

        noun_to_be_replaced = noun_replace[0] if noun_replace else None
        if noun_to_be_replaced is None:
            # print("No noun found in the sentence to replace.")
            pass
        else:
            print(f"Noun to be replaced: {noun_to_be_replaced}")
            translation_noun_to_be_replaced = dict.get(
                noun_to_be_replaced, "No translation available"
            )
            if translation_noun_to_be_replaced == "No translation available":
                print(f"Warning: No translation found for noun '{noun_to_be_replaced}'")
            else:
                # Clean translation, split by comma and / , remove bracket
                translation_noun_to_be_replaced = (
                    translation_noun_to_be_replaced.split(",")[0]
                    .split("/")[0]
                    .split("(")[0]
                    .strip()
                )
            print(f"Translation: {translation_noun_to_be_replaced}")
            # Get spacy POS mapping for the sentence
            noun_stem_info = None
            for word_info in spacy_info:
                if word_info["lemma"] == translation_noun_to_be_replaced:
                    noun_stem_info = word_info
                    break

            if noun_stem_info:
                print(f"Found noun stem info: {noun_stem_info}")
                morph = noun_stem_info.get("morph", None)
                if morph:
                    plural = morph.get("Number", None)
                    if plural == "Plur":
                        print(f"The noun '{noun_to_be_replaced}' is plural.")
                    else:
                        print(f"The noun '{noun_to_be_replaced}' is singular.")
                    case = morph.get("Case", None)
                    if case:
                        print(
                            f"The noun '{noun_to_be_replaced}' is in the case: {case}."
                        )
                    gender = morph.get("Gender", None)
                    if gender:
                        print(
                            f"The noun '{noun_to_be_replaced} is of gender {gender}   ."
                        )
            else:
                print(f"No noun stem info found for: {translation_noun_to_be_replaced}")
                morph = None
                plural = None
                case = None
                gender = None

            # print(f"Morphology Info: {morphology_info}")

        # ruleset = filtered_noun_rules # TODO: Adjust this to use the full ruleset or a specific subset as needed
        ruleset = []
        rule_text = ""
        if gender == "Masc":
            ruleset += [
                rule
                for rule in filtered_noun_rules
                if "masculine" in rule["description"].lower()
            ]

            # print(ruleset)
            gender_ruleset_text_start = (
                "The word to be replaced is masculine, so the following rules apply:\n"
            )
            gender_ruleset_text = gender_ruleset_text_start + "\n".join(
                "- " + rule["description"] for rule in ruleset
            )
            rule_text += gender_ruleset_text
        elif gender == "Fem":
            ruleset += [
                rule
                for rule in filtered_noun_rules
                if "feminine" in rule["description"].lower()
            ]
            gender_ruleset_text_start = (
                "The word to be replaced is feminine, so the following rules apply:\n"
            )
            gender_ruleset_text = gender_ruleset_text_start + "\n".join(
                "- " + rule["description"] for rule in ruleset
            )
            rule_text += gender_ruleset_text
            # print(ruleset)
        if plural == "Plur":
            ruleset += [
                rule
                for rule in filtered_noun_rules
                if "plural" in rule["description"].lower()
            ]
            plural_ruleset_text_start = (
                "The word to be replaced is plural, so the following rules apply:\n"
            )
            plural_ruleset_text = plural_ruleset_text_start + "\n".join(
                "- " + rule["description"] for rule in ruleset
            )
            rule_text += plural_ruleset_text
            # print(ruleset)
        if case is not None:
            if case == "Nom":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "nominative" in rule["description"].lower()
                ]
                case_ruleset_text_start = "The word to be replaced is nominative, so the following rules apply:\n"
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)
            elif case == "Acc":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "accusative" in rule["description"].lower()
                ]
                case_ruleset_text_start = "The word to be replaced is accusative, so the following rules apply:\n"
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)
            elif case == "Dat":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "dative" in rule["description"].lower()
                ]
                case_ruleset_text_start = (
                    "The word to be replaced is dative, so the following rules apply:\n"
                )
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)
            elif case == "Gen":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "genitive" in rule["description"].lower()
                ]
                case_ruleset_text_start = "The word to be replaced is genitive, so the following rules apply:\n"
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)
            elif case == "Loc":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "locative" in rule["description"].lower()
                ]
                case_ruleset_text_start = "The word to be replaced is locative, so the following rules apply:\n"
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)
            elif case == "Abl":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "ablative" in rule["description"].lower()
                ]
                case_ruleset_text_start = "The word to be replaced is ablative, so the following rules apply:\n"
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)
            elif case == "Voc":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "vocative" in rule["description"].lower()
                ]
                case_ruleset_text_start = "The word to be replaced is vocative, so the following rules apply:\n"
                case_ruleset_text = case_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += case_ruleset_text
                # print(ruleset)

        if tense is not None:
            if tense == "Past":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "past" in rule["description"].lower()
                ]
                tense_ruleset_text_start = (
                    "The sentence in past tense, so the following rules apply:\n"
                )
                tense_ruleset_text = tense_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += tense_ruleset_text
                # print(ruleset)
            elif tense == "Pres":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "present" in rule["description"].lower()
                ]
                tense_ruleset_text_start = (
                    "The sentence is in present tense, so the following rules apply:\n"
                )
                tense_ruleset_text = tense_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += tense_ruleset_text
                # print(ruleset)
            elif tense == "Fut":
                ruleset += [
                    rule
                    for rule in filtered_noun_rules
                    if "future" in rule["description"].lower()
                ]
                tense_ruleset_text_start = (
                    "The sentence is in future tense, so the following rules apply:\n"
                )
                tense_ruleset_text = tense_ruleset_text_start + "\n".join(
                    "- " + rule["description"] for rule in ruleset
                )
                rule_text += tense_ruleset_text
                # print(ruleset)

        ruleset_text = "\n".join("- " + rule["description"] for rule in ruleset)
        if not ruleset_text:
            ruleset_text = (
                "No specific rules apply for the given noun and sentence context."
            )

        # OVerride

        print("\n")
        # ruleset_text = "\n".join("- " + rule["description"] for rule in ruleset)

        pos_info_text = ""
        # print(pos_info_text)

        for noun in random_nouns.keys():
            noun = noun.strip()
            noun_translation = dict.get(noun, "No translation available")
            if noun_translation == "No translation available":
                print(f"Warning: No translation found for noun '{noun}'")

            prompt = base_prompt.format(
                rules_data=ruleset_text,
                sentence_text=sentence_text,
                translated_sentence=translated_sentence,
                pos_unimorph_info=pos_info_text,
                lang=base_filename.split("_")[0],  # Extract language from base_filename
                noun=noun,
                noun_translation=noun_translation,
            )
            # print(prompt)

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
                    "noun": noun,
                    "generated_text": gen_sentence,
                    "generated_text_translation": translation_generation,
                    "ruleset_text": ruleset_text,
                }
            )

    # Save generated sentences to a file
    output_filename = (
        Path("generated_sentences") / f"{base_filename}_generated_sentences.json"
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(generated_sentences, f, ensure_ascii=False, indent=4)

    print(f"Generated sentences saved to {output_filename}")


if __name__ == "__main__":
    app()
