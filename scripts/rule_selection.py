#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from pathlib import Path
import random
from tqdm import tqdm
import re


def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(f"Set the {var} environment variable. See .env.example.")
    return val


# In[ ]:


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')


# In[ ]:


import tiktoken

def get_token_length(prompt):
    """
    Estimates the token length of a given prompt using tiktoken.

    Args:
        prompt (str): The input prompt string.

    Returns:
        int: Estimated number of tokens in the prompt.
    """

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(prompt)
    return len(tokens)


# In[ ]:


# Load functions

def load_rules(rules_filename):
    with open(rules_filename, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    all_rules = []
    for rule in rules_data:
        if rule is not None:  # Check if rule is not None
            all_rules.append(rule)
    return all_rules

def load_dictionary(dict_path,reverse=False):
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


# In[ ]:


# Filter functions

from typing import Dict, List, Any

def get_contextual_target_pos_from_preprocessed(
    spacy_results: List[Dict[str, Any]], 
    target_english_gloss: str, 
    default_pos: str
) -> str:
    """
    Determines the contextual POS by searching the pre-processed spaCy results 
    for the target LRL word's English gloss/lemma.

    Args:
        spacy_results (List[Dict[str, Any]]): The list of token results from spacy_info['res'].
        target_english_gloss (str): The base English word from the LRL gloss (e.g., 'run').
        default_pos (str): The POS from the LRL dictionary (e.g., 'NOUN').

    Returns:
        str: The inferred contextual UPOS (e.g., 'VERB'), or the default POS.
    """

    # Ensure the gloss is cleaned for matching (e.g., 'run' vs 'Run')
    target_lemma = target_english_gloss.lower().strip()

    if not target_lemma:
        return default_pos

    for token_info in spacy_results:
        # We match against the lemma for robustness (e.g., 'running' should match 'run')
        token_lemma = token_info.get("lemma", "").lower()

        if token_lemma == target_lemma:
            # Found a match! Return the Universal POS (UPOS) tag
            return token_info.get("upos", default_pos)

    # Fallback: If no match is found, stick with the POS from the LRL dictionary
    return default_pos

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

        if count/len(senses) >= 0.5:  # at least half the senses are POS-like
            pos_like = True

        if pos_like:
            filtered_dict[word] = dictionary_with_metadata.get(word, word_metadata)

    return filtered_dict

def find_lrl_words_by_pos(sentence_text: str, sentence_gloss: str, pos_dicts: Dict[str, Dict[str, Any]], target_pos: List[str]) -> List[Dict[str, Any]]:
    """
    Identifies LRL words in a sentence that match specified POS tags and extracts 
    their dictionary lemma, translation, and contextual gloss/features.

    This revised version includes 'contextual_gloss_word' for polysemy resolution.
    """
    sentence_clean = sentence_text.lower()
    matches = []

    # Tracking occupied indices to prevent overlapping matches (greedy matching)
    occupied = [False] * len(sentence_clean) 

    for pos_tag in target_pos:
        # Get dictionary words for the current POS category (e.g., 'NOUN')
        pos_words = pos_dicts.get(pos_tag, {})

        # Iterate over words, longest first, to prioritize multi-word lexemes
        for word in sorted(pos_words.keys(), key=len, reverse=True):
            word_lc = word.lower()

            # Skip pure morphology (like rules that might use '=' or '-')
            if word_lc.startswith(("=", "-")):
                continue

            # Pattern to match the word stem followed by a boundary, a clitic/affix, or end of string
            # (?<!\w) checks for preceding non-word boundary (start of word)
            # (?=[=\-~]|\b|$) checks for following clitic/affix or word boundary
            pattern = rf"(?<!\w){re.escape(word_lc)}(?=[=\-~]|\b|$)"

            for m in re.finditer(pattern, sentence_clean):
                start, end = m.span()

                # Skip if this span is already part of a longer, matched word
                if any(occupied[start:end]):
                    continue

                # Mark the indices as occupied
                for i in range(start, end):
                    occupied[i] = True

                senses = pos_words[word]["senses"]

                if len(senses) == 0:
                    continue

                # Use the first sense found in the dictionary for the base translation/lemma
                sense = senses[0]

                # --- Contextual Gloss Extraction ---
                # Find the corresponding gloss word for the LRL word in the sentence
                sentence_words = sentence_text.split()
                gloss_words = sentence_gloss.split()
                gloss = ""

                for i, sentence_word in enumerate(sentence_words):
                    # Check if the matched LRL word is a substring of the current sentence token
                    if word_lc in sentence_word.lower():
                        if i < len(gloss_words):
                            gloss = gloss_words[i]
                        break

                if gloss == "":
                    gloss_split = []
                    contextual_gloss_word = ""
                else:
                    # Split the contextual gloss (e.g., 'run' or 'fish=obj')
                    gloss_split = re.split(r'[=.-]', gloss)

                    # Extract the base English word from the gloss (e.g., 'run' from 'run' or 'fish' from 'fish=obj')
                    # We take the first element of the split, provided it's not pure morphology.
                    base_word = gloss_split[0]
                    contextual_gloss_word = base_word if base_word and not base_word.startswith(("=", "-")) else ""

                # --- Final Match Object ---
                matches.append({
                    "pos": pos_tag, # Dictionary POS
                    "surface": sentence_text[start:end],
                    "lemma": word,
                    "translation": sense.get("cleaned_translation", ""), # Dictionary Translation
                    "gloss": gloss,
                    "gloss_split": gloss_split,
                    "contextual_gloss_word": contextual_gloss_word, # New Field for SpaCy Check
                    "start": start,
                    "end": end,
                })

    matches.sort(key=lambda x: x["start"])
    return matches


def get_random_words_from_dict(pos_dicts, pos_to_replace, no_of_random_words):
    """
    Get random words from the dictionary for the specified POS categories.

    Args:
        pos_dicts (dict): Dictionary of POS categories with words and metadata.
        pos_to_replace (list[str]): List of POS categories to select random words from.
        no_of_random_words (int): Number of random words to select per POS category.

    Returns:
        list[dict]: List of dictionaries with keys: 'lemma', 'senses', 'metadata'.
    """
    results = []

    for pos in pos_to_replace:
        pos_dict = pos_dicts.get(pos, {})
        random_words = random.sample(
            list(pos_dict.keys()),
            min(no_of_random_words, len(pos_dict))
        )
        for word in random_words:
            results.append({
                "lemma": word,
                "senses": pos_dict[word].get("senses", []),
                "metadata": pos_dict[word]
            })

    return results

def get_top_k_similar_words_from_dict(
    source_word,
    word,
    pos,
    pos_dicts,
    pos_dict_embeddings,
    model,
    k=5,
):
    """
    Get top-k similar words from the dictionary for a given word and POS using sentence embeddings.
    Args:
    source_word: str (the original word in the source language)
    word: str (English word or short phrase)
    pos: str (e.g. 'NOUN', 'VERB')
    pos_dicts: {POS: {lemma: metadata}}
    pos_dict_embeddings: {POS: tensor}
    model: SentenceTransformer model
    k: int (number of similar words to return)
    Returns:
    list of dicts with keys: 'lemma', 'senses', 'score', 'metadata'
    """



    if pos not in pos_dicts:
        return []

    # Embed input
    word_embedding = model.encode(word, convert_to_tensor=True)

    # Get dictionary for this POS only
    pos_dict = pos_dicts[pos]
    dict_words = list(pos_dict.keys())
    dict_senses = [pos_dict[w]["senses"] for w in dict_words]
    dict_senses_text = [
        "; ".join(s.get("cleaned_translation", "") for s in senses)
        for senses in dict_senses
    ]
    dict_first_sense_text = [
        senses[0].get("cleaned_translation", "") if senses else ""
        for senses in dict_senses
    ]


    # Get / compute embeddings
    if pos not in pos_dict_embeddings or pos_dict_embeddings[pos] is None:
        print(f"Computing embeddings for POS: {pos}...")
        pos_dict_embeddings[pos] = model.encode(
            dict_senses_text, convert_to_tensor=True
        )

    dict_embeddings = pos_dict_embeddings[pos]

    # Cosine similarity
    cosine_scores = util.cos_sim(word_embedding, dict_embeddings)[0]

    # Get top-k (+1 to skip identity match if needed)
    top_k = min(k + 1, len(dict_words))
    scores, indices = torch.topk(cosine_scores, k=len(dict_words))

    results = []
    translations = []
    for score, idx in zip(scores, indices):
        candidate = dict_words[idx]


        # Skip identical word (case-insensitive)
        if candidate.lower() == source_word.lower():
            continue

        if dict_first_sense_text[idx] in translations:
            continue
        translations.append(dict_first_sense_text[idx])

        results.append({
            "lemma": candidate,
            "senses": dict_senses[idx],
            "score": float(score),
            "metadata": pos_dict[candidate]
        })

        if len(results) == k:
            break

    return results


# In[ ]:


# Get word metadata

def get_lrl_words_meta(lrl_words_by_pos, spacy_info, dictionary, sentence):
        gender = None
        plural = None
        case = None
        morph = None
        translation_words_to_be_replaced = {}
        if lrl_words_by_pos == {}:
            print("No words found in the sentence to replace.")


        else:
            print(f"Words to be replaced: {lrl_words_by_pos}")
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
                    word: translation.split(",")[0]
                    .split("/")[0]
                    .split("(")[0]
                    .strip()
                    for word, translation in translation_words_to_be_replaced.items()
                }
            print(f"Translation: {translation_words_to_be_replaced}")

            # Check if each selected word's translation is present in the target sentence
            # TODO: Fix naive search, this can result in words infixed in others.
            words_present = []
            for word,word_translation in translation_words_to_be_replaced.items():
                if word_translation in sentence["translation"]:
                    print(f"{word_translation} ({word}) present in sentence")
                    words_present.append(word)
            # Randomly select one word to be replaced from present, or else from all
            if words_present!=[]:
                word_to_be_replaced = random.choice(words_present)
            else:
                word_to_be_replaced = random.choice(list(translation_words_to_be_replaced.keys()))
            translation_words_to_be_replaced = translation_words_to_be_replaced[word_to_be_replaced]
            print(f"Selected word to be replaced: {word_to_be_replaced} - {translation_words_to_be_replaced}")
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
                        print(
                            f"The word '{word_to_be_replaced}' is in the case: {case}."
                        )
                    gender = morph.get("Gender", None)
                    if gender:
                        print(
                            f"The word '{word_to_be_replaced}' is of gender {gender}."
                        )
            else:
                print(f"No word stem info found for: {translation_words_to_be_replaced}")

            # print(f"Morphology Info: {morphology_info}")
        return gender, plural, case, translation_words_to_be_replaced


# # Rule select function

# In[ ]:


TAG_MAPPING = {
    # Case and Grammatical Relations
    'obj':  {'feature': 'CASE', 'value': 'OBJ'},
    'acc':  {'feature': 'CASE', 'value': 'ACC'},
    'lat':  {'feature': 'CASE', 'value': 'LAT'},
    'dat':  {'feature': 'CASE', 'value': 'DAT'},
    'gen':  {'feature': 'CASE', 'value': 'GEN'},
    'nom':  {'feature': 'CASE', 'value': 'NOM'},
    'poss': {'feature': 'GRAM_REL', 'value': 'POSS'},

    # Number
    'sg':   {'feature': 'NUMBER', 'value': 'SG'},
    'pl':   {'feature': 'NUMBER', 'value': 'PLUR'},
    'du':   {'feature': 'NUMBER', 'value': 'DU'},

    # Person
    '1':    {'feature': 'PERSON', 'value': '1'},
    '2':    {'feature': 'PERSON', 'value': '2'},
    '3':    {'feature': 'PERSON', 'value': '3'},

    # Tense and Aspect
    'pres': {'feature': 'TENSE', 'value': 'PRES'},
    'past': {'feature': 'TENSE', 'value': 'PST'},
    'fut':  {'feature': 'TENSE', 'value': 'FUT'},
    'perf': {'feature': 'ASPECT', 'value': 'PERF'},
    'prog': {'feature': 'ASPECT', 'value': 'PROG'},

    # Mood and Voice
    'imp':  {'feature': 'MOOD', 'value': 'IMP'},
    'pass': {'feature': 'VOICE', 'value': 'PASS'},

    # Other Features
    'fem':  {'feature': 'GENDER', 'value': 'FEM'},
    'masc': {'feature': 'GENDER', 'value': 'MASC'},
    'def':  {'feature': 'DEFINITE', 'value': 'DEF'},

    # Add more mappings as needed
    'ins':  {'feature': 'CASE', 'value': 'INS'},
    'sim':  {'feature': 'ASPECT', 'value': 'SIM'},
    "1sg": [{'feature': 'PERSON', 'value': '1'}, {'feature': 'NUMBER', 'value': 'SG'}],
    "2sg": [{'feature': 'PERSON', 'value': '2'}, {'feature': 'NUMBER', 'value': 'SG'}],
    "3sg": [{'feature': 'PERSON', 'value': '3'}, {'feature': 'NUMBER', 'value': 'SG'}],
    "1pl": [{'feature': 'PERSON', 'value': '1'}, {'feature': 'NUMBER', 'value': 'PLUR'}],
    "2pl": [{'feature': 'PERSON', 'value': '2'}, {'feature': 'NUMBER', 'value': 'PLUR'}],
    "3pl": [{'feature': 'PERSON', 'value': '3'}, {'feature': 'NUMBER', 'value': 'PLUR'}],
    "loc": {'feature': 'CASE', 'value': 'LOC'},
    "1poss": [{'feature': 'PERSON', 'value': '1'}, {'feature': 'GRAM_REL', 'value': 'POSS'}],
    "2poss": [{'feature': 'PERSON', 'value': '2'}, {'feature': 'GRAM_REL', 'value': 'POSS'}],
    "3poss": [{'feature': 'PERSON', 'value': '3'}, {'feature': 'GRAM_REL', 'value': 'POSS'}],
    "foc": {'feature': 'FOCUS', 'value': 'FOC'},
    "irr": {'feature': 'MOOD', 'value': 'IRR'},
    "neg": {'feature': 'POLARITY', 'value': 'NEG'},
    "ben": {'feature': 'VOICE', 'value': 'BEN'},
    "prox": {'feature': 'DEICTIC', 'value': 'PROX'},
    "dist": {'feature': 'DEICTIC', 'value': 'DIST'},
    "nfin": {'feature': 'FINITENESS', 'value': 'NONFIN'},


}

def get_feature_set(gloss_split):
    atomic_tags = [tag.strip().lower() for tag in gloss_split[1:] if tag.strip()]

    target_feature_set = set()

    # 3. Mapping to Standardized Unimorph Pairs
    for tag in atomic_tags:
        if tag in TAG_MAPPING:
            mapping = TAG_MAPPING[tag]
            # Format the feature as 'FEATURE:VALUE', e.g., 'CASE:OBJ'
            if isinstance(mapping, list):
                for m in mapping:
                    standardized_feature = f"{m['feature']}:{m['value']}"
                    target_feature_set.add(standardized_feature)
            else:
                standardized_feature = f"{mapping['feature']}:{mapping['value']}"
                target_feature_set.add(standardized_feature)
        else:
            # Important: If a tag isn't mapped (e.g., a number '20'), log it or skip it.
            #print(f"Warning: Unmapped tag '{tag}' found in gloss.")
            continue

    return target_feature_set


# In[ ]:


import numpy as np

def select_rules(
    all_rules,
    target_pos,             
    target_feature_set,     # Local required feature/value pairs: e.g., {'CASE:OBJ', 'NUMBER:SG'}

    # --- GLOBAL AGREEMENT INPUTS (Derived from sentence-level Spacy analysis) ---
    sentence_tense=None,
    sentence_number=None,
    sentence_gender=None,
    # --------------------------------------------------------------------------

    source_sentence=None,   # Full English sentence context
    source_gloss=None,      # Word-for-word gloss context
    word_to_replace=None,   # The LRL STEM being modified (e.g., 'sor')

    top_k=10,
    w_embed=0.7,            
    w_feature=0.3,          
    rule_embeddings=None,   # Pre-computed rule embeddings (Tensors)
    model=None,             # Sentence Transformer model instance

    granularity="rule",     # 'rule', 'section', or 'book'
    sections_data=None,     # The full source text structure (for granularity='section'/'book')
):
    """
    Hybrid rule selection: combines embedding similarity of codified rules 
    and exact feature-set matching (local and global) to select top_k rules.
    """

    # 1. QUERY BUILDING (Semantic Score Input)
    target_features_str = " ".join(target_feature_set)
    global_context = f"TENSE:{sentence_tense or 'N/A'} NUMBER:{sentence_number or 'N/A'} GENDER:{sentence_gender or 'N/A'}"

    features = (
        f"CONTEXT: {source_sentence or ''} GLOSS: {source_gloss or ''} "
        f"GLOBAL AGREEMENT: {global_context} "
        f"TASK: Find the rule to modify {target_pos} '{word_to_replace or ''}' "
        f"for local features: {target_features_str}"
    ).strip()

    # Input validation (must have embeddings/model to proceed with scoring)
    if not features or rule_embeddings is None or model is None:
        print("Error: Missing features, model, or rule embeddings.")
        return []

    rule_objs = all_rules

    # 2. ENCODING & SEMANTIC SCORING (Task 2.1)
    feat_emb = model.encode(features, convert_to_tensor=True)
    sims = util.cos_sim(feat_emb, rule_embeddings)[0].cpu().numpy() # Semantic Score

    # 3. STRUCTURED FEATURE SCORING (Task 2.2)
    feature_scores = []

    # Prepare target feature map (e.g., {'CASE': 'OBJ', 'NUMBER': 'SG'})
    target_tags_map = {}
    for item in target_feature_set:
        try:
            feat, val = item.split(':')
            target_tags_map[feat.strip()] = val.strip()
        except ValueError:
            continue

    for rule in rule_objs:
        score = 0

        # Base Match Score (POS)
        if rule.get('target_pos') == target_pos:
            score += 1.0

        # Local Specific Feature Match (High-Value Match)
        rule_feature = rule.get('unimorph_feature')
        rule_value = rule.get('unimorph_value')

        # Check if the rule's local feature/value is required by the target set
        if (rule_feature in target_tags_map and 
            rule_value == target_tags_map[rule_feature]):
            score += 2.0 

        # Global Agreement Checks (Global Feature Match)
        # Assuming your codified rules have 'required_tense', 'required_gender', etc. fields

        # Tense Check
        if rule.get('required_tense') and rule.get('required_tense') == sentence_tense:
             score += 0.5

        # Number Check
        if rule.get('required_number') and rule.get('required_number') == sentence_number:
             score += 0.5

        # Gender Check
        if rule.get('required_gender') and rule.get('required_gender') == sentence_gender:
             score += 0.5

        # Normalize the score (Max possible score is 4.5: 1.0 + 2.0 + 3*0.5)
        normalized_feature_score = score / 4.5
        feature_scores.append(normalized_feature_score)

    feature_scores = np.array(feature_scores) # Structured Score

    # 4. FUSION & RANKING (Task 3)
    final_scores = (w_embed * sims) + (w_feature * feature_scores)

    top_idx = np.argsort(final_scores)[::-1][:top_k]
    top_rules = [all_rules[i] for i in top_idx]

    # 5. FORMAT OUTPUT based on Granularity (Task 4)

    if granularity == "rule":
        # Returns codified rule and code for LLM execution (standard mode)

        formatted_rules = []

        for i, rule in enumerate(top_rules):
            description = rule.get('description', 'Rule Description Not Available.')
            lrl_code = rule.get('lrl_code', 'FUNCTION ApplyRule(STEM, POS): RETURN STEM')

            formatted_rules.append(
                f"--- CANDIDATE RULE {i + 1} ---\n"
                f"Description: {description.strip()}\n"
                f"Code:\n{lrl_code.strip()}\n"
            )

        # Return the clean, single string block
        return "\n".join(formatted_rules)

    elif granularity == "section":
        # Returns source paragraph text for the rule
        if sections_data is None:
            raise ValueError("sections_data must be provided when granularity is 'section'.")

        selected_paragraphs = []
        MAX_PARA_TOKENS = 1000  # Safety ceiling per paragraph
        parallel_sentence_threshold = 2 # Max number of parallel sentence examples to keep per paragraph
        for r in top_rules:
            # 1. Get all refs for THIS specific rule
            refs = r.get("section_paragraph_ids", [])
            refs = [(int(sec_id), int(para_id)) for sec_id, para_id in refs]

            rule_candidates = []

            # 2. Find every paragraph associated with this rule and measure them
            for sec_id, para_id in refs:
                for section in sections_data:
                    if section.get("section_id") == sec_id:
                        paragraphs = section.get("paragraphs", [])
                        if 0 <= para_id < len(paragraphs):
                            text = paragraphs[para_id]
                            token_count = get_token_length(text)


                            # Find parallel sentences in the paragraph by searching for | (each parallel sentence is between | )
                            split_sentences = text.split("|")
                            start,split_sentences,end = split_sentences[0],split_sentences[1:-1],split_sentences[-1]
                            len_split_sentences = len(split_sentences)
                            if len_split_sentences > parallel_sentence_threshold:
                                # Pick the smallest sentences among them
                                smallest_sentences = sorted(split_sentences, key=lambda s: get_token_length(s))[:parallel_sentence_threshold]
                                text = start + " | ".join(smallest_sentences) + end
                            token_count = get_token_length(text)
                            # Only consider it if it's below our limit
                            if token_count <= MAX_PARA_TOKENS:
                                rule_candidates.append({
                                    "text": text,
                                    "tokens": token_count
                                })
                                break # Found the section for this ref

            # 3. Pick the winner for this rule (the smallest one)
            if rule_candidates:
                # Sort by token count ascending
                rule_candidates.sort(key=lambda x: x["tokens"])
                winner = rule_candidates[0]["text"]

                # Avoid duplicates if different rules point to the same small paragraph
                if winner not in selected_paragraphs:
                    selected_paragraphs.append(winner)

            # 4. Stop if we've filled our top_k slots
            if len(selected_paragraphs) >= top_k:
                break

        return "\n".join(selected_paragraphs)

    elif granularity == "book":
        # Returns the full source section text
        if sections_data is None:
            raise ValueError("sections_data must be provided when granularity is 'book'.")

        all_section_texts = []
        for section in sections_data:
            heading = section.get("heading", "")
            heading_level = section.get("heading_level",-1)
            if heading:
                all_section_texts.append(f"{'#' * (heading_level + 1)} {heading}")

            for paragraph in section.get("paragraphs", []):
                all_section_texts.append(paragraph)



        return "\n".join(all_section_texts)

    else:
        raise ValueError(f"Unknown granularity mode: {granularity}")


# # Driver code

# In[ ]:


filename = "kalamang.pdf"
language_name = filename.split(".pdf")[0]


print(language_name)


# In[ ]:


base_filename = Path(filename).stem

rules_filename = Path("../data/extracted_rules") / f"{base_filename}_extracted_rules_final.json"
all_rules = load_rules(rules_filename)

# Flattened list of all rules
print(f"No of rules: {len(all_rules)}")

# Load sections split
with open(f"../data/sections_split/{language_name}_sections_classified_split.json", "r", encoding="utf-8") as f:
    sections_split = json.load(f)

print(f"No of sections split loaded: {len(sections_split)}")

# Use the dynamically constructed parallel sentences filename

parallel_sentences_file = (
    Path("../data/parallel_sentences")
    / f"{base_filename}_parallel_sentences_cleaned_enriched.json"
)

with open(parallel_sentences_file, "r", encoding="utf-8") as f:
    parallel_sentences_with_pos_and_unimorph = json.load(f)

print(f"No of parallel sentences: {len(parallel_sentences_with_pos_and_unimorph)}")

# Load dictionary file based on base_filename if available, else use default path
dict_path = Path("../data/dictionary") / f"{base_filename}_dictionary.json"
dictionary = load_dictionary(dict_path)
dictionary_with_metadata_path = Path("../data/dictionary") / f"{base_filename}_dictionary_with_metadata.json"
dictionary_with_metadata = load_dictionary_with_meta(dictionary_with_metadata_path)

# TODO: Make this filtering for each POS
nouns_only_dict = filter_dict_by_pos(dictionary_with_metadata, target_pos=["NOUN", "PROPN"])
verbs_only_dict = filter_dict_by_pos(dictionary_with_metadata, target_pos=["VERB"])
adjectives_only_dict = filter_dict_by_pos(dictionary_with_metadata, target_pos=["ADJ","NUM"])
adverbs_only_dict = filter_dict_by_pos(dictionary_with_metadata, target_pos=["ADV"])
pos_dicts = {
    "NOUN": nouns_only_dict,
    "VERB": verbs_only_dict,
    "ADJ": adjectives_only_dict,
    "ADV": adverbs_only_dict,
}


# In[11]:


all_rules


# In[12]:


# Embed rules
import torch
embed_rules = True

if embed_rules == True:
    rule_texts = []
    for r in all_rules:
        # We assume 'r' is a dictionary containing the codified rule data:
        # 'description', 'lrl_code', 'target_pos', etc.
        try:
            # 1. Use the clean, synthesized description (semantic meaning)
            desc = str(r.get("description") or "")

            # 2. Use the lrl_code (functional logic/conditions)
            lrl_code = str(r.get("lrl_code") or "")

            # 3. Add explicit tags for key features (POS and Unimorph tags)
            pos_tag = f"POS:{r.get('target_pos')}"
            unimorph_tag = f"{r.get('unimorph_feature')}:{r.get('unimorph_value')}"

            # Combine all parts into a single, comprehensive text string
            text_parts = [
                desc,
                pos_tag,
                unimorph_tag,
                lrl_code
            ]

            rule_texts.append(" ".join(text_parts))

        except Exception as e:
            # Note: Since your rules are now in a standardized format, this exception should be rare.
            print(f"Error processing rule for embedding: {r.get('rule_id', 'Unknown')}: {e}")
            continue

    # Encode the list of rich text strings
    rule_embeddings = model.encode(rule_texts, convert_to_tensor=True, normalize_embeddings=True)
    # Save embeddings for future use
    torch.save(rule_embeddings, f"../models/embeddings/{Path(filename).stem}_rule_embeddings.pt")
    print(f"Saved rule embeddings shape: {rule_embeddings.shape}")
else:
    rule_embeddings = torch.load(f"../models/embeddings/{Path(filename).stem}_rule_embeddings.pt")
print(f"Rule embeddings shape: {rule_embeddings.shape}")


# In[13]:


base_prompt = f"""You are a professional linguist for {{lang}}. 
Task: Replace the {{pos}} in the sentence using provided rules. 
Output must be concise. Use fragments, not full paragraphs.

---

### Constraints
- Deduce existing clitics by comparing the original sentence to the provided stem.
- **Allomorphy**: Apply the correct variant based on the new stem's ending.
- Citation: You MUST cite Rule #s for all changes.

---

### INPUTS
- **{{granularity_statement}}**: 
{{rules_data}}

- **Sentence**: {{sentence_text}}
- **Translation**: {{translation}}
- **Target**: Replace {{stem_to_replace}} ({{original_gloss_features}}) with {{word}} ("{{word_translation}}") [POS: {{pos}}]

---

### Step-by-step reasoning (Direct & Brief)
1. **Context**: [Target word in sentence] + [Duced clitics/morphology].
2. **Rule Selection**: [Rule #s] used + [1-phrase justification]. 
3. **Transformation**: [New stem] + [Applied clitics/changes].
4. **Final Sentence**: [{{lang}} result].
5. **Translation**: [English result].

---

### Final Result
```yaml
final_sentence: "the_generated_lrl_sentence"
english_translation: "the_english_translation"
```
"""


# In[14]:


granularity_statement_template = {
    "rule": "- **Codified Rules**: Use the relevant codified rules below to infer relevant grammatical rules for the transformation.",
    "section": "- **Source Text**: Use the relevant source text paragraphs below to infer relevant grammatical rules for the transformation.",
    "book": "- **Full Grammar Book**: Use the entire source text below to infer relevant grammatical rules for the transformation."
}


# # Generate batch functions

# In[15]:


def is_replaceable_lexeme(match):
    surface = match["surface"]

    # Reject bound morphemes
    if surface.startswith("=") or surface.startswith("-"):
        return False

    # Reject very short tokens (particles, vowels)
    if len(surface) < 3:
        return False

    # Reject pure function morphology
    FUNCTIONAL_POS = {
        "CLITIC",
        "PARTICLE",
        "ADPOSITION",
        "AUX",
    }
    if match["pos"] in FUNCTIONAL_POS:
        return False

    return True


# In[16]:


def resolve_contextual_pos_for_all_matches(
    lrl_word_matches: List[Dict[str, Any]], 
    spacy_res: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Iterates through all LRL word matches and resolves their contextual POS 
    using the pre-processed spaCy results.
    """
    resolved_matches = []

    for match in lrl_word_matches:
        current_dict_pos = match["pos"]
        contextual_gloss_word = match["contextual_gloss_word"]

        # Use the existing contextual POS lookup function
        inferred_pos = get_contextual_target_pos_from_preprocessed(
            spacy_results=spacy_res, 
            target_english_gloss=contextual_gloss_word, 
            default_pos=current_dict_pos
        )

        # Add the resolved POS to the match object
        match["inferred_pos"] = inferred_pos
        resolved_matches.append(match)

    return resolved_matches

# (Ensure get_contextual_target_pos_from_preprocessed is defined somewhere accessible)


# In[ ]:


#########################################
# Main generation loop
#########################################

from typing import List, Dict, Any, Optional


pos_dict_embeddings = {
    k: None for k in pos_dicts.keys()
}

def generate_sentences(
    sentence_limit: int = 400, 
    filter_rules: bool = True, 
    VERBOSE: bool = False, 
    pos_to_replace: Optional[List[str]] = None, 
    no_of_random_nouns: int = 20, 
    sections_split: Optional[Any] = None, 
    granularity: str = "rule", 
    rule_embeddings: Optional[Any] = None, 
    top_k: int = 10
) -> None:
    """
    Generate sentences by replacing words of specified POS using grammatical rules.

    Args:
        sentence_limit (int): Maximum number of sentences to process. Set to -1 for no limit.
        filter_rules (bool): Whether to filter rules based on morphology.
        VERBOSE (bool): Whether to print detailed logs.
        pos_to_replace (List[str], optional): The POS tags targeted for replacement.
        # ... (other Args remain the same)

    Returns:
        None
    """

    # --- 1. Initialization and Setup ---
    all_batch_requests = []

    # Initialize sentence limit
    # Assuming parallel_sentences_with_pos_and_unimorph is a globally accessible list
    total_sentences = len(parallel_sentences_with_pos_and_unimorph)
    LIMIT = min(max(sentence_limit, 0), total_sentences)
    print(f"Using {LIMIT} sentences for generation out of {total_sentences}.")
    sentences_processed = 0

    # Ensure target POS list is available
    if not pos_to_replace:
        print("Error: pos_to_replace must be specified.")
        return

    debug_num = 0

    total_lrl_word_matches = 0
    multiple_lrl_word_sentences = 0

    # --- 2. Main Sentence Processing Loop ---
    skipped_parallel_sentences = []
    additional_prompts = []
    for idx, sentence in enumerate(parallel_sentences_with_pos_and_unimorph):

        if sentences_processed >= LIMIT:
            print("Reached sentence limit.")
            break

        # --- A. Extract Sentence Data ---
        #print(f"Processing sentence {idx + 1}")
        if VERBOSE:
            print(f"Selected sentence for generation {idx + 1}:\n{sentence['source']} \n{sentence['translation']} \n")

        sentence_text = sentence["source"]
        gloss_text = sentence.get("gloss", "")
        translation = sentence["translation"]
        spacy_info = sentence["spacy_info"].get("res", None) # List of token dicts
        tense = sentence["spacy_info"].get("tense", None)
        number = sentence["spacy_info"].get("number", None)
        genders = sentence["spacy_info"].get("genders", None)

        if VERBOSE:
            print(f"Tense: {tense}, Number: {number}, Genders: {genders}")

        # --- B. Identify, Resolve, and Filter LRL Words ---

        # 1. Find all candidate LRL words based on dictionary POS
        lrl_word_matches = find_lrl_words_by_pos(
            sentence_text, gloss_text, pos_dicts, target_pos=pos_to_replace
        )


        # 2. Resolve the contextual POS for ALL identified matches (Fixes 'kiem' polysemy)
        lrl_word_matches_resolved = resolve_contextual_pos_for_all_matches(
            lrl_word_matches, 
            spacy_info
        )

        # 3. Filter matches by replaceability AND correct INFERRED POS
        replaceable_lrl_words = {
            match["surface"]: match
            for match in lrl_word_matches_resolved
            if is_replaceable_lexeme(match) and match["inferred_pos"] in pos_to_replace
        }

        total_lrl_word_matches += len(replaceable_lrl_words)
        if len(replaceable_lrl_words) > 1:
            multiple_lrl_word_sentences += 1

        # Verbose printing of candidates
        lrl_words_for_print = {match["surface"]:(match["translation"], match["gloss_split"])
            for match in replaceable_lrl_words.values()
        }
        if VERBOSE:
            print(f"LRL candidates (final selection): {lrl_words_for_print} - {sentence_text}")

        if not replaceable_lrl_words:
            if VERBOSE: 
                print("No replaceable LRL words found matching target POS, skipping.")
            skipped_parallel_sentences.append(sentence)
            continue

        # --- C. Select Target Word and Extract Variables ---

        # 4. Replace all lrl words but use only one for generation (randomly)
        # Iterate through them and store the rest of the prompts incase number of sentences is not met. 
        randint = random.randint(0, len(replaceable_lrl_words) - 1)
        #print(f"Using LRL word index {randint} for generation.")
        for i,lrl_word in enumerate(replaceable_lrl_words.keys()):
            lrl_word_data = replaceable_lrl_words[lrl_word]

            # Extract variables
            lrl_word_translation = lrl_word_data["translation"]
            lrl_word_gloss_split = lrl_word_data["gloss_split"]
            target_pos_for_rule_selection = lrl_word_data["inferred_pos"] # Corrected POS

            #print(f"Selected LRL word: {lrl_word} ({target_pos_for_rule_selection}): {lrl_word_translation}")

            # --- D. Find Similar Words ---
            similar_words = get_top_k_similar_words_from_dict(
                source_word=lrl_word,
                word=lrl_word_translation,
                pos=target_pos_for_rule_selection, # Use the correct, inferred POS
                pos_dicts=pos_dicts,
                pos_dict_embeddings=pos_dict_embeddings,
                model=model,
                k=no_of_random_nouns,
            )

            similar_words = {w["lemma"]: w["metadata"]["senses"][0]["translation"] for w in similar_words}
            if VERBOSE: print(f"Similar words found: {similar_words}")

            # --- E. Rule Selection ---

            # Quick check for kalamang (moved here for efficiency, applied before rule selection)
            if base_filename == "kalamang" and target_pos_for_rule_selection == "NOUN" and lrl_word == "se":
                # Assuming 'se' is a special NOUN to be filtered in this language
                print(f"Skipping special NOUN 'se' for {base_filename}.")
                skipped_parallel_sentences.append(sentence)
                continue 

            target_feature_set = get_feature_set(lrl_word_gloss_split)

            if VERBOSE:
                print("\n###### RULE SELECTION ######")
                print(f"Target POS: {target_pos_for_rule_selection}")
                print(f"Target features: {target_feature_set}")

            rule_text = ""
            if filter_rules:
                rule_text = select_rules(
                    all_rules=all_rules,
                    target_pos=target_pos_for_rule_selection, # Use the corrected POS
                    target_feature_set=target_feature_set,
                    sentence_tense=tense,
                    sentence_number=number,
                    sentence_gender=genders,
                    source_sentence=sentence_text,
                    source_gloss=gloss_text,
                    word_to_replace=lrl_word,
                    top_k=top_k,
                    w_embed=0.7,
                    w_feature=0.3,
                    rule_embeddings=rule_embeddings,
                    model=model,
                    granularity=granularity,
                    sections_data=sections_split,
                )
            else:
                # Fallback if no filtering is requested
                rule_text = "\n".join("- " + rule["description"] for rule in all_rules)

            if VERBOSE:
                print(rule_text)
                print("----------------------------------------------\n")

            # --- F. Build Prompts for Replacement Words ---

            # Prepare POS/Morph info for the prompt (simplified, without complex token counts)
            pos_info_text = ""
            if spacy_info:
                for token in spacy_info:
                    morph_info = {k: v for k, v in token.get("morph", {}).items() if k in ["Case", "Number", "Gender"]}
                    pos_info_text += f"- {token['lemma']}: {morph_info}\n"

            if VERBOSE:
                print("--- English POS/Morphology Info ---")
                print(pos_info_text)

            for word in similar_words.keys():
                replacement_word = word
                word_translation = similar_words[word]

                if VERBOSE:
                    print(f"Generating prompt to replace '{lrl_word}' with '{replacement_word}' ({word_translation})")
                    print("With variables:")
                    print(f"  rules_data: {rule_text}")
                    print(f"  sentence_text: {sentence_text}")
                    print(f"  translation: {translation}")
                    print(f"  lang: {base_filename.split('_')[0]}")
                    print(f"  word: {replacement_word}")
                    print(f"  word_translation: {word_translation}")
                    print(f"  pos: {target_pos_for_rule_selection}")
                    print(f"  stem_to_replace: {lrl_word}")
                    print(f"  original_gloss_features: {' '.join(lrl_word_gloss_split)}")

                prompt = base_prompt.format(
                    rules_data=rule_text,
                    sentence_text=sentence_text,
                    translation=translation,
                    lang=language_name,
                    word=replacement_word,
                    word_translation=word_translation,
                    pos=target_pos_for_rule_selection,
                    granularity_statement=granularity_statement_template[granularity],
                    # === Missing variables added here ===
                    stem_to_replace=lrl_word,
                    original_gloss_features=" ".join(lrl_word_gloss_split),
                    # ==================================
                )

                tokens = get_token_length(prompt)
                #if tokens > 3000:
                #    debug_num += 1
                #    if debug_num < 30:
                #        print(f"Warning: Prompt length {tokens} exceeds 3000 tokens. Consider shortening the rules or context.")
                #        print(f"Prompt: {prompt}\n")


                if i == randint:
                    all_batch_requests.append(
                        {
                            "sentence_index": idx + 1,
                            "sentence_text": sentence_text,
                            "translation": translation,
                            "word_to_replace": lrl_word, # Original LRL word
                            "replacement_word": replacement_word, # New LRL word
                            "word_translation": word_translation,
                            "prompt": prompt,
                            "pos": target_pos_for_rule_selection,
                        }
                    )
                else:
                    additional_prompts.append(
                        {
                            "sentence_index": idx + 1,
                            "sentence_text": sentence_text,
                            "translation": translation,
                            "word_to_replace": lrl_word, # Original LRL word
                            "replacement_word": replacement_word, # New LRL word
                            "word_translation": word_translation,
                            "prompt": prompt,
                            "pos": target_pos_for_rule_selection,
                        }
                    )

            if i == randint:
                sentences_processed += 1



    #print(f"TOTAL LRL WORD MATCHES FOUND: {total_lrl_word_matches}")
    print(f"ADDITIONAL PROMPTS STORED: {len(additional_prompts)}")
    #print(f"SENTENCES WITH MULTIPLE LRL WORDS: {multiple_lrl_word_sentences}")
    print(f"SENTENCES PROCESSED: {sentences_processed}")
    # --- 3. Stage 2: SpaCy-Driven Deduction (Slot-in for Quota Fill) ---

    if sentences_processed < LIMIT and skipped_parallel_sentences:

        #print(f"\n##### STAGE 2: SpaCy-Driven Deduction (Filling Quota) #####")
        # Adding additional prompts from skipped sentences
        additional_prompt_sentences = len(additional_prompts)//no_of_random_nouns
        add_len = min(LIMIT - sentences_processed, additional_prompt_sentences)
        all_batch_requests.extend(additional_prompts[:add_len * no_of_random_nouns])
        print(f"Added {len(additional_prompts)} additional prompts from Stage 1.")
        sentences_processed += add_len

        print(f"Remaining quota: {LIMIT - sentences_processed} sentences.")

        additional_prompts = [] # Clear to free memory and add new ones
        additional_prompts_backup = []

        for idx_skip, sentence in enumerate(skipped_parallel_sentences):

            if sentences_processed >= LIMIT:
                print("Quota filled. Stopping Stage 2.")
                break

            # A. Extract Data (Repetition of extraction, but only for skipped sentences)
            sentence_text = sentence["source"]
            translation = sentence["translation"]
            spacy_info = sentence.get("spacy_info", {}).get("res", None)

            if not spacy_info: 
                # No SpaCy info available, skip
                continue

            # B. Check for target POS in SpaCy Data


            english_targets = [
                token for token in spacy_info if token["upos"] in pos_to_replace
            ]

            if not english_targets: 
                # Target POS is not found in English SpaCy, so we skip but add it to all_sentence_prompts
                random_words = get_random_words_from_dict(
                        pos_dicts=pos_dicts, 
                        pos_to_replace=pos_to_replace,
                        no_of_random_words=no_of_random_nouns,
                    )
                similar_words = {w["lemma"]: w["metadata"]["senses"][0]["translation"] for w in random_words}

                # E. Rule Selection (Features are UNKNOWN in deduction mode)
                target_feature_set = set() # Force empty set
                target_pos_for_rule_selection = pos_to_replace[0] # Just pick the first POS
                lrl_word = ""
                rule_text = select_rules(
                    all_rules=all_rules, target_pos=target_pos_for_rule_selection, 
                    target_feature_set=target_feature_set, # UNKNOWN
                    word_to_replace=lrl_word, # Placeholder
                    top_k=top_k, rule_embeddings=rule_embeddings, 
                    model=model, granularity=granularity, sections_data=sections_split,

                )

                # F. Build Prompts for Replacement Words (Using Deduction Variables)

                # Prepare POS/Morph info for the prompt
                pos_info_text = ""
                if spacy_info:
                    for token in spacy_info:
                        morph_info = {k: v for k, v in token.get("morph", {}).items() if k in ["Case", "Number", "Gender"]}
                        pos_info_text += f"- {token['lemma']}: {morph_info}\n"

                # Variables for the prompt (Stage 2 is DEDUCTION mode)
                stem_to_replace_var = "[[UNKNOWN LRL STEM: DEDUCE BASE FORM FROM CONTEXT]]"
                original_gloss_features_var = "[[UNKNOWN FEATURES: DEDUCE MORPHOLOGY FROM CONTEXT]]"
                lrl_word_for_prompt = lrl_word # Placeholder


                for word in similar_words.keys():
                    replacement_word = word
                    word_translation = similar_words[word]

                    prompt = base_prompt.format(
                        rules_data=rule_text,
                        sentence_text=sentence_text,
                        translation=translation,
                        lang=language_name,
                        word=replacement_word,
                        word_translation=word_translation,
                        pos=target_pos_for_rule_selection,
                        stem_to_replace=stem_to_replace_var, # Deduction placeholder
                        original_gloss_features=original_gloss_features_var, # Deduction placeholder
                        granularity_statement=granularity_statement_template[granularity],
                        english_pos_info=pos_info_text,
                        lrl_word_to_replace=lrl_word_for_prompt # Placeholder
                    )
                    additional_prompts_backup.append(
                            {
                                "sentence_index": sentence.get("index", idx_skip + 1),
                                "sentence_text": sentence_text,
                                "translation": translation,
                                "word_to_replace": "[[NO TARGET POS FOUND]]",
                                "replacement_word": replacement_word,
                                "word_translation": word_translation,
                                "prompt": prompt,
                                "pos": pos_to_replace,
                                "deduction_mode": True,
                            }
                        )

                continue

            # C. Synthesize Deduction Variables

            # 1. Select the English target (to guide the LLM replacement words)
            randint = random.randint(0, len(english_targets) - 1)
            prompts_generated_count = 0
            for i, token in enumerate(english_targets):
                english_target_token = token
                target_pos_for_rule_selection = english_target_token["upos"]

                # 2. Set placeholders (LRL word is UNKNOWN/to be deduced)
                lrl_word = f"[[LRL TOKEN ALIGNED TO: {english_target_token['word']}]]"
                lrl_word_translation = english_target_token["word"]

                # D. Find Similar Words (Need LRL replacements for this POS)
                similar_words = get_top_k_similar_words_from_dict(
                    source_word=lrl_word, # Placeholder
                    word=lrl_word_translation, 
                    pos=target_pos_for_rule_selection, 
                    pos_dicts=pos_dicts, 
                    pos_dict_embeddings=pos_dict_embeddings, 
                    model=model, k=no_of_random_nouns,
                )
                similar_words = {w["lemma"]: w["metadata"]["senses"][0]["translation"] for w in similar_words}

                if not similar_words: continue

                # E. Rule Selection (Features are UNKNOWN in deduction mode)
                target_feature_set = set() # Force empty set
                rule_text = select_rules(
                    all_rules=all_rules, target_pos=target_pos_for_rule_selection, 
                    target_feature_set=target_feature_set, # UNKNOWN
                    word_to_replace=lrl_word, # Placeholder
                    top_k=top_k, rule_embeddings=rule_embeddings, 
                    model=model, granularity=granularity, sections_data=sections_split,

                )

                # F. Build Prompts for Replacement Words (Using Deduction Variables)

                # Prepare POS/Morph info for the prompt
                pos_info_text = ""
                if spacy_info:
                    for token in spacy_info:
                        morph_info = {k: v for k, v in token.get("morph", {}).items() if k in ["Case", "Number", "Gender"]}
                        pos_info_text += f"- {token['lemma']}: {morph_info}\n"

                # Variables for the prompt (Stage 2 is DEDUCTION mode)
                stem_to_replace_var = "[[UNKNOWN LRL STEM: DEDUCE BASE FORM FROM CONTEXT]]"
                original_gloss_features_var = "[[UNKNOWN FEATURES: DEDUCE MORPHOLOGY FROM CONTEXT]]"
                lrl_word_for_prompt = lrl_word # Placeholder


                for word in similar_words.keys():
                    replacement_word = word
                    word_translation = similar_words[word]

                    prompt = base_prompt.format(
                        rules_data=rule_text,
                        sentence_text=sentence_text,
                        translation=translation,
                        lang=language_name,
                        word=replacement_word,
                        word_translation=word_translation,
                        pos=target_pos_for_rule_selection,
                        stem_to_replace=stem_to_replace_var, # Deduction placeholder
                        original_gloss_features=original_gloss_features_var, # Deduction placeholder
                        granularity_statement=granularity_statement_template[granularity],
                        english_pos_info=pos_info_text,
                        lrl_word_to_replace=lrl_word_for_prompt # Placeholder
                    )

                    if i == randint:
                        prompts_generated_count += 1
                        all_batch_requests.append(
                            {
                                "sentence_index": sentence.get("index", idx_skip + 1),
                                "sentence_text": sentence_text,
                                "translation": translation,
                                "word_to_replace": lrl_word,
                                "replacement_word": replacement_word,
                                "word_translation": word_translation,
                                "prompt": prompt,
                                "pos": target_pos_for_rule_selection,
                                "deduction_mode": True, # Stage 2 is DEDUCTION
                            }
                        )
                    else:
                        additional_prompts.append(
                            {
                                "sentence_index": sentence.get("index", idx_skip + 1),
                                "sentence_text": sentence_text,
                                "translation": translation,
                                "word_to_replace": lrl_word,
                                "replacement_word": replacement_word,
                                "word_translation": word_translation,
                                "prompt": prompt,
                                "pos": target_pos_for_rule_selection,
                                "deduction_mode": True, # Stage 2 is DEDUCTION
                            }
                        )



            if prompts_generated_count > 0:
                sentences_processed += 1
                if VERBOSE: print(f"✅ Processed sentence {idx_skip+1} (Stage 2). Total unique sentences: {sentences_processed}")

    sentences_processed = len(all_batch_requests)//no_of_random_nouns
    print(f"\nTotal sentences processed: {sentences_processed}, length of all_batch_requests: {len(all_batch_requests)}")
    if sentences_processed < LIMIT:
        print(f"⚠️  Warning: Only {sentences_processed} sentences processed, below the limit of {LIMIT}.")
        # Adding additional prompts if available
        remaining_quota = LIMIT - sentences_processed
        additional_prompt_sentences = len(additional_prompts)//no_of_random_nouns
        add_len = min(remaining_quota, additional_prompt_sentences)
        print(f"Add length calculated: {add_len}")
        all_batch_requests.extend(additional_prompts[:add_len * no_of_random_nouns])
        sentences_processed += add_len
        print(f"Added {len(additional_prompts)} additional prompts to fill quota after stage 2.")

    if sentences_processed < LIMIT:
        print(f"⚠️  Warning: Still below limit after adding additional prompts. Final count: {sentences_processed}/{LIMIT}.")
        # Adding random sentences from additional_prompts_backup if available
        remaining_quota = LIMIT - sentences_processed
        additional_prompt_sentences = len(additional_prompts_backup)//no_of_random_nouns
        add_len = min(remaining_quota, additional_prompt_sentences)
        all_batch_requests.extend(additional_prompts_backup[:add_len * no_of_random_nouns])

        sentences_processed += add_len
        print(f"Added {len(additional_prompts_backup)} backup prompts to fill quota. Final count: {sentences_processed}/{LIMIT}.")
    # --- 3. Final Output ---
    output_dir = Path("../data/generation_prompts") / base_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure pos_to_replace is used correctly in the filename
    pos_str = "-".join(pos_to_replace) if isinstance(pos_to_replace, list) else str(pos_to_replace)
    batch_file = Path(output_dir) / f"{base_filename}_all_batch_prompts_{pos_str}_{sentence_limit}_{granularity}.json"

    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(all_batch_requests, f, ensure_ascii=False, indent=4)

    PRINT_TOKEN_STATS = True
    if PRINT_TOKEN_STATS:
        # Get number of tokens and print a summary
        prompt_tokens_counts = [
            get_token_length(request["prompt"]) for request in all_batch_requests
        ]
        total_tokens = sum(prompt_tokens_counts)
        avg_tokens = total_tokens / len(all_batch_requests) if all_batch_requests else 0
        print(f"Total prompts: {len(all_batch_requests)}")
        print(f"Total tokens across all prompts: {total_tokens}")
        print(f"Average tokens per prompt: {avg_tokens:.2f}")

    print(f"✅ Saved {len(all_batch_requests)} prompts to {batch_file}")


# # Driver code FINAL

# In[230]:


# Load sections
sections_file = Path("../data/sections_split") / f"{base_filename}_sections_classified_split.json"
with open(sections_file, "r", encoding="utf-8") as f:
    sections = json.load(f) 

print("Loaded sections for rule selection, length:", len(sections))
print(sections[0:2])  # Print first 2 sections for verification


# In[ ]:


pos_to_replace_list = [["NOUN","PNOUN"], ["VERB"], ["ADJ","NUM"], ["ADV"]]
sentence_limit = 400
no_of_random_nouns = 20
top_k_rules = 5
granularities = ["rule","section"]

for pos_to_replace in pos_to_replace_list:
    print(f"\n=== Generating sentences for POS: {pos_to_replace} ===")
    for granularity in granularities:
        print(f"\n--- Granularity: {granularity} ---")
        generate_sentences(sentence_limit=sentence_limit, filter_rules=True, VERBOSE=False, 
                       pos_to_replace=pos_to_replace, no_of_random_nouns=no_of_random_nouns, sections_split=sections_split, granularity=granularity, 
                       rule_embeddings=rule_embeddings, top_k=top_k_rules)


# # Batching and LLM Calls

# ## Gemini batching

# In[139]:


from google import genai
import json
from tqdm import tqdm
import os
from pathlib import Path


# In[140]:


google_api_key = _require_env("GEMINI_API_KEY")


# Configure Gemini
client = genai.Client(api_key=google_api_key)
api_model = "gemini-2.5-flash"   # or "gemini-1.5-flash", etc.




# In[141]:


def create_gemini_batch_jsonl(
    all_batch_requests: dict, 
    output_dir: Path,
    base_filename: str,
    api_model: str = "gemini-2.5-flash",
) -> Path:
    """
    Creates a JSONL file formatted for the Gemini Batch API from a dictionary of prompts.

    Args:
        all_batch_requests: A dictionary where keys are unique IDs 
                            (e.g., 'NOUN_s_001_p_01') and values are the full prompt strings.
        output_dir: The directory where the JSONL file will be saved.
        base_filename: The base name for the output JSONL file.
        api_model: The model to specify (optional, but good practice).

    Returns:
        The path to the generated JSONL file.
    """
    batch_requests = []

    DEBUG = True  # Set to True to enable token length debugging
    if DEBUG:
        total_tokens = 0
    # Iterate through the dictionary of prompts
    for i,row in enumerate(all_batch_requests):
        prompt = row["prompt"]
        if DEBUG:
            tokens = get_token_length(prompt)
            #print(f"Prompt {i+1} token length: {tokens}")
            total_tokens += tokens

        # The unique_id acts as the 'key' for the batch response mapping
        unique_id = f"{base_filename}_item_{i+1}"
        item = {
            "key": unique_id,
            "method": "generateContent",  # Specify the method for the batch API
            "model": api_model,
            "request": {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }
        }
        batch_requests.append(item)
        # print(f"Processing batch item: {unique_id}") # Optional: for verbose output

    if DEBUG:
        print(f"For base filename: {base_filename}")
        print(f"Total prompts: {len(all_batch_requests)}")
        print(f"Total estimated tokens across all prompts: {total_tokens}")
        print(f"Average tokens per prompt: {total_tokens / len(all_batch_requests):.2f}")
    # --- File Saving Logic ---
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_file_path = output_dir / f"{base_filename}_gemini_batch_requests.jsonl"

    with open(batch_file_path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            # The Gemini Batch API requires each request to be a single line of JSONL
            f.write(json.dumps(req) + "\n")

    print(f"✅ Saved {len(batch_requests)} Gemini batch requests to {batch_file_path}")
    return batch_file_path


# In[142]:


# This path should point to the directory containing your saved JSON files of prompts

BASE_FILENAME =  language_name
OUTPUT_DIR = Path("../data/gemini_batch_requests") / BASE_FILENAME
DRY_RUN = False  # Set to True to skip actual file creation

prompt_files_dir = Path("../data/generation_prompts") / BASE_FILENAME


if not prompt_files_dir.exists():
    print(f"Error: Prompt directory not found at {prompt_files_dir}")
else:
    all_files = os.listdir(prompt_files_dir)

    for file in all_files:
        if "ADJ" not in file:
            continue
        # Assuming your saved prompt files are named like: 
        # "kalamang_substitution_all_batch_prompts_NOUN_400.json"
        if file.startswith(f"{BASE_FILENAME}_all_batch_prompts_") and file.endswith(".json"):

            # 1. Load the prompts dictionary
            input_file_path = prompt_files_dir / file
            print(f"Loading prompts from: {input_file_path}")
            with open(input_file_path, "r", encoding="utf-8") as f:
                # The loaded content must be a dictionary: {'unique_id': 'prompt_string'}
                all_batch_requests = json.load(f)

            # 2. Extract necessary parts for the output filename
            # The new base filename will include the POS and sentence limit for organization
            new_base_filename = file.replace(".json", "")

            # 3. Call the CORRECT function to build the Gemini JSONL file
            if not DRY_RUN:
                create_gemini_batch_jsonl(
                    all_batch_requests,
                    output_dir=OUTPUT_DIR,
                    base_filename=new_base_filename,
                    api_model=api_model,
                )
            else:
                 print(f"Dry run: Would have created batch for {new_base_filename}")


# In[143]:


all_rule_prompts_lines = []
all_section_prompts_lines = []
all_book_prompts_lines = []
prompt_files_dir = Path("../data/gemini_batch_requests") / BASE_FILENAME
for file in os.listdir(prompt_files_dir):
    if "ADV" not in file:
        continue
    if file.endswith(".jsonl"):
        input_file_path = prompt_files_dir / file
        print(f"Loading prompts from: {input_file_path}")
        with open(input_file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                if "rule" in file:
                    all_rule_prompts_lines.append(line)
                elif "section" in file:
                    all_section_prompts_lines.append(line)
                elif "book" in file:
                    all_book_prompts_lines.append(line)
print(f"Total rule prompts: {len(all_rule_prompts_lines)}")
print(f"Total section prompts: {len(all_section_prompts_lines)}")
print(f"Total book prompts: {len(all_book_prompts_lines)}")



# In[144]:


# Store combined prompts
combined_output_dir = Path("../data/gemini_batch_requests_combined") / BASE_FILENAME
combined_output_dir.mkdir(parents=True, exist_ok=True)
all_combined_prompts_lines = all_rule_prompts_lines + all_section_prompts_lines + all_book_prompts_lines
print(len(all_combined_prompts_lines))
LIMIT = -1
if LIMIT==-1:
    LIMIT = len(all_combined_prompts_lines)
with open(combined_output_dir / f"{BASE_FILENAME}_combined_prompts.jsonl", "w", encoding="utf-8") as f:
    for line in all_combined_prompts_lines[:LIMIT]:
        f.write(json.dumps(json.loads(line)) + "\n")

print(f"✅ Saved {LIMIT} combined prompts to {combined_output_dir / f'{BASE_FILENAME}_combined_prompts.jsonl'}")


# In[145]:


# import json
import time
import os
import sys
from pathlib import Path
from google import genai

# --- CONFIGURATION ---
INPUT_FILE = Path(f"../data/gemini_batch_requests_combined/{language_name}/{language_name}_combined_prompts.jsonl")
RESULTS_DIR = Path(f"../data/batch_results_optimized/{language_name}")
LOG_FILE = Path(f"{language_name}_batch_status_log.json")
RESULTS_DIR.mkdir(exist_ok=True)

TOKEN_LIMIT_PER_BATCH = 2000000 
API_KEY = _require_env("GEMINI_API_KEY")

MODEL_ID = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY)

# --- UTILITIES ---
def load_log():
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def update_log(chunk_id, status, job_name=None, error=None):
    log = load_log()
    log[str(chunk_id)] = {
        "status": status, 
        "job_name": job_name, 
        "error": str(error) if error else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=4)

def estimate_tokens(text):
    # Accurate enough for grouping; Gemini uses Byte Pair Encoding
    return len(text) // 4 

def get_optimized_chunks(input_path, limit):
    print("📏 Analyzing file to calculate optimal token chunks...")
    chunks = []
    current_chunk = []
    current_tokens = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt_text = data['request']['contents'][0]['parts'][0]['text']
                prompt_tokens = estimate_tokens(prompt_text)
                if current_tokens + prompt_tokens > limit:
                    chunks.append(current_chunk)
                    current_chunk = [line]; current_tokens = prompt_tokens
                else:
                    current_chunk.append(line); current_tokens += prompt_tokens
            except Exception: continue
        if current_chunk: chunks.append(current_chunk)
    return chunks

# --- CORE LOGIC ---
def process_batch(chunk_lines, chunk_idx):
    log = load_log()
    chunk_id = str(chunk_idx)

    if log.get(chunk_id, {}).get("status") == "COMPLETED" and (RESULTS_DIR / f"results_{chunk_id}.jsonl").exists():
        print(f"✅ [Chunk {chunk_id}] Already completed with existing results. Skipping.")
        return


    job_name = log.get(chunk_id, {}).get("job_name")

    if not job_name:
        temp_file = f"temp_{chunk_id}.jsonl"
        with open(temp_file, 'w', encoding='utf-8') as f: f.writelines(chunk_lines)

        # --- NEW RETRY LOGIC FOR SUBMISSION ---
        while True:
            try:
                print(f"📤 [Chunk {chunk_id}] Attempting Upload...")
                uploaded = client.files.upload(file=temp_file, config={'mime_type': 'application/jsonl'})

                job = client.batches.create(
                    model=MODEL_ID, 
                    src=uploaded.name,
                    config={'display_name': f"Kalamang_Part_{chunk_id}"}
                )
                job_name = job.name
                update_log(chunk_id, "RUNNING", job_name)
                os.remove(temp_file)
                break # Exit the retry loop once successful

            except Exception as e:
                if "429" in str(e):
                    print(f"🛑 Quota Full (429). Waiting 15 minutes to retry Chunk {chunk_id}...")
                    time.sleep(900) # Wait 15 mins before trying to enqueue again
                else:
                    print(f"❌ Unexpected Error: {e}")
                    return # Exit if it's a different kind of error

    print(f"⏳ [Chunk {chunk_id}] Monitoring Job: {job_name}")

    wait_time = 600 # Start polling every 10 minutes
    while True:
        try:
            status = client.batches.get(name=job_name)
            state = status.state.name

            if state == 'JOB_STATE_SUCCEEDED':
                print(f"✅ [Chunk {chunk_id}] Success! Downloading results...")
                content = client.files.download(file=status.dest.file_name)
                with open(RESULTS_DIR / f"results_{chunk_id}.jsonl", 'wb') as f:
                    f.write(content)
                update_log(chunk_id, "COMPLETED", job_name)
                break

            elif state in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']:
                print(f"🛑 [Chunk {chunk_id}] Job ended with state: {state}")
                update_log(chunk_id, state, job_name, error=status.error)
                break

            # Reset wait time on successful poll
            wait_time = 600 

        except Exception as e:
            print(f"⚠️ Connection lost/Server busy: {e}. Retrying in {wait_time}s...")
            wait_time = min(wait_time * 2, 600) # Exponential backoff up to 10 minutes

        time.sleep(wait_time)

# --- MAIN ---
chunks = get_optimized_chunks(INPUT_FILE, TOKEN_LIMIT_PER_BATCH)
print(f"🚀 Total Chunks: {len(chunks)}")


# In[146]:


def load_completed_chunks():
    """Reads the log and returns a set of IDs that are already COMPLETED."""
    if not LOG_FILE.exists():
        return set()

    with open(LOG_FILE, 'r') as f:
        try:
            log_data = json.load(f)
            # Filter for chunks marked as 'COMPLETED'
            return {chunk_id for chunk_id, info in log_data.items() 
                    if info.get("status") == "COMPLETED"}
        except json.JSONDecodeError:
            return set()


# In[147]:


def delete_job(job_name):
    """Deletes a Gemini batch job by its name."""
    try:
        # To Cancel a running/pending job
        client.batches.cancel(name=job_name)

        client.batches.delete(name=job_name)
        print(f"🗑️ Deleted job: {job_name}")
    except Exception as e:
        print(f"⚠️ Failed to delete job {job_name}: {e}")


# In[148]:


completed_ids = load_completed_chunks()
print(f"📜 Found {len(completed_ids)} completed chunks in log. They will be skipped.")


# In[149]:


import json
from pathlib import Path
# Read log file
LOG_FILE = Path(f"{language_name}_batch_status_log.json")
with open(LOG_FILE, 'r') as f:
    log_data = json.load(f)

#for row in log_data.values():
#    if row["job_name"]:
#        print(f"🗑️ Cleaning up RUNNING job: {row['job_name']}")
#        delete_job(row["job_name"])


# In[150]:


for idx, chunk_lines in enumerate(chunks):
    chunk_id = str(idx)

    # --- SKIP LOGIC ---
    completed_ids =[]
    if chunk_id in completed_ids:

        # Check if the result file actually exists on disk before skipping
        result_file = Path(f"{RESULTS_DIR}/results_{chunk_id}.jsonl")
        if result_file.exists():
            print(f"✅ Chunk {chunk_id} already completed with existing results. Skipping.")
            continue
        else:
            print(f"⚠️ Chunk {chunk_id} marked COMPLETED but file is missing. Re-running.")

    # Process the batch as normal
    process_batch(chunk_lines, idx)

    # 300-second breather
    print(f"🔋 Cooling down 5m after Chunk {chunk_id}...")
    time.sleep(300)


# ## Cleaning

# In[129]:


import re

def extract_from_edge_cases(lines_cleaned):
    source = None
    translation = None

    # Priority keys
    source_keys = ['final_sentence', 'final sentence', 'final output', 'source', 'sentence','final fragment', 'result']
    trans_keys = ['english_translation', 'translation']

    lines = [l.strip() for l in lines_cleaned if l.strip()]

    # 1. Extraction Loop
    for i, line in enumerate(lines):
        line_lower = line.lower()

        # --- Extract Source ---
        if not source:
            for key in source_keys:
                if key in line_lower and ":" in line:
                    content = "".join(line.split(":", 1)[1:]).strip()
                    if (not content or content == "*") and i + 1 < len(lines):
                        content = lines[i+1]

                    potential_s = content.strip('"').strip('`').strip('*').strip()

                    # --- VALIDATION STEP ---
                    # If the content looks like English reasoning, skip it and keep looking
                    if "original sentence" in potential_s.lower():
                        continue
                    english_indicators = ['modifies', 'modifying','context','clitic','inflection','ADJ','NOUN','VERB','ADV', 'replace']
                    is_reasoning = any(word in potential_s.lower() for word in english_indicators)

                    if not is_reasoning or len(potential_s.split()) < 20: # Sentences are usually shorter than reasoning
                        source = potential_s
                        break
                elif key in line_lower and "*" in line:
                    content = line.split("*", 1)[-1].strip()
                    potential_s = content.strip('"').strip('`').strip('*').strip()

                    # --- VALIDATION STEP ---
                    english_indicators = ['modifies', 'modifying','context','clitic','inflection','ADJ','NOUN','VERB','ADV', 'replace']
                    is_reasoning = any(word in potential_s.lower() for word in english_indicators)

                    if not is_reasoning or len(potential_s.split()) < 20:
                        source = potential_s
                        break

        # --- Extract Translation ---
        if not translation:
            for key in trans_keys:
                if key in line_lower and ":" in line:
                    content = "".join(line.split(":", 1)[1:]).strip()
                    if (not content or content == "*") and i + 1 < len(lines):
                        content = lines[i+1]
                    translation = content.strip('"').strip('`').strip('*').strip()
                    break
                elif key in line_lower and "*" in line:
                    content = line.split("*", 1)[-1].strip()
                    translation = content.strip('"').strip('`').strip('*').strip()
                    break

    # 2. Post-Process Cleanup
    if source:
        # Remove markdown bolding and Rule citations
        source = source.replace("**", "")
        source = re.sub(r"[\(\[]?Rule\s*#?\d+[\)\]]?", "", source)
        # Remove trailing ellipses often used in reasoning
        source = source.replace('[...].', '').replace('[...]', '').strip()


    return source, translation


# In[130]:


import re
import yaml

def get_final_result(generated_text):
    extracted_yaml = ""
    parsed_yaml = None

    # 1. Try Standard YAML Block (Your existing logic)
    yaml_start = generated_text.rfind("```yaml")
    if yaml_start != -1:
        yaml_text = generated_text[yaml_start + len("```yaml"):].strip()
        yaml_end = yaml_text.rfind("```")
        if yaml_end != -1:
            yaml_text = yaml_text[:yaml_end].strip()
        extracted_yaml = yaml_text
        try:
            parsed_yaml = yaml.safe_load(yaml_text)
        except:
            pass

    if parsed_yaml is not None:
        if parsed_yaml is not isinstance(parsed_yaml, dict):
            parsed_yaml = None  # Discard if not a dict
        else:
            keys_str = ", ".join(parsed_yaml.keys())
            if "reasoning" in keys_str.lower() or "explanation" in keys_str.lower():
                parsed_yaml = None  # Discard if reasoning/explanation is present
    if not parsed_yaml: 
        # Split into lines
        # Clean text from ``` markers and empty lines
        generated_text = generated_text.replace("```", "")
        # Remove text between [] 


        lines = generated_text.splitlines()
        lines = [line for line in lines if line.strip() != '']
        lines_cleaned = []
        prev_line = ""
        for line in lines:
            # Remove markdown code block markers
            if line.endswith(":") or line.endswith("*"):
                prev_line += line + " "
                continue
            elif line.lower().endswith("translation") or line.lower().endswith("sentence"):
                prev_line += ":" + line + " "
                continue
            else:
                if prev_line:
                    lines_cleaned.append(prev_line + line)
                    prev_line = ""
                else:
                    lines_cleaned.append(line)


        source = None
        translation = None
        # 2. Edge Case Extraction
        source, translation = extract_from_edge_cases(lines_cleaned)
        if source and translation:
            parsed_yaml = {"source": source, "translation": translation}
            extracted_yaml = f"source: {source}\ntranslation: {translation}"
        elif source and not translation:
            parsed_yaml = {"source": source}
            extracted_yaml = f"source: {source}"
            #print(f"Translation missing in lines: {lines_cleaned}")
        elif translation and not source:
            parsed_yaml = {"translation": translation}
            extracted_yaml = f"translation: {translation}"
            #print(f"Source missing in lines: {lines_cleaned}")

        else:
            #print(lines_cleaned)
            pass


    # 3. Final normalization for keys
    if parsed_yaml and isinstance(parsed_yaml, dict):
        # Ensure we return 'source' and 'translation' regardless of what the LLM called them
        if 'final_sentence' in parsed_yaml:
            parsed_yaml['source'] = parsed_yaml.pop('final_sentence')
        if 'english_translation' in parsed_yaml:
            parsed_yaml['translation'] = parsed_yaml.pop('english_translation')

    return extracted_yaml, parsed_yaml


# In[131]:


def hardcoded_extraction_adj_400_section(generated_text):
    # Find "Final output" and get two lines after that
    # Remove double \n if present
    generated_text = generated_text.replace("\n\n", "\n")
    keys = ["final output", "generated sentence", "final sentence","final new sentence","final generated sentence","resulting sentence", "translation", "output"]
    lines = generated_text.splitlines()

    final_output_index = -1
    parsed_yaml = {}
    for i, line in enumerate(lines):
        if any(key in line.lower() for key in keys):
            final_output_index = i
            break

    if final_output_index == -1:
        return None  # Not found
    # Remove any sentences with the key words
    final_lines = lines[final_output_index + 1:]

    final_lines = [line for line in final_lines if not any(key in line.lower() for key in keys)]
    #print(final_lines[-2:])

     # Extract the two lines after "Final output"
    source = final_lines[-2].strip() if len(final_lines) >= 2 else None
    translation = final_lines[-1].strip() if len(final_lines) >= 1 else None


    if source:
        parsed_yaml['source'] = source
    if translation:
        parsed_yaml['translation'] = translation
    return parsed_yaml


# In[132]:


def mandan_ADV_parse(generated_text):
    # Look for lines starting with "Source:" and "Translation:"
    source = None
    translation = None
    lines = generated_text.splitlines()
    for line in lines:
        if line.lower().startswith("source:"):
            source = line.split(":", 1)[1].strip()
        elif line.lower().startswith("translation:"):
            translation = line.split(":", 1)[1].strip()

    if source or translation:
        parsed_yaml = {}
        if source:
            parsed_yaml['source'] = source
        if translation:
            parsed_yaml['translation'] = translation
        return parsed_yaml
    return None


# In[137]:


failures = 0
incomplete_source = 0
incomplete_translation = 0
success = 0 

index = 0

result_dict = {}
language_name = "mandan"
RESULTS_DIR = Path(f"../data/batch_results_optimized/{language_name}")
print(f"\n--- Parsing results from directory: {RESULTS_DIR} ---")

for file in os.listdir(RESULTS_DIR):
    if file.endswith(".jsonl"):
        input_file_path = RESULTS_DIR / file
        #print(f"\n--- Sample results from: {input_file_path} ---")
        with open(input_file_path, "r", encoding="utf-8") as f:

            for i, line in enumerate(f):

                item = json.loads(line)
                key = item.get("key", "N/A")
                base_file = key.split("_item_")[0]

                if "ADJ" not in base_file:
                    continue

                if base_file not in result_dict:
                    result_dict[base_file] = []
                generated_text = item.get("response", {}).get("candidates", [{}])[0].get("content", "").get("parts", [{}])[0].get("text", "")
                #print(f"\nResponse Text:\n{generated_text}\n")
                #print(f"\nResult Key: {key}"
                if base_file == "sursilvan_romansh_all_batch_prompts_ADJ_400_section":
                    parsed_yaml = hardcoded_extraction_adj_400_section(generated_text)
                    if not parsed_yaml:
                        failures += 1
                        if random.random() < 0.1:  # Print 10% of failures for review
                            print(f"Failed to parse YAML from Text (hardcoded) :\n{generated_text}\n")

                else:
                    extracted_yaml, parsed_yaml = get_final_result(generated_text)
                    if parsed_yaml is None:
                        failures += 1
                        if random.random() < 0.1:  # Print 10% of failures for review
                            print(f"Failed to parse YAML from Text:\n{generated_text}\n")

                        #print(f"Could not parse Text:\n{generated_text}\n")

                if parsed_yaml and ("source" not in parsed_yaml or "translation" not in parsed_yaml):
                    if "source" not in parsed_yaml:
                        incomplete_source += 1
                    if "translation" not in parsed_yaml:
                        incomplete_translation += 1
                    if random.random() < 0.1:  # Print 10% of incomplete cases for review
                        missing_keys = []
                        if "source" not in parsed_yaml:
                            missing_keys.append("source")
                        if "translation" not in parsed_yaml:
                            missing_keys.append("translation")
                        print(f"Incomplete parse (missing {', '.join(missing_keys)}) from Text:\n{generated_text}\nParsed YAML:\n{parsed_yaml}\n")
                elif parsed_yaml:
                    success += 1
                    parsed_yaml["index"] = index

                    result_dict[base_file].append(parsed_yaml)
                    if random.random() < 0.05:  # Print 5% of successful cases for review
                        pass
                index += 1
                        #print(f"Successful parse from Text:\n{parsed_yaml}\n")
                    #print(f"Parsed YAML:\n{parsed_yaml}\n")


print(f"\nTotal failures to parse: {failures}")    
print(f"Total incomplete parses (missing source): {incomplete_source}")
print(f"Total incomplete parses (missing translation): {incomplete_translation}")
print(f"Total successful parses: {success}")


# In[134]:


# Remove duplicates based on 'source' and 'translation' for each base_file
for base_file, entries in result_dict.items():
    unique_entries = {}
    for entry in entries:
        try:
            key = (entry.get("source", "").strip(), entry.get("translation", "").strip())
            if key not in unique_entries:
                unique_entries[key] = entry
        except Exception as e:
            print(f"Error processing entry for deduplication: {e}")
    result_dict[base_file] = list(unique_entries.values())[:8000]  # Limit to first 8000 unique entries
    print(f"After deduplication, {base_file} has {len(result_dict[base_file])} unique entries.")


# In[135]:


for key in result_dict:
    print(key,len(result_dict[key]))


# In[119]:


lemmas = [5,10,15,20]

for key in result_dict:

    result_dict_batched = {key+f"_batch_{l}": [] for l in lemmas}
    for row in result_dict[key]:
        idx = row["index"]
        temp = idx%20

        for l in lemmas:
            if temp < l:
                result_dict_batched[key+f"_batch_{l}"].append(row)

    for batch_key in result_dict_batched:
        output_file = Path(f"../data/generated_results_batched/{language_name}/{batch_key}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_dict_batched[batch_key], f, ensure_ascii=False, indent=4)
        print(f"✅ Saved parsed results to {output_file} with {len(result_dict_batched[batch_key])} entries.")


# In[120]:


# Store them in separate files
output_dir = RESULTS_DIR / "cleaned_final_results"
output_dir.mkdir(exist_ok=True)
for base_file in result_dict:
    output_file = output_dir / f"{base_file}_final_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict[base_file], f, ensure_ascii=False, indent=4)
    print(f"✅ Saved final results for {base_file} to {output_file}")


# ### Individual batch process (test generation)

# In[215]:


language_name = "mandan"


# In[ ]:





# In[216]:


from google import genai
import json
from pathlib import Path

BASE_FILENAME = language_name
combined_output_dir = Path(f"../data/gemini_finetune_data/{language_name}")
file = combined_output_dir / f"test_inference.jsonl"

print(f"Loading prompts from: {file}")

with open(file, "r", encoding="utf-8") as f:
    all_batch_requests = [json.loads(line) for line in f]
print(f"Total prompts loaded: {len(all_batch_requests)}")


# In[217]:


for row in all_batch_requests:
    prompt = row["request"]["contents"][0]["parts"][0]["text"]
    prompt_addition = "\nProvide the final English translation in YAML format as shown:\n```yaml\ntranslation: <english translation>\n```"
    prompt = prompt + prompt_addition
    # Here you can process each prompt as needed
    print(f"Prompt: {prompt}\n")  # Print first 100 characters of each prompt
    row["request"]["contents"][0]["parts"][0]["text"] = prompt

# Save modified prompts back to a new JSONL file
output_file = combined_output_dir / f"test_inference_baseline.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in all_batch_requests:
        f.write(json.dumps(row) + "\n")


# In[218]:


google_api_key = _require_env("GEMINI_API_KEY")


# Configure Gemini
client = genai.Client(api_key=google_api_key)
api_model = "gemini-2.5-flash"   # or "gemini-1.5-flash", etc.




# In[219]:


uploaded_file = client.files.upload(
    file=str(output_file),
    config=genai.types.UploadFileConfig(display_name=f"{language_name}_gemini_batch_requests", mime_type='jsonl')
)

print(f"Uploaded file: {uploaded_file.name}")


# In[ ]:





# In[220]:


# Assumes `uploaded_file` is the file object from the previous step
client = genai.Client(api_key=google_api_key)
file_batch_job = client.batches.create(
    model="gemini-2.5-flash",
    src=uploaded_file.name,
    config={
        'display_name': f"{language_name}_sentence_gen_batch_job_test",
    },
)

print(f"Created batch job: {file_batch_job.name}")


# In[221]:


import time

batch_job = client.batches.get(name=file_batch_job.name)

completed_states = set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
])

print(f"Polling status for job: {file_batch_job.name}")
batch_job = client.batches.get(name=file_batch_job.name) # Initial get
while batch_job.state.name not in completed_states:
  print(f"Current state: {batch_job.state.name}")
  time.sleep(180) # Wait for 3 minutes before polling again
  batch_job = client.batches.get(name=file_batch_job.name)

print(f"Job finished with state: {batch_job.state.name}")
if batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")


# In[222]:


if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

    # If batch job was created with a file
    if batch_job.dest and batch_job.dest.file_name:
        # Results are in a file
        result_file_name = batch_job.dest.file_name
        print(f"Results are in file: {result_file_name}")

        print("Downloading result file content...")
        file_content = client.files.download(file=result_file_name)
        # Process file_content (bytes) as needed
        print(file_content.decode('utf-8'))

    # If batch job was created with inline request
    # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
    elif batch_job.dest and batch_job.dest.inlined_responses:
        # Results are inline
        print("Results are inline:")
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            print(f"Response {i+1}:")
            if inline_response.response:
                # Accessing response, structure may vary.
                try:
                    print(inline_response.response.text)
                except AttributeError:
                    print(inline_response.response) # Fallback
            elif inline_response.error:
                print(f"Error: {inline_response.error}")
    else:
        print("No results found (neither file nor inline).")
else:
    print(f"Job did not succeed. Final state: {batch_job.state.name}")
    if batch_job.error:
        print(f"Error: {batch_job.error}")


# In[223]:


# Save file content to local file
with open(f"scratch/{language_name}_gemini_batch_results_sentence_gen.txt", "wb") as f:
    f.write(file_content)


# In[224]:


with open(f"scratch/{language_name}_gemini_batch_results_sentence_gen.txt", "r", encoding="utf-8") as f:

    file_content = f.read()



# In[225]:


gemini_generated_text = []
generated_tokens = []
for idx, res in enumerate(file_content.splitlines()):
    res = json.loads(res)
    response = res["response"]
    key = res["key"]
    print(key)
    base_filename, item_id = key.split("_sent_")
    print(f"Processing result for file, item: {base_filename}, {item_id}")
    text = response.get("candidates")[0].get("content").get("parts")[0].get("text")

    gemini_generated_text.append({
        "item_id": item_id,
        "base_filename": base_filename,
        "generated_text": text
    })
    #print(f"Result {idx}")
    print(text)



# In[226]:


from pathlib import Path
output_dir = Path(f"../data/gemini_finetune_data/{language_name}/test_inference_results_baseline")
output_dir.mkdir(parents=True, exist_ok=True)

result_data = []
# Test file cleaning
for item in gemini_generated_text:
    parsed_yaml = None
    text = item["generated_text"]
    # Simple cleanup: get yaml block if present
    yaml_start = text.rfind("```yaml")
    if yaml_start != -1:
        yaml_text = text[yaml_start + len("```yaml"):].strip()
        yaml_end = yaml_text.rfind("```")
        if yaml_end != -1:
            yaml_text = yaml_text[:yaml_end].strip()
        cleaned_text = yaml_text
        try:
            parsed_yaml = yaml.safe_load(yaml_text)
            if isinstance(parsed_yaml, dict):
                print("YAML block found and parsed successfully.")
            else:
                print("YAML block found but is not a dictionary, storing raw YAML.")
                cleaned_text = "uncleaned_output: |\n" + text.strip()
        except Exception as e:
            print(f"Error parsing YAML: {e}, storing raw YAML.")
            cleaned_text = "uncleaned_output: |\n" + text.strip()
    else:
        print("No YAML block found, storing uncleaned output.")
        print(text.strip())
        cleaned_text = "uncleaned_output: |\n" + text.strip()
    item_id = item["item_id"]
    base_filename = item["base_filename"]
    result_data.append({
        "item_id": item_id,
        "base_filename": base_filename,
        "parsed_yaml": parsed_yaml,
        "cleaned_text": cleaned_text
    })

output_file = output_dir / f"{base_filename}_test_inference_results_cleaned.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)

print(f"✅ Saved cleaned results to {output_file}")


# In[ ]:





# In[196]:


import yaml

all_processed_data = {}

for row in gemini_generated_text:
    #print(f"Item ID: {row['item_id']}")
    base_filename = row['base_filename']
    #print(f"Base Filename: {base_filename}")
    if base_filename not in all_processed_data:
        all_processed_data[base_filename] = []
    #print(f"Generated Text: {row['generated_text']}\n")
    # Try to find yaml block at the end
    generated_text = row['generated_text']
    yaml_start = generated_text.find("```yaml")
    if yaml_start != -1:
        yaml_text = generated_text[yaml_start + len("```yaml"):].strip()
        yaml_end = yaml_text.find("```")
        if yaml_end != -1:
            yaml_text = yaml_text[:yaml_end].strip()
        #print(f"Extracted YAML:\n{yaml_text}\n")
        row["extracted_yaml"] = yaml_text
        # Parse YAML
        try:
            yaml_data = yaml.safe_load(yaml_text)
            #print(f"Parsed YAML Data:\n{yaml_data}\n")
            row["parsed_yaml"] = yaml_data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}\n")
    else:
        print("No YAML block found in the generated text.\n")
        # Try to see what last two lines are
        last_2_lines = generated_text.strip().splitlines()[-2:]
        if "translation" in last_2_lines[1].lower():
            source = last_2_lines[0].split(":", 1)[1].strip()
            translation = last_2_lines[1].split(":", 1)[1].strip()
            yaml_data = {
                "source": source,
                "translation": translation
            }
            row["parsed_yaml"] = yaml_data
            print(f"Extracted source and translation from last two lines:\n{yaml_data}\n")
        else:
            print("Could not extract source and translation from last two lines.\n")

    all_processed_data[base_filename].append(row)




# In[67]:


append = False

for base_filename in all_processed_data:
    print(f"Saving processed data for {base_filename}...")
    rows = all_processed_data[base_filename]
    output_file = Path(f"scratch/gemini_generated_sentences/{base_filename}_gemini_generated_sentences.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [row["parsed_yaml"] for row in rows if "parsed_yaml" in row]
    print(f"Total parsed YAML entries to save: {len(rows)}")


    if output_file.exists() and append:
        print(f"Warning: appending existing file {output_file}")
        # Load existing data
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        rows = existing_data + rows


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved {len(rows)} entries to {output_file}")


# 

# ## Openai BATCHING

# In[227]:


from pathlib import Path
import json
from tqdm import tqdm

def build_gemini_batch_jsonl(
    all_batch_requests,
    output_dir="./outputs",
    base_filename="no_filename_provided",
    model="gemini-2.5-flash",
):
    """
    Converts all_batch_requests into Gemini batch-compatible .jsonl file
    """
    batch_lines = []
    for i, req in enumerate(tqdm(all_batch_requests, desc="Building Gemini batch JSONL")):
        prompt = req["prompt"]
        entry = {
            "key": f"{base_filename}_{i+1}",
            "request": {
                "contents": [{"parts": [{"text": prompt}]}]
            },
        }
        batch_lines.append(entry)

    batch_path = Path(output_dir) / f"{base_filename}_gemini_batch.jsonl"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(batch_path, "w", encoding="utf-8") as f:
        for line in batch_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Saved Gemini batch file with {len(batch_lines)} requests to {batch_path}")
    return batch_path


# In[ ]:


import os
all_batch_requests_file = Path("generation_prompts") / base_filename 

all_files = os.listdir(all_batch_requests_file)
for file in all_files:
    if file.startswith(f"{base_filename}_all_batch_prompts_") and file.endswith(".json"):
        with open(all_batch_requests_file / file, "r", encoding="utf-8") as f:
            all_batch_requests = json.load(f)
        pos_part = file.split("_all_batch_prompts_")[1].rsplit("_",1)[0]
        sentence_limit_part = file.rsplit("_",1)[1].replace(".json","")
        build_gemini_batch_jsonl(
            all_batch_requests,
            output_dir=Path("generation_prompts") / base_filename,
            base_filename=file.replace(".json", ""),
        )


# In[ ]:


granularity


# In[118]:


json1_files = []


# In[119]:


# Check jsonl and print number of lines
jsonl_files = os.listdir(all_batch_requests_file)
jsonl_files = [f for f in jsonl_files if f.endswith(".jsonl")]
jsonl_files = [f for f in jsonl_files if granularity in f]
jsonl_files = [f for f in jsonl_files if "part" not in f]  # Exclude partial files
print(f"Found {len(jsonl_files)} jsonl files with granularity '{granularity}': {jsonl_files}")
for file in jsonl_files:
    if file.endswith(".jsonl"):
        with open(all_batch_requests_file / file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"File {file} has {len(lines)} lines.")
        token_length = get_token_length(lines[0])
        print(f"Estimated token length of first prompt: {token_length} tokens.")
        total_character_length = sum(len(line) for line in lines)
        print(f"Total character length of file: {total_character_length} characters.")
        print(f"------------------------------------")


# In[93]:


# show an example of a prompt
example_file = jsonl_files[0]
with open(all_batch_requests_file / example_file, "r", encoding="utf-8") as f:
    first_line = f.readline()
example_prompt = json.loads(first_line)["body"]["messages"][0]["content"]
example_prompt_token_length = get_token_length(example_prompt)
print(f"Example prompt from {example_file} has {example_prompt_token_length} tokens.")
print("Example prompt content:")
print(example_prompt)


# In[97]:


# Delete all temp files with _part in the filename
for file in os.listdir(all_batch_requests_file):
    print(file)
    if "_part" in file:
        os.remove(all_batch_requests_file / file)
        print(f"Deleted temporary file: {file}")


# In[56]:


# Split each file into N line chunks if larger than N lines
N = 800

TOKENLIMIT = 2000000  # 2 million tokens approx
for file in jsonl_files:
    if file.endswith(".jsonl"):
        if granularity not in file:
            continue
        with open(all_batch_requests_file / file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > N:
            chunk_size = N
            num_chunks = (len(lines) + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_lines = lines[i*chunk_size:(i+1)*chunk_size]
                chunk_file = all_batch_requests_file / f"{file.replace('.jsonl','')}_part{i+1}.jsonl"
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    cf.writelines(chunk_lines)
                # Check number of tokens in chunkfile
                total_tokens = sum(get_token_length(line) for line in chunk_lines)
                if total_tokens > TOKENLIMIT:
                    print(f"⚠️  Warning: Chunk file {chunk_file} exceeds token limit with {total_tokens} tokens.")
                print(f"Created chunk file {chunk_file} with {len(chunk_lines)} lines and {total_tokens} tokens.")
            # Optionally remove the original large file



# In[57]:


import os
import time
from pathlib import Path
from openai import OpenAI

# --- Load API key ---
client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

# --- Folder with batch .jsonl files ---
batch_folder = Path("./generation_prompts/kalamang")

# --- Output file to store batch IDs ---
batch_log_path = Path("./outputs/batch_job_ids.txt")
batch_log_path.parent.mkdir(parents=True, exist_ok=True)

# --- Batch settings ---
COMPLETION_WINDOW = "24h"
POLL_INTERVAL = 10 * 60  # 10 minutes
CONFIRM_SUBMIT = True  # Set to True to actually submit batches`

# --- Prepare log file ---
with open(batch_log_path, "w", encoding="utf-8") as log:
    log.write("# Batch Jobs Created\n")

# --- Helper function to wait for a batch to finish ---
def wait_for_completion(client, batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"🔍 Batch {batch_id} status: {batch.status}")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            return batch.status
        print(f"⏳ Waiting {POLL_INTERVAL/60} minutes...")
        time.sleep(POLL_INTERVAL)

# --- Main loop ---
for file in sorted(os.listdir(batch_folder)):
    if file.endswith(".jsonl") and "_part" in file:
        batch_path = batch_folder / file
        with open(batch_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)

        print(f"📦 {file}: {line_count} requests")

        if not CONFIRM_SUBMIT:
            print("⚠️ Dry run mode — skipping upload.")
            continue

        # --- Upload and create batch ---
        uploaded = client.files.create(file=open(batch_path, "rb"), purpose="batch")
        batch_job = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window=COMPLETION_WINDOW,
        )

        print(f"🚀 Submitted batch job {batch_job.id} for {file}")
        print(f"🕒 Initial status: {batch_job.status}\n")

        # --- Save to log ---
        with open(batch_log_path, "a", encoding="utf-8") as log:
            log.write(f"{file}\t{batch_job.id}\t{line_count} requests\n")

        # --- Wait until it finishes ---
        final_status = wait_for_completion(client, batch_job.id)
        print(f"✅ Batch {batch_job.id} finished with status: {final_status}\n")

print("🎉 All batch jobs processed sequentially!")
print(f"🧾 Log saved at: {batch_log_path}")


# In[ ]:





# In[78]:


# Find out the token length of a prompt
prompt = "Hello, how are you?"
token_length = get_token_length(prompt)
print(f"Token length of prompt: {token_length}")


# # Batch process test

# In[ ]:


import os
import openai
import requests

def send_openai_batch_request(batch_file_path, api_key):
    """
    Sends a batch request to OpenAI using the provided batch file.
    Args:
        batch_file_path (str): Path to the batch .jsonl file.
        api_key (str): OpenAI API key.
    Returns:
        dict: Response from the OpenAI API.
    """
    url = "https://api.openai.com/v1/batches"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/jsonl",
    }

    with open(batch_file_path, "r", encoding="utf-8") as f:
        batch_data = f.read()

    response = requests.post(url, headers=headers, data=batch_data)

    if response.status_code == 200:
        print("✅ Batch request sent successfully.")
        return response.json()
    else:
        print(f"❌ Failed to send batch request. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return None


# In[ ]:


# Batch send example (online, non-batch)
from google import genai as _genai

_client = _genai.Client(api_key=_require_env("GEMINI_API_KEY"))

results = []
for batch_request in all_batch_requests[20:40]:  # Send only first 20 for testing
    _response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=batch_request["prompt"],
    )
    print(f"Response for word '{batch_request['word']}':")
    print(_response.text)
    results.append({
        "word": batch_request["word"],
        "response": _response.text,
    })

print(f"✅ Completed sending {len(results)} requests.")


# In[ ]:


# Parse and save results
import os


final_results_file = Path("generation_results") / f"{base_filename}_generation_results_{'-'.join(pos_to_replace)}_{sentence_limit}.json"
os.makedirs(final_results_file.parent, exist_ok=True)
with open(final_results_file, "w") as f:
    json.dump(results, f, indent=2)


# In[ ]:


final_results = []
for res in results:
    # Find "Final Output:"
    content = res["response"]
    final_output_match = re.search(r'Final Output:\s*"(.*?)"\s*"(.*?)"', content, re.DOTALL)
    if final_output_match:
        lrl_sentence = final_output_match.group(1).strip()
        eng_translation = final_output_match.group(2).strip()
        print(f"Generated sentence for word '{res['word']}':")
        print(f"{lrl_sentence}")
        print(f"{eng_translation}\n")
        # Save the results
        final_results.append({
            "word": res["word"],
            "lrl_sentence": lrl_sentence,
            "eng_translation": eng_translation
        })





# In[ ]:


# show final results
for res in final_results:
    print(f"Generated sentence for word '{res['word']}':")
    print(f"{res['lrl_sentence']}")
    print(f"{res['eng_translation']}\n")


# In[ ]:


import os
from pathlib import Path


# In[196]:


# Read openai batch output
from tqdm import tqdm
file_path = "outputs/completions/"

file_list = os.listdir(file_path)

result_data = {}

for file_name in file_list:
    if file_name.endswith(".jsonl"):
        print(f"Reading file: {file_name}")
        idx = file_name.find("part")
        if idx !=-1:
            file_name_base = file_name[:idx-1]
            print(f"File base name: {file_name_base}")

        if result_data.get(file_name_base) is None:
            result_data[file_name_base] = []
            #print(data)

        with open(os.path.join(file_path, file_name), "r", encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                response = data.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                custom_id = data.get("custom_id", "No ID")
                #print(response)
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

                    #print(content)
                    final_output_match = re.search(r'Final Output:\s*"(.*?)"\s*"(.*?)"', content, re.DOTALL)
                    if final_output_match:
                        lrl_sentence = final_output_match.group(1).strip()
                        eng_translation = final_output_match.group(2).strip()
                        #print(f"Generated sentence:")
                        #print(f"{lrl_sentence}")
                        #print(f"{eng_translation}\n")
                        result_data[file_name_base].append({
                            "custom_id": custom_id,
                            "generated_sentence": lrl_sentence,
                            "generated_eng_translation": eng_translation
                        })



# In[197]:


result_data.keys()


# In[198]:


# Store each result data item as a separate JSON file with name based on the key
output_results_path = Path("generation_results_combined") / base_filename
output_results_path.mkdir(parents=True, exist_ok=True)
for key, results in result_data.items():
    output_file = output_results_path / f"{key}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved results to {output_file}")


# In[214]:


import json
from pathlib import Path

# --- Configuration ---
INPUT_DIR = Path("generation_results_combined/kalamang/")             # Folder with .json files
OUTPUT_DIR = Path("generation_results_combined_lemma/kalamang")   # Output folder
GROUP_SIZE = 20                             # Each set size (e.g., 20)
TAKE_COUNTS = [20]                       # How many to take from each segment
OFFSET_STEP = 20                            # Jump size for next block (usually same as GROUP_SIZE)

# Create output folder if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔍 Processing JSON files in {INPUT_DIR}...")

files = os.listdir(INPUT_DIR)
print(files)
print(f"Found {len(files)} files.")

for json_file in files:
    json_file = Path(json_file)
    print(f"📄 Processing {json_file}")

    # --- Read JSON file ---
    with open(INPUT_DIR / json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"⚠️ Skipping {json_file.name}: not a list")
        continue

    selected = []

    # --- Loop through the list in groups ---
    for i in range(0, len(data), GROUP_SIZE):
        # Example: if TAKE_COUNTS = [5, 10], we’ll take first 5, then next 10
        start = i
        for take in TAKE_COUNTS:
            selected.extend(data[start:start + take])
            start += OFFSET_STEP  # move 20 forward for the next block

    # --- Save filtered data ---
    output_path = OUTPUT_DIR / f"{json_file.stem}_{TAKE_COUNTS[0]}_subset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(selected)} items to {output_path.name}\n")


# In[ ]:





# In[ ]:


all_results = []

for res in batch_results:
    response = res.get("response", {})
    body = response.get("body", {})
    choices = body.get("choices", [])
    custom_id = res.get("custom_id", "No ID")
    #print(response)
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        print(f"Response for {custom_id}:")
        print(content)
        final_output_match = re.search(r'Final Output:\s*"(.*?)"\s*"(.*?)"', content, re.DOTALL)
        if final_output_match:
            lrl_sentence = final_output_match.group(1).strip()
            eng_translation = final_output_match.group(2).strip()
            print(f"Generated sentence:")
            print(f"{lrl_sentence}")
            print(f"{eng_translation}\n")
            all_results.append({
                "custom_id": custom_id,
                "lrl_sentence": lrl_sentence,
                "eng_translation": eng_translation
            })  



# In[ ]:


all_results


# In[ ]:


# Save block table to a JSON file
output_file = "generation_results/final_generation_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)


# In[ ]:





# # NLLB dataprep

# In[ ]:


english_sentences = []
translations = []

for sentence in parallel_sentences_with_pos_and_unimorph:
    eng = sentence["translation"]
    source = sentence["source"]

    if eng!="" and source!="":
        english_sentences.append(eng)
        translations.append(source)





# In[ ]:


english_sentences[:5], translations[:5]


# In[ ]:


# Store english sentences as eng.txt and translated sentences as lrl.txt

if os.path.exists("translation_data") == False:
    os.makedirs("translation_data")
with open("translation_data/eng.txt", "w", encoding="utf-8") as f:
    for line in english_sentences:
        f.write(line + "\n")

with open("translation_data/ckb.txt", "w", encoding="utf-8") as f:
    for line in translations:
        f.write(line + "\n")


# In[ ]:





# # Test

# In[ ]:


import json

filename = "generation_results\kalamang_generation_results_NOUN-PNOUN_80.json"
with open(filename, encoding="utf-8") as f:
    data = json.load(f)



# In[12]:


print(len(data))


# In[11]:


count = 10

ignore = 20

for item in data:
    if ignore>0:
        ignore-=1
        continue
    word = item.get("word",None)
    print(f"Word: {word}")
    response = item.get("response",None)
    print(response)
    count-=1
    if(count<=0):
        break


# In[ ]:




