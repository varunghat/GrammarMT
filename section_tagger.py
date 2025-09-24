import re
from collections import defaultdict
import json
import typer
from tqdm import tqdm
from pathlib import Path

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def simplify_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    elif tag.startswith("V"): return wordnet.VERB
    elif tag.startswith("N"): return wordnet.NOUN
    elif tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def lemmatize(tokens):
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(tok, simplify_pos(pos)) for tok, pos in pos_tags]

# Keywords for each grammar concept
TAG_KEYWORDS = {
    "Noun": ["noun", "object", "subject"],
    "Plural": ["plural"],
    "Singular": ["singular"],
    "Verb": ["verb", "action", "do", "does", "did"],
    "Past": ["past", "was", "were", "had", "did"],
    "Present": ["present"],
    "Future": ["future", "will", "shall"],
    "Adjective": ["adjective", "describes", "modifies noun"],
    "Adverb": ["adverb", "modifies verb", "manner"],
    "Pronoun": ["pronoun",],
    "Conjugation": ["conjugate", "conjugation", "form of verb", "verb table"],
    "Declension": ["declension", "form of noun"],
    "Case": ["nominative", "accusative", "genitive", "dative", "instrumental", "locative"],
    "Tense": ["tense", "past", "present", "future"],
    "1st Person": ["1st person", ],
    "2nd Person": ["2nd person",],
    "3rd Person": ["3rd person", ],
    "Article": ["article","definite","indefinite"],
    "Gender": ["gender","male","female","neutral","masculine","feminine","neuter"],
    "Position": ["position", "preposition", "postposition", "prepositional phrase", "locative"],
    "Conjunction": ["conjunction"],
    "Interjection": ["interjection", "exclamation", "emotion", "surprise", "greeting", "response"],

    # Book related tags
    "Publishing information": ["publishing", "publisher", "publication", "editor", "copyright", "ISBN", "edition"],
    "Structure": ["table of contents","contents", "index", "appendix", "bibliography", "glossary", "chapter", "section","page", "content", "introduction", "conclusion", "preface", "acknowledgment","appendix", "footnote", "endnote", "reference", "citation",],
    "History": ["history", "origin", "etymology", "development", "evolution", "ancient", "classical", "historical", "linguistic history", "language family"],
    "Culture": ["culture", "tradition", "festival", "ritual", "belief", "custom", "regional", "cultural context", "social context", "cultural significance"],




    # Linguistic tags
    "Phonetics": ["phonetics", "pronunciation", "sounds", "phonetic","phonology","phoneme","pronounce","retroflex","alveolar","dental","labial",
        "voiced", "voiceless", "aspirated", "unaspirated", "nasalized", "unreleased", "fricative", "plosive", "affricate", "approximant", "lateral", "tap", "flap",
        "glottal", "palatal", "velar", "uvular", "pharyngeal", "labiodental", "bilabial", "diphthong", "monophthong", "schwa", "vowel sound", "consonant sound",
        "vowel", "consonant", "syllable", "intonation", "stress", "pitch", "tone"],
    "Morphology": ["morphology", "structure of words", "word formation","morpheme","compound","affixation","derivation","inflection","root","stem","prefix","suffix"
        "infix","circumfix","clitic","allomorph","inflectional morphology","derivational morphology", "agglutinative", "fusional", "isolating", "polysynthetic"],
    "Semantics": ["semantics", "meaning", "interpretation", "context", "lexical semantics", "compositional semantics", "pragmatics"],
    "Syntax": ["syntax", "sentence structure", "word order", "grammar rules","syntactic"],



}

def classify_text_simple(text):

    tags = []
    
    # Normalize text to lowercase
    text = text.lower()
    
    for tag, keywords in TAG_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                tags.append(tag)
                  # No need to check other keywords for this tag
    
    # Count occurrences of each tag
    tag_counts = defaultdict(int)
    for tag in tags:
        tag_counts[tag] += 1

    return set(tags), dict(tag_counts)

def classify_text_with_lemmatization(text):
    # Tokenize and lemmatize the text
    tokens = text.split()
    lemmatized_tokens = lemmatize(tokens)
    # lowercase the lemmatized tokens
    lemmatized_tokens = [token.lower() for token in lemmatized_tokens]

    tags = []
    tag_words = []
    
    for tag, keywords in TAG_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', ' '.join(lemmatized_tokens)):
                # Find all matching words
                for token in lemmatized_tokens:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', token):
                        tags.append(tag)
                        tag_words.append((tag, token))
                  
    # Count occurrences of each tag
    # Count each tag occurrence from tag_words list since it contains the actual word-tag pairs
    tag_counts = defaultdict(int)
    for tag, _ in tag_words:
        tag_counts[tag] += 1

    return tags, dict(tag_counts), tag_words

app = typer.Typer()

@app.command()
def classify_sections(filename: str):
    print("Classifying sections in", filename)
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Data loaded with", len(data), "sections")

    LEMMATIZED_TAG_KEYWORDS = []

    for tag in TAG_KEYWORDS.keys():
        lemmatized_keywords = []
        for keyword in TAG_KEYWORDS[tag]:
            # Split by spaces and lemmatize each word
            words = keyword.split()
            lemmatized_words = lemmatize(words)
            lemmatized_keywords.append(" ".join(lemmatized_words))
        LEMMATIZED_TAG_KEYWORDS.append((tag, lemmatized_keywords))


    LEMMATIZED_TAG_KEYWORDS = dict(LEMMATIZED_TAG_KEYWORDS)
    # Remove duplicates
    LEMMATIZED_TAG_KEYWORDS = {k: list(set(v)) for k, v in LEMMATIZED_TAG_KEYWORDS.items()}



    # Classify each section's text
    classified_data = []
    for section in tqdm(data):
        text = section["text"]
        if len(text) < 10:
            continue
        tags, tag_counts, tag_words = classify_text_with_lemmatization(text)
        tags_heading, tag_counts_heading, tag_words_heading = classify_text_with_lemmatization(section["heading"])

        # Combine the tags with more weight to the heading
        heading_weight = 2.0
        combined_tags = {tag: tag_counts.get(tag, 0) + heading_weight * tag_counts_heading.get(tag, 0) for tag in set(tags).union(set(tags_heading))}
        combined_tag_counts = {tag: tag_counts.get(tag, 0) + heading_weight * tag_counts_heading.get(tag, 0) for tag in set(tag_counts).union(set(tag_counts_heading))}

        threshold = 2.0
        strong_threshold = 4.0
        
        # Filter out tags with low counts and convert to set
        filtered_tags = set(tag for tag, count in combined_tag_counts.items() if count >= threshold)

        # Tags sorted by count
        sorted_tags = sorted(combined_tag_counts.items(), key=lambda x: x[1], reverse=True)

        # Count occurrences of tag words
        tag_word_counts = defaultdict(int)
        for tag, word in tag_words:
            tag_word_counts[f"{tag}:{word}"] += 1  # Convert tuple to string key
        tag_word_counts_heading = defaultdict(int)
        for tag, word in tag_words_heading:
            tag_word_counts_heading[f"{tag}:{word}"] += 1  # Convert tuple to string key

        classified_data.append({
            "heading": section["heading"],
            "text": text,
            "tags": list(set(tags)),  # Keep counts for tags
            "tag_counts": tag_counts,
            "tag_word_counts": dict(tag_word_counts),
            "tags_heading": list(set(tags_heading)), 
            "tag_counts_heading": tag_counts_heading,
            "tag_word_counts_heading": dict(tag_word_counts_heading),
            "combined_tags": combined_tags,
            "combined_tag_counts": combined_tag_counts,
            "filtered_tags": list(filtered_tags),  # Convert set to list for JSON serialization
            "sorted_tags": sorted_tags,
            "strong_tags": {tag: count for tag, count in combined_tag_counts.items() if count >= strong_threshold},
            "page": section.get("page", ""),  # Add page number if available
            "type": section.get("type", "unknown"),  # Add type if available
        })

    Path("classified_json").mkdir(exist_ok=True)
    # Save classified data to a new JSON file
    with open(f"classified_json/{Path(filename).stem}_classified.json", "w", encoding="utf-8") as f:
        json.dump(classified_data, f, ensure_ascii=False, indent=4)
    
    print("Classified data saved to", f"classified_json/{Path(filename).stem}_classified.json")


if __name__ == "__main__":
    app()

