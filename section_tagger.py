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


def simplify_pos(tag: str):
    """
    Map Penn Treebank POS tags to WordNet POS tags.
    Default fallback is NOUN.
    """
    tag = tag.upper()

    if tag.startswith("J"):  # Adjective
        return wordnet.ADJ
    elif tag.startswith("V"):  # Verb
        return wordnet.VERB
    elif tag.startswith("N"):  # Noun
        return wordnet.NOUN
    elif tag.startswith("R"):  # Adverb
        return wordnet.ADV

    # Explicit extra mappings for clarity:
    elif tag in {"PRP", "PRP$", "WP", "WP$"}:  # pronouns
        return wordnet.NOUN
    elif tag in {"DT", "PDT", "WDT"}:  # determiners
        return wordnet.ADJ
    elif tag in {"IN"}:  # prepositions/subordinating conjunctions
        return wordnet.ADV
    elif tag in {"CC"}:  # coordinating conjunctions
        return wordnet.ADV
    elif tag in {"CD"}:  # cardinal numbers
        return wordnet.NOUN
    elif tag in {"UH"}:  # interjections
        return wordnet.NOUN

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
    "Pronoun": [
        "pronoun",
    ],
    "Conjugation": ["conjugate", "conjugation", "form of verb", "verb table"],
    "Declension": ["declension", "form of noun"],
    "Case": [
        "nominative",
        "accusative",
        "genitive",
        "dative",
        "instrumental",
        "locative",
    ],
    "Tense": ["tense", "past", "present", "future"],
    "1st Person": [
        "1st person",
    ],
    "2nd Person": [
        "2nd person",
    ],
    "3rd Person": [
        "3rd person",
    ],
    "Article": ["article", "definite", "indefinite"],
    "Gender": [
        "gender",
        "male",
        "female",
        "neutral",
        "masculine",
        "feminine",
        "neuter",
    ],
    "Position": [
        "position",
        "preposition",
        "postposition",
        "prepositional phrase",
        "locative",
    ],
    "Conjunction": ["conjunction"],
    "Interjection": [
        "interjection",
        "exclamation",
        "emotion",
        "surprise",
        "greeting",
        "response",
    ],
    # Book related tags
    "Publishing information": [
        "publishing",
        "publisher",
        "publication",
        "editor",
        "copyright",
        "ISBN",
        "edition",
    ],
    "Structure": [
        "table of contents",
        "contents",
        "index",
        "appendix",
        "bibliography",
        "glossary",
        "chapter",
        "section",
        "page",
        "content",
        "introduction",
        "conclusion",
        "preface",
        "acknowledgment",
        "appendix",
        "footnote",
        "endnote",
        "reference",
        "citation",
    ],
    "History": [
        "history",
        "origin",
        "etymology",
        "development",
        "evolution",
        "ancient",
        "classical",
        "historical",
        "linguistic history",
        "language family",
    ],
    "Culture": [
        "culture",
        "tradition",
        "festival",
        "ritual",
        "belief",
        "custom",
        "regional",
        "cultural context",
        "social context",
        "cultural significance",
    ],
    # Linguistic tags
    "Phonetics": [
        "phonetics",
        "pronunciation",
        "sounds",
        "phonetic",
        "phonology",
        "phoneme",
        "pronounce",
        "retroflex",
        "alveolar",
        "dental",
        "labial",
        "voiced",
        "voiceless",
        "aspirated",
        "unaspirated",
        "nasalized",
        "unreleased",
        "fricative",
        "plosive",
        "affricate",
        "approximant",
        "lateral",
        "tap",
        "flap",
        "glottal",
        "palatal",
        "velar",
        "uvular",
        "pharyngeal",
        "labiodental",
        "bilabial",
        "diphthong",
        "monophthong",
        "schwa",
        "vowel sound",
        "consonant sound",
        "vowel",
        "consonant",
        "syllable",
        "intonation",
        "stress",
        "pitch",
        "tone",
    ],
    "Morphology": [
        "morphology",
        "structure of words",
        "word formation",
        "morpheme",
        "compound",
        "affixation",
        "derivation",
        "inflection",
        "root",
        "stem",
        "prefix",
        "suffix",
        "infix",
        "circumfix",
        "clitic",
        "allomorph",
        "inflectional morphology",
        "derivational morphology",
        "agglutinative",
        "fusional",
        "isolating",
        "polysynthetic",
    ],
    "Semantics": [
        "semantics",
        "meaning",
        "interpretation",
        "context",
        "lexical semantics",
        "compositional semantics",
        "pragmatics",
    ],
    "Syntax": [
        "syntax",
        "sentence structure",
        "word order",
        "grammar rules",
        "syntactic",
    ],
}


def _prepare_keywords(tag_keywords):
    norm = {}
    for tag, kws in tag_keywords.items():
        singles, phrases = [], []
        for kw in kws:
            kw_norm = kw.strip().lower()
            if " " in kw_norm:
                phrases.append(kw_norm)
            else:
                singles.append(kw_norm)
        norm[tag] = {"singles": singles, "phrases": phrases}
    return norm


KEY_INDEX = _prepare_keywords(TAG_KEYWORDS)  # module global


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def _norm_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def classify_text(text: str, use_lemma: bool = True):
    raw = text or ""
    if use_lemma:
        tokens = [t for t in _WORD_RE.findall(raw)]
        lemmas = lemmatize(tokens) if tokens else []
        tokens_norm = [t.lower() for t in lemmas]
        full_norm = " ".join(tokens_norm)
    else:
        tokens_norm = [t.lower() for t in _WORD_RE.findall(raw)]
        full_norm = " ".join(tokens_norm)

    token_set = set(tokens_norm)

    tag_counts = defaultdict(float)
    tag_words = []

    for tag, parts in KEY_INDEX.items():
        # singles: count via token membership
        for w in parts["singles"]:
            if w in token_set:

                tag_counts[tag] += tokens_norm.count(
                    w
                )  # or +=1 if you want presence only
                # optional: add each occurrence
                tag_words.extend((tag, w) for _ in range(tokens_norm.count(w)))

        # phrases: count via regex on full normalized text
        for ph in parts["phrases"]:
            # \b doesn't work across spaces reliably; anchor with (?<!\w) (?!\w)
            pattern = rf"(?<!\w){re.escape(ph)}(?!\w)"
            for _ in re.finditer(pattern, full_norm):
                tag_counts[tag] += 1
                tag_words.append((tag, ph))

    # sorted tags by score
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return list({t for t, _ in sorted_tags}), dict(tag_counts), tag_words


app = typer.Typer(
    help="Classify sections of text into grammar concepts based on keyword matching.",
    pretty_exceptions_enable=False,
    add_completion=False,
)


@app.command()
def classify_sections(
    filename: str = typer.Argument(
        None, help="Path to the JSON file output from pdf_parser.py"
    ),
    heading_weight: float = typer.Option(
        2.0, help="Weight multiplier for heading tag counts"
    ),
    threshold: float = typer.Option(
        2.0, help="Minimum combined tag count to include a tag"
    ),
    strong_count: float = typer.Option(
        4.0, help="Minimum combined tag count to consider a tag as strong"
    ),
):
    """
    Classify sections of text into grammar concepts based on keyword matching.
    """
    path = Path(filename)
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    out_dir = Path("classified_json")
    out_dir.mkdir(exist_ok=True)

    # Check for nltk averaged_perceptron_tager, if not download
    # TODO
    

    classified = []
    for section in tqdm(data, desc="Classifying"):
        text = section.get("text", "")
        if len(text) < 10:
            continue

        tags_body, counts_body, words_body = classify_text(text, use_lemma=True)
        tags_head, counts_head, words_head = classify_text(
            section.get("heading", ""), use_lemma=True
        )

        # weighted combine
        combined = defaultdict(float)
        for k, v in counts_body.items():
            combined[k] += v
        for k, v in counts_head.items():
            combined[k] += heading_weight * v

        filtered = [k for k, v in combined.items() if v >= threshold]
        strong = {k: v for k, v in combined.items() if v >= strong_count}
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # Build outputs
        classified.append(
            {
                "heading": section.get("heading", ""),
                "text": text,
                "tags": tags_body,
                "tag_counts": counts_body,
                "tag_word_counts": {
                    f"{t}:{w}": c for (t, w), c in _count_pairs(words_body).items()
                },
                "tags_heading": tags_head,
                "tag_counts_heading": counts_head,
                "tag_word_counts_heading": {
                    f"{t}:{w}": c for (t, w), c in _count_pairs(words_head).items()
                },
                "combined_tag_counts": dict(
                    sorted(combined.items(), key=lambda x: x[1], reverse=True)
                ),
                "filtered_tags": filtered,
                "strong_tags": strong,
                "sorted_tags": sorted_combined,
                "page": section.get("page", ""),
                "type": section.get("type", "unknown"),
            }
        )

    out = out_dir / f"{path.stem}_classified.json"
    out.write_text(
        json.dumps(classified, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    print("Classified data saved to", str(out))


def _count_pairs(pairs):
    c = defaultdict(int)
    for p in pairs:
        c[p] += 1
    return c


if __name__ == "__main__":
    app()
