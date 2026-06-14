# GrammarMT: A Factorial Study of Synthetic Data Generation for
Low-Resource Machine Translation using Grammar Books

This repository contains the code accompanying the paper. It implements a pipeline that extracts grammatical rules from descriptive grammar books and uses them to guide translation of low-resource languages into English.

The approach was evaluated on three languages: **Kalamang** (Papua New Guinea), **Mandan** (North America), and **Sursilvan Romansh** (Switzerland).

---

## Method Overview

```
Grammar Book (PDF)
        │
        ▼
[1. pdf_parser.py]              Extract text → structured sections
        │
        ▼
[2. section_tagger.py]          Tag sections by grammatical topic
        │
        ▼
[3. rule_extraction.py]         Gemini-2.5-flash batch → YAML grammatical rules
        │
        ▼
[4. rule_codification.py]       Deduplicate + merge + codify rules (second Gemini pass)
        │
        ▼
[5. sentence_generation.py]     Generate training pairs using Gemini-2.5-flash
        │
        ▼
[6. gemini_finetune.py]         Fine-tune Gemini on generated pairs via Vertex AI SFT
        │
        ▼
[7. icl_baseline_eval.py]       ICL baseline: rule retrieval + Gemini translation → BLEU/ChrF
```

**Parallel sentence extraction** (needed before step 5) is a separate sub-pipeline:
```
Grammar Book (PDF)
        │
        ▼
[lsp_igt_classifier.py]                 BERT IGT classifier → SOURCE / GLOSS / TRANSLATION
        │
        ▼
[parallel_sentence_preprocess.py]       Clean, enrich with spacy tags, compute corpus statistics
        │
        ▼
[dictionary_extraction.py]              Extract dictionary columns from PDF pages → CSV
        │
        ▼
[dictionary_wordlist_enhancement.py]    Add spacy morphological analysis to dictionary entries → JSON
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download models
```bash
python scripts/utils/download_models.py
```
This downloads the BERT IGT classifier weights into `models/combined_bert_parallel_line_classifier_final/`.

### 3. Set API keys
```bash
cp .env.example .env
# Edit .env and fill in your keys
```
Then export before running:
```bash
export GEMINI_API_KEY=your_key_here

# For Vertex AI fine-tuning only:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account_key.json
```

---

## Scripts

| Script | Description | Inputs | Outputs |
|--------|-------------|--------|---------|
| `scripts/lsp_igt_classifier.py` | BERT-based IGT line classifier. Classifies each line as SOURCE / GLOSS / TRANSLATION and assembles parallel sentences. | `grammar_books/` PDFs | `data/igt_classifier_results/` JSON + annotated PDFs |
| `scripts/parallel_sentence_preprocess.py` | Cleans extracted parallel sentences (quote normalization, citation removal) and enriches with spacy morphological tags. Generates corpus statistics figures. | `data/parallel_sentences/` JSON | Cleaned + enriched JSON, `images/corpus_statistics.pdf` |
| `scripts/dictionary_extraction.py` | Extracts dictionary headwords and translations from columnar dictionary pages in a grammar book PDF. | Grammar book PDF | `data/extracted_dictionaries/` CSV |
| `scripts/dictionary_wordlist_enhancement.py` | Adds spacy morphological analysis to dictionary entries, producing the JSON format used by sentence generation. | `data/extracted_dictionaries/` CSV | `data/dictionary/{lang}_dictionary_with_metadata.json` |
| `scripts/pdf_parser.py` | Parses a grammar book PDF into structured sections (headings + body text). | PDF path | `data/sections/` JSON |
| `scripts/section_tagger.py` | Tags each section with grammatical topic labels using embedding similarity. | `data/sections/` JSON | `data/section_tagged/` JSON |
| `scripts/rule_extraction.py` | Extracts YAML grammatical rules from tagged sections using Gemini-2.5-flash batch API. Also splits sections by semantic similarity. | Tagged sections JSON | `data/sections_split/` JSON, `data/extracted_rules/{lang}_extracted_rules.json` |
| `scripts/rule_codification.py` | Deduplicates and merges extracted rules, then runs a second Gemini-2.5-flash batch pass to produce canonical descriptions and `ApplyRule()` pseudo-code. | `data/extracted_rules/{lang}_extracted_rules.json` | `data/extracted_rules/{lang}_extracted_rules_final.json` |
| `scripts/sentence_generation.py` | Generates training sentences by combining retrieved rules with seed parallel sentences using Gemini-2.5-flash. | Rules JSON + parallel sentences + dictionary | `data/generated_sentences/` JSON |
| `scripts/gemini_finetune.py` | Prepares fine-tuning JSONL files and launches Gemini supervised fine-tuning jobs via Vertex AI SFT (`sft.train()`). Requires `GOOGLE_APPLICATION_CREDENTIALS`. | Generated sentences | Fine-tuned Gemini model on Vertex AI |
| `scripts/icl_baseline_eval.py` | Full ICL evaluation pipeline (3 phases): build Gemini batch prompts, submit + poll, compute BLEU/ChrF/ChrF++. Fully resumable. | Rules, parallel sentences, dictionary | `data/icl_baseline/` results + metrics |
| `scripts/rule_selection.py` | Rule selection and ranking utilities used by the generation pipeline. | Rules JSON | Selected rules |
| `scripts/run_batches.py` | Submits generation prompts as OpenAI batch jobs and polls for completion. | Batch JSONL files | Downloaded results JSONL |
| `pipeline.py` | Orchestrates steps 1–4 (parse → tag → extract → generate) via a single CLI command. | PDF path + optional config YAML | All intermediate outputs |

---

## Reproducing Results

### Step 1: Extract parallel sentences from grammar books

```bash
python scripts/lsp_igt_classifier.py \
    --books grammar_books/comprehensive_grammar_library/
```

This produces `data/igt_classifier_results/{book}_parallel_sentences.json`. Add `--force` to re-run existing books. Add `--no-render` to skip annotated PDF generation.

### Step 2: Clean and enrich parallel sentences

Edit the `language` and `train_or_test` variables at the top of the script, then:
```bash
python scripts/parallel_sentence_preprocess.py
```

### Step 3: Extract and enhance dictionary

Edit `pdf_file` and `pageno_wordlist_start` at the top of `dictionary_extraction.py` to point to the dictionary section of the grammar book:
```bash
python scripts/dictionary_extraction.py
python scripts/dictionary_wordlist_enhancement.py \
    data/extracted_dictionaries/kalamang_extracted_word_list_columns.csv \
    kalamang
```

### Step 4: Run the grammar rule extraction pipeline

```bash
# Full pipeline for a single grammar book
python pipeline.py path/to/grammar_book.pdf --config config.yaml

# Or step by step:
python scripts/pdf_parser.py path/to/grammar_book.pdf
python scripts/section_tagger.py data/sections/grammar_book_sections.json
python scripts/rule_extraction.py data/section_tagged/grammar_book_sections_classified.json
```

### Step 5: Codify rules

```bash
python scripts/rule_codification.py \
    data/extracted_rules/kalamang_extracted_rules.json
```

### Step 6: Generate training sentences

```bash
python scripts/sentence_generation.py \
    data/extracted_rules/kalamang_extracted_rules_final.json \
    --model-provider gemini \
    --granularity rule \
    --sentence-limit 100
```

### Step 7: Fine-tune Gemini

Requires a Google Cloud project with Vertex AI enabled and a service account key:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account_key.json
python scripts/gemini_finetune.py
```

### Step 8: Run ICL baseline evaluation

```bash
python scripts/icl_baseline_eval.py
```

This runs all three phases and saves results to `data/icl_baseline/icl_baseline_metrics.json`.

---

## Configuration

A sample YAML config for `pipeline.py`:

```yaml
max_heading_number: 3
min_heading_occ_count: 2
min_heading_total_char_length: 30
main_body_tolerance: 0.8
heading_weight: 1.5
threshold: 0.7
strong_count: 2
target_words: 500
min_gap: 1
sentence_limit: 100
no_of_random_nouns: 2
```

---

## Results

Results (BLEU, ChrF, ChrF++) are reported across three conditions:
- **SDFT**: Standard direct fine-tuning (baseline, no grammar rules)
- **ICL-Rule**: In-context learning with retrieved grammatical rules
- **ICL-Section**: In-context learning with retrieved grammar sections

Metric outputs are saved to `data/icl_baseline/icl_baseline_metrics.json`.

---

## Citation

```bibtex
```
