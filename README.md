# Grammar book processing pipeline

This repository provides a modular pipeline for processing PDF documents of Grammar books, extracting structured information, and generating sentences from extracted rules. The pipeline is orchestrated using a Typer CLI application and consists of four main stages, each handled by a separate script.

## Pipeline Overview

The pipeline performs the following steps:

1. **PDF Parsing**: Converts a PDF document into a structured JSON format.
2. **Section Tagging**: Identifies and tags sections within the parsed JSON.
3. **Rule Extraction**: Extracts rules from the tagged sections.
4. **Sentence Generation**: Generates sentences based on the extracted rules.

Each stage can be configured via a YAML config file.

## Usage

Make sure your OpenAI API key is set in `api_key.txt`.
(Temporary, will be fixed later and more APIs will be added)

Run the pipeline using:

```bash
python pipeline.py <PDF_FILENAME> --config <CONFIG_FILE> --download-models
```

- `<PDF_FILENAME>`: Path to the input PDF file.
- `<CONFIG_FILE>` (optional): Path to a YAML configuration file with pipeline parameters.
- `--download-models` (optional): Flag to download required models before running the pipeline. Run this for the first time to ensure all models are available.

If no config file is provided, default settings are used.

## Pipeline Steps in Detail

### 1. PDF Parsing (`pdf_parser.py`)

- **Input**: PDF file
- **Output**: JSON file in `parsed_grammar_json/`
- **Config Parameters**:
    - `max_heading_number`
    - `min_heading_occ_count`
    - `min_heading_total_char_length`
    - `main_body_tolerance`

### 2. Section Tagging (`section_tagger.py`)

- **Input**: Parsed JSON from previous step
- **Output**: Tagged JSON in `classified_json/`
- **Config Parameters**:
    - `heading_weight`
    - `threshold`
    - `strong_count`

### 3. Rule Extraction (`rule_extraction.py`)

- **Input**: Tagged JSON from previous step
- **Output**: Rules JSON in `extracted_rules_json/`
- **Config Parameters**:
    - `target_words`
    - `min_gap`

### 4. Sentence Generation (`sentence_generation.py`)

- **Input**: Rules JSON from previous step
- **Output**: Generated sentences
- **Config Parameters**:
    - `sentence_limit`
    - `no_of_random_nouns`

## Configuration

A sample YAML config file:

```yaml
max_heading_number: 3
min_heading_occ_count: 2
min_heading_total_char_length: 30
main_body_tolerance: 0.8
heading_weight: 1.5
threshold: 0.7
strong_count: 2
target_words: 10
min_gap: 5
sentence_limit: 100
no_of_random_nouns: 3
```

## Error Handling

- The pipeline checks for required parameters in the config file and will stop with an error message if any are missing.
- Each stage must complete successfully for the pipeline to continue.

## Requirements

- Python 3.7+
- [Typer](https://typer.tiangolo.com/)
- [PyYAML](https://pyyaml.org/)
- [spaCy](https://spacy.io/)
- [tqdm](https://tqdm.github.io/)
- [langdetect](https://pypi.org/project/langdetect/)
- [OpenAI](https://pypi.org/project/openai/)
- [PyTorch](https://pytorch.org/) (`torch`)
- [sentence-transformers](https://www.sbert.net/)
- [NLTK](https://www.nltk.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)

Install dependencies:

```bash
pip install -r requirements.txt
```

## License

See `LICENSE` file for details.
