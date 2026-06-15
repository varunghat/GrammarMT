import typer
import subprocess
import sys
from pathlib import Path
import yaml


app = typer.Typer(
    name="pipeline",
    help="Pipeline to run the full document processing sequence",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def run(command: list, step: str) -> None:
    result = subprocess.run(command)
    if result.returncode != 0:
        typer.echo(f"Error in {step}, stopping pipeline.")
        raise typer.Exit(code=result.returncode)


@app.command()
def run_pipeline(
    filename: str = typer.Argument(None, help="Path to the PDF file"),
    config_file: str = typer.Option(
        None, "--config", "-c", help="Path to the config YAML file"
    ),
    download_models: bool = typer.Option(
        False, "--download-models", "-d", help="Download required models before running"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build batch files but skip Gemini API submission"
    ),
):
    """
    Run the full document processing pipeline:
    1. Parse the PDF to JSON
    2. Tag sections in the JSON
    3. Extract grammatical rules (Gemini batch)
    4. Codify and merge rules (Gemini batch)
    5. Generate training sentences
    """
    if download_models:
        typer.echo("Downloading required models...")
        run(["python", "scripts/utils/download_models.py"], "download_models")
        typer.echo("Models downloaded successfully.")

    if not filename or not Path(filename).is_file():
        typer.echo("Please provide a valid PDF filename.")
        raise typer.Exit(code=1)

    config = {}
    if config_file:
        if not Path(config_file).is_file():
            typer.echo(f"Config file not found: {config_file}")
            raise typer.Exit(code=1)
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    else:
        typer.echo("No config file provided, using defaults.")

    stem = Path(filename).stem
    # Language name is the first underscore-delimited token of the PDF filename
    # e.g. "kalamang_grammar.pdf" → "kalamang"
    language = stem.split("_")[0]

    ######################################################
    # Step 1: pdf_parser.py
    typer.echo(f"\n[1/5] Parsing PDF: {filename}")
    command = ["python", "scripts/pdf_parser.py", filename]
    if config:
        for flag, key in [
            ("--max-heading-number",          "max_heading_number"),
            ("--min-heading",                 "min_heading_occ_count"),
            ("--min-heading-total-char-length","min_heading_total_char_length"),
            ("--main-body-tolerance",         "main_body_tolerance"),
        ]:
            if key in config:
                command.extend([flag, str(config[key])])
    run(command, "pdf_parser.py")

    ######################################################
    # Step 2: section_tagger.py
    sections_json = Path("data/sections") / f"{stem}_sections.json"
    typer.echo(f"\n[2/5] Tagging sections: {sections_json}")
    command = ["python", "scripts/section_tagger.py", str(sections_json)]
    if config:
        for flag, key in [
            ("--heading-weight", "heading_weight"),
            ("--threshold",      "threshold"),
            ("--strong-count",   "strong_count"),
        ]:
            if key in config:
                command.extend([flag, str(config[key])])
    run(command, "section_tagger.py")

    ######################################################
    # Step 3: rule_extraction.py
    tagged_json = Path("data/section_tagged") / f"{stem}_sections_classified.json"
    typer.echo(f"\n[3/5] Extracting rules: {tagged_json}")
    command = ["python", "scripts/rule_extraction.py", str(tagged_json)]
    if config:
        for flag, key in [
            ("--target-words", "target_words"),
            ("--min-gap",      "min_gap"),
        ]:
            if key in config:
                command.extend([flag, str(config[key])])
    if dry_run:
        command.append("--dry-run")
    run(command, "rule_extraction.py")

    ######################################################
    # Step 4: rule_codification.py
    extracted_rules = Path("data/extracted_rules") / f"{language}_extracted_rules.json"
    typer.echo(f"\n[4/5] Codifying rules: {extracted_rules}")
    command = ["python", "scripts/rule_codification.py", str(extracted_rules)]
    if dry_run:
        command.append("--dry-run")
    run(command, "rule_codification.py")

    ######################################################
    # Step 5: sentence_generation.py
    final_rules = Path("data/extracted_rules") / f"{language}_extracted_rules_final.json"
    typer.echo(f"\n[5/5] Generating sentences: {final_rules}")
    command = ["python", "scripts/sentence_generation.py", str(final_rules)]
    if config:
        for flag, key in [
            ("--sentence-limit",      "sentence_limit"),
            ("--no-of-random-nouns",  "no_of_random_nouns"),
            ("--output-dir",          "output_dir"),
            ("--granularity",         "granularity"),
            ("--model-provider",      "model_provider"),
        ]:
            if key in config:
                command.extend([flag, str(config[key])])
    run(command, "sentence_generation.py")

    typer.echo("\nPipeline completed successfully.")


if __name__ == "__main__":
    app()
