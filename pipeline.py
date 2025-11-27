import typer
import subprocess
from pathlib import Path
import yaml


app = typer.Typer(
    name="pipeline",
    help="Pipeline to run the full document processing sequence",
    pretty_exceptions_enable=False,
    add_completion=False,
)


@app.command()
def run_pipeline(
    filename: str = typer.Argument(None, help="Path to the PDF file"),
    config_file: str = typer.Option(
        None, "--config", "-c", help="Path to the config file"
    ),
    download_models: bool = typer.Option(
        False, "--download-models", "-d", help="Download required models"
    ),
):
    """
    Run the full document processing pipeline:
    1. Parse the PDF to JSON
    2. Tag sections in the JSON
    3. Extract rules from the tagged JSON
    4. Generate sentences from the extracted rules
    """
    if download_models:
        typer.echo("Downloading required models...")
        error_code = subprocess.run(["python", "download_models.py"], check=True)
        if error_code.returncode != 0:
            typer.echo("Error downloading models, stopping pipeline.")
            raise typer.Exit(code=error_code.returncode)
        typer.echo("Models downloaded successfully.")
    if not filename or not Path(filename).is_file():
        typer.echo("Please provide a valid PDF filename.")
        raise typer.Exit(code=1)

    if config_file:
        if not Path(config_file).is_file():
            typer.echo("Please provide a valid config file.")
            raise typer.Exit(code=1)
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

    else:
        typer.echo("No config file provided, using default settings.")
        config = {}

    ######################################################
    # Step 1: Call pdf_parser.py with the PDF filename
    typer.echo(f"Running pdf_parser.py on {filename}")
    command = ["python", "src/pdf_parser.py", filename]
    if config_file:
        # Get additional arguments from config or use defaults
        max_heading_number = config.get("max_heading_number", None)
        min_heading_occ_count = config.get("min_heading_occ_count", None)
        min_heading_total_char_length = config.get(
            "min_heading_total_char_length", None
        )
        main_body_tolerance = config.get("main_body_tolerance", None)
        if (
            max_heading_number is None
            or min_heading_occ_count is None
            or min_heading_total_char_length is None
            or main_body_tolerance is None
        ):
            typer.echo("Config file is missing required parameters.")
            if max_heading_number is None:
                typer.echo("Missing: max_heading_number")
            if min_heading_occ_count is None:
                typer.echo("Missing: min_heading_occ_count")
            if min_heading_total_char_length is None:
                typer.echo("Missing: min_heading_total_char_length")
            if main_body_tolerance is None:
                typer.echo("Missing: main_body_tolerance")
            raise typer.Exit(code=1)

        # Extend the command with additional arguments
        command.extend(
            [
                "--max-heading-number",
                str(max_heading_number),
                "--min-heading",
                str(min_heading_occ_count),
                "--min-heading-total-char-length",
                str(min_heading_total_char_length),
                "--main-body-tolerance",
                str(main_body_tolerance),
                "--config",
                config_file,
            ]
        )
    error_code = subprocess.run(command, check=True)
    if error_code.returncode != 0:
        typer.echo("Error in pdf_parser.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    ######################################################
    # Step 2: Call section_tagger.py with the output JSON
    sections_json = Path("classified_json") / (
        Path(filename).stem + "_sections_classified.json"
    )
    typer.echo(f"Running section_tagger.py on {sections_json}")
    command = ["python", "src/section_tagger.py", str(sections_json)]
    if config_file:
        heading_weight = config.get("heading_weight", None)
        threshold = config.get("threshold", None)
        strong_count = config.get("strong_count", None)
        if heading_weight is None or threshold is None or strong_count is None:
            typer.echo(
                "Config file is missing required parameters for section_tagger.py."
            )
            if heading_weight is None:
                typer.echo("Missing: heading_weight")
            if threshold is None:
                typer.echo("Missing: threshold")
            if strong_count is None:
                typer.echo("Missing: strong_count")
            raise typer.Exit(code=1)
        command.extend(
            [
                "--heading-weight",
                str(heading_weight),
                "--threshold",
                str(threshold),
                "--strong-count",
                str(strong_count),
            ]
        )
    error_code = subprocess.run(command, check=True)
    if error_code.returncode != 0:
        typer.echo("Error in section_tagger.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    ######################################################
    # Step 3: Call rule_extraction.py with the next output JSON
    tagged_json = Path("classified_json") / (
        Path(filename).stem + "_sections_tagged.json"
    )
    typer.echo(f"Running rule_extraction.py on {tagged_json}")
    command = ["python", "src/rule_extraction.py", str(tagged_json)]
    if config_file:
        target_words = config.get("target_words", None)
        min_gap = config.get("min_gap", None)
        if target_words is None or min_gap is None:
            typer.echo(
                "Config file is missing required parameters for rule_extraction.py."
            )
            if target_words is None:
                typer.echo("Missing: target_words")
            if min_gap is None:
                typer.echo("Missing: min_gap")
            raise typer.Exit(code=1)
        command.extend(
            [
                "--target-words",
                str(target_words),
                "--min-gap",
                str(min_gap),
            ]
        )
    error_code = subprocess.run(command, check=True)
    if error_code.returncode != 0:
        typer.echo("Error in rule_extraction.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    ######################################################
    # Step 4: Call sentence_generation.py with all the output JSONs

    rules_json = Path("extracted_rules_json") / (
        Path(filename).stem + "_sections_tagged_rules.json"
    )
    typer.echo(f"Running sentence_generation.py on {rules_json}")
    command = ["python", "src/sentence_generation.py", str(rules_json)]
    if config_file:
        sentence_limit = config.get("sentence_limit", None)
        no_of_random_nouns = config.get("no_of_random_nouns", None)
        output_dir = config.get("output_dir", "generated_sentences")
        if sentence_limit is None or no_of_random_nouns is None or output_dir is None:
            typer.echo(
                "Config file is missing required parameters for sentence_generation.py."
            )
            if sentence_limit is None:
                typer.echo("Missing: sentence_limit")
            if no_of_random_nouns is None:
                typer.echo("Missing: no_of_random_nouns")

            if output_dir is None:
                typer.echo("Missing: output_dir")
            raise typer.Exit(code=1)
        command.extend(
            [
                "--sentence-limit",
                str(sentence_limit),
                "--no-of-random-nouns",
                str(no_of_random_nouns),
                "--output-dir",
                output_dir,
            ]
        )
    error_code = subprocess.run(command, check=True)
    if error_code.returncode != 0:
        typer.echo("Error in sentence_generation.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    typer.echo("Pipeline completed successfully.")


if __name__ == "__main__":
    app()
