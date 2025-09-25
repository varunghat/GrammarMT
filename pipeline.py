import typer
import subprocess
from pathlib import Path

app = typer.Typer(
    name="pipeline",
    help="Pipeline to run the full document processing sequence",
    pretty_exceptions_enable=False,
    add_completion=False,
)


@app.command()
def run_pipeline(filename: str):
    # Step 1: Call pdf_parser.py with the PDF filename
    typer.echo(f"Running pdf_parser.py on {filename}")
    error_code = subprocess.run(["python", "pdf_parser.py", filename], check=True)
    if error_code.returncode != 0:
        typer.echo("Error in pdf_parser.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    # Step 2: Call section_tagger.py with the output JSON
    sections_json = Path("parsed_grammar_json") / (
        Path(filename).stem + "_sections.json"
    )
    typer.echo(f"Running section_tagger.py on {sections_json}")
    error_code = subprocess.run(
        ["python", "section_tagger.py", str(sections_json)], check=True
    )
    if error_code.returncode != 0:
        typer.echo("Error in section_tagger.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    # Step 3: Call rule_extraction.py with the next output JSON
    tagged_json = Path("classified_json") / (
        Path(filename).stem + "_sections_tagged.json"
    )
    typer.echo(f"Running rule_extraction.py on {tagged_json}")
    error_code = subprocess.run(
        ["python", "rule_extraction.py", str(tagged_json)], check=True
    )
    if error_code.returncode != 0:
        typer.echo("Error in rule_extraction.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    # Step 4: Call sentence_generation.py with all the output JSONs

    rules_json = Path("extracted_rules_json") / (
        Path(filename).stem + "_sections_tagged_rules.json"
    )
    typer.echo(f"Running sentence_generation.py on {rules_json}")
    error_code = subprocess.run(
        ["python", "sentence_generation.py", str(rules_json)], check=True
    )
    if error_code.returncode != 0:
        typer.echo("Error in sentence_generation.py, stopping pipeline.")
        raise typer.Exit(code=error_code.returncode)

    typer.echo("Pipeline completed successfully.")


if __name__ == "__main__":
    app()
