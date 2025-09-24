import typer
import subprocess
from pathlib import Path

app = typer.Typer()

@app.command()
def run_pipeline(filename: str):
    # Step 1: Call pdf_parser.py with the PDF filename
    typer.echo(f"Running pdf_parser.py on {filename}")
    subprocess.run(["python", "pdf_parser.py", filename], check=True)

    # Step 2: Call section_tagger.py with the output JSON
    sections_json = Path("parsed_grammar_json") / (Path(filename).stem + "_sections.json")
    typer.echo(f"Running section_tagger.py on {sections_json}")
    subprocess.run(["python", "section_tagger.py", str(sections_json)], check=True)

    # Step 3: Call rule_extraction.py with the next output JSON
    tagged_json = Path("classified_json") / (Path(filename).stem + "_sections_tagged.json")
    typer.echo(f"Running rule_extraction.py on {tagged_json}")
    subprocess.run(["python", "rule_extraction.py", str(tagged_json)], check=True)

if __name__ == "__main__":
    app()