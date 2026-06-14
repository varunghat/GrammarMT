import nltk, spacy, subprocess, sys

print("Downloading NLTK resources...")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

print("Downloading spaCy model...")
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

print("All resources downloaded successfully.")
