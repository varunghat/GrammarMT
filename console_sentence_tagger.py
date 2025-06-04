import json
import os

# Load existing tagged data if it exists
tagged_data = []
if os.path.exists("sentences_data_tagged.json"):
    with open("sentences_data_tagged.json", "r", encoding="utf-8") as f:
        tagged_data = json.load(f)

with open("sentences_data.json","r",encoding="utf-8") as f:
    sentences_data = json.load(f)

    # Skip sections that have already been processed
    processed_count = len(tagged_data)
    sentences_processed = 0
    total_sentences = sum(len(section["sentences"]) for section in sentences_data)

    for i, section in enumerate(sentences_data):
        # Skip if this section was already tagged
        if i < processed_count:
            continue
            
        # Create a copy of the current document
        tagged_document = section.copy()
        sentences = section["sentences"]
        
        print(f"\nProcessing section {i+1} of {len(sentences_data)}...")
        for sentence in sentences:
            # Skip already processed sentences
            if sentences_processed < processed_count:
                sentences_processed += 1
                continue

            # Clear screen before showing each sentence
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\nProcessing sentence {sentences_processed + 1} of {total_sentences}...")
            print("\nSentence:", sentence)
            print("Commands: y (yes), ENTER (skip), z (undo last), Q (quit)")
            response = input("Does this contain a grammar rule? ").lower()
            
            if response == 'q':
                # Save and exit
                with open("sentences_data_tagged.json", "w", encoding="utf-8") as f:
                    json.dump(tagged_data, f, ensure_ascii=False, indent=4)
                exit()
            
            if response == 'z' and tagged_data:
                # Remove the last entry and decrement counters
                tagged_data.pop()
                sentences_processed -= 1
                # Save the updated data
                with open("sentences_data_tagged.json", "w", encoding="utf-8") as f:
                    json.dump(tagged_data, f, ensure_ascii=False, indent=4)
                continue
            
            # Create a tagged sentence entry
            tagged_sentence = {
                "sentence": sentence,
                "rule_tag": response == 'y'
            }
                
            # Save after each response to preserve progress
            tagged_data.append(tagged_sentence)
            with open("sentences_data_tagged.json", "w", encoding="utf-8") as f:
                json.dump(tagged_data, f, ensure_ascii=False, indent=4)
            
            sentences_processed += 1
                
    print("\nCompleted tagging all sentences!")
