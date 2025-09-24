import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import typer
import spacy
from sentence_transformers import util
import re
from openai import OpenAI
import yaml

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



def main(
    filename: str = typer.Argument(..., help="Path to the input 'tagged' JSON file")
):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))

    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    ###
    sentences_data = []
    filtered_out_data = []

    # Tags to filter out with the threshold
    tags_to_filter = {
        "Publishing information": 1.0,
        "Structure": 2.0,
        "Culture": 2.0,
        "History": 2.0,
        "Phonetics": 2.0
    }

    filtered_idx=0
    section_to_filtererd_mapping = {}

    for idx,section in enumerate(data):
        sorted_tags = section["sorted_tags"]
        sorted_tag_names = [tag[0] for tag in sorted_tags]
        tag_word_counts = section["tag_word_counts"]
        tag_counts_heading = section["tag_counts_heading"]
    
        # Track filtering reason
        filter_reason = None
        
        # Filter out sections with high scores in filtered tags
        if any(tag[0] in tags_to_filter and tag[1] >= tags_to_filter[tag[0]] for tag in sorted_tags):
            filter_reason = f"High score in filtered tag: {[tag[0] for tag in sorted_tags if tag[0] in tags_to_filter and tag[1] >= tags_to_filter[tag[0]]]}"
            section['filtered'] = {'status': True, 'reason': filter_reason}
            filtered_out_data.append(section)
            continue

        # Check if there are any other tags that are not in tags_to_filter
        other_tags = [tag for tag in sorted_tag_names if tag not in tags_to_filter]
        if not other_tags:
            filter_reason = "No tags other than filtered tags"
            section['filtered'] = {'status': True, 'reason': filter_reason}
            filtered_out_data.append(section)
            continue

        text = section["text"]
        
        # Use spacy to split the text into sentences
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        
        # Create sentence data structure with filtering info
        section_data = {
            "text": text,
            "sentences": sentences,
            "sorted_tags": sorted_tags,
            "tagged_words": tag_word_counts,
            "tagged_words_heading": tag_counts_heading,
            "filtered": {"status": False, "reason": None}
        }
        
        sentences_data.append(section_data)
        section_to_filtererd_mapping[idx] = filtered_idx
        filtered_idx += 1

        section['filtered'] = {'status': False, 'reason': None}
        section["sentences"] = sentences

    print(f"Total sections: {len(data)}")
    print(f"Filtered sections: {len(filtered_out_data)}")
    print(f"Remaining sections: {len(sentences_data)}")

    # Save the filtered mapping to a JSON file
    output_file = Path(f"scratch/{Path(filename).stem}_filtered_mapping.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(section_to_filtererd_mapping, f, ensure_ascii=False, indent=4)
    print(f"Filtered mapping saved to {output_file}")


    all_sections = []
    for section in data:
        text = section.get("text", "")
        if not text:
            continue
        filtered = section.get("filtered",None)
        if filtered["status"] == True:
            continue
        all_sections.append(text)

    print(f"Total sections after filtering: {len(all_sections)}")



    sections_split = []

    for idx,section in tqdm(enumerate(all_sections)):
        section_length = len(section.split())
        if section_length > 250:
            print("Length = ",section_length)
            #print(section[:30] + "......" + section[-30:])
            # split into sentences
            sentences = [sent.text.strip() for sent in nlp(section).sents]
            print(sentences)
            print(len(sentences))
            
        
            embeddings = model.encode(sentences)
            similarities = [util.cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(embeddings)-1)]
            print(len(similarities),similarities)

            # Get indices of similarities in ascending order of similarity
            # Get indices sorted by similarity in ascending order (least similar pairs first)
            split_indices = [i for i in range(len(similarities))]
            split_indices = sorted(split_indices, key=lambda x: similarities[x])
            split_indices = [i+1 for i in split_indices]
            print(split_indices)

            # No of splits
            n_splits = section_length // 200
            remainder = section_length % 200

            #print(n_splits,end='-')
            threshold = 50
            if (remainder / n_splits > threshold):
                n_splits+=1

            print("N_splits:",n_splits)

            temp = 0
            split_region = len(split_indices)/n_splits

            split_regions = [i*split_region for i in range(1,n_splits)]
            #print(split_regions)

            split_region_threshold = n_splits-1

            split_regions_range = []
            for region in split_regions:
                split_region = (region - split_region_threshold, region + split_region_threshold)
            
                split_region = (int(split_region[0]), int(split_region[1]))
                split_regions_range.append(split_region)


            print(split_regions_range)
            

            #print("Split region: ", split_region)
            #print(split_regions)

            iter = 0
            split_indices_result = []
            while(n_splits!=0 and iter < len(split_indices)):
    
                idx = split_indices[iter]
                # Check if the index is in the split region of any of the split regions
                for split_region in split_regions_range:
                    if idx >= split_region[0] and idx <= split_region[1]:
                        split_indices_result.append(idx)
                        n_splits -= 1
                        # Delete the region from the split regions
                        split_regions_range.remove(split_region)
                        print(f"Adding index {idx} to split indices result. Remaining splits: {n_splits}")
                        break
                    
                    #split_indices_result.append(split_indices[idx])
                    #n_splits -= 1
                iter += 1
            split_indices_result = sorted(set(split_indices_result))  # Remove duplicates and sort
            print("Split indices: ",split_indices_result)

            # Split the sentences into multiple paragraphs based on the split indices and then combine them into a single string 
            paragraphs = []
            start_idx = 0
            for idx in split_indices_result:
                if start_idx < idx:
                    paragraphs.append(" ".join(sentences[start_idx:idx]))
                    start_idx = idx
            if start_idx < len(sentences):
                paragraphs.append(" ".join(sentences[start_idx:]))
            #print("No of paragraphs: ",len(paragraphs))
            
            # Length of each paragraph
            #for i, paragraph in enumerate(paragraphs):
            #    print(f"{i+1} - Paragraph: {paragraph} \nlength: {len(paragraph.split())}")  
            # Replace the section with the paragraphs

            sections_split.append(paragraphs)
            #print("====================================")
        else:
            sections_split.append([section])

    print("No of sections: ",len(sections_split))
    all_sections = []
    for section in sections_split:
        all_sections.extend(section)
    print("No of total paragraphs: ",len(all_sections))

    ####################################################################################
    # Use gpt to extract rules from each paragraph using the prompt

    base_prompt = f"""You are an expert in computational linguistics.

    You will be given a paragraph from a descriptive grammar of the Kannada language.

    Your task is to extract all explicit or implicit grammatical rules described in the paragraph. Each rule should be expressed in both plain language and formal terms.
    If the rule cannot be expressed in a formal rewrite rule, leave the fst_rule field empty. If the paragraph does not describe any rules, return a description stating that no rules are described.
    Do not make up rules or provide explanations beyond the scope of the paragraph. Do not deduce rules from the sentence grammar, only extract rules that are explicitly stated in the paragraph.

    In addition, represent relevant morphological features using the UniMorph standard.

    The UniMorph format is a standardized, language-independent way of representing morphological features using uppercase tags separated by semicolons. 
    For example, a verb in past tense, third person singular is annotated as: V;PST;3;SG

    Follow these conventions for UniMorph:
    - Verbs (V): Annotate with Tense (PST, PRS, FUT), Person (1, 2, 3), Number (SG, PL), Mood (IND, SBJV, IMP), Aspect (IPFV, PFV), Voice (ACT, PASS), and Polarity (POS, NEG)
    - Nouns (N): Annotate with Number (SG, PL), Case (NOM, ACC, DAT, etc.), Gender (MASC, FEM, NEUT), Possession (POSS) where applicable
    - Adjectives (ADJ): Annotate with Degree (POS, CMP, SPRL), and if applicable, Number, Case, Gender
    - Adverbs (ADV): Annotate with Degree (POS, CMP, SPRL), and if applicable, Polarity (POS, NEG)
    - Pronouns (PRON): Annotate with Person (1, 2, 3), Number (SG, PL), Case (NOM, ACC, DAT, etc
    - Use POS tags like V (verb), N (noun), ADJ (adjective), ADV (adverb), PRON (pronoun), DET (determiner), etc.

    For each rule, output the following fields in YAML:

    - description: A simple plain-language summary of what the rule does
    - condition: When or where the rule applies
    - action: What transformation is made
    - fst_rule: A formal rewrite rule (use the format: X → Y / L ___ R).
    Include the relevant UniMorph feature(s) in brackets at the end of the rule, using the format:
    
    stem → stem+SUFFIX [UNIMORPH] / L ___ R

    For example:
    stem → stem+gaLu [N;PL] / ___ #

    Leave the field blank if the transformation cannot be expressed with a finite-state rewrite rule.

    - usage: Whether the rule is used in practice
    - unimorph: A UniMorph feature string (e.g., V;PST;3;SG)
    - grammar_tags: A list of grammar tags that apply to the rule (e.g., "Gender", "Plural", "Past Tense", "Third Person", "Singular", "Indicative Mood", "Active Voice", etc.) 

    If multiple rules are described in the paragraph, output multiple YAML blocks — one per rule.

    Do not include any explanation. Output only the YAML and nothing else.

    Paragraph: {{input_paragraph}}
    """

    
    with open("api_key.txt", "r") as f:
        openai_api_key = f.read().strip()

    
    client = OpenAI(api_key=openai_api_key)

    gpt_extracted_rules_direct = []
    # Set limit for API calls
    LIMIT = -1
    # Iterate through the sections
    print("Total sections to process:", len(sections_split))
    api_model = "gpt-4o-mini"
    for section in tqdm(sections_split):
        if LIMIT == 0:
            break
        
        LIMIT -= 1   
        # Process each section
        temp = []
        for paragraph in section:
            response = client.responses.create(
                model=api_model,
                input=base_prompt.format(input_paragraph=paragraph)
            )
            
            temp.append(response.output[0].content[0].text)
        gpt_extracted_rules_direct.append(temp)

    with open("scratch/gpt_extracted_rules_direct.json", "w", encoding="utf-8") as f:
        json.dump(gpt_extracted_rules_direct, f, ensure_ascii=False, indent=4)



    # clean the data
    for i, section in enumerate(gpt_extracted_rules_direct):
        if section is None:
            continue
        for j, response in enumerate(section):
            if response is None:
                continue
            print(response)
            # Remove code block markers and extra spaces
            cleaned_response = response.strip().replace("```yaml", "").replace("```", "")
            # Parse the cleaned YAML response
            try:
                parsed_response = yaml.safe_load(cleaned_response)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML for section {i}, paragraph {j}: {e}")
                print(parsed_response)
                parsed_response = None
            # Update the response in the list
            if parsed_response is not None:
                # Ensure the cleaned response is a dictionary
                if isinstance(parsed_response, dict):
                    parsed_response = [parsed_response]
                elif not isinstance(parsed_response, list):
                    print(f"Unexpected format for section {i}, paragraph {j}: {parsed_response}")
                    parsed_response = []
            gpt_extracted_rules_direct[i][j] = parsed_response
    # Store the responses in a JSON file
    with open(f"extracted_rules/{Path(filename).stem}_gpt_extracted_rules_direct_parsed.json", "w", encoding="utf-8") as f:
        json.dump(gpt_extracted_rules_direct, f, ensure_ascii=False, indent=4)

    print("Extraction complete. Results saved to", f"extracted_rules/{Path(filename).stem}_gpt_extracted_rules_direct_parsed.json")



if __name__ == "__main__":
    typer.run(main)

    