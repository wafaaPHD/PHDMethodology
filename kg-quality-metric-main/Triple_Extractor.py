import os
from collections import defaultdict
import json
import ast  # Use ast.literal_eval for safer string evaluation
import jellyfish  # for Jaro-Winkler similarity
from negation_detector import NegationDetector

# Create /Json folder if it doesn't exist
os.makedirs('Json', exist_ok=True)

# Initialize negation detector
negation_detector = NegationDetector()

def load_polarity_info(polarity_file):
    """
    Load sentence polarity information from polarity file.
    
    Args:
        polarity_file (str): Path to polarity file
        
    Returns:
        dict: Mapping of sentence number to (polarity, sentence_text)
    """
    polarity_map = {}
    try:
        with open(polarity_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and '||' in line:
                    parts = line.split('||', 2)
                    if len(parts) >= 3:
                        sentence_num = int(parts[0])
                        polarity = parts[1]
                        sentence_text = parts[2]
                        polarity_map[sentence_num] = (polarity, sentence_text)
    except FileNotFoundError:
        print(f"Warning: Polarity file {polarity_file} not found. Using default positive polarity.")
    return polarity_map

# def extract_triples(triple_file, pos_file):
#     triples_map = defaultdict(list)
#     with open(triple_file, 'r') as f_triple, open(pos_file, 'r') as f_pos:
#         triple_lines = f_triple.readlines()
#         pos_lines = f_pos.readlines()
        
#         for triple_line in triple_lines:
#             triple_parts = triple_line.strip().split("||")
#             sentence_number = int(triple_parts[0])
#             triple_data = ast.literal_eval(triple_parts[1])  # Safely evaluate the triple data

#             newtriple = []
            
#             if len(triple_data) >= 3:  # Check if triple_data has at least three elements
#                 if sentence_number <= len(pos_lines):
#                     extractsentence = pos_lines[sentence_number - 1].replace('[', '').replace(']', '').replace('\n','')
#                     elements = extractsentence.split(", ")
#                     sentence = [element.strip() for element in elements]

#                     # Check if the entire subject and object are present in the sentence
#                     if any(word in triple_data[0] for word in sentence) and any(word in triple_data[2] for word in sentence):
#                         newtriple.append(triple_data)
            
#             if newtriple:
#                 triples_map[sentence_number].append(triple_data)
            
#     return triples_map

import jellyfish  # for Jaro-Winkler similarity

def has_direct_or_similar_match(text, pos_nouns, threshold=0.50):
    for noun in pos_nouns:
        if noun in text:
            return True  # direct match
        for word in text.split():
            if jellyfish.jaro_winkler_similarity(noun.lower(), word.lower()) >= threshold:
                return True  # similarity match
    return False

def extract_triples_with_polarity(triple_file, pos_file, polarity_file):
    """
    Extract triples and enhance them with polarity information to create 4-tuples.
    
    Args:
        triple_file (str): Path to file containing raw triples
        pos_file (str): Path to file containing extracted noun phrases
        polarity_file (str): Path to file containing sentence polarity information
        
    Returns:
        dict: Mapping of sentence numbers to lists of 4-tuples (subject, predicate, object, polarity)
    """
    triples_map = defaultdict(list)
    
    # Load polarity information
    polarity_map = load_polarity_info(polarity_file)
    
    with open(triple_file, 'r', encoding='utf-8', errors='ignore') as f_triple, \
         open(pos_file, 'r', encoding='utf-8', errors='ignore') as f_pos:
        
        triple_lines = f_triple.readlines()
        pos_lines = f_pos.readlines()

        for triple_line in triple_lines:
            try:
                triple_parts = triple_line.strip().split("||")
                sentence_number = int(triple_parts[0])
                raw = triple_parts[1].replace(r"\/", "/")
                triple_data = ast.literal_eval(raw)
            except:
                triple_parts = triple_line.strip().split("\t")
                sentence_number = int(triple_parts[0])
                triple_data = (triple_parts[1], triple_parts[2], triple_parts[3])
            newtriple = []

            if len(triple_data) >= 3:
                if sentence_number <= len(pos_lines):
                    extractsentence = pos_lines[sentence_number - 1].replace('[', '').replace(']', '').replace('\n','')
                    pos_nouns = [element.strip() for element in extractsentence.split(", ")]

                    subject = triple_data[0]
                    predicate = triple_data[1]
                    obj = triple_data[2]

                    subject_valid = has_direct_or_similar_match(subject, pos_nouns)
                    object_valid = has_direct_or_similar_match(obj, pos_nouns)

                    if subject_valid and object_valid:
                        # Get polarity information for this sentence
                        sentence_polarity, sentence_text = polarity_map.get(sentence_number, ("positive", ""))
                        
                        # If we have the sentence text, try to get more accurate predicate polarity
                        if sentence_text:
                            predicate_polarity = negation_detector.extract_predicate_polarity(sentence_text, predicate)
                        else:
                            predicate_polarity = sentence_polarity
                        
                        # Create 4-tuple: (subject, predicate, object, polarity)
                        enhanced_triple = [subject, predicate, obj, predicate_polarity]
                        newtriple.append(enhanced_triple)

            if newtriple:
                triples_map[sentence_number].extend(newtriple)

    return triples_map

def extract_triples(triple_file, pos_file):
    """
    Extract triples (backward compatibility function - returns 3-tuples).
    
    Args:
        triple_file (str): Path to file containing raw triples
        pos_file (str): Path to file containing extracted noun phrases
        
    Returns:
        dict: Mapping of sentence numbers to lists of 3-tuples (subject, predicate, object)
    """
    triples_map = defaultdict(list)
    with open(triple_file, 'r', encoding='utf-8', errors='ignore') as f_triple, open(pos_file, 'r', encoding='utf-8', errors='ignore') as f_pos:
        triple_lines = f_triple.readlines()
        pos_lines = f_pos.readlines()

        for triple_line in triple_lines:
            try:
                triple_parts = triple_line.strip().split("||")
                sentence_number = int(triple_parts[0])
                raw = triple_parts[1].replace(r"\/", "/")
                triple_data = ast.literal_eval(raw)
            except:
                triple_parts = triple_line.strip().split("\t")
                sentence_number = int(triple_parts[0])
                triple_data = (triple_parts[1], triple_parts[2], triple_parts[3])
            newtriple = []

            if len(triple_data) >= 3:
                if sentence_number <= len(pos_lines):
                    extractsentence = pos_lines[sentence_number - 1].replace('[', '').replace(']', '').replace('\n','')
                    pos_nouns = [element.strip() for element in extractsentence.split(", ")]

                    subject = triple_data[0]
                    obj = triple_data[2]

                    subject_valid = has_direct_or_similar_match(subject, pos_nouns)
                    object_valid = has_direct_or_similar_match(obj, pos_nouns)

                    if subject_valid and object_valid:
                        newtriple.append(triple_data)

            if newtriple:
                triples_map[sentence_number].append(triple_data)

    return triples_map


# List of tool names and input file names
tools_and_files = [    
    ("BAllMyMethod", "BAllMyMethod.txt"),
    ("clausie_benchie_form", "clausie_benchie_form.txt"),
    ("compactie_benchie_form", "compactie_benchie_form.txt"),
    ("imojie_benchie_form","imojie_benchie_form.txt"),
    ("m2oie_benchie_form","m2oie_benchie_form.txt"),
    ("minie_benchie_form","minie_benchie_form.txt"),
    ("openie6_benchie_form","openie6_benchie_form.txt"),
    ("reverb_benchie_form","reverb_benchie_form.txt")
]

# Process tools and files with enhanced polarity-aware extraction
for tool, file in tools_and_files:
    # Check if file exists before processing
    if not os.path.exists(file):
        print(f"Skipping {tool}: File {file} not found")
        continue
    
    print(f"Processing {tool} with polarity detection...")
    
    # Extract triples with polarity (4-tuples)
    triples_with_polarity = extract_triples_with_polarity(
        file, 
        "output_pos_ideal_benchie.txt",
        "output_polarity_ideal_benchie.txt"
    )
    
    # Save the polarity-enhanced output in /Json folder
    polarity_output_file = os.path.join('Json', f"{tool}_with_polarity.json")
    try:
        with open(polarity_output_file, 'w', encoding='utf-8') as json_file:
            json.dump(triples_with_polarity, json_file, ensure_ascii=False, indent=2)
        print(f"Polarity-enhanced triples from {tool} saved to {polarity_output_file}")
    except IOError as e:
        print(f"Failed to write {polarity_output_file}: {e}")
    
    # Also save backward-compatible 3-tuple version
    triples_3tuple = extract_triples(file, "output_pos_ideal_benchie.txt")
    output_file = os.path.join('Json', f"{tool}.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(triples_3tuple, json_file, ensure_ascii=False, indent=2)
        print(f"Traditional triples from {tool} saved to {output_file}")
    except IOError as e:
        print(f"Failed to write {output_file}: {e}")

print("\n=== Triple extraction with polarity detection completed ===")
print("4-tuple files (with polarity): *_with_polarity.json")
print("3-tuple files (backward compatible): *.json")
