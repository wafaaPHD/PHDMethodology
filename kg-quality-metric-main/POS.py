import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from negation_detector import NegationDetector

# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize negation detector
negation_detector = NegationDetector()

# Function to process a sentence and extract relevant phrases with polarity
def extract_phrases_with_polarity(sentence):
    """
    Extract phrases and determine sentence polarity using negation detection.
    
    Args:
        sentence (str): Input sentence to process
        
    Returns:
        tuple: (phrases_list, sentence_polarity)
    """
    # Get sentence polarity information
    polarity_info = negation_detector.get_sentence_polarity_info(sentence)
    sentence_polarity = "negative" if polarity_info["has_negation"] else "positive"
    
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    
    # Perform part-of-speech tagging
    tagged_words = pos_tag(words)
    
    # Extract relevant phrases (nouns/adjectives/CDs)
    n1 = ""   # Temporary string to hold concatenated nouns/adjectives/CDs
    prev = "" # Holds the part-of-speech (POS) of the previous word
    vec = []  # List to store final extracted phrases
    
    # Iterate through tagged words to extract relevant phrases
    for word, pos in tagged_words:
        if prev == "":
            if pos.startswith('N') or pos.startswith('J') or pos == 'CD':
                prev = pos
                n1 += (word + " ")
        elif prev.startswith('N') or prev.startswith('J') or prev == 'CD':
            if pos.startswith('N') or pos.startswith('J') or pos == 'CD':
                n1 += (word + " ")
            else:
                vec.append(n1.strip())
                prev = ""
                n1 = ""
    
    if n1:
        vec.append(n1.strip())
    
    return vec, sentence_polarity

# Function to process a sentence and extract relevant phrases (backward compatibility)
def extract_phrases(sentence):
    """
    Extract phrases from sentence (backward compatibility function).
    
    Args:
        sentence (str): Input sentence to process
        
    Returns:
        list: List of extracted phrases
    """
    phrases, _ = extract_phrases_with_polarity(sentence)
    return phrases

# File paths
input_file_path = 'sentences_benchie.txt'
output_file_path = 'output_pos_ideal_benchie.txt'
polarity_output_file_path = 'output_polarity_ideal_benchie.txt'

# Read sentences from input file and process each line
with open(input_file_path, 'r') as input_file, \
     open(output_file_path, 'w') as output_file, \
     open(polarity_output_file_path, 'w') as polarity_file:
    
    for line_num, line in enumerate(input_file, 1):
        line = line.strip()
        if line:
            phrases, sentence_polarity = extract_phrases_with_polarity(line)
            
            # Write phrases without quotes and in desired format (backward compatibility)
            output_file.write(f"[{', '.join(phrases)}]\n ")
            
            # Write polarity information for each sentence
            polarity_file.write(f"{line_num}||{sentence_polarity}||{line}\n")

print(f"Phrase extraction completed. Output written to {output_file_path}")
print(f"Polarity information written to {polarity_output_file_path}")
