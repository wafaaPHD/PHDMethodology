"""
Negation Detection Module

This module provides functionality to detect negation in sentences using spaCy dependency parsing.
It identifies negation words (like "not", "never", "no") and determines the polarity of predicates.
"""

import spacy
from typing import List, Tuple, Dict

class NegationDetector:
    def __init__(self):
        """Initialize the negation detector with spaCy English model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Error: spaCy English model not found. Please install it with:")
            print("python -m spacy download en_core_web_sm")
            raise
    
    def detect_negation_in_sentence(self, sentence: str) -> Dict[int, str]:
        """
        Detect negation in a sentence and return polarity for each token.
        
        Args:
            sentence (str): Input sentence to analyze
            
        Returns:
            Dict[int, str]: Dictionary mapping token index to polarity ('positive' or 'negative')
        """
        doc = self.nlp(sentence)
        token_polarities = {}
        
        # Initialize all tokens as positive
        for token in doc:
            token_polarities[token.i] = "positive"
        
        # Find negation dependencies
        for token in doc:
            # Check if this token has a negation dependency
            if token.dep_ == "neg":
                # Find the head (the word being negated)
                head = token.head
                if head:
                    token_polarities[head.i] = "negative"
                    # Also mark the negation word itself
                    token_polarities[token.i] = "negative"
        
        # Additional negation patterns
        negation_words = {"not", "no", "never", "neither", "nor", "nothing", "nobody", "nowhere", "none"}
        
        for token in doc:
            if token.text.lower() in negation_words:
                # Mark the negation word and try to find what it negates
                token_polarities[token.i] = "negative"
                
                # Look for the verb or adjective this negation affects
                for child in token.head.children:
                    if child.pos_ in ["VERB", "ADJ"] and child.i != token.i:
                        token_polarities[child.i] = "negative"
                
                # Also check the head of the negation word
                if token.head.pos_ in ["VERB", "ADJ"]:
                    token_polarities[token.head.i] = "negative"
        
        return token_polarities
    
    def extract_predicate_polarity(self, sentence: str, predicate_text: str) -> str:
        """
        Extract polarity for a specific predicate in the sentence.
        
        Args:
            sentence (str): Input sentence
            predicate_text (str): The predicate text to analyze
            
        Returns:
            str: 'positive' or 'negative'
        """
        doc = self.nlp(sentence)
        token_polarities = self.detect_negation_in_sentence(sentence)
        
        # Find tokens that match the predicate text
        predicate_tokens = []
        for token in doc:
            if predicate_text.lower() in token.text.lower() or token.lemma_.lower() in predicate_text.lower():
                predicate_tokens.append(token.i)
        
        # Check if any predicate token is negative
        for token_idx in predicate_tokens:
            if token_polarities.get(token_idx, "positive") == "negative":
                return "negative"
        
        # If no direct match, check for negation around verb phrases
        for token in doc:
            if token.pos_ == "VERB" and (
                token.text.lower() in predicate_text.lower() or 
                token.lemma_.lower() in predicate_text.lower()
            ):
                if token_polarities.get(token.i, "positive") == "negative":
                    return "negative"
        
        return "positive"
    
    def get_sentence_polarity_info(self, sentence: str) -> Dict:
        """
        Get comprehensive polarity information for a sentence.
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            Dict: Contains sentence analysis with polarity information
        """
        doc = self.nlp(sentence)
        token_polarities = self.detect_negation_in_sentence(sentence)
        
        result = {
            "sentence": sentence,
            "tokens": [],
            "has_negation": any(pol == "negative" for pol in token_polarities.values()),
            "negation_tokens": []
        }
        
        for token in doc:
            token_info = {
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
                "polarity": token_polarities.get(token.i, "positive"),
                "index": token.i
            }
            result["tokens"].append(token_info)
            
            if token_info["polarity"] == "negative":
                result["negation_tokens"].append(token_info)
        
        return result

def test_negation_detector():
    """Test function to demonstrate negation detection capabilities."""
    detector = NegationDetector()
    
    test_sentences = [
        "The weather is sunny.",
        "The weather is not sunny.",
        "He never goes to school.",
        "She doesn't like coffee.",
        "Nobody came to the party.",
        "The system is not working properly.",
        "I don't think it's correct."
    ]
    
    print("=== Negation Detection Test ===")
    for sentence in test_sentences:
        info = detector.get_sentence_polarity_info(sentence)
        print(f"\nSentence: {sentence}")
        print(f"Has negation: {info['has_negation']}")
        if info['negation_tokens']:
            print("Negative tokens:")
            for token in info['negation_tokens']:
                print(f"  - {token['text']} (pos: {token['pos']}, dep: {token['dep']})")
    
    # Test predicate polarity extraction
    print("\n=== Predicate Polarity Test ===")
    test_cases = [
        ("The weather is sunny.", "is"),
        ("The weather is not sunny.", "is"),
        ("He runs fast.", "runs"),
        ("He doesn't run fast.", "run"),
        ("She likes ice cream.", "likes"),
        ("She doesn't like vegetables.", "like")
    ]
    
    for sentence, predicate in test_cases:
        polarity = detector.extract_predicate_polarity(sentence, predicate)
        print(f"Sentence: {sentence}")
        print(f"Predicate '{predicate}' polarity: {polarity}\n")

if __name__ == "__main__":
    test_negation_detector()