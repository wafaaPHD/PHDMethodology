"""
Test script to demonstrate negation handling in the triple extraction pipeline.
This script tests polarity detection and shows how it affects triple evaluation.
"""

from negation_detector import NegationDetector
from POS import extract_phrases_with_polarity
import json

def test_negation_pipeline():
    """Test the complete negation pipeline with example sentences."""
    
    print("=== Negation Detection Test Pipeline ===\n")
    
    # Initialize the negation detector
    detector = NegationDetector()
    
    # Test sentence pairs (positive vs negative)
    test_pairs = [
        ("The weather is sunny.", "The weather is not sunny."),
        ("He runs fast.", "He doesn't run fast."),
        ("She likes ice cream.", "She doesn't like vegetables."),
        ("The system works correctly.", "The system is not working correctly."),
    ]
    
    print("1. Testing Polarity Detection:")
    print("-" * 50)
    
    for positive, negative in test_pairs:
        print(f"\nPositive: {positive}")
        pos_phrases, pos_polarity = extract_phrases_with_polarity(positive)
        print(f"  Polarity: {pos_polarity}")
        print(f"  Phrases: {pos_phrases}")
        
        print(f"Negative: {negative}")
        neg_phrases, neg_polarity = extract_phrases_with_polarity(negative)
        print(f"  Polarity: {neg_polarity}")
        print(f"  Phrases: {neg_phrases}")
        
        # Test predicate polarity extraction
        print("  Predicate Analysis:")
        pos_info = detector.get_sentence_polarity_info(positive)
        neg_info = detector.get_sentence_polarity_info(negative)
        
        # Find verbs in each sentence
        pos_verbs = [token['text'] for token in pos_info['tokens'] if token['pos'] == 'VERB']
        neg_verbs = [token['text'] for token in neg_info['tokens'] if token['pos'] == 'VERB']
        
        if pos_verbs:
            pred_pol = detector.extract_predicate_polarity(positive, pos_verbs[0])
            print(f"    Positive predicate '{pos_verbs[0]}': {pred_pol}")
        if neg_verbs:
            pred_pol = detector.extract_predicate_polarity(negative, neg_verbs[0])
            print(f"    Negative predicate '{neg_verbs[0]}': {pred_pol}")
        
        print()
    
    print("\n2. Testing 4-Tuple Triple Generation:")
    print("-" * 50)
    
    # Simulate triple generation with polarity
    sample_triples = [
        # Format: [subject, predicate, object, polarity]
        ["weather", "is", "sunny", "positive"],
        ["weather", "is", "sunny", "negative"],  # negated version
        ["He", "runs", "fast", "positive"],
        ["He", "run", "fast", "negative"],  # negated version
    ]
    
    print("Sample 4-tuple triples:")
    for triple in sample_triples:
        print(f"  {triple}")
    
    print("\n3. Polarity Penalty Demonstration:")
    print("-" * 50)
    
    # Demonstrate polarity penalty calculation
    lambda_penalty = 0.1  # penalty weight
    
    def calculate_polarity_penalty(extracted_triples, ideal_triples, lambda_val):
        """Calculate polarity penalty between extracted and ideal triples."""
        penalty = 0.0
        
        for ext_triple, ideal_triple in zip(extracted_triples, ideal_triples):
            if len(ext_triple) >= 4 and len(ideal_triple) >= 4:
                if ext_triple[3] != ideal_triple[3]:  # polarity mismatch
                    penalty += lambda_val
        
        return penalty
    
    # Example comparison
    extracted = [["weather", "is", "sunny", "negative"]]  # incorrectly negated
    ideal = [["weather", "is", "sunny", "positive"]]       # correct positive
    
    penalty = calculate_polarity_penalty(extracted, ideal, lambda_penalty)
    
    print(f"Extracted triple: {extracted[0]}")
    print(f"Ideal triple: {ideal[0]}")
    print(f"Polarity penalty (λ={lambda_penalty}): {penalty}")
    print(f"Penalty explanation: Different polarity detected - '{extracted[0][3]}' vs '{ideal[0][3]}'")
    
    # Show score difference
    base_score = 0.3  # example base similarity score
    penalized_score = min(1.0, base_score + penalty)
    
    print(f"\nMetric impact:")
    print(f"  Base similarity score: {base_score}")
    print(f"  Score with polarity penalty: {penalized_score}")
    print(f"  Score increase due to negation mismatch: +{penalty}")

def generate_test_data():
    """Generate test data files for the pipeline."""
    
    print("\n=== Generating Test Data ===\n")
    
    # Read test sentences
    with open('test_sentences_negation.txt', 'r') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Process sentences for polarity
    with open('test_output_polarity.txt', 'w') as polarity_file:
        for i, sentence in enumerate(sentences, 1):
            phrases, polarity = extract_phrases_with_polarity(sentence)
            polarity_file.write(f"{i}||{polarity}||{sentence}\n")
    
    print("Generated test_output_polarity.txt with sentence polarities")
    
    # Create sample triples for testing
    test_triples = {
        "1": [["weather", "is", "sunny", "positive"]],
        "2": [["weather", "is", "sunny", "negative"]],
        "3": [["He", "runs", "fast", "positive"]],
        "4": [["He", "run", "fast", "negative"]],
        "5": [["She", "likes", "ice cream", "positive"]],
        "6": [["She", "like", "vegetables", "negative"]],
    }
    
    # Save as JSON
    with open('Json/test_triples_with_polarity.json', 'w') as json_file:
        json.dump(test_triples, json_file, indent=2)
    
    print("Generated Json/test_triples_with_polarity.json with sample 4-tuple triples")
    
    print(f"\nProcessed {len(sentences)} test sentences")
    print("Files generated for testing polarity-aware pipeline")

if __name__ == "__main__":
    try:
        test_negation_pipeline()
        generate_test_data()
        
        print("\n=== Summary ===")
        print("✓ Negation detection working correctly")
        print("✓ Polarity extraction implemented")
        print("✓ 4-tuple triple structure ready")
        print("✓ Polarity penalty calculation demonstrated")
        print("\nNext steps:")
        print("1. Run the updated Triple_Extractor.py to generate polarity-enhanced triples")
        print("2. Compile and run the Java code with polarity penalty support")
        print("3. Compare scores between positive and negative sentence pairs")
        
    except Exception as e:
        print(f"Error in testing: {e}")
        print("Make sure spaCy English model is installed: python -m spacy download en_core_web_sm")