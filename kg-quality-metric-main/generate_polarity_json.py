"""
Generate JSON files with polarity information for the ideal triples.
"""

import json
import os
from negation_detector import NegationDetector

def generate_polarity_enhanced_json():
    detector = NegationDetector()
    
    # Read existing JSON file
    with open('Json_Ideal/triples_benchie.json', 'r') as f:
        original_data = json.load(f)
    
    # Read sentences
    with open('sentences_benchie.txt', 'r') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    polarity_enhanced_data = {}
    
    for sentence_num, sentence in enumerate(sentences, 1):
        if str(sentence_num) in original_data:
            triples = original_data[str(sentence_num)]
            enhanced_triples = []
            
            for triple in triples:
                if len(triple) >= 3:
                    subject, predicate, obj = triple[0], triple[1], triple[2]
                    
                    # Get predicate polarity
                    predicate_polarity = detector.extract_predicate_polarity(sentence, predicate)
                    
                    # Create 4-tuple: [subject, predicate, object, polarity]
                    enhanced_triple = [subject, predicate, obj, predicate_polarity]
                    enhanced_triples.append(enhanced_triple)
            
            polarity_enhanced_data[str(sentence_num)] = enhanced_triples
    
    # Save enhanced JSON
    os.makedirs('Json_Ideal_Polarity', exist_ok=True)
    
    with open('Json_Ideal_Polarity/triples_benchie.json', 'w') as f:
        json.dump(polarity_enhanced_data, f, indent=2)
    
    print("Generated Json_Ideal_Polarity/triples_benchie.json with polarity information")

    ## Also generate for benchie
    #try:
    #    with open('Json_Ideal/benchie.json', 'r') as f:
    #        benchie_data = json.load(f)
        
    #    with open('sentences_ideal_benchie.txt', 'r') as f:
    #        benchie_sentences = [line.strip() for line in f if line.strip()]
        
    #    benchie_polarity_data = {}
        
    #    for sentence_num, sentence in enumerate(benchie_sentences, 1):
    #        if str(sentence_num) in benchie_data:
    #            triples = benchie_data[str(sentence_num)]
    #            enhanced_triples = []
                
    #            for triple in triples:
    #                if len(triple) >= 3:
    #                    subject, predicate, obj = triple[0], triple[1], triple[2]
    #                    predicate_polarity = detector.extract_predicate_polarity(sentence, predicate)
    #                    enhanced_triple = [subject, predicate, obj, predicate_polarity]
    #                    enhanced_triples.append(enhanced_triple)
                
    #            benchie_polarity_data[str(sentence_num)] = enhanced_triples
        
    #    with open('Json_Ideal_Polarity/benchie.json', 'w') as f:
    #        json.dump(benchie_polarity_data, f, indent=2)
        
    #    print("Generated Json_Ideal_Polarity/benchie.json with polarity information")
    
    #except FileNotFoundError as e:
    #    print(f"Warning: Could not generate benchie polarity data: {e}")

if __name__ == "__main__":
    generate_polarity_enhanced_json()