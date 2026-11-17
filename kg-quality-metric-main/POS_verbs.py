import spacy
import json
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def extract_triples_with_and_replacement(text):
    text_with_and = text.replace(",", " and")
    sentences = []
    doc = nlp(text_with_and)
    for sent in doc.sents:
        sentences.append(str(sent))
    return sentences

def extract_triples_for_sentence(sentence):
    doc = nlp(sentence)
    triples = []
    last_subject = None

    for sent in doc.sents:
        subject = None
        for token in sent:
            if token.dep_ == "nsubj":
                subject = " ".join([tok.text for tok in token.subtree])
                last_subject = subject
                break

        if not subject:
            subject = last_subject

        for token in sent:
            if token.pos_ in ["VERB", "AUX"]:
                verb = token.text
                obj = None

                for child in token.children:
                    if child.dep_ in ["attr", "dobj"]:
                        obj = " ".join([tok.text for tok in child.subtree])
                        break
                    elif child.dep_ == "prep":
                        pobj = next((tok for tok in child.children if tok.dep_ == "pobj"), None)
                        if pobj:
                            obj = f"{child.text} " + " ".join([tok.text for tok in pobj.subtree])
                            break

                if not obj:
                    for child in token.children:
                        if child.text.lower() == "as":
                            pobj = next((tok for tok in child.children if tok.dep_ == "pobj"), None)
                            if pobj:
                                obj = f"as " + " ".join([tok.text for tok in pobj.subtree])
                                break

                if verb == "was" and not any(triple for triple in triples if triple[1] == verb):
                    if "by" in [child.text for child in token.children]:
                        agent = next((tok for tok in token.children if tok.dep_ == "agent"), None)
                        if agent:
                            obj = f"by {agent.text}"

                if subject and obj:
                    triples.append((subject, verb, obj))
                elif subject:
                    triples.append((subject, verb, None))
                elif obj:
                    triples.append((None, verb, obj))

    return triples

def refine_triples_for_kg_with_and(triples):
    refined_triples = []
    for subject, verb, obj in triples:
        if obj and "and" in obj:
            obj_parts = [part.strip() for part in obj.split(" and ")]
            for part in obj_parts:
                if "as" in part:
                    refined_triples.append((subject, verb, part))
                else:
                    refined_triples.append((subject, verb, f"as {part}"))
        else:
            refined_triples.append((subject, verb, obj))
    return refined_triples

# Main driver: Read input, process, write output
output_lines = []

with open("sentences_benchie.txt", "r") as infile:
    lines = infile.readlines()

for line in lines:
    line = line.strip()
    if not line:
        output_lines.append("[]")
        continue

    sentences = extract_triples_with_and_replacement(line)
    all_refined_triples = []

    for sentence in sentences:
        triples = extract_triples_for_sentence(sentence)
        refined_triples = refine_triples_for_kg_with_and(triples)
        all_refined_triples.extend(refined_triples)

    predicates = [verb for _, verb, _ in all_refined_triples]
    output_lines.append(json.dumps(predicates))

with open("output_verbs_ideal_sentences_benchie.txt", "w") as outfile:
    for line in output_lines:
        outfile.write(line + "\n")
