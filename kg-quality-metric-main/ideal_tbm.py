# import csv

# def process_csv(input_file, output_file):
#     with open(input_file, mode='r', encoding='utf-8') as infile, \
#          open(output_file, mode='w', encoding='utf-8') as outfile:
        
#         reader = csv.reader(infile)

#         line_count = 0
#         for row in reader:
#             if len(row) >= 3:
#                 combined = row[1].strip() + " " + row[2].strip()
#                 outfile.write(combined + '\n')
#                 line_count += 1

#         print(f"✅ Total combined sentences written: {line_count}")

# # Run
# process_csv('data.csv', 'sentences_ideal_tinybutmighty.txt')

import json
import re

def clean_phrase(phrase):
    # Remove optional markers like [the], [a] etc.
    return ' '.join(re.sub(r"[\[\]]", "", word).strip() for word in phrase.split())

def parse_triples_from_file(input_file):
    output = {}
    current_id = None

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Match new sentence ID and sentence
            sent_match = re.match(r"sent_id:(\d+)\s+(.*)", line)
            if sent_match:
                current_id = sent_match.group(1)
                output[current_id] = []
                continue

            # Skip cluster headers
            if re.match(r"\d+-->\s*Cluster\s*\d+:", line):
                continue

            # Match triple line: Subject --> Predicate --> Object
            triple_match = re.match(r"(.*?)-->(.*?)-->(.*)", line)
            if triple_match and current_id:
                subj = clean_phrase(triple_match.group(1).strip())
                pred = clean_phrase(triple_match.group(2).strip())
                obj = clean_phrase(triple_match.group(3).strip())

                output[current_id].append([subj, pred, obj])

    return output

def main():
    input_file = 'triples_benchie.txt'   # <-- Replace with your actual file name
    output_file = 'Json/triples_benchie.json'

    triples_json = parse_triples_from_file(input_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(triples_json, f, indent=2)

    print(f"Triples written to {output_file}")

if __name__ == "__main__":
    main()
