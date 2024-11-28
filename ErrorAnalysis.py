# Sample system-generated triples
system_generated_triples = []
with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\carb_sentences\\benchie_gold_annotations_enMyMethod.txt", "r") as file:
    text_data = file.readlines()


for x in text_data:
  tri=x.split('\t')
  for x2 in tri[2].split(', '):  
   if (len(x2)<6 or len(tri[1]) < 6) or (len(tri[3].split('\n')[0]) < 6):       
    tribles = {}    
    tribles["subject"]=tri[1]
    tribles["predicate"]=x2
    tribles["object"]=tri[3].split('\n')[0]
    system_generated_triples.append(tribles)
# Calculate Average Constituent Length (ACL)
acl_scores = [len(triple["subject"]) + len(triple["predicate"]) + len(triple["object"]) for triple in system_generated_triples]
average_acl = sum(acl_scores) / len(acl_scores)

# Calculate Number of Constituent Clauses (NCC)
ncc_scores = [len(triple["predicate"].split()) for triple in system_generated_triples]
average_ncc = sum(ncc_scores) / len(ncc_scores)

# Calculate Repetitions Per Argument (RPA)
total_arguments = sum([len(triple["subject"]) + len(triple["object"]) for triple in system_generated_triples])
unique_arguments = len(set([triple["subject"] for triple in system_generated_triples] + [triple["object"] for triple in system_generated_triples]))
rpa_score = total_arguments / unique_arguments

# Output the calculated metrics
print("Average Constituent Length (ACL):", average_acl)
print("Number of Constituent Clauses (NCC):", average_ncc)
print("Repetitions Per Argument (RPA):", rpa_score)
