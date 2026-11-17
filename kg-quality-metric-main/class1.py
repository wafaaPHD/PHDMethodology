import json
import math
import os
from typing import List, Dict, Set, Tuple, Optional
from textdistance import JaroWinkler

# Initialize Jaro-Winkler similarity
jw = JaroWinkler()

# Class to represent a graph node
class GraphNode:
    def __init__(self, values: List[str]):
        self.values = values
        self.adjacent_nodes = []

# Class to represent an undirected graph
class Graph:
    def __init__(self):
        self.nodes = {}

    # Method to add an edge between two nodes (undirected)
    def add_edge(self, node1: List[str], node2: List[str]):
        key1 = self._get_key(node1)
        key2 = self._get_key(node2)

        if key1 not in self.nodes:
            self.nodes[key1] = GraphNode(node1)
        if key2 not in self.nodes:
            self.nodes[key2] = GraphNode(node2)
        
        self.nodes[key1].adjacent_nodes.append(key2)
        self.nodes[key2].adjacent_nodes.append(key1)

    # Helper method to get a string key from a list of values
    def _get_key(self, values: List[str]) -> str:
        return '_'.join([v.lower() for v in values]) + '_'

    # Method to perform depth-first search
    def _dfs(self, key: str, visited: Set[str], component: List[str]):
        visited.add(key)
        component.extend(self.nodes[key].values)
        for adj_key in self.nodes[key].adjacent_nodes:
            if adj_key not in visited:
                self._dfs(adj_key, visited, component)

    # Method to calculate components
    def get_components(self) -> List[List[str]]:
        visited = set()
        components = []
        for key in self.nodes:
            if key not in visited:
                component = []
                self._dfs(key, visited, component)
                components.append(component)
        return components

    # Method to check if an edge exists
    def edge_exists(self, node1: List[str], node2: List[str]) -> bool:
        key1 = self._get_key(node1)
        key2 = self._get_key(node2)
        return (key1 in self.nodes and key2 in self.nodes and 
                key2 in self.nodes[key1].adjacent_nodes)

    # Method to print the graph
    def print_graph(self):
        for key in self.nodes:
            values = self.nodes[key].values
            print(f"Node {values} --> ", end="")
            for adj_key in self.nodes[key].adjacent_nodes:
                adj_values = self.nodes[adj_key].values
                print(f"{adj_values} ", end="")
            print()

    # Method to get the total number of nodes
    def number_of_nodes(self) -> int:
        return len(self.nodes)

    # Method to get the number of components
    def number_of_components(self) -> int:
        return len(self.get_components())

    # Method to collect all pos elements present in the graph nodes
    def collect_pos_elements(self) -> Set[str]:
        pos_elements = set()
        for key in self.nodes:
            pos_elements.update(self.nodes[key].values)
        return pos_elements

    # Method to count the number of pos elements in the graph nodes
    def count_pos_elements(self) -> int:
        return len(self.collect_pos_elements())

# Jaro-Winkler similarity function
def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro-Winkler similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    return jw.similarity(s1, s2)

def calculate_d(graph: Graph, n_ideal: int, n_nonideal: int, 
                total_nouns_sum: float, noun_similarity_score: Dict[str, float]) -> float:
    number_of_nodes = float(graph.number_of_nodes())
    number_of_components = float(graph.number_of_components())

    term1 = abs(1.0 - number_of_nodes / float(n_nonideal))
    
    # Updated formula for Term-2
    sum_diff = sum(1.0 - similarity for similarity in noun_similarity_score.values())
    term2 = abs(n_ideal - n_nonideal) + sum_diff
    
    term3 = abs(1.0 - number_of_components)

    return (float(n_nonideal) / total_nouns_sum) * (term1 + term2 + term3)

def find_best_matching_ideal_polarity(pos_verb: str, ideal_relations: List[str], 
                                     ideal_polarities: List[str]) -> str:
    max_similarity = 0.0
    best_polarity = "positive"
    
    for i, ideal_relation in enumerate(ideal_relations):
        similarity = jaro_winkler_similarity(pos_verb.lower(), ideal_relation.lower())
        if similarity > max_similarity and i < len(ideal_polarities):
            max_similarity = similarity
            best_polarity = ideal_polarities[i]
    
    return best_polarity

def get_expected_polarity(polarity_triples: Dict, sentence_number: int, pos_verb: str) -> str:
    try:
        if str(sentence_number) in polarity_triples:
            triples = polarity_triples[str(sentence_number)]
            
            max_similarity = 0.0
            best_polarity = "positive"
            
            for triple in triples:
                if len(triple) >= 4:
                    predicate = triple[1]
                    polarity = triple[3]
                    
                    similarity = jaro_winkler_similarity(pos_verb.lower(), predicate.lower())
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_polarity = polarity
            
            return best_polarity
    except:
        pass
    
    return "positive"

def calculate_verb_metric_with_polarity(extracted_triples: List[List[str]], 
                                      ideal_triples: List[List[str]], 
                                      pos_verbs: List[str], 
                                      lambda_param: float) -> float:
    if not extracted_triples:
        return 1.0
    
    # Extract relations from extracted triples
    relations = []
    extracted_polarities = []
    
    for tuple_data in extracted_triples:
        if len(tuple_data) > 1:
            relations.append(tuple_data[1])  # predicate is at index 1
            if len(tuple_data) >= 4:
                extracted_polarities.append(tuple_data[3])
            else:
                extracted_polarities.append("positive")
    
    # Extract ideal relations and polarities
    ideal_relations = []
    ideal_polarities = []
    
    for tuple_data in ideal_triples:
        if len(tuple_data) > 1:
            ideal_relations.append(tuple_data[1])
            if len(tuple_data) >= 4:
                ideal_polarities.append(tuple_data[3])
            else:
                ideal_polarities.append("positive")
    
    # Filter relations that match POS verbs with similarity >= 0.3
    filtered_relations = []
    filtered_polarities = []
    
    for i, relation_verb in enumerate(relations):
        matched = False
        for pos_verb in pos_verbs:
            similarity = jaro_winkler_similarity(relation_verb.lower(), pos_verb.lower())
            if similarity >= 0.3 and not matched:
                filtered_relations.append(relation_verb)
                filtered_polarities.append(extracted_polarities[i])
                matched = True
    
    if not filtered_relations:
        return 1.0
    
    sum_val = 0.0
    polarity_penalty = 0.0
    
    if len(filtered_relations) > len(pos_verbs):
        # Match each POS verb to best relation
        for pos_verb in pos_verbs:
            max_similarity = 0.0
            best_match = -1
            
            for i, relation_verb in enumerate(filtered_relations):
                similarity = jaro_winkler_similarity(relation_verb.lower(), pos_verb.lower())
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = i
            
            sum_val += (1.0 - max_similarity)
            
            if best_match != -1:
                extracted_polarity = filtered_polarities[best_match]
                ideal_polarity = find_best_matching_ideal_polarity(pos_verb, ideal_relations, ideal_polarities)
                if extracted_polarity != ideal_polarity:
                    polarity_penalty += lambda_param
    else:
        # Match each relation to best POS verb
        for i, relation_verb in enumerate(filtered_relations):
            max_similarity = 0.0
            best_pos_verb = ""
            
            for pos_verb in pos_verbs:
                similarity = jaro_winkler_similarity(relation_verb.lower(), pos_verb.lower())
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pos_verb = pos_verb
            
            sum_val += (1.0 - max_similarity)
            
            if best_pos_verb:
                extracted_polarity = filtered_polarities[i]
                ideal_polarity = find_best_matching_ideal_polarity(best_pos_verb, ideal_relations, ideal_polarities)
                if extracted_polarity != ideal_polarity:
                    polarity_penalty += lambda_param
    
    if len(pos_verbs) > len(filtered_relations):
        sum_val += abs(len(pos_verbs) - len(filtered_relations))
    
    sum_val = sum_val / (2 * len(pos_verbs))
    normalized_polarity_penalty = polarity_penalty / len(pos_verbs)
    
    return min(1.0, sum_val + normalized_polarity_penalty)

def main():
    oie_tools = ["BAllMyMethod.json",
    "clausie_benchie_form.json",
    "compactie_benchie_form.json",
    "imojie_benchie_form.json",
    "m2oie_benchie_form.json",
    "minie_benchie_form.json",
    "openie6_benchie_form.json",
    "reverb_benchie_form.json"]
    oie_output = ["output_benchie_triple_AllMyMethod.txt",
    "output_benchie_triple_clausie_benchie_form.txt",
    "output_benchie_triple_compactie_benchie_form.txt",
    "output_benchie_triple_imojie_benchie_form.txt",
    "output_benchie_triple_m2oie_benchie_form.txt",
    "output_benchie_triple_minie_benchie_form.txt",
    "output_benchie_triple_openie6_benchie_form.txt",
    "output_benchie_triple_reverb_benchie_form.txt"]
    oie_predicate_output = ["predicate_output_benchie_AllMyMethod.txt",
    "predicate_output_benchie_clausie_benchie_form.txt",
    "predicate_output_benchie_compactie_benchie_form.txt",
    "predicate_output_benchie_imojie_benchie_form.txt",
    "predicate_output_benchie_m2oie_benchie_form.txt",
    "predicate_output_benchie_minie_benchie_form.txt",
    "predicate_output_benchie_openie6_benchie_form.txt",
    "predicate_output_benchie_reverb_benchie_form.txt"]
    oie_Combin_output = ["Combin_output_benchie_AllMyMethod.txt",
    "Combin_output_benchie_clausie_benchie_form.txt",
    "Combin_output_benchie_compactie_benchie_form.txt",
    "Combin_output_benchie_imojie_benchie_form.txt",
    "Combin_output_benchie_m2oie_benchie_form.txt",
    "Combin_output_benchie_minie_benchie_form.txt",
    "Combin_output_benchie_openie6_benchie_form.txt",
    "Combin_output_benchie_reverb_benchie_form.txt"]

    # Subject and object evaluation
    for tool in range(len(oie_tools)):
        finalsum = 0.0
        try:
            with open(f"Json_Ideal/{oie_tools[tool]}", 'r') as f:
                triples_data = json.load(f)
            
            # Create output directory if it doesn't exist
            os.makedirs("output_benchie_ideal_triple_Mymethod", exist_ok=True)
            
            with open(f"output_benchie_ideal_triple_Mymethod/{oie_output[tool]}", 'w') as writer:
                for target_sentence_number in range(1, 721):
                    if str(target_sentence_number) in triples_data:
                        graphs = []
                        
                        # Read POS from file
                        pos = []
                        try:
                            with open("output_pos_ideal_benchie.txt", 'r') as br:
                                lines = br.readlines()
                                if target_sentence_number - 1 < len(lines):
                                    line = lines[target_sentence_number - 1]
                                    # Remove brackets and split
                                    line = line.replace('[', '').replace(']', '').strip()
                                    if line:
                                        elements = line.split(', ')
                                        pos.extend([elem.strip() for elem in elements if elem.strip()])
                        except FileNotFoundError:
                            print(f"Warning: output_pos_ideal_benchie.txt not found")
                            pos = []
                        
                        # Retrieve input strings
                        new_input_strings = triples_data[str(target_sentence_number)]
                        input_strings = [list(inner_list) for inner_list in new_input_strings]
                        
                        # Map for storing similarity scores
                        noun_similarity_score = {}
                        
                        # Processing input strings
                        for tuple_data in input_strings:
                            if len(tuple_data) < 3:
                                continue
                                
                            node1 = tuple_data[0]
                            relation = tuple_data[1]
                            node2 = tuple_data[2]
                            
                            node_list1 = []
                            node_list2 = []
                            
                            # Find matching elements in pos list for node1
                            word_checked_subject = 0
                            for word in pos:
                                if node1.lower() in word.lower():
                                    word_checked_subject = 1
                                    node_list1.append(word)
                                    if word not in noun_similarity_score:
                                        noun_similarity_score[word] = 1.0
                            
                            if word_checked_subject == 0 and pos:
                                max_similarity = 0.0
                                best_match = ""
                                
                                for word in pos:
                                    similarity = jaro_winkler_similarity(node1.lower(), word.lower())
                                    if similarity > max_similarity:
                                        max_similarity = similarity
                                        best_match = word
                                
                                if max_similarity >= 0.3 and best_match:
                                    node_list1.append(best_match)
                                    if best_match not in noun_similarity_score:
                                        noun_similarity_score[best_match] = max_similarity
                            
                            # Find matching elements in pos list for node2
                            word_checked_object = 0
                            for word in pos:
                                if node2.lower() in word.lower():
                                    word_checked_object = 1
                                    node_list2.append(word)
                                    if word not in noun_similarity_score:
                                        noun_similarity_score[word] = 1.0
                            
                            if word_checked_object == 0 and pos:
                                max_similarity = 0.0
                                best_match = ""
                                
                                for word in pos:
                                    similarity = jaro_winkler_similarity(node2.lower(), word.lower())
                                    if similarity > max_similarity:
                                        max_similarity = similarity
                                        best_match = word
                                
                                if max_similarity >= 0.3 and best_match:
                                    node_list2.append(best_match)
                                    if best_match not in noun_similarity_score:
                                        noun_similarity_score[best_match] = max_similarity
                            
                            # Skip if either node list is empty
                            if not node_list1 or not node_list2:
                                continue
                            
                            edge_exists_in_all_graphs = True
                            for graph in graphs:
                                if not graph.edge_exists(node_list1, node_list2):
                                    edge_exists_in_all_graphs = False
                                    break
                            
                            if edge_exists_in_all_graphs:
                                new_graph = Graph()
                                new_graph.add_edge(node_list1, node_list2)
                                graphs.append(new_graph)
                            else:
                                for graph in graphs:
                                    if not graph.edge_exists(node_list1, node_list2):
                                        graph.add_edge(node_list1, node_list2)
                                        break
                        
                        # Calculate pos element counts
                        pos_element_counts = [graph.count_pos_elements() for graph in graphs]
                        
                        # Calculate and write the metric sum
                        if graphs and pos:
                            noun_sum = sum(pos_element_counts)
                            sum_val = 0.0
                            for i in range(len(graphs)):
                                sum_val += calculate_d(graphs[i], len(pos), pos_element_counts[i], noun_sum, noun_similarity_score)
                            
                            writer.write(f"{target_sentence_number} {sum_val/(2*len(pos))}\n")
                            finalsum += sum_val/(2*len(pos))
                        else:
                            # Handle case with no graphs
                            writer.write(f"{target_sentence_number} 0.0\n")
                    
                    else:
                        # Read POS from file
                        pos = []
                        try:
                            with open("output_pos_ideal_benchie.txt", 'r') as br:
                                lines = br.readlines()
                                if target_sentence_number - 1 < len(lines):
                                    line = lines[target_sentence_number - 1]
                                    line = line.replace('[', '').replace(']', '').strip()
                                    if line:
                                        elements = line.split(', ')
                                        pos.extend([elem.strip() for elem in elements if elem.strip()])
                        except FileNotFoundError:
                            print(f"Warning: output_pos_ideal_benchie.txt not found")
                            pos = []
                        
                        if pos:
                            sum_val = 2 * len(pos)
                            writer.write(f"{target_sentence_number} {sum_val/(2*len(pos))}\n")
                            finalsum += sum_val/(2*len(pos))
                        else:
                            writer.write(f"{target_sentence_number} 0.0\n")
                
                writer.write(f"\nAvg. value is: {finalsum/720.0}\n")
        
        except Exception as e:
            print(f"Error in subject/object evaluation: {e}")

    # Predicate - intrinsic evaluation against POS verbs with polarity support
    for tool in range(len(oie_tools)):
        finalsum = 0.0
        try:
            with open(f"Json_Ideal/{oie_tools[tool]}", 'r') as f:
                triples_data = json.load(f)
            
            # Load polarity information
            polarity_triples = {}
            try:
                with open(f"Json_Ideal_Polarity/{oie_tools[tool]}", 'r') as f1:
                    polarity_triples = json.load(f1)
            except:
                print("Warning: Could not load polarity data, using intrinsic evaluation only")
            
            os.makedirs("output_benchie_predicate_ideal_triple_Mymethod", exist_ok=True)
            
            with open(f"output_benchie_predicate_ideal_triple_Mymethod/{oie_predicate_output[tool]}", 'w') as writer:
                for target_sentence_number in range(1, 721):
                    sum_val = 0.0
                    
                    # Get POS verbs
                    pos_verbs = []
                    try:
                        with open("output_verbs_ideal_benchie.txt", 'r') as br:
                            lines = br.readlines()
                            if target_sentence_number - 1 < len(lines):
                                line = lines[target_sentence_number - 1]
                                line = line.replace('[', '').replace(']', '').strip()
                                if line:
                                    elements = line.split(', ')
                                    pos_verbs.extend([elem.strip() for elem in elements if elem.strip()])
                    except FileNotFoundError:
                        print(f"Warning: output_verbs_ideal_benchie.txt not found")
                        pos_verbs = []
                    
                    # Intrinsic evaluation
                    relations = []
                    extracted_polarities = []
                    
                    if str(target_sentence_number) in triples_data:
                        new_input_strings = triples_data[str(target_sentence_number)]
                        
                        # Extract relations and polarities
                        for tuple_data in new_input_strings:
                            if len(tuple_data) > 1:
                                relations.append(tuple_data[1])
                                if len(tuple_data) >= 4:
                                    extracted_polarities.append(tuple_data[3])
                                else:
                                    extracted_polarities.append("positive")
                        
                        # Filter relations
                        filtered_relations = []
                        filtered_polarities = []
                        
                        for i, relation_verb in enumerate(relations):
                            matched = False
                            for pos_verb in pos_verbs:
                                similarity = jaro_winkler_similarity(relation_verb.lower(), pos_verb.lower())
                                if similarity >= 0.3 and not matched:
                                    filtered_relations.append(relation_verb)
                                    filtered_polarities.append(extracted_polarities[i])
                                    matched = True
                        
                        if not filtered_relations or not pos_verbs:
                            sum_val = 1.0
                        else:
                            similarity_sum = 0.0
                            polarity_penalty = 0.0
                            lambda_param = 0.1
                            
                            if len(filtered_relations) > len(pos_verbs):
                                # Match each POS verb to best relation
                                for pos_verb in pos_verbs:
                                    max_similarity = 0.0
                                    best_match = -1
                                    
                                    for i, relation_verb in enumerate(filtered_relations):
                                        similarity = jaro_winkler_similarity(relation_verb.lower(), pos_verb.lower())
                                        if similarity > max_similarity:
                                            max_similarity = similarity
                                            best_match = i
                                    
                                    similarity_sum += (1.0 - max_similarity)
                                    
                                    if best_match != -1 and polarity_triples:
                                        extracted_polarity = filtered_polarities[best_match]
                                        expected_polarity = get_expected_polarity(polarity_triples, target_sentence_number, pos_verb)
                                        if extracted_polarity != expected_polarity:
                                            polarity_penalty += lambda_param
                            else:
                                # Match each relation to best POS verb
                                for i, relation_verb in enumerate(filtered_relations):
                                    max_similarity = 0.0
                                    best_pos_verb = ""
                                    
                                    for pos_verb in pos_verbs:
                                        similarity = jaro_winkler_similarity(relation_verb.lower(), pos_verb.lower())
                                        if similarity > max_similarity:
                                            max_similarity = similarity
                                            best_pos_verb = pos_verb
                                    
                                    similarity_sum += (1.0 - max_similarity)
                                    
                                    if best_pos_verb and polarity_triples:
                                        extracted_polarity = filtered_polarities[i]
                                        expected_polarity = get_expected_polarity(polarity_triples, target_sentence_number, best_pos_verb)
                                        if extracted_polarity != expected_polarity:
                                            polarity_penalty += lambda_param
                            
                            if len(pos_verbs) > len(filtered_relations):
                                similarity_sum += abs(len(pos_verbs) - len(filtered_relations))
                            
                            sum_val = (similarity_sum / (2 * len(pos_verbs))) + (polarity_penalty / len(pos_verbs))
                            sum_val = min(1.0, sum_val)
                    
                    else:
                        sum_val = 1.0
                    
                    writer.write(f"{target_sentence_number} {sum_val}\n")
                    finalsum += sum_val
                
                writer.write(f"\nAvg. value is: {finalsum/720.0}\n")
        
        except Exception as e:
            print(f"Error in predicate evaluation: {e}")

    # Combined metric calculation
    alpha = 0.5
    os.makedirs("output_benchie_combined_ideal_triple_Mymethod", exist_ok=True)
    for tool in range(len(oie_tools)):
        try:
            with open(f"output_benchie_ideal_triple_Mymethod/{oie_output[tool]}", 'r') as noun_reader, \
                 open(f"output_benchie_predicate_ideal_triple_Mymethod/{oie_predicate_output[tool]}", 'r') as pred_reader, \
                 open(f"output_benchie_combined_ideal_triple_Mymethod/{oie_Combin_output[tool]}", 'w') as combined_writer:
            
                noun_lines = noun_reader.readlines()
                pred_lines = pred_reader.readlines()
            
                combined_sum = 0.0
                count = 0
            
                for i in range(min(len(noun_lines), len(pred_lines))):
                    noun_line = noun_lines[i].strip()
                    pred_line = pred_lines[i].strip()
                
                    if (not noun_line.startswith("Avg.") and not pred_line.startswith("Avg.") and 
                        noun_line and pred_line):
                    
                        noun_parts = noun_line.split()
                        pred_parts = pred_line.split()
                    
                        if len(noun_parts) >= 2 and len(pred_parts) >= 2:
                            try:
                                sentence_num = int(noun_parts[0])
                                noun_metric = float(noun_parts[1])
                                pred_metric = float(pred_parts[1])
                            
                                combined_metric = alpha * noun_metric + (1.0 - alpha) * pred_metric
                            
                                combined_writer.write(f"{sentence_num} {combined_metric}\n")
                                combined_sum += combined_metric
                                count += 1
                            except ValueError:
                                continue
            
                if count > 0:
                    combined_writer.write(f"\nAvg. combined metric: {combined_sum / count}\n")
    
        except Exception as e:
            print(f"Error in combined metric calculation: {e}")

if __name__ == "__main__":
    main()