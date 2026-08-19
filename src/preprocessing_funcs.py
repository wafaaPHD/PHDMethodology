#from allennlp.predictors.predictor import Predictor
import os
import re
import spacy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .misc import save_as_pickle, load_pickle, get_subject_objects,processSubjectObjectPairs2
from tqdm import tqdm
import logging
from .SubjectObjectrelation import splitMergeSentences,chunktext,SubjectObjectrelation
import contractions
from .triple_extractor import TripleExtractor,TripleExtractorConfig
import sys
import time
import psutil
from datetime import datetime
import json

#from .graph4nlpEvaluation import extract_triples,evaluate_extraction


tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


def process_sent(sent):
    if sent not in [" ", "\n", ""]:
        sent = sent.strip("\n")            
        sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent)
        sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
        sent = re.sub("^ +", "", sent) # remove space in front
        sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
        sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
        #sent = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent) # Replace all CAPS with capitalize
        return sent
    return
def process_textlines(text):
    text = [process_sent(sent) for sent in text]
    text = " ".join([t for t in text if t is not None])
    text = re.sub(' {2,}', ' ', text) # remove extra spaces > 1
    return text    
def get_span_words(span, document):
    return ' '.join(document[span[0]:span[1]+1])
def replace_coreferences(document, clusters):
    """
    Replaces words in the document based on the given coreference clusters.
    Each cluster will be replaced by its representative word (the first word in the cluster).
    """
    # Convert document list to a copy to avoid modifying the original list
    updated_document = document[:]
    
    # Process each cluster
    for cluster in clusters:
        # The representative word is the first word in the cluster (span)
        representative_word = ' '.join(document[cluster[0][0]:cluster[0][1]+1])  # cluster[0] is the first span in the cluster
        # Replace all words in the cluster with the representative word
        for span in cluster:
            for index in span:
                if(index==cluster[0][0] or index==cluster[0][1]):
                    continue
                if(representative_word in ['He','he','She','she','It','it','We','we','They','they','you','You',"I"]):
                    continue
                updated_document[index] = representative_word
    return updated_document
def print_clusters(prediction):
    document, clusters = prediction['document'], prediction['clusters']
    for cluster in clusters:
        print(get_span_words(cluster[0], document) + ': ', end='')
        print(f"[{'; '.join([get_span_words(span, document) for span in cluster])}]")
def create_pretraining_corpus(args,modelt,raw_text, nlp,index, predictor=None, window_size=500, save_stats_to_file=True, stats_file="processing_stats.json"):

    '''
    Input: Chunk of raw text
    Output: modified corpus of triplets (relation statement, entity1, entity2)
    '''
    # Start timing and memory tracking
    save_Analysis_to_file=True
    analysis_file="Analysis_stats.json"
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    # Initialize statistics dictionary
    raw_text=raw_text.text
    stats = {
        "timestamp": datetime.now().isoformat(),
        "raw_text_length": len(raw_text),
        "raw_text_words": len(raw_text.split()),
        "window_size": window_size,
        "total_sentences": 0,
        "total_sentences_length": [],
        "total_triples_before_filter": 0,
        "total_triples_after_filter": 0,
        "inference_time_seconds": 0,
        "processing_time_seconds": 0,
        "memory_usage_mb": 0,
        "filter_stats": {}
    }
    
    logger.info("Processing sentences...")
    
    cluster = []
    analysisList=[]
    total_triples_before = 0
    total_triples_after = 0
    
    # Track inference time if predictor is used
    inference_start = time.time()
    #try:
    #    prediction = predictor.predict(document= raw_text)  # get prediction
    #    #updated_document = replace_coreferences(prediction['document'], prediction['clusters'])
    #    print("Clsuters:-")
    #    for clusterPredict in prediction['clusters']:
    #        coref_w_spans=(get_span_words(clusterPredict[0], prediction['document']),f"[{'; '.join([get_span_words(span, prediction['document']) for span in clusterPredict])}]",clusterPredict)
    #        if (coref_w_spans) not in cluster:
    #            cluster.append(coref_w_spans)
    #            
    #    print(cluster)  # list of clusters (the indices of spaCy tokens)
    #    print('\n') #Newline
    #except :
    #    pass  
    #updated_text = ' '.join(updated_document)
    #sents_doc = nlp(updated_text)
    inference_end = time.time()
    stats["inference_time_seconds"] = inference_end - inference_start
    
    # Process text with spaCy
    nlp_start = time.time()
    sents_doc = nlp(raw_text)
    nlp_end = time.time()
    
    ents = sents_doc.ents # get entities
    
    # Count total sentences
    doc_sents = [s for s in sents_doc.sents]
    stats["total_sentences"] = len(doc_sents)
    stats["total_sentences_length"] = [len(sent) for sent in doc_sents]
   
    logger.info("Processing relation statements by entities...")
    entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                            "WORK_OF_ART", "LAW", "LANGUAGE","DATE"]
    length_doc = len(sents_doc)
    D = []; ents_list = []
    ents_list_Position = []
    relations,sents = [],[]
    entities_pos=[];

    relationstring=''
    entityindex=[]
    reference_triples=[]
    extracted_triples=[]

    
    logger.info("Processing relation statements by dependency tree parsing...")
    sent_=sents_doc
    sents.append(sent_.text)
    reference_triples.append((0,0,0))
    
    # Track triples before filter
    triples_before = 0
    triples_after = 0
       
    left_r=sent_[0].i   
    # Configure
    # Configure extractor
    config = TripleExtractorConfig()
    #config.deduplication_level = 'semantic'#semantic
    config.deduplication_level = 'aggressive'
    config.clean_output = True
    config.use_fallback = True
    config.extract_cross_sentence = True
    config.use_semantic_similarity = True
    config.use_coref = True
    config.extract_interrogative = True
    config.extract_imperative = True
    config.extract_exclamatory = True
    config.extract_elliptical = True
    config.extract_comparative = True
    config.extract_cleft = True
    config.extract_nominalized = True
    config.extract_absolute = True
    config.extract_coordinated = True
    config.extract_negation = True
    config.extract_modality = True
    # Initialize
    extractor = TripleExtractor(nlp, config)


    all_sent_triples,tokensarr=extractor.extract(sent_)
    all_sent_triples2,tokensarr=processSubjectObjectPairs2(sent_)
    for itemr in all_sent_triples2:
         all_sent_triples.append(itemr)
    triples_before += len(all_sent_triples)
    converted = [item[:3] for item in all_sent_triples]
    for item in all_sent_triples:
        with open(args.PathDataset+'AllMyMethod.txt', 'a', encoding="utf-8") as f:
                                    f.write(str(index))
                                    f.write('\t')
                                    f.write(item[0]+'\t'+item[1]+'\t'+item[2])
                                    f.write('\n')
           
    
    
    for graphdata_e,tokensarr_row in zip(all_sent_triples,tokensarr):
            try:    
                e1, e2 = graphdata_e[0], graphdata_e[2]
                rel = graphdata_e[1]
                triple_dict=graphdata_e[3]
                triple_dictsub=graphdata_e[4]
                start_index=-1
                end_index=-1
                if len(triple_dict)==0:
                    first_word = e1.split()[0]
                    last_word = e1.split()[-1]
                    for i, token in enumerate(sent_):
                        if first_word in token.text:
                            start_index = i
                        if last_word in token.text:
                            end_index = i+1
                        if end_index != -1:
                                    triple_dict=(start_index,end_index)
                                    break
                if len(triple_dictsub)==0:
                    first_word = e2.split()[0]
                    last_word = e2.split()[-1]
                    for i, token in enumerate(sent_):
                                            if first_word in token.text:
                                                start_index = i
                                            if last_word in token.text:
                                                end_index = i+1
                                            if end_index != -1:
                                                        triple_dictsub=(start_index,end_index)
                                                        break
                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2
                if all(isinstance(triple_dictitem, tuple) for triple_dictitem in triple_dict):
                    e1start, e1end =triple_dict[len(triple_dict)-1][0],triple_dict[len(triple_dict)-1][1] 
                else:
                    e1start, e1end =triple_dict[0],triple_dict[len(triple_dict)-1] 

                if all(isinstance(triple_dictitem, tuple) for triple_dictitem in triple_dictsub):
                    e2start, e2end =triple_dictsub[len(triple_dictsub)-1][0],triple_dictsub[len(triple_dictsub)-1][1] 
                else:
                    e2start, e2end =triple_dictsub[0],triple_dictsub[len(triple_dictsub)-1] 

                relindex=graphdata_e[5]
                if len(triple_dict)==0:
                                    for r in e1.split():
                                        start_index = tokensarr_row.index(r)
                                        relindex.append(start_index)
                
                relindex[:] = [number - left_r for number in relindex]
                
                if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0):
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),rel,relindex)
                    if ((r, e1text, e2text) not in D):
                        D.append((r, e1text.split(' _ ')[-1], e2text))
                        ents_list.append((e1text, e2text))
                        relationstring+=rel+'('+e1+','+e2+')'+'|'
                        entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                    if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                    if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))

                    change=False
                    for x in cluster:        
                        if(e1text.split(' _ ')[-1] in x[1] and [e1start,e1end] in x[2]):
                            e1text=x[0]
                            change=True
                        if(e2text in x[1] and [e2start,e2end] in x[2]):
                            e2text=x[0]
                            change=True
                        if(change and e1text!=e2text):
                            if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                                extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                    if(change==False):
                        extracted_triples.append((e1text, rel, e2text))
            except:
                pass
    
    ents_list_Position.append(entityindex)
    relations.append(relationstring) 
    
    
    # Calculate final time and memory
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    stats["processing_time_seconds"] = end_time - start_time
    stats["memory_usage_mb"] = end_memory - start_memory
    stats["final_memory_mb"] = end_memory
    stats["total_relations_extracted"] = len(extracted_triples)
    stats["unique_relations"] = len(set([triple[1] for triple in extracted_triples if triple[1]]))
    stats["cosine_similarity"]=args.cosine_similarity
    stats["Z_scores"]=args.Z_scores    
    # Save statistics to file if requested
    if save_stats_to_file:
        try:
            # Load existing stats if file exists
            existing_stats = []
            if os.path.exists(args.PathDataset+stats_file):
                with open(args.PathDataset+stats_file, 'r') as f:
                    existing_stats = json.load(f)
            
            # Append new stats
            existing_stats.append(stats)
            
            # Save to file
            with open(args.PathDataset+stats_file, 'w') as f:
                json.dump(existing_stats, f, indent=2)
            logger.info(f"Statistics saved to {args.PathDataset+stats_file}")
        except Exception as e:
            logger.error(f"Error saving statistics to file: {e}")
    if save_stats_to_file:
            try:
                # Load existing stats if file exists
                existing_stats = []
                if os.path.exists(args.ResultPathDataset+stats_file):
                    with open(args.ResultPathDataset+stats_file, 'r') as f:
                        existing_stats = json.load(f)
                
                # Append new stats
                existing_stats.append(stats)
                
                # Save to file
                with open(args.ResultPathDataset+stats_file, 'w') as f:
                    json.dump(existing_stats, f, indent=2)
                logger.info(f"Statistics saved to {args.ResultPathDataset+stats_file}")
            except Exception as e:
                logger.error(f"Error saving statistics to file: {e}")
    
    if save_Analysis_to_file:
            try:
                # Load existing stats if file exists
                existing_stats = []
                if os.path.exists(args.ResultPathDataset+analysis_file):
                    with open(args.ResultPathDataset+analysis_file, 'r') as f:
                        existing_stats = json.load(f)
                
                # Append new stats
                existing_stats.append(analysisList)
                
                # Save to file
                with open(args.ResultPathDataset+analysis_file, 'w') as f:
                    json.dump(existing_stats, f, indent=2)
                logger.info(f"Statistics saved to {args.ResultPathDataset+analysis_file}")
            except Exception as e:
                logger.error(f"Error saving statistics to file: {e}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS SUMMARY:")
    print("=" * 60)
    print(f"Raw text length: {stats['raw_text_length']} characters")
    print(f"Raw text words: {stats['raw_text_words']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Average sentence length: {sum(stats['total_sentences_length']) / len(stats['total_sentences_length']) if stats['total_sentences_length'] else 0:.2f} tokens")
    print(f"Total triples before filter: {stats['total_triples_before_filter']}")
    print(f"Total triples after filter: {stats['total_triples_after_filter']}")
    print(f"Total relations extracted: {stats['total_relations_extracted']}")
    print(f"Unique relations: {stats['unique_relations']}")
    print(f"Inference time: {stats['inference_time_seconds']:.2f} seconds")
    print(f"Processing time: {stats['processing_time_seconds']:.2f} seconds")
    print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
    print("=" * 60)
    
    return D, sents, relations, ents_list_Position, cluster, entities_pos, reference_triples, extracted_triples, analysisList, stats
def create_pretraining_corpus_(raw_text, nlp,predictor=None, window_size=500):
    '''
    Input: Chunk of raw text
    Output: modified corpus of triplets (relation statement, entity1, entity2)
    '''
    logger.info("Processing sentences...")
    
    cluster = []
    analysisList=[]
    #try:
    #    prediction = predictor.predict(document= raw_text)  # get prediction
    #    #updated_document = replace_coreferences(prediction['document'], prediction['clusters'])
    #    print("Clsuters:-")
    #    for clusterPredict in prediction['clusters']:
    #        coref_w_spans=(get_span_words(clusterPredict[0], prediction['document']),f"[{'; '.join([get_span_words(span, prediction['document']) for span in clusterPredict])}]",clusterPredict)
    #        if (coref_w_spans) not in cluster:
    #            cluster.append(coref_w_spans)
            
    #    print(cluster)  # list of clusters (the indices of spaCy tokens)
    #    print('\n') #Newline
    #except :
    #    pass  
    #updated_text = ' '.join(updated_document)
    #sents_doc = nlp(updated_text)
    sents_doc = nlp(raw_text)
    ents = sents_doc.ents # get entities
   
    logger.info("Processing relation statements by entities...")
    entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                            "WORK_OF_ART", "LAW", "LANGUAGE","DATE"]
    length_doc = len(sents_doc)
    D = []; ents_list = []
    ents_list_Position = []
    relations,sents = [],[]
    entities_pos=[];

    relationstring=''
    entityindex=[]
    reference_triples=[]
    extracted_triples=[]

    
    logger.info("Processing relation statements by dependency tree parsing...")
    doc_sents = [s for s in sents_doc.sents]
    sent_=sents_doc
    #for sent_ in tqdm(doc_sents, total=len(doc_sents)):
    #if len(sent_) > (window_size + 1):
    #        continue
    sents.append(sent_.text)
    reference_triples.append((0,0,0))
    #triples = extract_triples(sent_.text)
    #if('edge_attributes' not in dir(triples)):
    #       #if(len(dir(triples))==46):
    #        reference_triples.append((0,0,0))
    #else:
    #        edge_attributes=[d.get('token') for d in triples.edge_attributes]
    #        node_attributes=[d.get('token') for d in triples.node_attributes]
    #        mapping=triples._nids_eid_mapping
        
    #        for x in mapping:
    #            d=mapping[x]
    #            reference_triples.append((node_attributes[x[0]], edge_attributes[d], node_attributes[x[1]]))
        
    left_r=sent_[0].i   
    all_sent_triples,tokensarr=processSubjectObjectPairs2(sent_)#sents_doc
    converted = [item[:3] for item in all_sent_triples]
    filter_engine = AdaptiveTripleFilter(modelt,z_threshold=args.Z_scores, min_similarity=args.cosine_similarity)
    AllSentTriplesAdaptiveTripleFilter, analysis1 = filter_engine.filter_and_analyze(sent_.text, converted)
    all_sent_triples[:] = [triple for triple in all_sent_triples if tuple(triple[:3]) in AllSentTriplesAdaptiveTripleFilter]
    if 'error' not in analysis1:
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS:")
        print(f"Total: {analysis1['total_triples']}")
        print(f"Kept: {analysis1['kept_triples']}")
        print(f"Filtered out: {analysis1['filtered_out']}")
        print(f"Precision: {analysis1['precision']:.2f}")
        print("\nError breakdown:")
        for err, count in analysis1['error_breakdown'].items():
            print(f"   - {err}: {count}")
    
        print("\n" + "=" * 60)
        print("DETAILED SCORES (first 4 triples):")
        for item in analysis1['per_triple_scores'][:4]:
            print(f"\n{item['triple']}")
            print(f"  → Kept: {item['kept']}")
            if item['error_type']:
                print(f"  → Error: {item['error_type']}")
            print(f"  → Relevance: {item['relevance']:.2f}")
            print(f"  → Consistency: {item['consistency']:.2f}")
            print(f"  → Combined: {item['combined']:.2f}")
    for graphdata_e in all_sent_triples:
            
            e1, e2 = graphdata_e[0], graphdata_e[2]
            rel = graphdata_e[1]
            triple_dict=graphdata_e[3]
            triple_dictsub=graphdata_e[4]
            #if (len(e1) > 5) or (len(e2) > 5): # don't want entities that are too long
            #        continue
            e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2
            e1start, e1end =triple_dict[0],triple_dict[1] 
            e2start, e2end = triple_dictsub[0],triple_dictsub[1]
            if(e2start==e2end):
                e2end=e2end+1
            if(e1start==e1end):
                e1end=e1end+1
            relindex=graphdata_e[5]
            relindex[:] = [number - left_r for number in relindex]
            
            if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0): #and ((e1text, e2text) not in ents_list)
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),rel,relindex)
                    if ((r, e1text, e2text) not in D):
                        D.append((r, e1text, e2text))
                        ents_list.append((e1text, e2text))
                        relationstring+=rel+'('+e1+','+e2+')'+'|'
                        entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                    if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                    if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))  
                    change=False
                    for x in cluster:        
                        if(e1text in x[1] and [e1start,e1end] in x[2]):
                            e1text=x[0]
                            change=True
                        if(e2text in x[1] and [e2start,e2end] in x[2]):
                            e2text=x[0]
                            change=True
                        if(change and e1text!=e2text):
                            if ((e1text,rel, e2text)) not in extracted_triples:
                                extracted_triples.append((e1text, rel, e2text))
                    if(change==False):
                        if ((e1text,rel, e2text)) not in extracted_triples:
                            extracted_triples.append((e1text, rel, e2text))
    extractor = TripleExtractor(nlp)
    all_sent_triples,tokensarr=extractor.extract(sent_)
    converted = [item[:3] for item in all_sent_triples]
    AllSentTriplesAdaptiveTripleFilter, analysis = filter_engine.filter_and_analyze(sent_.text, converted)
    all_sent_triples[:] = [triple for triple in all_sent_triples if tuple(triple[:3]) in AllSentTriplesAdaptiveTripleFilter]
    if 'error' in analysis:
        print("")
    if 'error' not in analysis:
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS:")
        print(f"Total: {analysis['total_triples']}")
        print(f"Kept: {analysis['kept_triples']}")
        print(f"Filtered out: {analysis['filtered_out']}")
        print(f"Precision: {analysis['precision']:.2f}")
        print("\nError breakdown:")
        for err, count in analysis['error_breakdown'].items():
            print(f"   - {err}: {count}")
    
        print("\n" + "=" * 60)
        print("DETAILED SCORES (first 4 triples):")
        for item in analysis['per_triple_scores'][:4]:
            print(f"\n{item['triple']}")
            print(f"  → Kept: {item['kept']}")
            if item['error_type']:
                print(f"  → Error: {item['error_type']}")
            print(f"  → Relevance: {item['relevance']:.2f}")
            print(f"  → Consistency: {item['consistency']:.2f}")
            print(f"  → Combined: {item['combined']:.2f}")
    
    for graphdata_e,tokensarr_row in zip(all_sent_triples,tokensarr):
                e1, e2 = graphdata_e[0], graphdata_e[2]
                rel = graphdata_e[1]
                triple_dict=graphdata_e[3]
                triple_dictsub=graphdata_e[4]
                #if (len(e1) > 5) or (len(e2) > 5): # don't want entities that are too long
                #    continue
                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2
                if all(isinstance(triple_dictitem, tuple) for triple_dictitem in triple_dict):
                    e1start, e1end =triple_dict[len(triple_dict)-1][0],triple_dict[len(triple_dict)-1][1] 
                else:
                    e1start, e1end =triple_dict[0],triple_dict[len(triple_dict)-1] 
                e2start, e2end = triple_dictsub[0],triple_dictsub[1]
                relindex=graphdata_e[5]
                relindex[:] = [number - left_r for number in relindex]
                
                if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0) : #and ((e1text, e2text) not in ents_list)
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),rel,relindex)
                    if ((r, e1text, e2text) not in D):
                        D.append((r, e1text.split(' _ ')[-1], e2text))
                        ents_list.append((e1text, e2text))
                        relationstring+=rel+'('+e1+','+e2+')'+'|'
                        entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                    if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                    if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))

                    change=False
                    for x in cluster:        
                        if(e1text.split(' _ ')[-1] in x[1] and [e1start,e1end] in x[2]):
                            e1text=x[0]
                            change=True
                        if(e2text in x[1] and [e2start,e2end] in x[2]):
                            e2text=x[0]
                            change=True
                        if(change and e1text!=e2text):
                            if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                                extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                    if(change==False):
                        # if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                            # extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                        extracted_triples.append((e1text, rel, e2text))
    
    all_sent_triples,tokensarr=SubjectObjectrelation(sent_)
    converted = [item[:3] for item in all_sent_triples]
    AllSentTriplesAdaptiveTripleFilter, analysis2 = filter_engine.filter_and_analyze(sent_.text, converted)
    all_sent_triples[:] = [triple for triple in all_sent_triples if tuple(triple[:3]) in AllSentTriplesAdaptiveTripleFilter]
    if 'error' in analysis2:
        print("")
    if 'error' not in analysis2:
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS:")
        print(f"Total: {analysis2['total_triples']}")
        print(f"Kept: {analysis2['kept_triples']}")
        print(f"Filtered out: {analysis2['filtered_out']}")
        print(f"Precision: {analysis2['precision']:.2f}")
        print("\nError breakdown:")
        for err, count in analysis2['error_breakdown'].items():
            print(f"   - {err}: {count}")
    
        print("\n" + "=" * 60)
        print("DETAILED SCORES (first 4 triples):")
        for item in analysis2['per_triple_scores'][:4]:
            print(f"\n{item['triple']}")
            print(f"  → Kept: {item['kept']}")
            if item['error_type']:
                print(f"  → Error: {item['error_type']}")
            print(f"  → Relevance: {item['relevance']:.2f}")
            print(f"  → Consistency: {item['consistency']:.2f}")
            print(f"  → Combined: {item['combined']:.2f}")
    
    for graphdata_e,tokensarr_row in zip(all_sent_triples,tokensarr):
                e1, e2 = graphdata_e[0], graphdata_e[2]
                rel = graphdata_e[1]
                triple_dict=graphdata_e[3]
                triple_dictsub=graphdata_e[4]
                #if (len(e1) > 5) or (len(e2) > 5): # don't want entities that are too long
                #    continue
                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2
                e1start, e1end =triple_dict[len(triple_dict)-1][0],triple_dict[len(triple_dict)-1][1] 
                e2start, e2end = triple_dictsub[0],triple_dictsub[1]
                relindex=graphdata_e[5]
                relindex[:] = [number - left_r for number in relindex]
                
                if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0) : #and ((e1text, e2text) not in ents_list)
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),rel,relindex)
                    if ((r, e1text, e2text) not in D):
                        D.append((r, e1text.split(' _ ')[-1], e2text))
                        ents_list.append((e1text, e2text))
                        relationstring+=rel+'('+e1+','+e2+')'+'|'
                        entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                    if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                    if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))

                    change=False
                    for x in cluster:        
                        if(e1text.split(' _ ')[-1] in x[1] and [e1start,e1end] in x[2]):
                            e1text=x[0]
                            change=True
                        if(e2text in x[1] and [e2start,e2end] in x[2]):
                            e2text=x[0]
                            change=True
                        if(change and e1text!=e2text):
                            if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                                extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                    if(change==False):
                        # if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                            # extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                        extracted_triples.append((e1text, rel, e2text))
    pairs = get_subject_objects(sent_)        
    if len(pairs) > 0:
            for pair in pairs:
                e1, e2 = pair[0], pair[1]
                
                if (len(e1) > 3) or (len(e2) > 3): # don't want entities that are too long
                    continue
                
                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1.text,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2.text
                e1start, e1end = e1[0].i if isinstance(e1, list) else e1.i, e1[-1].i + 1 if isinstance(e1, list) else e1.i + 1
                e2start, e2end = e2[0].i if isinstance(e2, list) else e2.i, e2[-1].i + 1 if isinstance(e2, list) else e2.i + 1
                if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0) and ((e1text, e2text) not in ents_list):
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),'','')
                    D.append((r, e1text, e2text))
                    ents_list.append((e1text, e2text))
                    entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))
                if ((e1text,"", e2text)) not in extracted_triples:
                                extracted_triples.append((e1text,"", e2text))           

    print("Processed dataset samples from named entity extraction:")
    for i in tqdm(range(len(ents))):
        e1 = ents[i]
        e1start = e1.start; e1end = e1.end
        if e1.label_ not in entities_of_interest:
            continue
        #if re.search("[\d+]", e1.text): # entities should not contain numbers
        #    continue
        
        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            e2start = e2.start; e2end = e2.end
            if e2.label_ not in entities_of_interest:
                continue
            if re.search("[\d+]", e2.text): # entities should not contain numbers
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (e1end < e2start) and ((e1.text, e2.text) not in ents_list):
            
              if (1 <= (e2start - e1end) <= window_size): # check if next nearest entity within window_size
                # Find start of sentence
                punc_token = False
                start = e1start - 1
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0
                
                # Find end of sentence
                punc_token = False
                start = e2end
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc
                
                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue
                
                x = [token.text for token in sents_doc[left_r:right_r]]
                
                ### empty strings check ###
                for token in x:
                    assert len(token) > 0
                assert len(e1.text) > 0
                assert len(e2.text) > 0
                assert e1start != e1end
                assert e2start != e2end
                assert (e2start - e1end) > 0
                
                r = (x, (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),'','')
                D.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))
                entityindex.append((e1.text,e2.text,(e1start,e1end),(e2start,e2end)))
                if ((e1.text, e1start,e1end)) not in entities_pos:
                        entities_pos.append((e1.text, e1start,e1end))
                if ((e2.text, e2start,e2end)) not in entities_pos:
                        entities_pos.append((e2.text, e2start,e2end))
                if ((e1.text,"", e2.text)) not in extracted_triples:
                                extracted_triples.append((e1.text,"", e2.text))


    ents_list_Position.append(entityindex)
    relations.append(relationstring)  
    analysisList.append((raw_text,analysis1,analysis,analysis2))
    return D,sents,relations,ents_list_Position,cluster,entities_pos,reference_triples,extracted_triples,analysisList
def create_pretraining_corpus_OLD(raw_text, nlp,predictor=None, window_size=500):
    '''
    Input: Chunk of raw text
    Output: modified corpus of triplets (relation statement, entity1, entity2)
    '''
    logger.info("Processing sentences...")
    
    cluster = []
    #try:
    #    prediction = predictor.predict(document= raw_text)  # get prediction
    #    #updated_document = replace_coreferences(prediction['document'], prediction['clusters'])
    #    print("Clsuters:-")
    #    for clusterPredict in prediction['clusters']:
    #        coref_w_spans=(get_span_words(clusterPredict[0], prediction['document']),f"[{'; '.join([get_span_words(span, prediction['document']) for span in clusterPredict])}]",clusterPredict)
    #        if (coref_w_spans) not in cluster:
    #            cluster.append(coref_w_spans)
            
    #    print(cluster)  # list of clusters (the indices of spaCy tokens)
    #    print('\n') #Newline
    #except :
    #    pass  
    #updated_text = ' '.join(updated_document)
    #sents_doc = nlp(updated_text)
    sents_doc = nlp(raw_text)
    ents = sents_doc.ents # get entities
   
    logger.info("Processing relation statements by entities...")
    entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                            "WORK_OF_ART", "LAW", "LANGUAGE","DATE"]
    length_doc = len(sents_doc)
    D = []; ents_list = []
    ents_list_Position = []
    relations,sents = [],[]
    entities_pos=[];

    relationstring=''
    entityindex=[]
    reference_triples=[]
    extracted_triples=[]

    
    logger.info("Processing relation statements by dependency tree parsing...")
    doc_sents = [s for s in sents_doc.sents]
    for sent_ in tqdm(doc_sents, total=len(doc_sents)):
        if len(sent_) > (window_size + 1):
            continue
        sents.append(sent_.text)
        triples = extract_triples(sent_.text)
        if('edge_attributes' not in dir(triples)):
           #if(len(dir(triples))==46):
            reference_triples.append((0,0,0))
        else:
            edge_attributes=[d.get('token') for d in triples.edge_attributes]
            node_attributes=[d.get('token') for d in triples.node_attributes]
            mapping=triples._nids_eid_mapping
        
            for x in mapping:
                d=mapping[x]
                reference_triples.append((node_attributes[x[0]], edge_attributes[d], node_attributes[x[1]]))
        
        left_r = sent_[0].i    
        all_sent_triples,tokensarr=processSubjectObjectPairs2(sent_)#sents_doc
        for graphdata_e in all_sent_triples:
            
            e1, e2 = graphdata_e[0], graphdata_e[2]
            rel = graphdata_e[1]
            triple_dict=graphdata_e[3]
            triple_dictsub=graphdata_e[4]
            #if (len(e1) > 5) or (len(e2) > 5): # don't want entities that are too long
            #        continue
            e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2
            e1start, e1end =triple_dict[0],triple_dict[1] 
            e2start, e2end = triple_dictsub[0],triple_dictsub[1]
            if(e2start==e2end):
                e2end=e2end+1
            if(e1start==e1end):
                e1end=e1end+1
            relindex=graphdata_e[5]
            relindex[:] = [number - left_r for number in relindex]
            
            if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0): #and ((e1text, e2text) not in ents_list)
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),rel,relindex)
                    if ((r, e1text, e2text) not in D):
                        D.append((r, e1text, e2text))
                        ents_list.append((e1text, e2text))
                        relationstring+=rel+'('+e1+','+e2+')'+'|'
                        entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                    if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                    if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))  
                    change=False
                    for x in cluster:        
                        if(e1text in x[1] and [e1start,e1end] in x[2]):
                            e1text=x[0]
                            change=True
                        if(e2text in x[1] and [e2start,e2end] in x[2]):
                            e2text=x[0]
                            change=True
                        if(change and e1text!=e2text):
                            if ((e1text,rel, e2text)) not in extracted_triples:
                                extracted_triples.append((e1text, rel, e2text))
                    if(change==False):
                        if ((e1text,rel, e2text)) not in extracted_triples:
                            extracted_triples.append((e1text, rel, e2text))
        all_sent_triples,tokensarr=SubjectObjectrelation(sent_)
        for graphdata_e,tokensarr_row in zip(all_sent_triples,tokensarr):
                e1, e2 = graphdata_e[0], graphdata_e[2]
                rel = graphdata_e[1]
                triple_dict=graphdata_e[3]
                triple_dictsub=graphdata_e[4]
                #if (len(e1) > 5) or (len(e2) > 5): # don't want entities that are too long
                #    continue
                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2
                e1start, e1end =triple_dict[len(triple_dict)-1][0],triple_dict[len(triple_dict)-1][1] 
                e2start, e2end = triple_dictsub[0],triple_dictsub[1]
                relindex=graphdata_e[5]
                relindex[:] = [number - left_r for number in relindex]
                
                if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0) : #and ((e1text, e2text) not in ents_list)
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),rel,relindex)
                    if ((r, e1text, e2text) not in D):
                        D.append((r, e1text.split(' _ ')[-1], e2text))
                        ents_list.append((e1text, e2text))
                        relationstring+=rel+'('+e1+','+e2+')'+'|'
                        entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                    if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                    if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))

                    change=False
                    for x in cluster:        
                        if(e1text.split(' _ ')[-1] in x[1] and [e1start,e1end] in x[2]):
                            e1text=x[0]
                            change=True
                        if(e2text in x[1] and [e2start,e2end] in x[2]):
                            e2text=x[0]
                            change=True
                        if(change and e1text!=e2text):
                            if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                                extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                    if(change==False):
                        # if ((e1text.split(' _ ')[-1],rel, e2text)) not in extracted_triples:
                            # extracted_triples.append((e1text.split(' _ ')[-1], rel, e2text))
                        extracted_triples.append((e1text, rel, e2text))

        pairs = get_subject_objects(sent_)        
        if len(pairs) > 0:
            for pair in pairs:
                e1, e2 = pair[0], pair[1]
                
                if (len(e1) > 3) or (len(e2) > 3): # don't want entities that are too long
                    continue
                
                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1.text,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2.text
                e1start, e1end = e1[0].i if isinstance(e1, list) else e1.i, e1[-1].i + 1 if isinstance(e1, list) else e1.i + 1
                e2start, e2end = e2[0].i if isinstance(e2, list) else e2.i, e2[-1].i + 1 if isinstance(e2, list) else e2.i + 1
                if (e1end < e2start and e1start != e1end and e2start != e2end and (e2start - e1end) > 0) and ((e1text, e2text) not in ents_list):
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),'','')
                    D.append((r, e1text, e2text))
                    ents_list.append((e1text, e2text))
                    entityindex.append((e1text,e2text,(e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r)))
                if ((e1text, e1start - left_r,e1end - left_r)) not in entities_pos:
                        entities_pos.append((e1text, e1start - left_r,e1end - left_r))
                if ((e2text,e2start - left_r, e2end - left_r)) not in entities_pos:
                        entities_pos.append((e2text,e2start - left_r, e2end - left_r))
                if ((e1text,"", e2text)) not in extracted_triples:
                                extracted_triples.append((e1text,"", e2text))           

    print("Processed dataset samples from named entity extraction:")
    for i in tqdm(range(len(ents))):
        e1 = ents[i]
        e1start = e1.start; e1end = e1.end
        if e1.label_ not in entities_of_interest:
            continue
        #if re.search("[\d+]", e1.text): # entities should not contain numbers
        #    continue
        
        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            e2start = e2.start; e2end = e2.end
            if e2.label_ not in entities_of_interest:
                continue
            if re.search("[\d+]", e2.text): # entities should not contain numbers
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (e1end < e2start) and ((e1.text, e2.text) not in ents_list):
            
              if (1 <= (e2start - e1end) <= window_size): # check if next nearest entity within window_size
                # Find start of sentence
                punc_token = False
                start = e1start - 1
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0
                
                # Find end of sentence
                punc_token = False
                start = e2end
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc
                
                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue
                
                x = [token.text for token in sents_doc[left_r:right_r]]
                
                ### empty strings check ###
                for token in x:
                    assert len(token) > 0
                assert len(e1.text) > 0
                assert len(e2.text) > 0
                assert e1start != e1end
                assert e2start != e2end
                assert (e2start - e1end) > 0
                
                r = (x, (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r),'','')
                D.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))
                entityindex.append((e1.text,e2.text,(e1start,e1end),(e2start,e2end)))
                if ((e1.text, e1start,e1end)) not in entities_pos:
                        entities_pos.append((e1.text, e1start,e1end))
                if ((e2.text, e2start,e2end)) not in entities_pos:
                        entities_pos.append((e2.text, e2start,e2end))
                if ((e1.text,"", e2.text)) not in extracted_triples:
                                extracted_triples.append((e1.text,"", e2.text))


    ents_list_Position.append(entityindex)
    relations.append(relationstring)

    return D,sents,relations,ents_list_Position,cluster,entities_pos,reference_triples,extracted_triples
class pretrain_dataset(Dataset):
    def __init__(self, args, D, relations_mapper=None, batch_size=None):
        self.internal_batching = True
        self.batch_size = batch_size # batch_size cannot be None if internal_batching == True
        self.alpha = 0.7
        self.mask_probability = 0.15
        
        self.df = pd.DataFrame(D, columns=['r','e1','e2'])
        self.e1s = list(self.df['e1'].unique())
        self.e2s = list(self.df['e2'].unique())
        self.args=args

        # --- relation label wiring (was previously dropped entirely) ---
        # `relations_mapper` is the Relations_Mapper (rel2idx/idx2rel) built from
        # the whole corpus. We precompute one relation_id per row here so it can
        # ride along through tokenize()/__getitem__()/collate() instead of being
        # silently discarded before it ever reaches the trainer.
        self.relations_mapper = relations_mapper
        if self.relations_mapper is not None:
            def _lookup_relation_id(row):
                rel_phrase = row['r'][3]
                key = f"{rel_phrase}({row['e1']},{row['e2']})"
                if key in self.relations_mapper.rel2idx:
                    return self.relations_mapper.rel2idx[key]
                # Fallback used by the original preprocessing for empty-phrase
                # relations, which were stored under the bare '' key.
                if rel_phrase == '' and '' in self.relations_mapper.rel2idx:
                    return self.relations_mapper.rel2idx['']
                # Genuinely unmappable (e.g. entity text drifted due to
                # upstream coref merging) -> mark as ignored, not silently wrong.
                return -1
            self.df['relation_id'] = self.df.apply(_lookup_relation_id, axis=1)
            n_unmapped = int((self.df['relation_id'] == -1).sum())
            if n_unmapped > 0:
                logger.info("%d/%d examples have no matching relation id and will be "
                            "excluded from the triple/relation loss." % (n_unmapped, len(self.df)))
        else:
            self.df['relation_id'] = -1
        if args.model_no == 0:
            from .model.BERT.tokenization_bert import BertTokenizer as Tokenizer
            model = args.model_size #'bert-base-uncased'
            lower_case = True
            model_name = 'BERT'
        elif args.model_no == 1:
            from .model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
            model = args.model_size #'albert-base-v2'
            lower_case = False
            model_name = 'ALBERT'
        elif args.model_no == 2:
            from .model.BERT.tokenization_bert import BertTokenizer as Tokenizer
            model = 'bert-base-uncased'
            lower_case = False
            model_name = 'BioBERT'
        
        tokenizer_path =args.PathDataset+ '%s_tokenizer.pkl' % (model_name)
        if os.path.isfile(tokenizer_path):
            self.tokenizer = load_pickle('%s_tokenizer.pkl' % (model_name),self.args)
            logger.info("Loaded tokenizer from saved path.")
        else:
            if args.model_no == 2:
                self.tokenizer = Tokenizer(vocab_file='./additional_models/biobert_v1.1_pubmed/vocab.txt',
                                           do_lower_case=False)
            else:
                self.tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
            self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
            save_as_pickle("%s_tokenizer.pkl" % (model_name), self.tokenizer,args)
            logger.info("Saved %s tokenizer at %s_tokenizer.pkl" % (model_name, model_name))
        e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        assert e1_id != e2_id != 1
            
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1_token_id = self.tokenizer.encode("[E1]")[1:-1][0]
        self.E1s_token_id = self.tokenizer.encode("[/E1]")[1:-1][0]
        self.E2_token_id = self.tokenizer.encode("[E2]")[1:-1][0]
        self.E2s_token_id = self.tokenizer.encode("[/E2]")[1:-1][0]
        self.PS = Pad_Sequence(seq_pad_value=self.tokenizer.pad_token_id,\
                               label_pad_value=self.tokenizer.pad_token_id,\
                               label2_pad_value=-1,\
                               label3_pad_value=-1,\
                               label4_pad_value=-1)
        
    def put_blanks(self, D):
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        # if blank_e1 >= self.alpha:
        #    r, e1, e2 = D
        #    D = (r, "[BLANK]", e2)
        
        # if blank_e2 >= self.alpha:
        #    r, e1, e2 = D
        #    D = (r, e1, "[BLANK]")
        return D
        
    def tokenize(self, D, relation_id=-1):
        (x, s1, s2, rOI,relIndex), e1, e2 = D
        s=e1 
        o=e2
        s1start=s1[0]
        s1end=s1[1]
        if isinstance(s1,list):
            s1start=s1[len(s1)-1][0]
            s1end=s1[len(s1)-1][1]
        if isinstance(x[0],list):
            x=x[0]
        x = [w.lower() for w in x if x != '[BLANK]'] # we are using uncased model
        
        ### Include random masks for MLM training
        forbidden_idxs = [i for i in range(s1start, s1end)] + [i for i in range(s2[0], s2[1])]
        
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]

        if(rOI!=''):
            #for xindex in relIndex:
            #    if xindex not in pool_idxs:
            #        pool_idxs.append(xindex)
            masked_idxs=relIndex           

        else:
            masked_idxs = np.random.choice(pool_idxs,\
                                        size=round(self.mask_probability*len(pool_idxs)),\
                                        replace=False)

       
        # BUGFIX: masked_for_pred used to drop punctuation-only masked tokens
        # while x still replaced every position in masked_idxs with [MASK] --
        # so len(masked_for_pred) could be less than the number of [MASK]
        # tokens actually in x. Downstream, `lm_logits[x == mask_id]` (all
        # masked positions) and `masked_for_pred` (punctuation excluded) are
        # assumed to line up 1:1 for the LM cross-entropy loss and for
        # evaluate_'s debug decode -- a mismatch there either crashes
        # CrossEntropyLoss (shape mismatch) or misaligns the debug printout.
        # Fix: filter masked_idxs itself to non-punctuation tokens *first*,
        # then use that same filtered set for both masked_for_pred and the
        # x replacement, so the two are always in lockstep.
        masked_idxs = [idx for idx in masked_idxs
                       if bool(re.search(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", x[idx])) == False]

        masked_for_pred = [token.lower() for idx, token in enumerate(x) if idx in masked_idxs]

        #masked_for_pred = [w.lower() for w in masked_for_pred] # we are using uncased model
        x = [token if (idx not in masked_idxs) else self.tokenizer.mask_token \
             for idx, token in enumerate(x)]

        if(rOI!=''):
             ### replace x spans with '[BLANK]' if e is '[BLANK]'
         if (e1 == '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1start] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1end:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]         
        
         elif (e1 == '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1start] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1end:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]            
        
         elif (e1 != '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1start] + ['[E1]'] + x[s1start:s1end] + ['[/E1]'] + \
                x[s1end:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]
           
         elif (e1 != '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1start] + ['[E1]'] + x[s1start:s1end] + ['[/E1]'] + \
                x[s1end:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]
        else:
        ### replace x spans with '[BLANK]' if e is '[BLANK]'
         if (e1 == '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]
        
         elif (e1 == '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]
        
         elif (e1 != '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]
        
         elif (e1 != '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]

        e1_e2_start = ([i for i, e in enumerate(x) if e == '[E1]'][0],\
                        [i for i, e in enumerate(x) if e == '[E2]'][0])
        
        #relation={"edge_tokens:":masked_for_pred,"src":s,"tgt":o}
        #import json
        #with open(self.args.PathDataset+'relation.txt', 'a') as f:
        #    json.dump(relation, f)
        #    f.write('\n')
        
        x = self.tokenizer.convert_tokens_to_ids(x)
        masked_for_pred = self.tokenizer.convert_tokens_to_ids(masked_for_pred)
        '''
        e1 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E1_token_id) + 1, x.index(self.E1s_token_id))]]
        e2 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E2_token_id) + 1, x.index(self.E2s_token_id))]]
        '''
        return x, masked_for_pred, e1_e2_start, relation_id #, e1, e2
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ### implements standard batching
        if not self.internal_batching:
            r, e1, e2 = self.df.iloc[idx][['r', 'e1', 'e2']]
            relation_id = self.df.iloc[idx]['relation_id']
            x, masked_for_pred, e1_e2_start, relation_id = self.tokenize(self.put_blanks((r, e1, e2)), relation_id)
            x = torch.tensor(x)
            masked_for_pred = torch.tensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            #e1, e2 = torch.tensor(e1), torch.tensor(e2)
            return x, masked_for_pred, e1_e2_start, e1, e2, relation_id
        
        ### implements noise contrastive estimation
        else:
            ### get positive samples
            r, e1, e2 = self.df.iloc[idx][['r', 'e1', 'e2']]# positive sample
            relation_id = self.df.iloc[idx]['relation_id']
            pool = self.df[((self.df['e1'] == e1) & (self.df['e2'] == e2))].index
            pool = pool.append(self.df[((self.df['e1'] == e2) & (self.df['e2'] == e1))].index)
            pos_idxs = np.random.choice(pool, \
                                        size=min(int(self.batch_size//2), len(pool)), replace=False)
            ### get negative samples
            '''
            choose from option: 
            1) sampling uniformly from all negatives
            2) sampling uniformly from negatives that share e1 or e2
            '''
            if np.random.uniform() > 0.5:   
                pool = self.df[((self.df['e1'] != e1) | (self.df['e2'] != e2))].index
                neg_idxs = np.random.choice(pool, \
                                            size=min(int(self.batch_size//2), len(pool)), replace=False)
                Q = 1/len(pool)
            
            else:
                if np.random.uniform() > 0.5: # share e1 but not e2
                    pool = self.df[((self.df['e1'] == e1) & (self.df['e2'] != e2))].index
                    if len(pool) > 0:
                        neg_idxs = np.random.choice(pool, \
                                                    size=min(int(self.batch_size//2), len(pool)), replace=False)
                    else:
                        neg_idxs = []

                else: # share e2 but not e1
                    pool = self.df[((self.df['e1'] != e1) & (self.df['e2'] == e2))].index
                    if len(pool) > 0:
                        neg_idxs = np.random.choice(pool, \
                                                    size=min(int(self.batch_size//2), len(pool)), replace=False)
                    else:
                        neg_idxs = []
                        
                if len(neg_idxs) == 0: # if empty, sample from all negatives
                    pool = self.df[((self.df['e1'] != e1) | (self.df['e2'] != e2))].index
                    neg_idxs = np.random.choice(pool, \
                                            size=min(int(self.batch_size//2), len(pool)), replace=False)
                Q = 1/len(pool)
            
            batch = []
            ## process positive sample
            pos_df = self.df.loc[pos_idxs]
            for idx, row in pos_df.iterrows():
                try:
                    r, e1, e2 =row[0], row[1], row[2]
                except:
                        r, e1, e2 = row['r'], row['e1'], row['e2']
                relation_id = row['relation_id']

                x, masked_for_pred, e1_e2_start, relation_id = self.tokenize(self.put_blanks((r, e1, e2)), relation_id)
                x = torch.LongTensor(x)
                masked_for_pred = torch.LongTensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                #e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.FloatTensor([1.0]),\
                              torch.LongTensor([1]), torch.LongTensor([relation_id])))
            
            ## process negative samples
            negs_df = self.df.loc[neg_idxs]
            for idx, row in negs_df.iterrows():
                try:
                    r, e1, e2 =row[0], row[1], row[2]
                except:
                    r, e1, e2 = row['r'], row['e1'], row['e2']
                relation_id = row['relation_id']
                x, masked_for_pred, e1_e2_start, relation_id = self.tokenize(self.put_blanks((r, e1, e2)), relation_id)
                x = torch.LongTensor(x)
                masked_for_pred = torch.LongTensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                #e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.FloatTensor([Q]), torch.LongTensor([0]),
                              torch.LongTensor([relation_id])))
            batch = self.PS(batch)
            return batch
    
class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=1, label2_pad_value=-1,\
                 label3_pad_value=-1, label4_pad_value=-1, label5_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value
        self.label5_pad_value = label5_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        labels3 = list(map(lambda x: x[3], sorted_batch))
        labels3_padded = pad_sequence(labels3, batch_first=True, padding_value=self.label3_pad_value)
        y3_lengths = torch.LongTensor([len(x) for x in labels3])
        
        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(labels4, batch_first=True, padding_value=self.label4_pad_value)
        y4_lengths = torch.LongTensor([len(x) for x in labels4])

        # relation_id (real relation label for the triple-prediction head; -1 = ignore)
        labels5 = list(map(lambda x: x[5], sorted_batch))
        labels5_padded = pad_sequence(labels5, batch_first=True, padding_value=self.label5_pad_value)

        return seqs_padded, labels_padded, labels2_padded, labels3_padded, labels4_padded,\
                x_lengths, y_lengths, y2_lengths, y3_lengths, y4_lengths, labels5_padded
    
def load_dataloaders(args,modelt, max_length=50000):   
    if not os.path.isfile(args.PathDataset+args.pretrain_data+'_'+'D.pkl'):
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        #model = "en_coreference_web_trf" if "--coref" in sys.argv else "en_core_web_hftrf"
        #model = "en_coreference_web_trf" if "--coref" in sys.argv else "en_core_web_trf"
        model='en_core_web_sm'
        print(f"Loading '{model}' …")  
        try:
            nlp = spacy.load(model)
        except OSError:
            print(f"Model '{model}' not found. Downloading...")
            spacy.cli.download(model)
            nlp = spacy.load(model)      
        #nlp = spacy.load("en_core_web_lg")#       
        nlp.add_pipe("merge_entities")
        nlp.add_pipe("merge_noun_chunks")
        from spacy.tokenizer import Tokenizer
        from spacy.symbols import ORTH
       # Add the special case rule
        special_case = [{ORTH: "Milligan 1-0"}]
        special_case1 = [{ORTH: "Grand View 3-0"}]
        special_case4 = [{ORTH: "Webber International 1-0"}]
        special_case2 = [{ORTH: "Azusa Pacific 0-0"}]
        special_case3=[{ORTH:"step-by-step"}]
        nlp.tokenizer.add_special_case("Milligan 1-0", special_case)
        nlp.tokenizer.add_special_case("Grand View 3-0", special_case1)
        nlp.tokenizer.add_special_case("Azusa Pacific 0-0", special_case2)
        nlp.tokenizer.add_special_case("Webber International 1-0", special_case4)
        nlp.tokenizer.add_special_case("step-by-step",special_case3)
        
        #predictor=Predictor.from_path("./allenepi/coref-spanbert-large-2021.03.10.tar.gz")#("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")#("./allenepi/coref-spanbert-large-2021.03.10.tar.gz")#("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")#
        predictor=None
        logger.info("\nLoading pre-training data...")
        logger.info("\nLoading Spacy NLP...")
        D = []
        Allsents,relations,ents_list_Position,cluster,entities_pos,reference_triples, extracted_triples=[],[],[],[],[],[],[]
        analysislist=[]
        if(args.pretrain_data=='NYT_Dataset.csv'):
            df = pd.read_csv(args.PathDataset+args.pretrain_data)
            print(df)
            print()
            df = df[~df['abstract'].isna()]
            df = df.reset_index()  # make sure indexes pair with number of rows
            for index, row in tqdm(df.iterrows(), total=len(df)):
               linesen=row['abstract']
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
                
        elif(args.pretrain_data=='Mendeley_Climate_Change_Library.csv'):
            df = pd.read_csv(args.PathDataset+args.pretrain_data)
            print(df)
            print()
            df = df[~df['Abstract'].isna()]
            df = df.reset_index()  # make sure indexes pair with number of rows
            for index, row in tqdm(df.iterrows(), total=len(df)):
               linesen=row['Abstract']
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
        elif(args.pretrain_data=='sentences_webnlg.txt'):
                  with open("./dataset/sentences_webnlg/"+args.pretrain_data, "r", encoding="utf8") as f:
                    text = f.readlines()
                    
                    for line in tqdm(text, total=len(text)):
                       print (line)
                       linesen=line
                       if linesen not in [" ", "\n", ""]:
                            index=len(Allsents)
                            D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                            D.extend(D1)
                            Allsents.append(sents1)
                            relations.extend(relations1)
                            ents_list_Position.extend(ents_list_Position1)
                            cluster.append(cluster1)
                            entities_pos.append(entities_pos1)
                            reference_triples.append(referse_triples1)
                            extracted_triples.append(extract_triples1)        
        elif(args.pretrain_data=='cnn.txt'):
          with open("./dataset/OIE/"+args.pretrain_data, "r", encoding="utf8") as f:
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               print (line)
               linesen=line
               if linesen not in [" ", "\n", ""]:
                    index=len(Allsents)
                    D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                    D.extend(D1)
                    Allsents.append(sents1)
                    relations.extend(relations1)
                    ents_list_Position.extend(ents_list_Position1)
                    cluster.append(cluster1)
                    entities_pos.append(entities_pos1)
                    reference_triples.append(referse_triples1)
                    extracted_triples.append(extract_triples1)

        elif(args.pretrain_data=='wire57_sentences.txt'):
          with open('./dataset/wire57/'+args.pretrain_data, "r", encoding="utf8") as f:
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               print (line)
               linesen=line
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
                
        elif(args.pretrain_data=='carb_sentences.txt'):
          with open('./dataset/carb_sentences/'+args.pretrain_data, "r", encoding="utf8") as f:
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               print (line)
               linesen=line
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                analysislist.extend(analysislist1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)

        elif(args.pretrain_data=='Re-OIE2016.json'):
          with open('./dataset/Re-OIE2016/'+args.pretrain_data, "r", encoding="utf8") as f:
            
            #D = load_pickle('DORg.pkl',args)
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
              if(line.find('.": [')>=0 or line.find('. \'\'": [\n')>0):
                sentOR=line.replace(': [\n','').replace('"','').replace('  ','')
                linesen=sentOR
                index=len(Allsents)
                with open(args.PathDataset+'sentenses.txt', 'a', encoding="utf-8") as f:
                                                    f.write(linesen)
                                                    f.write('\n')
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysis1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
                
        elif(args.pretrain_data=='train_filter.data'):
          with open('./dataset/CDR/'+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()            
            for line in tqdm(text, total=len(text)):
               lines = line.rstrip().split('\t')
               print('\n')
               print (lines[1])
               linesen=lines[1]
               with open(args.PathDataset+'sentenses.txt', 'a', encoding="utf-8") as f:
                            f.write(linesen)
                            f.write('\n')
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)

                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysis1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)                
        elif(args.pretrain_data=='dev_filter.data'):
          with open(args.PathDataset+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               lines = line.rstrip().split('\t')
               print (lines[1])
               linesen=lines[1]
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)                
        elif(args.pretrain_data=='test_filter.data'):
          with open(args.PathDataset+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               lines = line.rstrip().split('\t')
               print (lines[1])
               linesen=lines[1]
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)

        elif(args.pretrain_data=='train.data'):
          with open(args.PathDataset+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               lines = line.rstrip().split('\t')
               print (lines[1])
               linesen=lines[1]
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
        elif(args.pretrain_data=='dev.data'):
          with open(args.PathDataset+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               lines = line.rstrip().split('\t')
               print (lines[1])
               linesen=lines[1]
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
        elif(args.pretrain_data=='test.data'):
          with open(args.PathDataset+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               lines = line.rstrip().split('\t')
               print (lines[1])
               linesen=lines[1]
               if linesen not in [" ", "\n", ""]:
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)

        logger.info("\nTotal number of relation statements in pre-training corpus: %d" % len(D))
        
        #evaluate_extraction(reference_triples, extracted_triples,args,Allsents,nlp,ev1="2")
        print(analysislist)
        with open(args.PathDataset+'analysislist.txt', 'a', encoding="utf-8") as f:
                                    f.write(str(analysislist))
                                    f.write('\n')     
        dfClustre = pd.DataFrame(cluster)
        df_train = pd.DataFrame(data={'sents': Allsents, 'relations': relations})
        df_train1 = pd.DataFrame(data={'sents': Allsents, 'relations': relations,'ents_list_Position':ents_list_Position,'Cluster':cluster,'Entities_pos':entities_pos})
        df_train.insert(1, 'ID', range(1, 1 + len(df_train)))
        df_train1.insert(1, 'ID', range(1, 1 + len(df_train)))
        rm = Relations_Mapper(df_train['relations'])

        df_train['relations_id'] = df_train.progress_apply(lambda x: [rm.rel2idx[j] for j in  x['relations'].split('|')] , axis=1)
        df_train1['relations_id'] = df_train1.progress_apply(lambda x: [rm.rel2idx[j] for j in  x['relations'].split('|')] , axis=1)
        
        save_as_pickle(args.pretrain_data+'_'+'relations.pkl', rm,args)
        save_as_pickle(args.pretrain_data+'_'+'dfClustre.pkl', dfClustre,args)
        save_as_pickle(args.pretrain_data+'_'+'df_train.pkl', df_train,args)
        save_as_pickle(args.pretrain_data+'_'+'df_train1.pkl', df_train1,args)
        save_as_pickle(args.pretrain_data+'_'+'D.pkl', D,args)

        logger.info("\nSaved pre-training corpus to %s" % args.PathDataset+"D.pkl")
        # BUGFIX: num_relations must be the number of distinct relation classes
        # (rm.n_classes), not len(D) (the number of training examples). Using
        # len(D) previously made the relation-classifier/embedding size track
        # dataset size instead of the actual label space.
        args.num_relations = rm.n_classes
        relation = rm
    else:
        logger.info("\nLoaded pre-training data from saved file")
        D = load_pickle(args.PathDataset+args.pretrain_data+'_'+'D.pkl',args)
        relation=load_pickle(args.PathDataset+args.pretrain_data+'_'+'relations.pkl',args)
        # BUGFIX: was `len(D)` (number of examples, e.g. 583 for wire57) instead
        # of the number of relation classes (rm.n_classes / len(rel2idx), e.g.
        # 487 for wire57). This must match the size of the model's relation
        # classifier / relation embedding table.
        args.num_relations = relation.n_classes

        ## Get the top 3000 records (assuming data is a list or similar structure)
        #top_3000_data = D[:3000]
        #D=top_3000_data

        
    train_set = pretrain_dataset(args, D, relations_mapper=relation, batch_size=args.batch_size)
    
     
    return train_set

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            relationlist=relation.split('|')
            for relationvalue in tqdm(relationlist):
               if relationvalue not in self.rel2idx.keys():
                  self.rel2idx[relationvalue] = self.n_classes
                  self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

def preprocessMyDataSet(linesen,nlp):

                linesen = re.sub(' {2,}', ' ', linesen) # remove extra spaces > 1
                linesen = re.sub('(\s){2,}', '', linesen) # remove extra spaces > 1                
                linesen = re.sub(r"[\|]", " ", linesen)
                linesen = re.sub(r'(?<=[0-9./])\s+(?=[0-9./])','',linesen)
                line=splitMergeSentences(linesen,nlp)
                # Regular expression to convert "Comdr. R. M. Metcalf" to "Comdr.R.M.Metcalf"
                line = re.sub(r'Comdr\.\s([A-Za-z])\.\s([A-Za-z])\.\s([A-Za-z]+)', r'Comdr\1\2\3', line)
                # Regular expression to convert "Mrs. Thomas J. Doyle" to "Mrs.Thomas J.Doyle"
                line = re.sub(r'Mrs\.\s([A-Za-z]+)\s([A-Za-z])\.\s([A-Za-z]+)', r'Mrs\1\2\3', line)
                line = re.sub(r'Mr\.\s([A-Za-z]+)\s([A-Za-z])\.\s([A-Za-z]+)', r'Mr\1\2\3', line)

                line = re.sub("^ +", "", line) # remove space in front
                line = re.sub(r"([\.\?,!]){2,}", r"\1", line) # remove multiple puncs
                line = re.sub(r" +([\.\?,!])", r"\1", line) # remove extra spaces in front of punc
                line = re.sub('<[A-Z]+/*>', '', line) # remove special tokens eg. <FIL/>, <S>\[\][]\/\\\\\-—
                #line = re.sub(r"[\*\"\n…•€\|♫#?]", " ", line)
                line = re.sub(r"[\*\n…•€\|♫#?]", " ", line)
                line = re.sub(r"[•€|♫#]", " ", line)
               
                line=re.sub(r'(?<=\/)\s+(?=\w+)', '', line)

                #line=re.sub(r'\s+([?,.!;%+/-])', r'\1', line)
                line=re.sub(r'\s+([?!%+/-])', r'\1', line)
                
                line=re.sub(r'(?<=\d)\s+(?=(?![or,but,and,or,of,by,we,so,if,in,is,he,go])\w+)', '', line)
                line = re.sub(r'(?<=\))\s+(?=\b[\w\d]{1}\b)', '', line)
                line = re.sub(r'(?<=\))\s+(?=\b[\w\d^[or,of,by,we,so,if,in,is,he,go]]{2}\b)', '', line)                
                line = re.sub(r'(?=(?![or,but,and])\w+)\s+(?=\b[\d]{1,2}\b)', '', line)                
                line = re.sub(r'(?<=[\w\d])\s+(?=\))', '', line)
                line=re.sub(r'(?<=\-)\s+(?=[\w\d]+)', '', line)
                line=re.sub(r'(?<=[0-9]\))\s+(?=\b[to]{2}\b)', '', line)
                line=re.sub(r'(?<=[0-9])\s+(?=\b[to]{2}\b)', '', line)
                line = re.sub(r'(?<=\d\)to)\s+(?=[0-9])', '', line)
                line = re.sub(r'(?=\d)\s+(?=to{2})', '', line)
                line = re.sub(r'(?<=to)\s+(?=\d)', '', line)
                
                line = re.sub(r'(?<=\d)\s+(?=\()', '', line)
                line = re.sub(r'(?<=\d%)\s+(?=\()', '', line)
                
                line = re.sub(r'(?<=\-)\s+(?=\d\()', '', line)
                line = re.sub(r'(?<=\-)\s+(?=\d\%\()', '', line)
                line=re.sub(r'(?<=\\)\s+(?=[\w\d\-]+)', '', line)
                line = re.sub('(\s){2,}', ' ', line) # remove extra spaces > 1
                regex = r'\b(\w+)(?:\W+\1\b)+' 
                line=re.sub(regex, r'\1', line, flags=re.IGNORECASE)

                flage=True
                ns = ""
                allnc=""
                lineList=line.split("(")
                textbractes= re.findall(r'\((.*?)\)',line)
                for index,xc in enumerate(lineList):
                    textfind=re.findall(r'(.*?)\)',xc)
                    if(len(textfind)>0):
                     if(textfind[0] in textbractes):
                        allnc=allnc+'('
                        ns=''
                        for char in textbractes[textbractes.index(re.findall(r'(.*?)\)',xc)[0])]:
                         if not char.isspace():
                             if flage:
                                 ns+=char.capitalize()
                                 flage=False
                             else:
                                 ns += char
                         else:
                              flage=True
                        allnc+=ns+')'+xc.split(")")[1]
                     else:
                         allnc+='('+xc.lstrip()                    
                    else:
                        if index>0:
                            allnc+='('+xc.lstrip()
                        else:
                            allnc+=xc.lstrip()
                line=allnc
                
                flage=True
                ns = ""
                allnc=""
                lineList=line.split("[")
                textbractes= re.findall(r'\[(.*?)\]',line)
                for index,xc in enumerate(lineList):
                    textfind=re.findall(r'(.*?)\]',xc)
                    if(len(textfind)>0):
                     if(textfind[0] in textbractes):
                        allnc=allnc+'['
                        ns=''
                        for char in textbractes[textbractes.index(re.findall(r'(.*?)\]',xc)[0])]:
                         if not char.isspace():
                             if flage:
                                 ns+=char.capitalize()
                                 flage=False
                             else:
                                 ns += char
                         else:
                              flage=True
                        allnc+=ns+']'+xc.split("]")[1]
                     else:
                         allnc+='['+xc.lstrip()                    
                    else:
                        if index>0:
                            allnc+='['+xc.lstrip()
                        else:
                            allnc+=xc.lstrip()
                line=allnc
                return line

def LoadDataPreprocess(args,modelt,linesen,nlp,index,predictor):
                #linesen = re.sub(r'(?<=[a-z])\s+(?=\')','',linesen)
                linesen=contractions.fix(linesen,leftovers=False,slang=False)
                #linesen = re.sub(r"[\*\"\n\\…\+\/\=\(\)•€\[\]\|♫—~#]", " ", linesen)
                #linesen=preprocessMyDataSet(linesen,nlp)                
                text1=linesen                
                if text1 not in [" ", "\n", ""]:
        
                   text_chunks=chunktext(500,text1)
            
                   for text_chunk in tqdm(text_chunks, total=len(text_chunks)):
                      index+=1
                      D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1, stats=create_pretraining_corpus(args,modelt,text_chunk, nlp,index,predictor, window_size=500)
                      #D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1=create_pretraining_corpus(text_chunk, nlp,predictor, window_size=500)
                      #len(sents1)
                      with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a' , encoding="utf-8") as f:
                                f.write('sent_id:'+(index).__str__())
                                f.write('\t')
                      with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a', encoding="utf-8") as f:    
                                f.write(text_chunk.text)
                                f.write('\n')
                      relation=''
                      for indexcluster,x1 in enumerate(extract_triples1):
                          if(x1[1]!=""):
                                with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a', encoding="utf-8") as f:
                                    f.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
                                    f.write('\n')
                                strrela=x1[0]+' --> '+x1[1]+' --> '+x1[2]+"\n"
                                if(strrela not in relation):
                                 relation+=x1[0]+' --> '+x1[1]+' --> '+x1[2]+"\n"                            
                                 with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a', encoding="utf-8") as f:
                                    f.write(x1[0]+' --> '+x1[1]+' --> '+x1[2])
                                    f.write('\n')
                                 with open(args.PathDataset+str(args.cosine_similarity)+str(args.Z_scores)+'MyMethod.txt', 'a', encoding="utf-8") as f1:
                                    f1.write((index).__str__())
                                    f1.write('\t')
                                    f1.write(x1[0]+'\t'+x1[1]+'\t'+x1[2])
                                    f1.write('\n')     
                                 
                                 # if(len(x1[0].split(' _ ')) >1 or len(x1[1].split(', '))>1 or len(x1[2].split(' _ '))>1):
                                 #  for s in x1[0].split(' _ '):
                                 #    for r in x1[1].split(', '):
                                 #        for o in x1[2].split(' _ '):
                                 #            strrela=s+' --> '+r+' --> '+o+"\n"
                                 #            if(strrela not in relation):
                                 #                with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a') as f:
                                 #                    f.write(s+' --> '+r+' --> '+o)
                                 #                    f.write('\n')
                                 #                with open(args.PathDataset+'benchie_gold_annotations_enMyMethod.txt', 'a') as f:
                                 #                    f.write((index).__str__())
                                 #                    f.write('\t')
                                 #                    f.write(s+'\t'+r+'\t'+o)
                                 #                    f.write('\n')
                                 #                relation+=s+' --> '+r+' --> '+o+"\n"
                      for indexcluster,x1 in enumerate(referse_triples1):
                          if(x1[0]==0 and x1[2]==0):
                              break
                          if(x1[1]!=""):
                                strrela=str(x1[0])+' --> '+str(x1[1])+' --> '+str(x1[2])+"\n"
                                if(strrela not in relation):
                                 relation+=str(x1[0])+' --> '+str(x1[1])+' --> '+str(x1[2])+"\n"                            
                                 with open(args.PathDataset+'benchie_gold_annotations_en4nlp.txt', 'a', encoding="utf-8") as f:
                                    f.write((index).__str__())
                                    f.write('\t')
                                    f.write(str(x1[0])+'\t'+str(x1[1])+'\t'+str(x1[2]))
                                    f.write('\n')     
                                 
                                 # if(len(x1[0].split(' _ ')) >1 or len(x1[1].split(', '))>1 or len(x1[2].split(' _ '))>1):
                                 #  for s in x1[0].split(' _ '):
                                 #    for r in x1[1].split(', '):
                                 #        for o in x1[2].split(' _ '):
                                 #            strrela=s+' --> '+r+' --> '+o+"\n"
                                 #            if(strrela not in relation):
                                                
                                 #                with open(args.PathDataset+'benchie_gold_annotations_en4nlp.txt', 'a') as f:
                                 #                    f.write((index).__str__())
                                 #                    f.write('\t')
                                 #                    f.write(s+'\t'+r+'\t'+o)
                                 #                    f.write('\n')
                                 #                relation+=s+' --> '+r+' --> '+o+"\n"
                      with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a', encoding="utf-8") as f:    
                               f.write('\n')
                return D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1,analysislist1
