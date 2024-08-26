from asyncio.windows_events import NULL
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
from .SubjectObjectrelation import SubjectObjectrelation,splitMergeSentences,chunktext
from allennlp.predictors.predictor import Predictor
from .graph4nlpEvaluation import extract_triples,evaluate_extraction
import contractions

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

def print_clusters(prediction):
    document, clusters = prediction['document'], prediction['clusters']
    for cluster in clusters:
        print(get_span_words(cluster[0], document) + ': ', end='')
        print(f"[{'; '.join([get_span_words(span, document) for span in cluster])}]")
def create_pretraining_corpus(raw_text, nlp,predictor, window_size=500):
    '''
    Input: Chunk of raw text
    Output: modified corpus of triplets (relation statement, entity1, entity2)
    '''
    logger.info("Processing sentences...")
    sents_doc = nlp(raw_text)
    cluster = []
    try:
        prediction = predictor.predict(document= raw_text)  # get prediction
        print("Clsuters:-")
        for clusterPredict in prediction['clusters']:
            coref_w_spans=(get_span_words(clusterPredict[0], prediction['document']),f"[{'; '.join([get_span_words(span, prediction['document']) for span in clusterPredict])}]",clusterPredict)
            if (coref_w_spans) not in cluster:
                cluster.append(coref_w_spans)
            
        print(cluster)  # list of clusters (the indices of spaCy tokens)
        print('\n') #Newline
    except :
        pass  
   
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
            
            if (e1end < e2start): #and ((e1text, e2text) not in ents_list)
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
                
                if (e1end < e2start) : #and ((e1text, e2text) not in ents_list)
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
                if (e1end < e2start) and ((e1text, e2text) not in ents_list):
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
    def __init__(self, args, D, batch_size=None):
        self.internal_batching = True
        self.batch_size = batch_size # batch_size cannot be None if internal_batching == True
        self.alpha = 0.7
        self.mask_probability = 0.15
        
        self.df = pd.DataFrame(D, columns=['r','e1','e2'])
        self.e1s = list(self.df['e1'].unique())
        self.e2s = list(self.df['e2'].unique())
        self.args=args
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
        
    def tokenize(self, D):
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

       
        masked_for_pred = [token.lower() for idx, token in enumerate(x) if (idx in masked_idxs) and bool(re.search(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", token))==False]

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
        
        relation={"edge_tokens:":masked_for_pred,"src":s,"tgt":o}
        import json
        with open(self.args.PathDataset+'relation.txt', 'a') as f:
            json.dump(relation, f)
            f.write('\n')
        
        x = self.tokenizer.convert_tokens_to_ids(x)
        masked_for_pred = self.tokenizer.convert_tokens_to_ids(masked_for_pred)
        '''
        e1 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E1_token_id) + 1, x.index(self.E1s_token_id))]]
        e2 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E2_token_id) + 1, x.index(self.E2s_token_id))]]
        '''
        return x, masked_for_pred, e1_e2_start #, e1, e2
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ### implements standard batching
        if not self.internal_batching:
            r, e1, e2 = self.df.iloc[idx]
            x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((r, e1, e2)))
            x = torch.tensor(x)
            masked_for_pred = torch.tensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            #e1, e2 = torch.tensor(e1), torch.tensor(e2)
            return x, masked_for_pred, e1_e2_start, e1, e2
        
        ### implements noise contrastive estimation
        else:
            ### get positive samples
            r, e1, e2 = self.df.iloc[idx] # positive sample
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
                r, e1, e2 = row[0], row[1], row[2]
                x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((r, e1, e2)))
                x = torch.LongTensor(x)
                masked_for_pred = torch.LongTensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                #e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.FloatTensor([1.0]),\
                              torch.LongTensor([1])))
            
            ## process negative samples
            negs_df = self.df.loc[neg_idxs]
            for idx, row in negs_df.iterrows():
                r, e1, e2 = row[0], row[1], row[2]
                x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((r, e1, e2)))
                x = torch.LongTensor(x)
                masked_for_pred = torch.LongTensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                #e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.FloatTensor([Q]), torch.LongTensor([0])))
            batch = self.PS(batch)
            return batch
    
class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=1, label2_pad_value=-1,\
                 label3_pad_value=-1, label4_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value
        
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
        return seqs_padded, labels_padded, labels2_padded, labels3_padded, labels4_padded,\
                x_lengths, y_lengths, y2_lengths, y3_lengths, y4_lengths

def load_dataloaders(args, max_length=50000):
   
    if not os.path.isfile(args.PathDataset+args.pretrain_data+'_'+'D.pkl'):
        nlp = spacy.load("en_core_web_lg")#
       
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
        
        predictor=Predictor.from_path("./allenepi/coref-spanbert-large-2021.03.10.tar.gz")#("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")#

        logger.info("\nLoading pre-training data...")
        logger.info("\nLoading Spacy NLP...")
        D = []
        Allsents,relations,ents_list_Position,cluster,entities_pos,reference_triples, extracted_triples=[],[],[],[],[],[],[]
        if(args.pretrain_data=='NYT_Dataset.csv'):
            df = pd.read_csv(args.PathDataset+args.pretrain_data)
            print(df)
            print()
            df = df[~df['abstract'].isna()]
            df = df.reset_index()  # make sure indexes pair with number of rows
            for index, row in tqdm(df.iterrows(), total=len(df)):
                linesen=row['abstract']
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
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
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)
                
        elif(args.pretrain_data=='cnn.txt'):
          with open(args.PathDataset+args.pretrain_data, "r", encoding="utf8") as f:
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
                print (line)
                linesen=line
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess(linesen,nlp,args,index,predictor)
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)                      
                
        elif(args.pretrain_data=='train_filter.data'):
          with open(args.PathDataset+args.pretrain_data, "r") as f:#, encoding="utf8"
            text = f.readlines()            
            for line in tqdm(text, total=len(text)):
                lines = line.rstrip().split('\t')
                print('\n')
                print (lines[1])
                linesen=lines[1]
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
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
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
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
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
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
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
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
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
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
                index=len(Allsents)
                D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=LoadDataPreprocess((linesen,nlp,args,index,predictor))
                D.extend(D1)
                Allsents.append(sents1)
                relations.extend(relations1)
                ents_list_Position.extend(ents_list_Position1)
                cluster.append(cluster1)
                entities_pos.append(entities_pos1)
                reference_triples.append(referse_triples1)
                extracted_triples.append(extract_triples1)

        logger.info("\nTotal number of relation statements in pre-training corpus: %d" % len(D))
        
        evaluate_extraction(reference_triples, extracted_triples,args,Allsents,nlp,ev1="2")
        
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
    else:
        logger.info("\nLoaded pre-training data from saved file")
        D = load_pickle(args.pretrain_data+'_'+'D.pkl',args)
        
    train_set = pretrain_dataset(args, D, batch_size=args.batch_size)
    
     
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
                
                line = re.sub("^ +", "", line) # remove space in front
                line = re.sub(r"([\.\?,!]){2,}", r"\1", line) # remove multiple puncs
                line = re.sub(r" +([\.\?,!])", r"\1", line) # remove extra spaces in front of punc
                line = re.sub('<[A-Z]+/*>', '', line) # remove special tokens eg. <FIL/>, <S>\[\][]\/\\\\\-—
                line = re.sub(r"[\*\"\n…‘•€\|♫;”“~`#?]", " ", line)
                line = re.sub(r"[‘•€|♫—”“~`#]", " ", line)
               
                line=re.sub(r'(?<=\/)\s+(?=\w+)', '', line)

                line=re.sub(r'\s+([?,.!;%+/-])', r'\1', line)
                
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

def LoadDataPreprocess(linesen,nlp,args,index,predictor):
                linesen = re.sub(r'(?<=[a-z])\s+(?=\')','',linesen)
                linesen=contractions.fix(linesen,leftovers=False,slang=False)
                linesen = re.sub(r"[\*\"\n\\…\+\/\=\(\)‘•€\[\]\|♫;—”“~`#]", " ", linesen)
                line=preprocessMyDataSet(linesen,nlp)                
                text1=line                
                if text1 not in [" ", "\n", ""]:
        
                   text_chunks=chunktext(500,text1)
            
                   for text_chunk in tqdm(text_chunks, total=len(text_chunks)):
                      D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1=create_pretraining_corpus(text_chunk, nlp,predictor, window_size=500)
                      index+=len(sents1)
                      with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a') as f:
                                f.write('sent_id:'+(index).__str__())
                                f.write('\t')
                      with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a') as f:    
                                f.write(text_chunk)
                                f.write('\n')
                      relation=''
                      for indexcluster,x1 in enumerate(extract_triples1):
                          if(x1[1]!=""):
                                with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a') as f:
                                    f.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
                                    f.write('\n')
                                strrela=x1[0]+' --> '+x1[1]+' --> '+x1[2]+"\n"
                                if(strrela not in relation):
                                 relation+=x1[0]+' --> '+x1[1]+' --> '+x1[2]+"\n"                            
                                 with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a') as f:
                                    f.write(x1[0]+' --> '+x1[1]+' --> '+x1[2])
                                    f.write('\n')
                                 with open(args.PathDataset+'benchie_gold_annotations_enMyMethod.txt', 'a') as f:
                                    f.write((index).__str__())
                                    f.write('\t')
                                    f.write(x1[0]+'\t'+x1[1]+'\t'+x1[2])
                                    f.write('\n')     
                                 
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
                                 with open(args.PathDataset+'benchie_gold_annotations_en4nlp.txt', 'a') as f:
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
                      with open(args.PathDataset+'benchie_gold_annotations_en.txt', 'a') as f:    
                               f.write('\n')
                return D1,sents1,relations1,ents_list_Position1,cluster1,entities_pos1,referse_triples1,extract_triples1
