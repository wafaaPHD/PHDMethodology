from tracemalloc import start
import spacy
import re
import wordninja
import numpy as np
from chunkipy import TextChunker, TokenEstimator
from transformers import AutoTokenizer
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
def appendChunk(original, chunk):
    if(original==''):
        return chunk
    return original + ' ' + chunk
def appendSOChunk(original, chunk):
    if(original==''):
        return chunk
    return original + ' _ ' + chunk
def appendChunktoken(original, chunk):
    if(original==''):
        return chunk
    return original + ' ' + chunk
def appendChunkandtoken(original, chunk):
    if(original==''):
        return chunk
    return original + ' & ' + chunk
def isRelationCandidate(token):
    deps = ["ROOT", "adj","agent", "amod","ccomp","advcl","relcl"]
    return any(subs in token.dep_ for subs in deps)
def isConstructionCandidate(token):
    deps = ["test"]#conj"prep","mod",
    return any(subs in token.dep_ for subs in deps)
def SubjectObjectrelationtest(tokens):
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]#
    OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod","compound","npadvmod","advmod","mod"]
    ADJECTIVES = ["ROOT",'prep',"acomp", "advcl",  "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
                  "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod"
    tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ"]
    tagsubject=["WP","VBD","IN","WDT","WP$","WRB"]
    # SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    # OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod"]
    # ADJECTIVES = ['prep',"acomp", "advcl", "advmod", "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
    #               "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","compound","npadvmod","aux","neg"]
    # tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ"]
    # tagsubject=["WP","VBD","IN","WDT","WP$","WRB"]

    subject = ''
    object = ''
    relation = ''
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]
    relIndex=[]     

    #x = [token.lower_ for token in tokens]
    #rtoken.append(x)
    for token in tokens:
        #printToken(token)
        #rtoken.append(token)
        #if "punct" in token.dep_ or (token.text not in ('(',')','[',']') and "punct" in token.dep_):
        if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
            continue
        
        #Relation
        
        if (token.dep_ in ADJECTIVES or token.pos_ in ["VERB","ADP","ADV","PART","AUX"] or token.tag_ in tagRelation) and token.ent_type_ not in ['PERCENT','CARDINAL']:
            if'prep' in token.dep_ or token.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
             if relation == '':
                if(token.head.text==token.text):
                    relation = appendChunk('', token.text)
                    relIndex=[token.i]
                elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagRelation):
                    relation = appendChunk('', token.text)
                    relIndex=[token.i]
                elif(token.pos_ in ["PART"] and token.tag_ in ["TO"]):
                    relation = appendChunk('', token.text)
                    relIndex=[token.i]
                else:
                    relation = appendChunktoken(token.head.text, token.text)
                    relIndex=[token.head.i,token.i]
               
             else:
                 if(token.head.text==token.text):
                     relation = appendChunk(relation, token.text)
                     relIndex.append(token.i)
                 elif(token.head.text==relation):
                     relation = appendChunk(relation, token.text)
                     relIndex.append(token.i)
                 elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagRelation):
                     relation = appendChunk(relation, token.text)
                     relIndex.append(token.i)
                 elif(token.pos_ in ["PART"] and token.tag_ in ["TO","IN"]):
                     relation = appendChunk(relation+', ', token.text)
                     relIndex.append(token.i)
                 else: 
                     relation = appendChunktoken(relation+', '+token.head.text, token.text)
                     relIndex.append(token.head.i)
                     relIndex.append(token.i) 
               #relation = appendChunk(relation, token.lower_) 
               #relIndex.append(token.i) 
            elif relation == '':
                relation = appendChunktoken('', token.text)
                relIndex.append(token.i)
            else:
               relation = appendChunktoken(relation, token.text) 
               relIndex.append(token.i) 
        #SUBJECTS    and token.lower_ not in subject.lower()
        elif (((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
            #subject=''
            #subjectindex=[]
            subject=appendSOChunk(subject, token.text)
            start=token.i
            end=token.i+1
            if token.conjuncts:
                       conjuncts = token.conjuncts             # tuple of conjuncts
                       for conj in conjuncts:
                          if(conj.tag_ not in tagsubject):
                           spanconj = conj
                           subject = appendChunk(subject, spanconj.text)
                           end=spanconj.i+1
           
            subjectindex.append((start,end))
        #OBJECTS
        elif (token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS) or (token.ent_type_ in ['PERCENT','CARDINAL']):
            if(object!=''):
                if relation == '':
                    relation = appendChunktoken('', object)
                    relIndex.append(objectindex[0])
                else:
                   relation = appendChunk(relation, object) 
                   relIndex.append(objectindex[0])
            if(subject==''):
                subject=appendSOChunk(subject, token.text)
                start=token.i
                end=token.i+1
                if token.conjuncts:
                       conjuncts = token.conjuncts             # tuple of conjuncts
                       for conj in conjuncts:
                          if(conj.tag_ not in tagsubject):
                           spanconj = conj
                           subject = appendChunk(subject, spanconj.text)
                           end=spanconj.i+1
                subjectindex.append((start,end))
      
            object=''
            objectindex=[]
            object = appendSOChunk(object, token.text)
            start=token.i
            end=token.i+1            
           
            if token.tag_=='CD' or token.ent_type_ in ['CARDINAL']:
                object=''
                if(token.head.dep_ not in ADJECTIVES):
                    object = appendChunktoken(object, token.head.text)
                    if(start>token.head.i):
                        start=token.head.i
                object = appendChunktoken(object, token.text)
                
                
                end=token.i+1
                if token.conjuncts:
                       conjuncts = token.conjuncts             # tuple of conjuncts
                       for conj in conjuncts:
                          if(conj.tag_ not in tagsubject):
                           spanconj = conj
                           object = appendChunk(object, spanconj.text)
                           end=spanconj.i+1
            elif token.conjuncts:
                       conjuncts = token.conjuncts             # tuple of conjuncts
                       for conj in conjuncts:
                           if(conj.tag_ not in tagsubject):
                            spanconj = conj
                            object = appendChunk(object, spanconj.text)
                            end=spanconj.i+1
            
            objectindex=(start,end)  
            
        if relation == '' and object!='' and subject!='':
             if (token.head.dep_ in ADJECTIVES or token.head.pos_ == "VERB") and token.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.text)
                relIndex.append(token.head.i)
               else:
                relation = appendChunk(relation, token.head.lower_) 
                relIndex.append(token.head.i)
             elif (token.head.head.dep_ in ADJECTIVES or token.head.head.pos_ == "VERB") and token.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.lower_)
                relIndex.append(token.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.lower_) 
                relIndex.append(token.head.head.i) 
             elif (token.head.head.head.dep_ in ADJECTIVES or token.head.head.head.pos_ == "VERB") and token.head.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.head.lower_)
                relIndex.append(token.head.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.head.lower_) 
                relIndex.append(token.head.head.head.i) 

        if(subject.strip() !='' and relation.strip() !='' and object.strip() !=''):
            if subject == object:
             object = ''
             objectindex=[]
             relation = ''             
             rtoken=[]
             relIndex=[]
             continue
            else :
             #for xindex,xsubject in zip(subjectindex,subject.split(' _ ')):
                 #trible.append((xsubject.strip(), relation.strip(), object.strip(),xindex,objectindex,relIndex))
                 trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                 print (subject.strip(), ",", relation.strip(), ",", object.strip())
                 start=np.min(relIndex)
                 end=np.max(relIndex)
                 #if(xindex[0]<start):
                 #    start=xindex[0]
                 #if(objectindex[1]>end):
                 #    end=objectindex[1]
                 if(subjectindex[0][0]<start):
                     start=subjectindex[0][0]
                 if(objectindex[1]>end):
                     end=objectindex[1]

                 end=start+100
                 if(end>len(tokens)):
                     end=len(tokens)-1
                 start=0
                 end=len(tokens)
                 rtoken=[token.text for token in tokens[start:end]]
                 rtoken1.append(rtoken)
                 #print (xsubject.strip(), ",", relation.strip(), ",", object.strip())
                 object = ''
                 objectindex=[]
                 rtoken=[]
                 relation = ''
                 relIndex=[]
             #object = ''
             #objectindex=[]
             #rtoken=[]
             #relation = ''
             #relIndex=[]            
             
 
    return trible,rtoken1

def SubjectObjectrelation_(tokens):
    #SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    #OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod"]
    #ADJECTIVES = ['prep',"acomp", "advcl", "advmod", "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
    #              "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","compound","npadvmod","aux","neg"]
    #tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ"]
    #tagsubject=["WP","VBD","IN","WDT","WP$","WRB"]
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl","appos"]#
    OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod","compound","npadvmod","advmod","mod"]
    ADJECTIVES = ["ROOT",'prep',"acomp", "advcl",  "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
                  "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod"
    tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ","MD"]
    tagsubject=["WP","VBD","IN","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB"]

    subject = ''
    object = ''
    relation = ''
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]
    relIndex=[]
    #flage=False      

    #x = [token.lower_ for token in tokens]
    #rtoken.append(x)
    subjectflage=False
    for token in tokens:
        subjectflage=False
        object = ''
        objectindex=[]
        rtoken=[]
        relation = ''
        relIndex=[]
        if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
            continue
        subtree = list(token.subtree)
      
        for xtoken in subtree:
                            
                    if(relation in ['is','are','am'] and xtoken.text in ['is','are','am']):
                            relation = ''
                            relIndex=[]        
                    if(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
                      if (subject.split(' _ ')[-1]!=xtoken.text) and (xtoken.text not in subject.split(' _ ')): 
                        start = xtoken.i
                        end = xtoken.i + 1
                        if(subjectflage):
                            subject=appendSOChunk("", xtoken.text)
                        else:                            
                            subject=appendSOChunk(subject, xtoken.text)
                        if xtoken.conjuncts:
                            conjuncts = xtoken.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendSOChunk(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                        
                    if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject==""):
                          bflage=True
                          for x in trible:
                              if(xtoken.text in x[0] or xtoken.text in x[2]):
                                  bflage=False
                          if(bflage):
                         
                            start = xtoken.i
                            end = xtoken.i + 1
                            subject=appendSOChunk(subject, xtoken.text)
                            if xtoken.conjuncts:
                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendSOChunk(subject, spanconj.text)
                                        end=spanconj.i+1
           
                            subjectindex.append((start,end))
                        elif (object!=xtoken.text and subject!=xtoken.text):
                         object = appendSOChunk(object, xtoken.text)
                         start=xtoken.i
                         end=xtoken.i+1
                         if xtoken.conjuncts:
                            conjuncts = xtoken.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendSOChunk(object, spanconj.text)
                                    end=spanconj.i+1
                         objectindex=(start,end) 
                    #Relation
                    if (xtoken.dep_ in ADJECTIVES or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] or xtoken.tag_ in tagRelation):#and xtoken.ent_type_ not in ['PERCENT','CARDINAL']
                       if'prep' in xtoken.dep_ or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
                         if relation == '':
                            if(xtoken.head.text==xtoken.text):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO"]):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            else:
                                relation = appendChunktoken(xtoken.head.text, xtoken.text)
                                relIndex.append(xtoken.head.i)
                                relIndex.append(xtoken.i)
                         else:
                            if(xtoken.head.text==xtoken.text):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.head.text==relation):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO","IN"]):
                                relation = appendChunk(relation+', ', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.dep_ in["neg"]):    
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            else:
                                relation = appendChunktoken(relation+', '+xtoken.head.text, xtoken.text)
                                relIndex.append(xtoken.head.i)
                                relIndex.append(xtoken.i) 
                       elif relation == '':
                                relation = appendChunktoken('', xtoken.text)
                                relIndex.append(xtoken.i)
                       else:
                                relation = appendChunktoken(relation, xtoken.text) 
                                relIndex.append(xtoken.i)
                    
                    if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                          if(bflage):
                            subjectflage=True
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            
                         
                        else:
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]


        if(object=="" and (token.text not in subject) and token.tag_ in tagRelation and relation!="" and relation!=token.text):
            bflage=True
            for x in trible:
                              if(subject.strip() in x[0] or subject.strip() in x[2]):
                                  bflage=False
            if(bflage):
             if(len(subjectindex)>1):
                object = appendSOChunk(object, subject.split(' _ ')[0])
                objectindex=subjectindex[0]
                subject=subject.replace(subject.split(' _ ')[0]+' _ ','')
                subjectindex.remove(objectindex)
             else:
              relation=relation.replace(token.text, '')
              if(token.i in relIndex):
                relIndex.remove(token.i)
              object = appendSOChunk(object, token.text)
              start=token.i
              end=token.i+1
              if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ in tagRelation):
                                    spanconj = conj
                                    object = appendSOChunk(object, spanconj.text)
                                    end=spanconj.i+1
              objectindex=(start,end)
           
        
        if relation == '' and object!='' and subject!='':
             if (token.head.dep_ in ADJECTIVES or token.head.pos_ == "VERB") and token.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.text)
                relIndex.append(token.head.i)
               else:
                relation = appendChunk(relation, token.head.lower_) 
                relIndex.append(token.head.i)
             elif (token.head.head.dep_ in ADJECTIVES or token.head.head.pos_ == "VERB") and token.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.lower_)
                relIndex.append(token.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.lower_) 
                relIndex.append(token.head.head.i) 
             elif (token.head.head.head.dep_ in ADJECTIVES or token.head.head.head.pos_ == "VERB") and token.head.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.head.lower_)
                relIndex.append(token.head.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.head.lower_) 
                relIndex.append(token.head.head.head.i) 

        if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            subject=''
                            subjectindex=[]
                         
                        else:
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
    return trible,rtoken1
def SubjectObjectrelation__(tokens):
    #SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    #OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod"]
    #ADJECTIVES = ['prep',"acomp", "advcl", "advmod", "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
    #              "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","compound","npadvmod","aux","neg"]
    #tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ"]
    #tagsubject=["WP","VBD","IN","WDT","WP$","WRB"]
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]#
    OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod","compound","npadvmod","advmod","mod"]
    ADJECTIVES = ["ROOT",'prep',"acomp", "advcl",  "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
                  "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod"
    tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ","MD","RB"]
    tagsubject=["WP","VBD","IN","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT"]
    subject = ''
    object = ''
    relation = ''
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]
    relIndex=[]
    subjectflage=False
    for token in tokens:
        subjectflage=False
        object = ''
        objectindex=[]
        rtoken=[]
        # relation = ''
        # relIndex=[]
        subtree=[]
        if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
            continue
        
        if(((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=token.lower_) and (token.lower_ not in subject.lower().split(' _ ')): 
                        start = token.i
                        end = token.i + 1
                        if(subjectflage):
                            subject=appendSOChunk("", token.text)
                        else:                            
                            subject=appendSOChunk(subject, token.text)
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendSOChunk(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                        
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject==""):
                          bflage=True
                          for x in trible:
                              if(token.text in x[0] or token.text in x[2]):
                                  bflage=False
                          if(bflage):
                         
                            start = token.i
                            end = token.i + 1
                            subject=appendSOChunk(subject, token.text)
                            if token.conjuncts:
                                conjuncts = token.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendSOChunk(subject, spanconj.text)
                                        end=spanconj.i+1
           
                            subjectindex.append((start,end))
         #Relation
                          
        subtree = list(token.subtree)
        flagestree=False
        for xsub in subtree:
            if(subject.split(' _ ')[-1] in xsub.text):
                 flagestree=True        

        if(flagestree):
         relation=''
         for xtoken in subtree:
                            
                    if(relation in ['is','are','am'] and xtoken.text in ['is','are','am']):
                            relation = ''
                            relIndex=[] 
                    
                    if(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                        start = xtoken.i
                        end = xtoken.i + 1
                        if(subjectflage):
                            subject=appendSOChunk("", xtoken.text)
                        
                        else:                            
                            subject=appendSOChunk(subject, xtoken.text)
                        if xtoken.conjuncts:
                            conjuncts = xtoken.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendSOChunk(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))     
                    if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(xtoken.dep_ =='appos' and xtoken.head.dep_ in SUBJECTS and xtoken.head.tag_ not in tagsubject):
                        # if(subject==""):
                          bflage=True
                          for x in trible:
                              if(xtoken.text in x[0] or xtoken.text in x[2]):
                                  bflage=False
                          if(bflage):
                           if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                                              
                            start = xtoken.i
                            end = xtoken.i + 1
                            if(subjectflage):
                                subject=appendSOChunk("", xtoken.text)
                            else:
                                subject=appendSOChunk(subject, xtoken.text)
                            if xtoken.conjuncts:
                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendSOChunk(subject, spanconj.text)
                                        end=spanconj.i+1
           
                            subjectindex.append((start,end))
                        elif (object!=xtoken.text and subject!=xtoken.text):
                         object = appendSOChunk(object, xtoken.text)
                         start=xtoken.i
                         end=xtoken.i+1
                         if xtoken.conjuncts:
                            conjuncts = xtoken.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendSOChunk(object, spanconj.text)
                                    end=spanconj.i+1
                         objectindex=(start,end) 
                    #Relation
                    if (xtoken.dep_ in ADJECTIVES or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] or xtoken.tag_ in tagRelation):#and xtoken.ent_type_ not in ['PERCENT','CARDINAL']
                       if'prep' in xtoken.dep_ or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
                         if relation == '':
                            if(xtoken.head.text==xtoken.text):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO"]):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            else:
                                relation = appendChunktoken(xtoken.head.text, xtoken.text)
                                relIndex.append(xtoken.head.i)
                                relIndex.append(xtoken.i)
                         else:
                            if(xtoken.head.text==xtoken.text):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.head.text==relation):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO","IN"]):
                                relation = appendChunk(relation+', ', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.dep_ in["neg"]):    
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            else:
                                relation = appendChunktoken(relation+', '+xtoken.head.text, xtoken.text)
                                relIndex.append(xtoken.head.i)
                                relIndex.append(xtoken.i) 
                       elif relation == '':
                                relation = appendChunktoken('', xtoken.text)
                                relIndex.append(xtoken.i)
                       else:
                                relation = appendChunktoken(relation, xtoken.text) 
                                relIndex.append(xtoken.i)
                    
                    if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                              for xc in subject.split(' _ '):
                                   if(xc in x[0] and relation.strip() in x[1] and object.strip() in x[2]):
                                        bflage=False
                              
                          if(bflage):
                            subjectflage=True
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            # relation = ''
                            # relIndex=[]
                          else:  
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            # relation = ''
                            # relIndex=[]
                            
                         
                        else:
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            # relation = ''
                            # relIndex=[]

        if(object=="" and (token.text not in subject)  and relation!="" and relation!=token.text):
            #and token.tag_ in tagRelation
            bflage=True
            # Find all tokens before punctuation
            tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
            if tokenslist:
            # Get the last token
                last_token = tokenslist[-1]
            if(token.text!=last_token):
                bflage=False
            for x in trible:
                              if(subject.strip() in x[0] or subject.strip() in x[2]):
                                  bflage=False
            if(len(subjectindex)<=1):
                for x in subjectindex:
                  if(token.i==x[1]):
                      bflage=False
                      break          
            if(bflage):
             if(len(subjectindex)>1):
                
                object = appendSOChunk(object, subject.split(' _ ')[-1])
                objectindex=subjectindex[-1]
                subject=subject.replace(subject.split(' _ ')[-1]+' _ ','')
                subjectindex.remove(objectindex)
             else:                  
              relation=relation.replace(token.text, '')
              if(token.i in relIndex):
                relIndex.remove(token.i)
              object = appendSOChunk(object, token.text)
              start=token.i
              end=token.i+1
              if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ in tagRelation):
                                    spanconj = conj
                                    object = appendSOChunk(object, spanconj.text)
                                    end=spanconj.i+1
              objectindex=(start,end)
           
        
        if relation == '' and object!='' and subject!='':
             if (token.head.dep_ in ADJECTIVES or token.head.pos_ == "VERB") and token.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.text)
                relIndex.append(token.head.i)
               else:
                relation = appendChunk(relation, token.head.lower_) 
                relIndex.append(token.head.i)
             elif (token.head.head.dep_ in ADJECTIVES or token.head.head.pos_ == "VERB") and token.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.lower_)
                relIndex.append(token.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.lower_) 
                relIndex.append(token.head.head.i) 
             elif (token.head.head.head.dep_ in ADJECTIVES or token.head.head.head.pos_ == "VERB") and token.head.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.head.lower_)
                relIndex.append(token.head.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.head.lower_) 
                relIndex.append(token.head.head.head.i) 

        if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            subject=''
                            subjectindex=[]
                         
                        else:
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
    return trible,rtoken1
def SubjectObjectrelationOLd(tokens):
    #SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    #OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod"]
    #ADJECTIVES = ['prep',"acomp", "advcl", "advmod", "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
    #              "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","compound","npadvmod","aux","neg"]
    #tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ"]
    #tagsubject=["WP","VBD","IN","WDT","WP$","WRB"]
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]#
    OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod","compound","npadvmod","advmod","mod"]
    ADJECTIVES = ["ROOT",'prep',"acomp", "advcl",  "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
                  "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod"
    tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ","MD","RB","JJ","JJR","JJS","RBR","RBS","RP"]
    tagsubject=["WP","VBD","IN","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP"]
    subject = ''
    object = ''
    relation = ''
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]
    relIndex=[]
    subjectflageMain=False
    mainsubject=''
    mainsubjectindex=[]
    mainrelation=''
    mainrelationindex=[]
    for token in tokens:
        
        object = ''
        objectindex=[]
        rtoken=[]
        # relation = ''
        # relIndex=[]
        subtree=[]
        if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
            continue
        
        if(((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=token.lower_) and (token.lower_ not in subject.lower().split(' _ ')): 
                        start = token.i
                        end = token.i + 1
                        if(subjectflageMain):
                           subject=appendSOChunk("", token.text)
                        else:                            
                            subject=appendSOChunk(subject, token.text)
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendSOChunk(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                      else:                          
                          if(subjectflageMain):
                           start = token.i
                           end = token.i + 1
                           subject=appendSOChunk("", token.text)
                           subjectindex.append((start,end))
                        
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject==""):
                          bflage=True
                          for x in trible:
                              if(token.text in x[0] or token.text in x[2]):
                                  bflage=False
                          if(bflage):
                         
                            start = token.i
                            end = token.i + 1
                            subject=appendSOChunk(subject, token.text)
                            if token.conjuncts:
                                conjuncts = token.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendSOChunk(subject, spanconj.text)
                                        end=spanconj.i+1
           
                            subjectindex.append((start,end))
                             

        #Relation
        if (token.dep_ in ADJECTIVES or token.pos_ in ["VERB","ADP","ADV","PART","AUX"] or token.tag_ in tagRelation):#and token.ent_type_ not in ['PERCENT','CARDINAL']
                       if'prep' in token.dep_ or token.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
                         if relation == '':
                            if(token.head.text==token.text):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagRelation):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["PART"] and token.tag_ in ["TO"]):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            elif(token.head.text==subject):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            else:
                                relation = appendChunktoken(token.head.text, token.text)
                                relIndex.append(token.head.i)
                                relIndex.append(token.i)
                         else:
                           if(token.text not in relation.split(' ')):                                
                            if(token.head.text==token.text):
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)
                            elif(token.head.text==relation or token.head.text+' '==relation):
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagRelation):
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["PART"] and token.tag_ in ["TO","IN"]):
                                relation = appendChunk(relation+', ', token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["PART"] and token.dep_ in["neg"]):    
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)                            
                            else:
                                relation = appendChunktoken(relation+', '+token.head.text, token.text)
                                relIndex.append(token.head.i)
                                relIndex.append(token.i) 
                       elif relation == '':
                                relation = appendChunktoken('', token.text)
                                relIndex.append(token.i)
                       else:
                                relation = appendChunktoken(relation, token.text) 
                                relIndex.append(token.i)                     

        subtree = list(token.subtree)
        flagestree=False
        flagestree2=False
        for xsub in subtree:
           if(subject.split(' _ ')[-1] in xsub.text):
                  flagestree=True  
           if(relation.split(', ')[-1]!=''):
            for relationtoken in relation.split():
              if(relationtoken in xsub.text):
                 flagestree2=True   
           if(flagestree and flagestree2):
                 break
           #if(flagestree2):  
           #    break
        
        if(flagestree and flagestree2):
          mainsubject=subject
          mainsubjectindex=subjectindex
          mainrelation=relation
          mainrelationindex=relIndex
          
        #if(flagestree2):
           
          # if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
          #               if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
          #                 bflage=True
          #                 for x in trible:
          #                     if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
          #                         bflage=False
          #                 if(bflage):
          #                   trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
          #                   print (subject.strip(), "/", relation.strip(), "/", object.strip())
          #                   start=np.min(relIndex)
          #                   end=np.max(relIndex)
          #                   if(subjectindex[0][0]<start):
          #                       start=subjectindex[0][0]
          #                   if(objectindex[1]>end):
          #                        end=objectindex[1]
          #                   end=start+100
          #                   if(end>len(tokens)):
          #                       end=len(tokens)-1
          #                   start=0
          #                   end=len(tokens)
          #                   rtoken=[token.text for token in tokens[start:end]]
          #                   rtoken1.append(rtoken)
          #                   object = ''
          #                   objectindex=[]
          #                   rtoken=[]
          #                   relation = ''
          #                   relIndex=[]
          #                   # subject=''
          #                   # subjectindex=[]
          #                   subjectflageMain=True                        
                         
          #               else:
          #                   object = ''
          #                   objectindex=[]
          #                   rtoken=[]
          #                   relation = ''
          #                   relIndex=[]
          subjectflage=False
          for xtoken in subtree:
              #object
                    if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if (object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split(' _ ')):
                         if(xtoken.text in subject.split(' _ ')):
                                subjectlist =subject.split(' _ ')
                                if(len(subjectlist)>0):
                                    objectindex=subjectlist.index(xtoken.text)
                                    subjectindexobject=subjectindex[objectindex]
                                    subjectindex.remove(subjectindexobject)
                                    subject=subject.replace(xtoken.text+' _ ','')
                         if(xtoken.text in relation.split()):
                                relationlist =relation.replace(',','').split()
                                if(len(relationlist)>0):
                                    #relationindex=relationlist.index(xtoken.text)
                                    #relationindexobject=relIndex[relationindex]
                                    relIndex.remove(xtoken.i)
                                    relation=relation.replace(xtoken.text,'')         

                         object = appendSOChunk(object, xtoken.text)
                         start=xtoken.i
                         end=xtoken.i+1
                         if xtoken.conjuncts:
                            conjuncts = xtoken.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendSOChunk(object, spanconj.text)
                                    end=spanconj.i+1
                         objectindex=(start,end) 
              #subject      
                    if(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                        start = xtoken.i
                        end = xtoken.i + 1
                        if(subjectflage or subjectflageMain):
                            subject=appendSOChunk("", xtoken.text)
                        
                        else:                            
                            subject=appendSOChunk(subject, xtoken.text)
                        if xtoken.conjuncts:
                            conjuncts = xtoken.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendSOChunk(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                        #Relation
                    if(relation in ['is','are','am'] and xtoken.text in ['is','are','am']):
                            relation = ''
                            relIndex=[] 
              #relation      
                    if ((xtoken.dep_ in ADJECTIVES or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] or xtoken.tag_ in tagRelation) and xtoken.text not in relation.split()):#and xtoken.ent_type_ not in ['PERCENT','CARDINAL']
                       if'prep' in xtoken.dep_ or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
                         if relation == '' or subjectflage:
                            if(xtoken.head.text==xtoken.text):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO"]):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.head.text==subject):
                                relation = appendChunk('', xtoken.text)
                                relIndex.append(xtoken.i)
                            else:
                                relation = appendChunktoken(xtoken.head.text, xtoken.text)
                                relIndex.append(xtoken.head.i)
                                relIndex.append(xtoken.i)
                         else:
                           if(token.text not in relation.split(' ')):
                            if(xtoken.head.text==xtoken.text):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.head.text==relation or xtoken.head.text+' '==relation):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO","IN"]):
                                relation = appendChunk(relation+', ', xtoken.text)
                                relIndex.append(xtoken.i)
                            elif(xtoken.pos_ in ["PART"] and xtoken.dep_ in["neg"]):    
                                relation = appendChunk(relation, xtoken.text)
                                relIndex.append(xtoken.i)
                            else:
                                relation = appendChunktoken(relation+', '+xtoken.head.text, xtoken.text)
                                relIndex.append(xtoken.head.i)
                                relIndex.append(xtoken.i) 
                       elif relation == '':
                                relation = appendChunktoken('', xtoken.text)
                                relIndex.append(xtoken.i)
                       else:
                                relation = appendChunktoken(relation, xtoken.text) 
                                relIndex.append(xtoken.i)
                    
                    if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip() in x[2]):
                                  bflage=False
                              for xc in subject.split(' _ '):
                                   if(xc in x[0] and relation.strip() in x[1] and object.strip() in x[2]):
                                        bflage=False
                                   
                              
                          if(bflage):
                            subjectflage=True
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            # relation = ''
                            # relIndex=[]
                          else:  
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            # relation = ''
                            # relIndex=[]
                            
                         
                        else:
                            subjectflage=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            # relation = ''
                            # relIndex=[]
          subject=mainsubject
          subjectindex=mainsubjectindex
          relation=mainrelation
          relIndex=mainrelationindex
          for xtoken in subtree:
                        if xtoken.dep_ in ["punct","cc"]  or "PUNCT" in xtoken.pos_ or xtoken.is_punct:
                            continue
                    # if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if (object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split('_') and xtoken.text not in subject.split(' _ ') and xtoken.text not in relation.split()):
                         # if(xtoken.text in subject.split(' _ ')):
                         #        subjectlist =subject.split(' _ ')
                         #        if(len(subjectlist)>0):
                         #            objectindex=subjectlist.index(xtoken.text)
                         #            subjectindexobject=subjectindex[objectindex]
                         #            subjectindex.remove(subjectindexobject)
                         #            subject=subject.replace(xtoken.text+' _ ','')
                         object = appendChunk(object, xtoken.text)
                         start=xtoken.i
                         end=xtoken.i+1
                         # if xtoken.conjuncts:
                         #    conjuncts = xtoken.conjuncts             # tuple of conjuncts
                         #    for conj in conjuncts:
                         #        if(conj.tag_ not in tagsubject):
                         #            spanconj = conj
                         #            object = appendChunk(object, spanconj.text)
                         #            end=spanconj.i+1
                         objectindex=(start,end)

        if(object=="" and (token.text not in subject)  and relation!="" and relation!=token.text):
            #and token.tag_ in tagRelation
            bflage=True
            # Find all tokens before punctuation
            tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
            if tokenslist:
            # Get the last token
                last_token = tokenslist[-1]
            if(token.text!=last_token):
                bflage=False
            # for x in trible:
            #                   if(subject.strip() in x[0] or subject.strip() in x[2]):
            #                       bflage=False
            # if(len(subjectindex)<=1):
            #     for x in subjectindex:
            #       if(token.i==x[1]):
            #           bflage=False
            #           break          
            if(bflage):
             # if(len(subjectindex)>1):
                
             #    object = appendSOChunk(object, subject.split(' _ ')[-1])
             #    objectindex=subjectindex[-1]
             #    subject=subject.replace(subject.split(' _ ')[-1]+' _ ','')
             #    subjectindexobject=subjectindex[objectindex]
             #    subjectindex.remove(subjectindexobject)
             # else:                  
              relation=relation.replace(token.text, '')
              if(token.i in relIndex):
                relIndex.remove(token.i)
              object = appendSOChunk(object, token.text)
              start=token.i
              end=token.i+1
              if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ in tagRelation):
                                    spanconj = conj
                                    object = appendSOChunk(object, spanconj.text)
                                    end=spanconj.i+1
              objectindex=(start,end)          
        
        if relation == '' and object!='' and subject!='':
             if (token.head.dep_ in ADJECTIVES or token.head.pos_ == "VERB") and token.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.text)
                relIndex.append(token.head.i)
               else:
                relation = appendChunk(relation, token.head.lower_) 
                relIndex.append(token.head.i)
             elif (token.head.head.dep_ in ADJECTIVES or token.head.head.pos_ == "VERB") and token.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.lower_)
                relIndex.append(token.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.lower_) 
                relIndex.append(token.head.head.i) 
             elif (token.head.head.head.dep_ in ADJECTIVES or token.head.head.head.pos_ == "VERB") and token.head.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
               if relation == '':
                relation = appendChunk('', token.head.head.head.lower_)
                relIndex.append(token.head.head.head.i)
               else:
                relation = appendChunk(relation, token.head.head.head.lower_) 
                relIndex.append(token.head.head.head.i) 

        if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            subject=''
                            subjectindex=[]
                            subjectflageMain=True
                         
                        else:
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
    return trible,rtoken1

def SubjectObjectrelation____(tokens):
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]#
    OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod","compound","npadvmod","advmod","mod"]
    ADJECTIVES = ["ROOT",'prep',"acomp", "advcl",  "amod", "nn", "nmod", "ccomp","complm", "adj","agent",
                  "ccomp","advcl","relcl","hmod", "infmod", "xcomp", "rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod"
    tagRelation=["VBD","VB","VBG","VBN","VBP","VBZ","MD","RB","JJ","JJR","JJS","RBR","RBS","RP","IN"]
    tagsubject=["TO","MD","WP","VBD","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP","IN"]
    subject = ''
    object = ''
    relation = ''
    relIndex=[]
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]    
    subjectflageMain=False
    subjectflage=False
    mainsubject=''
    mainsubjectindex=[]
    mainrelation=''
    mainrelationindex=[]
    for token in tokens:
        
        object = ''
        objectindex=[]
        rtoken=[]
        subtree=[]
        if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
            continue
        if(token.ent_type_ in ['MONEY']):
            tokenEnter=token.left_edge.text+token.text
            start=token.left_edge.i
            end=token.i+1
        else:
            start = token.i
            end = token.i + 1
            tokenEnter=token.text
        
        if(((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=token.lower_) and (token.lower_ not in subject.lower().split(' _ ')): 
                        if(subjectflageMain or subjectflage):
                           subject=appendSOChunk("", tokenEnter)
                           subjectflageMain=False
                        else:                            
                            subject=appendSOChunk(subject, tokenEnter)
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendChunkandtoken(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                        relation = ''
                        relIndex=[]
                      else:                          
                          if(subjectflageMain):
                           subject=appendSOChunk("", tokenEnter)
                           subjectflageMain=False
                           subjectindex.append((start,end))
                           relation = ''
                           relIndex=[]
                        
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject=="" or subjectflageMain):
                            if(subjectflageMain):
                                subject=appendSOChunk("", tokenEnter)
                                subjectflageMain=False
                            else:
                                subject=appendSOChunk(subject, tokenEnter)
                            if token.conjuncts:
                                conjuncts = token.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendChunkandtoken(subject, spanconj.text)
                                        end=spanconj.i+1
           
                            subjectindex.append((start,end))
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
            if (object!=token.text and subject!=token.text and token.text not in object.split('_') and token.text not in subject.split(' _ ') and token.text not in relation.split() and token.text not in subject.split(' and ')):
                         object = appendChunk(object, tokenEnter)
                         if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendChunkandtoken(object, spanconj.text)
                                    end=spanconj.i+1
                         objectindex=(start,end) 
        if(token.pos_ in ['SYM'] and token.head.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
            if (object!=token.text and subject!=token.text and token.text not in object.split('_') and token.text not in subject.split(' _ ') and token.text not in relation.split() and token.text not in subject.split(' and ')):
                        
                         object = appendChunk(object, token.text+token.head.text)
                         start=token.i
                         end=token.head.i+1
                         if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendChunkandtoken(object, spanconj.text)
                                    end=spanconj.i+1
                         objectindex=(start,end)

        #Relation
        if (token.dep_ in ADJECTIVES or token.pos_ in ["VERB","ADP","ADV","PART","AUX"] or token.tag_ in tagRelation):#and token.ent_type_ not in ['PERCENT','CARDINAL']
                       if'prep' in token.dep_ or token.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
                         if relation == '':
                            if(token.head.text==token.text):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagRelation):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["PART"] and token.tag_ in ["TO"]):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            elif(token.head.text==subject):
                                relation = appendChunk('', token.text)
                                relIndex.append(token.i)
                            else:
                                relation = appendChunktoken(token.head.text, token.text)
                                relIndex.append(token.head.i)
                                relIndex.append(token.i)
                         else:
                           if(token.text not in relation.split(' ')):                                
                            if(token.head.text==token.text):
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)
                            elif(token.head.text==relation or token.head.text+' '==relation):
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagRelation):
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["PART"] and token.tag_ in ["TO","IN"]):
                                relation = appendChunk(relation+', ', token.text)
                                relIndex.append(token.i)
                            elif(token.pos_ in ["PART"] and token.dep_ in["neg"]):    
                                relation = appendChunk(relation, token.text)
                                relIndex.append(token.i)                            
                            else:
                                relation = appendChunktoken(relation+', '+token.head.text, token.text)
                                relIndex.append(token.head.i)
                                relIndex.append(token.i) 
                       elif relation == '':
                                relation = appendChunktoken('', token.text)
                                relIndex.append(token.i)
                       else:
                                relation = appendChunktoken(relation, token.text) 
                                relIndex.append(token.i)                     

        subtree = list(token.subtree)
        flagestree=False
        flagestree2=False
        for xsub in subtree:
           
           if((xsub.text in subject.split(' and ') or xsub.text in subject.split(' _ ')) and subject!=''):
                  flagestree=True  
           
           if(relation!=''):
            for relationtoken in relation.split():
              if(relationtoken in xsub.text):
                 flagestree2=True   
           if(flagestree and flagestree2):
                 break

        if(flagestree and flagestree2):
          mainsubject=subject
          mainsubjectindex=subjectindex
          mainrelation=relation
          mainrelationindex=relIndex
          relation=''
          relIndex=[]
          subject=''
          subjectindex=[]
          subjectflage=False
          #for xtoken in subtree:
          #    if(token.ent_type_ in ['MONEY']):
          #          tokenEnter=xtoken.left_edge.text+xtoken.text
          #          start=xtoken.left_edge.i
          #          end=xtoken.i+1
          #    else:
          #          start = xtoken.i
          #          end = xtoken.i + 1
          #          tokenEnter=xtoken.text
          #    #object                                  
          #          if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
          #              if(subject=='' or subjectflage):
          #                  subject=appendSOChunk(subject, tokenEnter)
          #                  subjectflage=False
          #                  if xtoken.conjuncts:
          #                      conjuncts = xtoken.conjuncts             # tuple of conjuncts
          #                      for conj in conjuncts:
          #                          if(conj.tag_ not in tagsubject):
          #                              spanconj = conj
          #                              subject = appendChunkandtoken(subject, spanconj.text)
          #                              end=spanconj.i+1
           
          #                  subjectindex.append((start,end))
          #              if (object!=tokenEnter and subject!=tokenEnter and tokenEnter not in object.split(' _ ') and tokenEnter not in subject.split(' and ') and tokenEnter not in subject.split(' _ ')):
          #               if(tokenEnter in subject.split(' _ ')):
          #                      subjectlist =subject.split(' _ ')
          #                      if(len(subjectlist)>0):
          #                          objectindex=subjectlist.index(tokenEnter)
          #                          subjectindexobject=subjectindex[objectindex]
          #                          subjectindex.remove(subjectindexobject)
          #                          subject=subject.replace(tokenEnter+' _ ','')
          #               if(xtoken.i in relIndex):
          #                      relationlist =relation.replace(',','').split()
          #                      if(len(relationlist)>0):
                                    
          #                          relIndex.remove(xtoken.i)
          #                          relation=relation.replace(tokenEnter,'')         

          #               object = appendSOChunk(object, tokenEnter)
                         
          #               if xtoken.conjuncts:
          #                  conjuncts = xtoken.conjuncts             # tuple of conjuncts
          #                  for conj in conjuncts:
          #                      if(conj.tag_ not in tagsubject):
          #                          spanconj = conj
          #                          object = appendChunkandtoken(object, spanconj.text)
          #                          end=spanconj.i+1
          #               objectindex=(start,end) 
          #    #subject      
          #          if(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
          #            if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                        
          #              if(subjectflage or subjectflageMain):
          #                  subject=appendSOChunk("", tokenEnter)
                        
          #              else:                            
          #                  subject=appendSOChunk(subject,tokenEnter)
          #              if xtoken.conjuncts:
          #                  conjuncts = xtoken.conjuncts             # tuple of conjuncts
          #                  for conj in conjuncts:
          #                      if(conj.tag_ not in tagsubject):
          #                          spanconj = conj
          #                          subject = appendChunkandtoken(subject, spanconj.text)
          #                          end=spanconj.i+1
           
          #              subjectindex.append((start,end))
          #              #Relation
          #          # if(relation in ['is','are','am'] and xtoken.text in ['is','are','am']):
          #          #         relation = ''
          #          #         relIndex=[] 
          #    #relation      
          #          if ((xtoken.dep_ in ADJECTIVES or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] or xtoken.tag_ in tagRelation) and (xtoken.text not in relation.split() or subjectflage)):#and xtoken.ent_type_ not in ['PERCENT','CARDINAL']
          #             if'prep' in xtoken.dep_ or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
          #               if relation == '' or subjectflage:
          #                  subjectflage=False
          #                  if(xtoken.head.text==xtoken.text):
          #                      relation = appendChunk('', xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
          #                      relation = appendChunk('', xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO"]):
          #                      relation = appendChunk('', xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.head.text==subject):
          #                      relation = appendChunk('', xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  else:
          #                      relation = appendChunktoken(xtoken.head.text, xtoken.text)
          #                      relIndex.append(xtoken.head.i)
          #                      relIndex.append(xtoken.i)
          #               else:
          #                 if(xtoken.text not in relation.split(' ')):
          #                  if(xtoken.head.text==xtoken.text):
          #                      relation = appendChunk(relation, xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.head.text==relation or xtoken.head.text+' '==relation):
          #                      relation = appendChunk(relation, xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] and xtoken.tag_ in tagRelation):
          #                      relation = appendChunk(relation, xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.pos_ in ["PART"] and xtoken.tag_ in ["TO","IN"]):
          #                      relation = appendChunk(relation+', ', xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  elif(xtoken.pos_ in ["PART"] and xtoken.dep_ in["neg"]):    
          #                      relation = appendChunk(relation, xtoken.text)
          #                      relIndex.append(xtoken.i)
          #                  else:
          #                      relation = appendChunktoken(relation+', '+xtoken.head.text, xtoken.text)
          #                      relIndex.append(xtoken.head.i)
          #                      relIndex.append(xtoken.i) 
          #             elif relation == '':
          #                      relation = appendChunktoken('', xtoken.text)
          #                      relIndex.append(xtoken.i)
          #             else:
          #                      relation = appendChunktoken(relation, xtoken.text) 
          #                      relIndex.append(xtoken.i)
                    
          #          if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
          #              if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
          #                bflage=True
          #                for x in trible:
          #                    if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip() in x[2]):
          #                        bflage=False
          #                    for xc in subject.split(' _ '):
          #                         if(xc in x[0] and relation.strip() in x[1] and object.strip() in x[2]):
          #                              bflage=False
                                   
                              
          #                if(bflage):
          #                  subjectflage=True
          #                  trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
          #                  print (subject.strip(), "/", relation.strip(), "/", object.strip())
          #                  start=np.min(relIndex)
          #                  end=np.max(relIndex)
          #                  if(subjectindex[0][0]<start):
          #                      start=subjectindex[0][0]
          #                  if(objectindex[1]>end):
          #                       end=objectindex[1]
          #                  end=start+100
          #                  if(end>len(tokens)):
          #                      end=len(tokens)-1
          #                  start=0
          #                  end=len(tokens)
          #                  rtoken=[token.text for token in tokens[start:end]]
          #                  rtoken1.append(rtoken)
          #                  object = ''
          #                  objectindex=[]
          #                  rtoken=[]
                            
          #                else:  
          #                  subjectflage=True
          #                  object = ''
          #                  objectindex=[]
          #                  rtoken=[]
                           
          #              else:
          #                  subjectflage=True
          #                  object = ''
          #                  objectindex=[]
          #                  rtoken=[]
                           
          subject=mainsubject
          subjectindex=mainsubjectindex
          relation=mainrelation
          relIndex=mainrelationindex
          for xtoken in subtree:
                      if xtoken.dep_ in ["punct","cc"]  or "PUNCT" in xtoken.pos_ or xtoken.is_punct:
                            continue
                      if(token.pos_ in ['SYM'] and token.head.dep_ in OBJECTS):
                        object = appendChunk(object, xtoken.text+xtoken.head.text)
                        start=xtoken.i
                        end=xtoken.head.i+1
                        objectindex=(start,end)
                      if(xtoken.tag_ in tagsubject):
                        if(xtoken.i>mainrelationindex[-1]):
                          relation = appendChunk(relation, xtoken.text)
                          relIndex.append(xtoken.i)
                      if (xtoken.tag_ not in tagsubject and object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split('_') and xtoken.text not in subject.split(' _ ') and xtoken.text not in relation.split() and xtoken.text not in subject.split(' and ')):
                       if(xtoken.i>mainrelationindex[-1]):
                        if(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
                            if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                                #if(subjectflage or subjectflageMain):
                                start=xtoken.i
                                end=xtoken.i+1
                                subject=appendSOChunk("", xtoken.text)
                                subjectflageMain=False
                                #else:
                                #    subject=appendSOChunk(subject,xtoken.text)
                                if xtoken.conjuncts:
                                    conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                    for conj in conjuncts:
                                        if(conj.tag_ not in tagsubject):
                                            spanconj = conj
                                            subject = appendChunkandtoken(subject, spanconj.text)
                                            end=spanconj.i+1
                                subjectindex.append((start,end))
                        else:
                            object = appendChunk(object, xtoken.text)
                            start=xtoken.i
                            end=xtoken.i+1
                            objectindex=(start,end)
                      if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            #subjectflageMain=True
                          else:
                              object = ''
                              objectindex=[]
                         
                        else:
                            #subjectflageMain=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                           

        if(object=="" and (token.text not in subject)  and relation!="" and relation!=token.text):
            #
            bflage=True
            # Find all tokens before punctuation
            tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
            if tokenslist:
            # Get the last token
                last_token = tokenslist[-1]
            if(token.text!=last_token):
                bflage=False
            
            if(bflage):
            
              relation=relation.replace(token.text, '')
              if(token.i in relIndex):
                relIndex.remove(token.i)
              object = appendSOChunk(object, token.text)
              start=token.i
              end=token.i+1
              if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ in tagRelation):
                                    spanconj = conj
                                    object = appendChunkandtoken(object, spanconj.text)
                                    end=spanconj.i+1
              objectindex=(start,end)          
        
        #if relation == '' and object!='' and subject!='':
        #     if (token.head.dep_ in ADJECTIVES or token.head.pos_ == "VERB") and token.head.ent_type_ not in ['PERCENT','CARDINAL']:
        #       if relation == '':
        #        relation = appendChunk('', token.head.text)
        #        relIndex.append(token.head.i)
        #       else:
        #        relation = appendChunk(relation, token.head.lower_) 
        #        relIndex.append(token.head.i)
        #     elif (token.head.head.dep_ in ADJECTIVES or token.head.head.pos_ == "VERB") and token.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
        #       if relation == '':
        #        relation = appendChunk('', token.head.head.lower_)
        #        relIndex.append(token.head.head.i)
        #       else:
        #        relation = appendChunk(relation, token.head.head.lower_) 
        #        relIndex.append(token.head.head.i) 
        #     elif (token.head.head.head.dep_ in ADJECTIVES or token.head.head.head.pos_ == "VERB") and token.head.head.head.ent_type_ not in ['PERCENT','CARDINAL']:
        #       if relation == '':
        #        relation = appendChunk('', token.head.head.head.lower_)
        #        relIndex.append(token.head.head.head.i)
        #       else:
        #        relation = appendChunk(relation, token.head.head.head.lower_) 
        #        relIndex.append(token.head.head.head.i) 

        if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2]):
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            # subject=''
                            # subjectindex=[]
                            subjectflageMain=True
                         
                        else:
                            subjectflageMain=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
    return trible,rtoken1

def SubjectObjectrelation_NewOLd(tokens):
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]#
    OBJECTS = ["dobj", "dative", "oprd","pobj","iobj","nummod","compound","npadvmod","advmod","mod","attr","acomp"]
    #OBJECTS = ["dobj", "dative", "oprd","pobj","appos","nummod"]
    ADJECTIVES = ["ROOT",'ccomp','prep', "advcl",  "amod", "nn", "nmod", "complm", "adj","agent",
                 "advcl","hmod", "infmod","rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod""acomp",
    #tagrelation=["TO","VBD","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP"]#
    tagrelation=["TO","VBD","VBD","VB","VBG","VBN","VBP","VBZ","JJR","JJS","RBR","RBS","RP","MD"]#
    tagsubject=["TO","VBD","VBD","VB","VBG","VBN","VBP","VBZ","JJR","JJS","RBR","RBS","RP","WP","VBD","WDT","WP$","WRB","MD"]#["TO","MD","WP","VBD","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP"]#
    subject = ''
    object = ''
    relation = ''
    relIndex=[]
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]    
    subjectflageMain=True
    subjectflage=False
    newSubject=''
    newSubjectIndex=[]
    relations=[]
    for token in tokens:
        xcomp_verbs=[]
        object = ''
        objectindex=[]
        rtoken=[]
        subtree=[]
        if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
            continue
        if(token.ent_type_ in ['MONEY']):
            tokenEnter=token.left_edge.text+token.text
            start=token.left_edge.i
            end=token.i+1
        else:
            start = token.i
            end = token.i + 1
            tokenEnter=token.text
        if token.dep_ == "xcomp":
                xcomp_verbs.append(token)        
        elif(((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=token.lower_) and (token.lower_ not in subject.lower().split(' _ ')): 
                        if(subjectflageMain or subjectflage):
                           subject=appendSOChunk("", tokenEnter)
                           subjectflageMain=False
                        else:                            
                            subject=appendSOChunk(subject, tokenEnter)
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendChunkandtoken(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                        lastIndex=end
                        relation = ''
                        relIndex=[]
                      else:                          
                          if(subjectflageMain):
                           subject=appendSOChunk("", tokenEnter)
                           subjectflageMain=False
                           subjectindex.append((start,end))
                           lastIndex=end
                           relation = ''
                           relIndex=[]
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject=='' or subjectflageMain):
                        #if(len(trible)==0 or subject==""):
                            if(subjectflageMain and subject==''):
                              subject=appendSOChunk("", tokenEnter)
                              subjectflageMain=False
                            # else:
                            #     subject=appendSOChunk(subject, tokenEnter)
                              if token.conjuncts:
                                conjuncts = token.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendChunkandtoken(subject, spanconj.text)
                                        end=spanconj.i+1
           
                              subjectindex.append((start,end))
                              relation = ''
                              relIndex=[]
                              lastIndex=end
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
            if (object!=token.text and subject!=token.text and token.text not in object.split('_') and token.text not in subject.split(' _ ') and token.text not in relation.split() and token.text not in subject.split(' and ')):
                 if(end>subjectindex[-1][1]):
                        #if(token.head.pos_ =='ADP' and token.head.tag_=='IN'):
                        #    if(all(token.head.i+1 in tup for tup in subjectindex) or (token.head.i in relIndex)):
                        #          relation = appendChunk(relation,token.text)
                        #          relIndex.append(token.i)                                                
                        #    else:
                        #          relation = appendChunk(relation,token.text)
                        #          relIndex.append(token.i)
                        #          object = appendChunk(object,token.right_edge.text)
                        #          start=token.right_edge.i
                        #          end=token.right_edge.i+1
                        #          objectindex=(start,end)  
                        #    #if(token.head.i in relIndex):
                        #    #           object = appendChunk(object,token.text)
                        #    #           start=token.i
                        #    #           end=token.i+1
                        #    #           objectindex=(start,end)  
                        #    #if(token.head.head.i in relIndex):
                        #    #           relation = appendChunk(relation, token.head.text)
                        #    #           relIndex.append(token.head.i)                                
                        #    #           object = appendChunk(object,token.text)
                        #    #           start=token.i
                        #    #           end=token.i+1
                        #    #           objectindex=(start,end) 
                        #    #else:
                                       
                        #    #    object = appendChunk(object,token.head.text+' '+tokenEnter)
                        #    #    start=xtoken.head.i
                        #    #    end=xtoken.i+1
                        #    #    objectindex=(start,end)  
                        #else:
                        object = appendChunk(object, tokenEnter)
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendChunkandtoken(object, spanconj.text)
                                    end=spanconj.i+1
                        objectindex=(start,end) 
        elif(token.pos_ in ['SYM'] and token.head.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
            if (object!=token.text and subject!=token.text and token.text not in object.split('_') and token.text not in subject.split(' _ ') and token.text not in relation.split() and token.text not in subject.split(' and ')):
                    if(end>subjectindex[-1][1]):
  
                         object = appendChunk(object, token.text+token.head.text)
                         start=token.i
                         end=token.head.i+1
                         if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendChunkandtoken(object, spanconj.text)
                                    end=spanconj.i+1
                         objectindex=(start,end)
        #Relation
        if token.dep_ == "relcl":  # Relative clause dependency tagsubject
                        obj_relcl = None
                        for child in token.children:
                            if child.dep_ in OBJECTS:
                               if(token.head.text.lower() not in subject.lower().split(' _ ') and token.head.tag_ not in tagsubject):#
                                subj_relcl = token.head.text  # The noun the relative clause refers to
                                start=token.head.i
                                end=token.head.i+1
                                subject=appendSOChunk("", subj_relcl)
                                subjectindex.append((start,end))
                                relation = ''
                                relIndex=[]
                               verb_relcl = token
                               if(token.i not in relIndex):
                                   if(token.left_edge.dep_=="aux" and token.left_edge.i not in relIndex):
                                       relation = appendChunk(relation, token.left_edge.text)
                                       relIndex.append(token.left_edge.i)
                                   relation = appendChunk(relation, token.text)
                                   relIndex.append(token.i)
                               obj_relcl = child.text
                               object = appendChunk("", obj_relcl)
                               start=child.i
                               end=child.i+1
                               objectindex=(start,end)
                        
                # Case 1: Handling direct SVO relations
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':  # The main verb in the root clause
            # Find the subject of the verb (nsubj)
            subject1 = [t for t in token.lefts if t.dep_ == 'nsubj']
            
            # Find the direct object (dobj)
            direct_object = [t for t in token.rights if t.dep_ == 'dobj']
            
            # Handle indirect object (iobj) if present
            indirect_object = [t for t in token.rights if t.dep_ == 'iobj']
            
            # Create relations for direct and indirect objects
            if subject and direct_object:
                relations.append((subject1[0], token, direct_object[0]))
            if subject and indirect_object:
                relations.append((subject1[0], token, indirect_object[0]))
        
        # Case 2: Handling passive sentences (where the object becomes the subject)
        if token.pos_ == 'VERB' and token.dep_ == 'nsubjpass':  # Passive voice, subject is the receiver
            # Find the subject (which is the object in passive voice)
            subject1 = [t for t in token.lefts if t.dep_ == 'nsubjpass']
            
            # Find the actual object of the verb (dobj in passive construction)
            direct_object = [t for t in token.rights if t.dep_ == 'dobj']
            
            # Create a triple for passive constructions
            if subject and direct_object:
                relations.append((subject1[0], token, direct_object[0]))
        
        # Case 3: Handling infinitive clauses (like "To allow this level...")
        if token.pos_ == 'VERB' and token.dep_ == 'xcomp':  # Open clausal complement
            subject1 = [t for t in token.lefts if t.dep_ == 'nsubj']
            if subject:
                relations.append((subject[0], token, token.text))  # Verb + infinitive (like "allow")
        
        #elif (token.dep_ in ADJECTIVES or token.pos_ in ["VERB","ADP","ADV","PART","AUX"] or token.tag_ in tagsubject):#and token.ent_type_ not in ['PERCENT','CARDINAL']
        #               if'prep' in token.dep_ or token.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
        #                 if relation == '':
        #                    if(token.head.text==token.text):
        #                        relation = appendChunk('', token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagsubject):
        #                        relation = appendChunk('', token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.pos_ in ["PART"] and token.tag_ in ["TO"]):
        #                        relation = appendChunk('', token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.head.text==subject):
        #                        relation = appendChunk('', token.text)
        #                        relIndex.append(token.i)
        #                    else:
        #                        relation = appendChunktoken(token.head.text, token.text)
        #                        relIndex.append(token.head.i)
        #                        relIndex.append(token.i)
        #                 else:
        #                   if(token.text not in relation.split(' ')):                                
        #                    if(token.head.text==token.text):
        #                        relation = appendChunk(relation, token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.head.text==relation or token.head.text+' '==relation):
        #                        relation = appendChunk(relation, token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.pos_ in ["VERB","ADP","ADV","PART","AUX"] and token.tag_ in tagsubject):
        #                        relation = appendChunk(relation, token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.pos_ in ["PART"] and token.tag_ in ["TO","IN"]):
        #                        relation = appendChunk(relation+', ', token.text)
        #                        relIndex.append(token.i)
        #                    elif(token.pos_ in ["PART"] and token.dep_ in["neg"]):    
        #                        relation = appendChunk(relation, token.text)
        #                        relIndex.append(token.i)                            
        #                    else:
        #                        relation = appendChunktoken(relation+', '+token.head.text, token.text)
        #                        relIndex.append(token.head.i)
        #                        relIndex.append(token.i) 
        #               elif relation == '':
        #                        relation = appendChunktoken('', token.text)
        #                        relIndex.append(token.i)
        #               else:
        #                        relation = appendChunktoken(relation, token.text) 
        #                        relIndex.append(token.i)                     
        if subject and relation and xcomp_verbs:
                       for xcomp_verb in xcomp_verbs:
                        # Check if xcomp has an object 
                        xcomp_obj = None
                        for child in xcomp_verb.children:
                         if child.dep_ in OBJECTS:
                            xcomp_obj = child
                         if child.tag_ in tagsubject:
                            if(child.i not in relIndex):
                             relation = appendChunk(relation,child.text)
                             relIndex.append(child.i)
                        if xcomp_obj:
                            object = appendChunk("", xcomp_obj.text)
                            start=xcomp_obj.i
                            end=xcomp_obj.i+1
                            objectindex=(start,end)
                        else:
                            object = appendChunk("", xcomp_verb.right_edge.text)
                            start=xcomp_verb.right_edge.i
                            end=xcomp_verb.right_edge.i+1
                            objectindex=(start,end)
        subtree = list(token.subtree)
        flagestree=False
        flagestree2=False
        for xsub in subtree:
           
           if((xsub.text in subject.split(' and ') or xsub.text in subject.split(' _ ')) and subject!=''):
                  flagestree=True  
           
           if(relation!=''):
            for relationtoken in relation.split():
              if(relationtoken in xsub.text):
                 flagestree2=True   
           if(flagestree and flagestree2):
                 break
        if(flagestree or flagestree2):
          #lastIndex=relIndex[-1]
          xcomp_verbs=[]  
          subjectflage=True
          newObject=''
          newObjectIndex=[]
          relation=''
          relIndex=[]
          for indx,xtoken in enumerate(subtree):                      
                      xcomp_verbs=[]
                      if xtoken.dep_ in ["punct","cc"]  or "PUNCT" in xtoken.pos_ or xtoken.is_punct:
                            continue
                      if(xtoken.i+1<subjectindex[-1][1]):
                          continue
                      if(xtoken.pos_ =='ADP' and xtoken.tag_=='IN'):
                          if(all(xtoken.head.i+1 in tup for tup in subjectindex) or (xtoken.head.i in relIndex)):
                                  
                                  relation = appendChunk(relation,xtoken.text)
                                  relIndex.append(xtoken.i)                                                
                          else:
                                  relation = appendChunk(relation,xtoken.text)
                                  relIndex.append(xtoken.i)
                                  object = appendChunk(object,xtoken.right_edge.text)
                                  #object = appendChunk(object,xtoken.text+' '+xtoken.right_edge.text)
                                  #start=xtoken.i
                                  start=xtoken.right_edge.i
                                  end=xtoken.right_edge.i+1
                                  objectindex=(start,end)  
                      # Step 2: Handle multiple xcomp dependencies
                      if xtoken.dep_ in  ["xcomp"]:
                         if(xtoken.i not in relIndex):
                          if(subjectflage):
                                relation = appendChunk('',xtoken.text)
                                relIndex=[]
                                relIndex.append(xtoken.i)
                                subjectflage=False
                          else:
                              relation = appendChunk(relation,xtoken.text)
                              relIndex.append(xtoken.i)
                         xcomp_verbs.append(xtoken)
                          # Handle multiple xcomp dependencies+
                      if(xtoken.dep_=="appos"):
                          if(xtoken.head.text.lower() in object.lower().split(' _ ')):
                            if(all(xtoken.i+1 not in tup for tup in objectindex)):
                              object = appendChunk(object, xtoken.text)
                              start=xtoken.i
                              end=xtoken.head.i+1
                              objectindex=(start,end) 
                          elif(xtoken.head.text.lower() in subject.lower().split(' _ ') and xtoken.tag_ not in tagsubject):
                              if(all(xtoken.i+1 not in tup for tup in subjectindex)):
                                start=xtoken.i
                                end=xtoken.i+1
                                subject=appendSOChunk(subject, xtoken.text)
                                subjectindex.append((start,end))
                                relation = ''
                                relIndex=[]
                      if(xtoken.pos_ in ['SYM'] and xtoken.head.dep_ in OBJECTS):
                        object = appendChunk(object, xtoken.text+xtoken.head.text)
                        start=xtoken.i
                        end=xtoken.head.i+1
                        objectindex=(start,end)                      
                      if(xtoken.tag_ in tagrelation or xtoken.dep_ in ADJECTIVES):
                       if xtoken.dep_ == "relcl":  # Relative clause dependency tagsubject
                        obj_relcl = None
                        for child in xtoken.children:
                            if child.dep_ in OBJECTS:
                               if(xtoken.head.text.lower() not in subject.lower().split(' _ ') and xtoken.head.tag_ not in tagsubject):#
                                subj_relcl = xtoken.head.text  # The noun the relative clause refers to
                                start=xtoken.head.i
                                end=xtoken.head.i+1
                                subjectindex=[]
                                subject=appendSOChunk("", subj_relcl)
                                subjectindex.append((start,end))
                                relation=''
                                relIndex=[]
                               newSubject=subject
                               newSubjectIndex.append((start,end))  
                               verb_relcl = xtoken
                               if(xtoken.i not in relIndex):
                                   if(xtoken.left_edge.dep_=="aux" and xtoken.left_edge.i not in relIndex):
                                       relation = appendChunk(relation, xtoken.left_edge.text)
                                       relIndex.append(xtoken.left_edge.i)
                                   relation = appendChunk(relation, xtoken.text)
                                   relIndex.append(xtoken.i)
                               obj_relcl = child.text
                               object = appendChunk("", obj_relcl)
                               start=child.i
                               end=child.i+1
                               objectindex=(start,end)
                       elif(xtoken.i+1>subjectindex[-1][1]):
                         if(xtoken.i not in relIndex): 
                             if(subjectflage and xtoken.pos_ !='ADP' and xtoken.tag_!='IN'):
                                 if(xtoken.dep_=='aux'):
                                     relation = appendChunk('',xtoken.text+' '+xtoken.head.text)
                                     relIndex=[]
                                     relIndex.append(xtoken.i)
                                     relIndex.append(xtoken.head.i)
                                 else:
                                    relation = appendChunk('',xtoken.text)
                                    relIndex=[]
                                    relIndex.append(xtoken.i)
                                 subjectflage=False
                             elif (xtoken.pos_ !='ADP' and xtoken.tag_!='IN'):
                                relation = appendChunk(relation,xtoken.text)
                                relIndex.append(xtoken.i)
                      if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                                  if (object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split('_') and xtoken.text not in subject.split(' _ ') and xtoken.text not in relation.split() and xtoken.text not in subject.split(' and ')):
                                      if(indx==0):
                                          start=xtoken.i
                                          end=xtoken.i+1
                                          subject=appendSOChunk("", xtoken.text)
                                          subjectindex.append((start,end))
                                          subjectflageMain=False
                                      if(xtoken.i+1>subjectindex[-1][1]):
                                              if(xtoken.head.pos_ =='ADP' and xtoken.head.tag_=='IN'):
                                                  if(xtoken.head.i in relIndex):
                                                             object = appendChunk(object,xtoken.text)
                                                             start=xtoken.i
                                                             end=xtoken.i+1
                                                             objectindex=(start,end)  
                                                  else:
                                       
                                                      object = appendChunk(object,xtoken.head.text+' '+xtoken.text)
                                                      start=xtoken.head.i
                                                      end=xtoken.i+1
                                                      objectindex=(start,end)  
                                              else:
                                               object = appendChunk(object, xtoken.text)
                                              if xtoken.conjuncts:
                                                  conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                                  for conj in conjuncts:
                                                      if(conj.tag_ not in tagsubject):
                                                          spanconj = conj
                                                          object = appendChunkandtoken(object, spanconj.text)
                                                          end=spanconj.i+1
                                              objectindex=(start,end)
                      if(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
                        if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                            if(subjectflageMain or subjectflage):
                                subjectindex=[]
                                subject=appendSOChunk("", xtoken.text)
                                subjectflageMain=False
                            else:                            
                                subject=appendSOChunk(subject, xtoken.text)
                            start=xtoken.i
                            end=xtoken.i+1
                            if xtoken.conjuncts:
                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendChunkandtoken(subject, spanconj.text)
                                        end=spanconj.i+1                            
                            subjectindex.append((start,end))
                            relation = ''
                            relIndex=[]
                        lastIndex=end                            
                      if xtoken.dep_=="attr":
                          obj = xtoken
                          newObject=xtoken.text
                          newObjectIndex=(xtoken.i,xtoken.i+1)
                          # Now, extract the entire object (subtree) of the "attr" token
                          if obj:
                              start1=0
                              end1=0
                              full_obj=''
                              # Extract the complete noun phrase (subtree) rooted at the 'attr' token
                              for child in obj.subtree:
                                  if(len(full_obj.split())==5):
                                      break;
                                  if(start1>child.i):
                                    start1=child.i
                                  if(end1<child.i+1):
                                    end1=child.i+1
                                  full_obj += " "+child.text
                              #full_obj = " ".join([child.text for child in obj.subtree])
                              object =appendChunk("", full_obj)  
                              objectindex=(start1,end1)
                      if subject and relation and xcomp_verbs:
                       relation=''
                       relIndex=[]
                       for xcomp_verb in xcomp_verbs:
                        # Check if xcomp has an object 
                        xcomp_obj = None
                        for child in xcomp_verb.children:
                         if child.dep_ in OBJECTS:
                            xcomp_obj = child
                         if child.tag_ in tagsubject:
                            if(child.i not in relIndex):
                             relation = appendChunk(relation,child.text)
                             relIndex.append(child.i)
                        if xcomp_obj:
                            object = appendChunk("", xcomp_obj.text)
                            start=xcomp_obj.i
                            end=xcomp_obj.i+1
                            objectindex=(start,end)
                        else:
                            object = appendChunk("", xcomp_verb.right_edge.text)
                            start=xcomp_verb.right_edge.i
                            end=xcomp_verb.right_edge.i+1
                            objectindex=(start,end)
                      if((subject.strip() !='' and object.strip() !='')):
                         if(relation==''):
                          if((xtoken.head.i not in relIndex) and (xtoken.head.tag_ in tagrelation or xtoken.head.dep_ in ADJECTIVES)):
                            relation = appendChunk(relation,xtoken.head.text)
                            relIndex.append(xtoken.head.i)
                      if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):

                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2] and len(object.strip()) ==len(x[2])):#
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            if(newObject!=''):
                              trible.append((subject.strip(), relation.strip(), newObject.strip(),subjectindex[:],newObjectIndex,relIndex))
                              rtoken1.append(rtoken)
                              print (subject.strip(), "/", relation.strip(), "/", newObject.strip())
                              newObject=''
                              newObjectIndex=[]
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            #relation = ''
                            #relIndex=[]
                            subjectflage=True
                            if(newSubject!=''):
                                subjectindex=[]
                                subject=appendSOChunk("", newSubject)
                                subjectindex.append((newSubjectIndex[0],newSubjectIndex[1]))
                                newSubject=''
                                newSubjectIndex=[]
                            
                                #relation = ''
                                #relIndex=[]
                            
                            #subjectflageMain=True
                          else:
                              object = ''
                              objectindex=[]
                              #relation = ''
                              #relIndex=[]
                              subjectflage=True
                        else:
                            #subjectflageMain=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            #relation = ''
                            #relIndex=[]
                            subjectflage=True
        if(object=="" and (token.text not in subject)  and relation!="" and relation!=token.text):
            #
            bflage=True
            # Find all tokens before punctuation
            tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
            if tokenslist:
            # Get the last token
                last_token = tokenslist[-1]
            if(token.text!=last_token):
                bflage=False
            
            if(bflage):
            
              relation=relation.replace(token.text, '')
              if(token.i in relIndex):
                relIndex.remove(token.i)
              object = appendSOChunk(object, token.text)
              start=token.i
              end=token.i+1
              if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    object = appendChunkandtoken(object, spanconj.text)
                                    end=spanconj.i+1
              objectindex=(start,end)  
        if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2] and len(object.strip()) ==len(x[2])):#
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            # subject=''
                            # subjectindex=[]
                            subjectflageMain=True
                         
                        else:
                            subjectflageMain=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
    return trible,rtoken1
def SubjectObjectrelation(tokens):
    SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl",'acomp']#
    OBJECTS = ["dobj", "dative", "oprd","pobj","iobj","nummod","compound","npadvmod","mod","attr","acomp"]
    #OBJECTS = ["dobj", "dative", "oprd","pobj","appos","nummod"]"advmod",,"advmod"
    ADJECTIVES = ["ROOT",'ccomp','prep', "advcl",  "amod", "nn", "nmod", "complm", "adj","agent",
                 "advcl","hmod", "infmod","rcmod", "poss","possessive","aux","neg"]#"compound","npadvmod","advmod","mod""acomp","ROOT",
    #tagrelation=["TO","VBD","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP"]#
    tagrelation=["TO","VBD","VBD","VB","VBG","VBN","VBP","VBZ","JJR","JJS","RB","RBR","RBS","RP","MD","WDT","WP$","WRB","JJ","IN"]#
    tagsubject=["TO","VBD","VBD","VB","VBG","VBN","VBP","VBZ","JJR","JJS","RBR","RBS","RP","WP","WDT","WP$","WRB","MD","JJ","IN"]#["TO","MD","WP","VBD","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP"]#
    subject = ''
    object = ''
    relation = ''
    relIndex=[]
    trible=[]
    rtoken=[]
    rtoken1=[]
    subjectindex=[]
    objectindex=[]    
    subjectflageMain=True
    subjectflage=False
    subjectfromsubtree=False
    newSubject=''
    newSubjectIndex=[]
    relations=[]
    subject1=[]
    direct_object3=[]
    flageRoot=False
    
    for token in tokens:
       flageRoot=False
       subject1=[]
       direct_object3=[]
       xcomp_verbs=[]
       object = ''
       objectindex=[]
       rtoken=[]
       subtree=[]
       if "punct" in token.dep_ or "PUNCT" in token.pos_ or token.is_punct:
                continue
       if(token.ent_type_ in ['MONEY']):
            tokenEnter=token.left_edge.text+token.text
            start=token.left_edge.i
            end=token.i+1
       else:
           start = token.i
           end = token.i + 1
           tokenEnter=token.text
       if(((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=token.lower_) and (token.lower_ not in subject.lower().split(' _ ')): 
                        #subject_phrase = tokens[token.left_edge.i:token.right_edge.i + 1]
                        #start=token.left_edge.i
                        #end=token.right_edge.i + 1
                        #tokenEnter=subject_phrase.text
                        if(subjectflageMain or subjectflage):
                           subject=appendSOChunk("", tokenEnter)
                           subjectflageMain=False
                        else:                            
                            subject=appendSOChunk(subject, tokenEnter)
                        subjectindex.append((start,end))
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendChunkandtoken(subject, spanconj.text)
                                    end=spanconj.i+1
                                    start=spanconj.i
                                    subjectindex.append((start,end))
                        
                        lastIndex=end
                        relation = ''
                        relIndex=[]
                      else:                          
                          if(subjectflageMain):
                           subject=appendSOChunk("", tokenEnter)
                           subjectflageMain=False
                           subjectindex.append((start,end))
                           lastIndex=end
                           relation = ''
                           relIndex=[]
       if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject=='' or subjectflageMain):
                            #subject_phrase = tokens[token.left_edge.i:token.right_edge.i + 1]
                            #start=token.left_edge.i
                            #end=token.right_edge.i + 1
                            #tokenEnter=subject_phrase.text
                            if(subjectflageMain and subject==''):
                              subject=appendSOChunk("", tokenEnter)
                              subjectflageMain=False
                              subjectindex.append((start,end))
                              if token.conjuncts:
                                    conjuncts = token.conjuncts             # tuple of conjuncts
                                    for conj in conjuncts:
                                        if(conj.tag_ not in tagsubject):
                                            spanconj = conj
                                            subject = appendChunkandtoken(subject, spanconj.text)
                                            end=spanconj.i+1
                                            start=spanconj.i
                                            subjectindex.append((start,end))
                              relation = ''
                              relIndex=[]
                              lastIndex=end
       subtree = list(token.subtree)
       flagestree=False
       flagestree2=False
       if((len(subtree)>=3)):
        for xsub in subtree:
           if((xsub.text in subject.split(' and ') or xsub.text in subject.split(' _ ')) and subject!=''):
                  flagestree=True
           if(any(xsub.i in tup for tup in subjectindex)):
                  flagestree=True 
           if(relation!=''):
            for relationtoken in relation.split():
              if(relationtoken in xsub.text):
                 flagestree2=True            
           if(flagestree and flagestree2):
                 break
       if token.pos_ == 'VERB' and token.dep_ == 'ROOT':  # The main verb in the root clause
                            # Find the subject of the verb (nsubj)'nsubj'
                            subject1 = [t for t in token.lefts if t.dep_ in SUBJECTS ]                            
                            # Find the direct object (dobj)
                            direct_object3 = [t for t in token.subtree if t not in subject1 and t!=token and t.dep_ not in ['appos','punct']]
                            #subtree=direct_object3 
                            #flageRoot=True
                            #relation = appendChunk('',token.text)
                            #relIndex.append(token.i)
       if(flagestree==False) and (token.lemma_ != 'say'):
           if((token in subtree) and (len(subtree)>=3)):
               subject=''
               subjectindex=[]
               flagestree=True
               subjectfromsubtree=True
       if(flagestree):#or flagestree2
          xcomp_verbs=[]  
          subjectflage=True
          newObject=''
          newObjectIndex=[]
          object = ''
          objectindex=[]
          relation=''
          relIndex=[]  
          for xtoken in subtree:
                      xcomp_verbs=[]
                      if xtoken.dep_ in ["punct","cc"]  or "PUNCT" in xtoken.pos_ or xtoken.is_punct:
                            continue
                      if(xtoken.lemma_ == 'say'):
                          continue
                      if(len(subjectindex)==0):
                            lastindex=0
                      else:
                          try:
                              if(len(subjectindex)>1):
                                  lastindex=subjectindex[0][1]
                              else:
                                lastindex=subjectindex[-1][1]
                          except :
                              lastindex=subjectindex[-1]                            
                      if xtoken.text.lower() in ["who","which"]:
                             # Find the antecedent of 'who', which is typically the subject before 'who'
                             # The antecedent should be the token with dependency relation "nsubj" or a noun phrase before 'who'
                             antecedent = None
                             for ancestor in xtoken.ancestors:
                                    if ancestor.dep_ == "nsubj" and ancestor.head == xtoken.head:
                                             antecedent=tokens[ancestor.left_edge.i:ancestor.right_edge.i + 1]
                                             start=ancestor.left_edge.i
                                             end=ancestor.right_edge.i + 1
                                             #antecedent = ancestor
                                             break
                                    if ancestor.dep_ == "ROOT" and ancestor == xtoken.head:
                                             antecedent=tokens[ancestor.left_edge.left_edge.i:ancestor.left_edge.right_edge.i + 1]
                                             #antecedent = ancestor.left_edge
                                             start=ancestor.left_edge.left_edge.i
                                             end=ancestor.left_edge.right_edge.i + 1
                                             break
                                    if (ancestor.head.dep_ == "ROOT") and ancestor.head == xtoken.head and ancestor.left_edge.tag_ not in tagsubject and ancestor.left_edge.left_edge.dep_ not in ["punct","cc"]:
                                                antecedent=tokens[ancestor.left_edge.left_edge.i:ancestor.left_edge.right_edge.i + 1]
                                                #antecedent = ancestor.left_edge
                                                start=ancestor.left_edge.left_edge.i
                                                end=ancestor.left_edge.right_edge.i + 1
                                                break 
                                    if ancestor == xtoken.head and ancestor.head.head.left_edge.tag_ not in tagsubject and ancestor.head.head.left_edge.left_edge.dep_ not in ["punct","cc"]:
                                             #antecedent = ancestor.head.head.left_edge
                                             antecedent=tokens[ancestor.head.head.left_edge.left_edge.i:ancestor.head.head.left_edge.right_edge.i + 1]
                                             start=ancestor.head.head.left_edge.left_edge.i
                                             end=ancestor.head.head.left_edge.right_edge.i + 1
                                             if('who'in antecedent.text.split()):
                                                 antecedent = ancestor.head.head.left_edge
                                                 start=ancestor.head.head.left_edge.i
                                                 end=ancestor.head.head.left_edge.i + 1
                                             if('which' in antecedent.text.split()):
                                                 antecedent = ancestor.head
                                                 start=ancestor.head.i
                                                 end=ancestor.head.i + 1  
                                             break                                            
                             if antecedent:
                                         # Extract the full noun phrase referring to the antecedent
                                         subject_phrase =antecedent
                                         #start=antecedent.left_edge.i
                                         #end=antecedent.right_edge.i + 1
                                         subject=appendSOChunk("", subject_phrase.text)
                                         subjectindex.append((start,end))
                                         subjectflageMain=False
                             #            print(f"Antecedent of 'who': {subject_phrase.text}")
                             #else:
                             #            print("No antecedent found for 'who'.")
                      # Step 2: Handle multiple xcomp dependencies
                      if xtoken.dep_ in  ["xcomp"]:
                          xcomp_verbs.append(xtoken)
                      # Handle multiple xcomp dependencies
                      # Handling infinitive clauses 
                      if subject and relation and xcomp_verbs:
                         
                       # relation=''
                       # relIndex=[]                       
                       for xcomp_verb in xcomp_verbs:
                        # Check if xcomp has an object 
                        xcomp_obj = None
                        for child in xcomp_verb.children:
                         if child.dep_ in OBJECTS:
                            xcomp_obj = child
                         else:# child.tag_ in tagsubject:
                            if(child.i not in relIndex and xtoken.i>child.i):#and xcomp_obj.i>child.i
                              if((child.i-max(relIndex))>1 and xtoken.i>child.i):
                                   relation=''
                                   relIndex=[] 
                                   if(xtoken.head.tag_ in tagrelation):
                                       relation = appendChunk(relation,xtoken.head.text)
                                       relIndex.append(xtoken.head.i)
                                   if(xtoken.left_edge.tag_ in tagrelation):
                                       relation = appendChunk(relation,xtoken.left_edge.text)
                                       relIndex.append(xtoken.left_edge.i)
                                   relation = appendChunk(relation,xtoken.text)
                                   relIndex.append(xtoken.i)
                              relation = appendChunk(relation,child.text)
                              relIndex.append(child.i)
                        if xcomp_obj:
                            object = appendChunk("", xcomp_obj.text)
                            start=xcomp_obj.i
                            end=xcomp_obj.i+1
                            objectindex=(start,end)
                            asd=[t for t in tokens[xtoken.head.left_edge.i:xtoken.head.right_edge.i+1] if t.tag_ not in tagrelation and t.pos_ not in ['NUM','CCONJ','PUNCT'] and t.i+1<start and t.lemma_!='who']
                            if(len(asd)>0):
                                subject=''
                                subjectindex=[]
                                for asdword in asd:
                                    start=asdword.i
                                    end=asdword.i+1
                                    subject=appendSOChunk(subject, asdword.text)
                                    subjectindex.append((start,end))
                            newSubjectIndex=[]
                            newSubject=subject
                            newSubjectIndex.append((start,end))
                        else:
                            asd=[t for t in tokens[xtoken.head.left_edge.i:xtoken.head.right_edge.i+1] if t.tag_ not in tagrelation and t.pos_ not in ['NUM','CCONJ','PUNCT'] and t.i+1<start and t.lemma_!='who']
                            if(len(asd)>0):
                                subject=''
                                subjectindex=[]
                                for asdword in asd:
                                    start=asdword.i
                                    end=asdword.i+1
                                    subject=appendSOChunk(subject, asdword.text)
                                    subjectindex.append((start,end))  
                            newSubject=subject
                            newSubjectIndex=[]
                            newSubjectIndex.append((start,end))
                            object = appendChunk("", xcomp_verb.right_edge.text)
                            start=xcomp_verb.right_edge.i
                            end=xcomp_verb.right_edge.i+1
                            objectindex=(start,end)
                       if(xtoken.i not in relIndex and xtoken.i != start):
                        relation = appendChunk(relation,xtoken.text)
                        relIndex.append(xtoken.i)
                      elif(xtoken.dep_=="advmod" and xtoken.ent_type_=='DATE'):
                          subjectindex=[]
                          subject=appendSOChunk("", xtoken.text)
                          subjectindex.append((xtoken.i,xtoken.i+1))
                          relation=''
                          relIndex=[]
                          if(xtoken.head.tag_=='VBN'):
                            relation = appendChunk(relation,tokens[xtoken.head.i-1].text)
                            relIndex.append((xtoken.head.i)-1)
                          relation = appendChunk(relation,xtoken.head.text)
                          relIndex.append(xtoken.head.i)
                          subjectflage=False
                      elif(xtoken.dep_=="advmod" and xtoken.pos_ == 'ADV' and xtoken.tag_ == 'RB'):
                          object = appendChunk(object, xtoken.text)
                          start=xtoken.i
                          end=xtoken.i+1
                          objectindex=(start,end) 
                      elif (xtoken.dep_ == 'dep' and xtoken.pos_=='NOUN'):
                          subjectindex=[]
                          subject=appendSOChunk("", xtoken.text)
                          subjectindex.append((xtoken.i,xtoken.i+1))
                          subjectflage=False
                      elif(xtoken.pos_ =='ADP' and xtoken.tag_=='IN'):
                          if(subjectflage):
                                      relation = ''
                                      relIndex=[]
                                      subjectflage=False                                      
                          if(xtoken.tag_=='IN' and xtoken.head.i not in relIndex and xtoken.lemma_=='in'):
                              if((any(xtoken.head.i in tup for tup in subjectindex))==False):
                                  if xtoken.head.dep_ in  ["xcomp"]:
                                      asd=[t for t in tokens[xtoken.head.head.left_edge.i:xtoken.head.head.right_edge.i+1] if t.tag_ not in tagrelation and t.pos_ not in ['NUM','CCONJ','PUNCT'] and t.i+1<start and t.lemma_!='who']
                                      if(len(asd)>0):
                                            subject=''
                                            subjectindex=[]
                                            relation=''
                                            relIndex=[]
                                            for asdword in asd:
                                                start=asdword.i
                                                end=asdword.i+1
                                                subject=appendSOChunk(subject, asdword.text)
                                                subjectindex.append((start,end))
                                      newSubjectIndex=[]
                                      newSubject=subject
                                      newSubjectIndex.append((start,end))
                                      relation = appendChunk(relation,xtoken.head.head.text)
                                      relIndex.append(xtoken.head.head.i)
                                      relation = appendChunk(relation,xtoken.head.text)
                                      relIndex.append(xtoken.head.i)
                                      for child in xtoken.head.children:
                                            if child.dep_ in OBJECTS:
                                                xcomp_obj = child
                                            else:# child.tag_ in tagsubject:
                                                if(child.i not in relIndex and (child.tag_!='TO' and xtoken.tag_!='IN')):
                                                    relation = appendChunk(relation,child.text)
                                                    relIndex.append(child.i)
                                  else:
                                      relation = appendChunk(relation,xtoken.head.text)
                                      relIndex.append(xtoken.head.i)
                          if((any(xtoken.head.i+1 in tup for tup in subjectindex) or (xtoken.head.i in relIndex)) and (xtoken.i not in relIndex)):
                                  relation = appendChunk(relation,xtoken.text)
                                  relIndex.append(xtoken.i)                                                
                          elif(xtoken.i not in relIndex):
                              if(xtoken.right_edge.head.text==xtoken.text):
                                  if((xtoken.head.i in relIndex)):
                                    #if(subjectflage):
                                    #  relation = ''
                                    #  relIndex=[]
                                    #  subjectflage=False
                                    relation = appendChunk(relation,xtoken.text)
                                    relIndex.append(xtoken.i)
                                  elif(xtoken.head.tag_ in tagrelation or xtoken.head.dep_ in ADJECTIVES):
                                    #if(subjectflage):
                                    #  relation = ''
                                    #  relIndex=[]
                                    #  subjectflage=False
                                    if(xtoken.head.i-1<len(tokens)):
                                     if(xtoken.head.tag_ == 'VBG' and tokens[xtoken.head.i-1].tag_ in tagrelation):
                                        x=tokens[xtoken.head.i-1:xtoken.head.i + 1]
                                        relation = appendChunk(relation,x.text+' '+xtoken.text)
                                        relIndex.append(xtoken.head.i-1)
                                        relIndex.append(xtoken.head.i)
                                        relIndex.append(xtoken.i)
                                    else:
                                        relation = appendChunk(relation,xtoken.head.text+' '+xtoken.text)
                                        relIndex.append(xtoken.head.i)
                                        relIndex.append(xtoken.i)
                                  elif(xtoken.text not in object.split()):
                                    #if(subjectflage):
                                    #  relation = ''
                                    #  relIndex=[]
                                    #  subjectflage=False
                                    # subject_phrase =xtoken.head # tokens[antecedent.left_edge.i:antecedent.right_edge.i + 1]
                                    # start=xtoken.head.i
                                    # end=xtoken.head.i + 1

                                    subject_phrase = tokens[xtoken.head.i:xtoken.head.i + 1]
                                    start=xtoken.head.i
                                    end=xtoken.head.i + 1
                                    subjectindex=[]
                                    subject=appendSOChunk("", subject_phrase.text)
                                    subjectindex.append((start,end))                                    
                               
                                    relation = appendChunk(relation,xtoken.text)
                                    relIndex.append(xtoken.i)                                    

                                  object = appendChunk(object,xtoken.right_edge.text)
                                  #object = appendChunk(object,xtoken.text+' '+xtoken.right_edge.text)
                                  #start=xtoken.i
                                  start=xtoken.right_edge.i
                                  end=xtoken.right_edge.i+1
                                  objectindex=(start,end)
                              else:
                                  relation = appendChunk(relation,xtoken.head.text+' '+xtoken.text)
                                  relIndex.append(xtoken.head.i)
                                  relIndex.append(xtoken.i)
                                  xobject = [t for t in xtoken.rights if t.dep_ in OBJECTS ] 
                                  #xobject=tokens[xtoken.left_edge.i+1:xtoken.right_edge.i + 1]
                                  if(len(xobject)>0):
                                    start=xobject[0].i
                                    object = appendChunk('',xobject[0].text)
                                    end=xobject[0].i + 1
                                    objectindex=(start,end)
                      elif(xtoken.dep_=="appos"):
                          if(xtoken.head.text.lower() in object.lower().split(' _ ')):
                            try:
                              if objectIndex is None:
                               break
                            except :
                               break                            
                            if(any(xtoken.i+1 not in tup for tup in objectindex)):
                              subject_phrase = tokens[xtoken.head.left_edge.i:xtoken.head.right_edge.i + 1]
                              start=xtoken.head.left_edge.i
                              end=xtoken.head.right_edge.i + 1
                              tokenEnter=subject_phrase.text
                              if(end-start<5):                                
                                    object = appendChunk(object, tokenEnter)
                                    #start=xtoken.i
                                    #end=xtoken.head.i+1
                                    objectindex=(start,end)                                    
                              else:
                                    object = appendChunk(object, xtoken.text)
                                    start=xtoken.i
                                    end=xtoken.head.i+1
                                    objectindex=(start,end) 
                              #object = appendChunk(object, tokenEnter)
                              ##start=xtoken.i
                              ##end=xtoken.head.i+1
                              #objectindex=(start,end) 
                          elif(xtoken.head.text.lower() in subject.lower().split(' _ ') and xtoken.tag_ not in tagsubject):
                              try:
                                if subjectindex is None:
                                 break
                              except :
                                  break                              
                              if(any(xtoken.i+1 not in tup for tup in subjectindex)):
                                subject=''
                                relation = ''
                                relIndex=[]
                                subjectindex=[]                                
                                subject_phrase = tokens[xtoken.head.left_edge.i:xtoken.head.right_edge.i + 1]
                                start=xtoken.head.left_edge.i
                                end=xtoken.head.right_edge.i + 1
                                tokenEnter=subject_phrase.text
                                if(end-start<5):                                
                                    subject=appendSOChunk(subject, tokenEnter)
                                    subjectindex.append((start,end))
                                    relation = ''
                                    relIndex=[]
                                else:
                                    subject_phrase = tokens[xtoken.head.i:xtoken.i+1]
                                    start=xtoken.head.i
                                    if(end-start>5):
                                        if(xtoken.i<len(tokens)):
                                            subject_phrase = tokens[xtoken.i]
                                            start=xtoken.i
                                        else:
                                            subject_phrase = xtoken.i
                                            start=xtoken.i
                                    tokenEnter=subject_phrase.text                                    
                                    end=xtoken.i+1
                                    subject=appendSOChunk(subject, tokenEnter)
                                    subjectindex.append((start,end))
                                    newSubject=tokenEnter
                                    newSubjectIndex=[]
                                    newSubjectIndex.append((start,end))                                    
                                    #relation = ''
                                    #relIndex=[]
                          else:
                              token_phrase = tokens[xtoken.left_edge.i:xtoken.right_edge.i]
                              for x in token_phrase:
                                  if x.dep_ in ["punct","cc"]  or "PUNCT" in x.pos_ or x.is_punct:
                                        continue
                                  if(subject=='' and x.tag_ not in tagrelation):
                                      start=x.i
                                      end=x.i+1
                                      subject=appendSOChunk(subject, xtoken.text)
                                      subjectindex.append((start,end))
                                  elif(x.tag_ in tagrelation and x.i not in relIndex):                                      
                                      relation = appendChunk(relation,x.text)
                                      relIndex.append(x.i)
                                  else:
                                      start=x.i
                                      end=x.i + 1
                                      tokenEnter=x.text
                                      object = appendChunk(object,tokenEnter)
                                      objectindex=(start,end)
                      elif(xtoken.pos_ in ['SYM'] and xtoken.head.dep_ in OBJECTS):
                        object = appendChunk(object, xtoken.text+xtoken.head.text)
                        start=xtoken.i
                        end=xtoken.head.i+1
                        objectindex=(start,end)
                      elif xtoken.dep_=="attr":
                          obj = xtoken
                          if(xtoken.pos_!='NUM'):
                           if(any(xtoken.text not in tup for tup in trible)):
                            newObject=xtoken.text
                            newObjectIndex=(xtoken.i,xtoken.i+1)
                          # Now, extract the entire object (subtree) of the "attr" token
                          if obj:
                              start1=-1
                              end1=0
                              full_obj=''
                              # Extract the complete noun phrase (subtree) rooted at the 'attr' token
                              for child in obj.subtree:
                                  if(len(full_obj.split())==5):
                                      break;
                                  if(child.text==subject):
                                      continue;
                                  if(child.tag_ in tagrelation):
                                      continue;
                                  if(child.lemma_ == 'who'):
                                      break;
                                  if(child.text in relation.split()):
                                        continue;
                                  if relIndex is not None:
                                    if(len(relIndex)>0):
                                     if(child.i <= max(relIndex)):
                                           continue;
                                  if(start1==-1):
                                      start1=child.i
                                  if(start1>child.i):
                                    start1=child.i
                                  if(end1<child.i+1):
                                    end1=child.i+1
                                  full_obj += " "+child.text
                              #full_obj = " ".join([child.text for child in obj.subtree])
                              object =appendChunk("", full_obj)  
                              objectindex=(start1,end1)
                      elif((xtoken.tag_ in tagrelation or xtoken.dep_ in ADJECTIVES)):#
                       #if((xtoken.dep_== 'ROOT' and xtoken.pos_ == 'VERB')):
                       #    continue
                       if xtoken.dep_ == "relcl":  # Relative clause dependency tagsubject
                        obj_relcl = None
                        relation=''
                        relIndex=[]
                        
                        verb_relcl = xtoken
                        if(xtoken.i not in relIndex):
                                   if(xtoken.left_edge.dep_=="aux" and xtoken.left_edge.i not in relIndex):
                                       relation = appendChunk(relation, xtoken.left_edge.text)
                                       relIndex.append(xtoken.left_edge.i)
                                   relation = appendChunk(relation, xtoken.text)
                                   relIndex.append(xtoken.i)
                        for child in xtoken.children:
                            if child.dep_ in OBJECTS:
                               if(xtoken.head.text.lower() not in subject.lower().split(' _ ') and xtoken.head.tag_ not in tagsubject):#
                                subj_relcl = xtoken.head.text  # The noun the relative clause refers to
                                start=xtoken.head.i
                                end=xtoken.head.i+1
                                subjectindex=[]
                                subject=appendSOChunk("", subj_relcl)
                                subjectindex.append((start,end))
                                # relation=''
                                # relIndex=[]
                               newSubject=subject
                               newSubjectIndex=[]
                               newSubjectIndex.append((start,end))  
                               obj_relcl = child.text
                               object = appendChunk("", obj_relcl)
                               start=child.i
                               end=child.i+1
                               objectindex=(start,end)

                       elif(xtoken.i+1>lastindex):
                         if(xtoken.i not in relIndex): 
                             if(subjectflage and xtoken.pos_ !='ADP' and xtoken.tag_!='IN'):
                                 if(xtoken.dep_=='aux'):
                                     relIndex=[]
                                     relation=''
                                     for childrenaux in xtoken.head.children:
                                        if(childrenaux.tag_ in tagrelation or childrenaux.dep_ in ADJECTIVES):
                                          if(childrenaux.i < xtoken.head.i):
                                           relation = appendChunk(relation,childrenaux.text)
                                           relIndex.append(childrenaux.i)
                                     relation = appendChunk(relation,xtoken.head.text)
                                     relIndex.append(xtoken.head.i)
                                     
                                     # relation = appendChunk('',xtoken.text+' '+xtoken.head.text)
                                     # relIndex=[]
                                     # relIndex.append(xtoken.i)
                                     # relIndex.append(xtoken.head.i)
                                     #التغديل
                                     #subj= tokens[xtoken.head.head.i:xtoken.head.head.i + 1]
                                     subj=xtoken.head.head
                                     if((subj.pos_ in ["NOUN","PRON","PROPN"] and subj.tag_ not in tagsubject and subj.dep_ in OBJECTS) or subj.dep_ in SUBJECTS):
                                        start=xtoken.head.head.i
                                        end=xtoken.head.head.i+1
                                        subjectindex=[]
                                        subject=appendSOChunk("", subj.text)
                                        subjectindex.append((start,end))
                                 elif(xtoken.pos_ =='ADP' and xtoken.tag_=='JJ'):
                                    relation = appendChunk(xtoken.head.text,xtoken.text)
                                    relIndex=[]
                                    relIndex.append(xtoken.head.i)
                                    relIndex.append(xtoken.i)
                                 else:
                                   # asd=[t for t in tokens[xtoken.head.left_edge.i:xtoken.head.right_edge.i+1] if t.tag_ not in tagrelation and t.pos_ not in ['NUM','CCONJ','PUNCT'] and t.i+1<start and t.lemma_!='who']
                                   # if(len(asd)>0):
                                   #      subject=''
                                   #      subjectindex=[]
                                   #      relation=''
                                   #      relIndex=[]
                                   #      for asdword in asd:
                                   #          start=asdword.i
                                   #          end=asdword.i+1
                                   #          subject=appendSOChunk(subject, asdword.text)
                                   #          subjectindex.append((start,end))
                                   # else:
                                    subject_phrase=[t for t in xtoken.lefts if t.dep_ in SUBJECTS and t.tag_ not in tagsubject]
                                    if(len(subject_phrase)==0):
                                        subject_phrase=[t for t in xtoken.head.lefts if t.dep_ in SUBJECTS and t.tag_ not in tagsubject]
                                    if(len(subject_phrase)==0):
                                        antecedent = None
                                        for ancestor in xtoken.ancestors:
                                            if ancestor.dep_ == "nsubj" and ancestor.head == xtoken.head:
                                                antecedent = ancestor
                                                break
                                            if (ancestor.dep_ == "ROOT") and ancestor.head == xtoken.head and ancestor.left_edge.tag_ not in tagsubject:
                                                antecedent = ancestor.left_edge
                                                break 
                                            if (ancestor.head.dep_ == "ROOT") and ancestor.head == xtoken.head and ancestor.left_edge.tag_ not in tagsubject:
                                                antecedent = ancestor.left_edge
                                                break     
                                        if antecedent:
                                                # Extract the full noun phrase referring to the antecedent
                                                subject_phrase =antecedent # tokens[antecedent.left_edge.i:antecedent.right_edge.i + 1]
                                                start=antecedent.i
                                                end=antecedent.i + 1
                                                subject=appendSOChunk("", subject_phrase.text)
                                                subjectindex.append((start,end))
                                        else:
                                            relation = appendChunk('',xtoken.text)
                                            relIndex=[]
                                            relIndex.append(xtoken.i)
                                    elif(len(subject_phrase)>0 and subject_phrase[0].lemma_ != 'who' and ((any(subject_phrase[0].i + 1 in tup for tup in subjectindex))==False)):
                                        subject_phrase =subject_phrase[0]
                                        start=subject_phrase.i
                                        end=subject_phrase.i + 1
                                        subjectindex=[]
                                        subject=appendSOChunk("", subject_phrase.text)
                                        subjectindex.append((start,end))
                                        relation = appendChunk('',xtoken.text)
                                        relIndex=[]
                                        relIndex.append(xtoken.i)
                                    else:
                                       if(xtoken.pos_ == 'AUX'):
                                            relation = appendChunk('',xtoken.text)
                                            relIndex=[]
                                            relIndex.append(xtoken.i)
                                       else:
                                        relation = appendChunk(relation,xtoken.text)
                                        relIndex.append(xtoken.i)                                    
                                 subjectflage=False
                                 
                             else:#if (xtoken.pos_ !='ADP' and xtoken.tag_!='IN'):
                                #allword=[t for t in xtoken.subtree] 
                                #subjectsubbool=True
                                #for subjectsubtree in allword:
                                #  if( subject == subjectsubtree.text):
                                #      subjectsubbool=False
                                #      break
                                #if(subjectsubbool and len(allword)>1):  
                                #      subject=''
                                #      subjectindex=[]
                                relation = appendChunk(relation,xtoken.text)
                                relIndex.append(xtoken.i)
                      elif(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                                  if (object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split('_') and xtoken.text not in subject.split(' _ ') and xtoken.text not in relation.split() and xtoken.text not in subject.split(' and ')):
                                      lastRelation=0
                                      if(len(relIndex)>0):
                                          lastRelation=max(relIndex)                                      
                                      if(subject==''):
                                          if(xtoken.i+1>lastRelation):
                                            relIndex=[]
                                            relation=''
                                          #subject_phrase = tokens[xtoken.left_edge.i:xtoken.right_edge.i + 1]
                                          #start=xtoken.left_edge.i
                                          #end=xtoken.right_edge.i + 1
                                          #tokenEnter=subject_phrase.text
                                          start=xtoken.i
                                          end=xtoken.i+1
                                          subject=appendSOChunk("", xtoken.text)
                                          subjectindex.append((start,end))
                                          if xtoken.conjuncts:
                                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                                for conj in conjuncts:
                                                    if(conj.tag_ not in tagsubject):
                                                        spanconj = conj
                                                        subject = appendChunkandtoken(subject, spanconj.text)
                                                        end=spanconj.i+1
                                                        start=spanconj.i
                                                        subjectindex.append((start,end))
                                          
                                          subjectflageMain=False
                                      elif(xtoken.i+1>lastindex):
                                              if(xtoken.head.pos_ =='ADP' and xtoken.head.tag_=='IN'):
                                                  if(xtoken.head.i in relIndex):
                                                      if(xtoken.ent_type_ in ['MONEY']):
                                                             tokenEnter=xtoken.left_edge.text+xtoken.text
                                                             object = appendChunk(object,tokenEnter)
                                                             start=xtoken.left_edge.i
                                                             end=xtoken.i+1
                                                             objectindex=(start,end)
                                                      else:
                                                             token_phrase = tokens[xtoken.head.left_edge.i+1:xtoken.head.right_edge.i + 1]
                                                             if('who'in token_phrase.text.split() or 'Who' in token_phrase.text.split()):
                                                                  obj = xtoken
                                                                  start1=-1
                                                                  end1=0
                                                                  full_obj=''
                                                                  # Extract the complete noun phrase (subtree) rooted at the 'attr' token
                                                                  for child in obj.subtree:
                                                                      if(len(full_obj.split())==5):
                                                                          break;
                                                                      if(child.text==subject):
                                                                          continue;
                                                                      if(child.tag_ in tagrelation):
                                                                          continue;
                                                                      if(child.lemma_ == 'who'):
                                                                          break;
                                                                      if(child.text in relation.split()):
                                                                          continue;
                                                                      if relIndex is not None:
                                                                        if(len(relIndex)>0):
                                                                         if(child.i <= max(relIndex)):
                                                                               continue;                                                           
                                                                      if(start1==-1):
                                                                          start1=child.i
                                                                      if(start1>child.i):
                                                                        start1=child.i
                                                                      if(end1<child.i+1):
                                                                        end1=child.i+1
                                                                      full_obj += " "+child.text
                                                                  #full_obj = " ".join([child.text for child in obj.subtree])
                                                                  object =appendChunk("", full_obj)  
                                                                  objectindex=(start1,end1)
                                                             else:                                                                 
                                                                 start=xtoken.head.left_edge.i+1
                                                                 end=xtoken.head.right_edge.i + 1
                                                                 tokenEnter=token_phrase.text
                                                                 object = appendChunk(object,tokenEnter)
                                                                 objectindex=(start,end)
                                                  else:
                                                      if(xtoken.head.dep_=='prep'):
                                                          if(xtoken.head.i not in relIndex):
                                                            relation = appendChunk(relation,xtoken.head.text)
                                                            relIndex.append(xtoken.head.i)
                                                          object = appendChunk(object,xtoken.text)
                                                          start=xtoken.i
                                                          end=xtoken.i+1
                                                          objectindex=(start,end) 
                                                      else:
                                                          object = appendChunk(object,xtoken.head.text+' '+xtoken.text)
                                                          start=xtoken.head.i
                                                          end=xtoken.i+1
                                                          objectindex=(start,end)  
                                              else:
                                               antecedent = None
                                               for ancestor in xtoken.ancestors:
                                                        if ancestor.dep_ == "nsubj" and ancestor.head == xtoken.head:
                                                                 antecedent=tokens[ancestor.left_edge.i:ancestor.right_edge.i + 1]
                                                                 start=ancestor.left_edge.i
                                                                 end=ancestor.right_edge.i + 1
                                                                 #antecedent = ancestor
                                                                 break
                                                        if ancestor.dep_ == "ROOT" and ancestor == xtoken.head:
                                                                 antecedent=tokens[ancestor.left_edge.left_edge.i:ancestor.left_edge.right_edge.i + 1]
                                                                 #antecedent = ancestor.left_edge
                                                                 start=ancestor.left_edge.left_edge.i
                                                                 end=ancestor.left_edge.right_edge.i + 1
                                                                 break
                                                        if (ancestor.head.dep_ == "ROOT") and ancestor.head == xtoken.head and ancestor.left_edge.tag_ not in tagsubject and ancestor.left_edge.left_edge.dep_ not in ["punct","cc"]:
                                                                    antecedent=tokens[ancestor.left_edge.left_edge.i:ancestor.left_edge.right_edge.i + 1]
                                                                    #antecedent = ancestor.left_edge
                                                                    start=ancestor.left_edge.left_edge.i
                                                                    end=ancestor.left_edge.right_edge.i + 1
                                                                    break 
                                                        if ancestor == xtoken.head and ancestor.head.head.dep_ == "ROOT" and ancestor.head.head.left_edge.tag_ not in tagsubject and ancestor.head.head.left_edge.left_edge.dep_ not in ["punct","cc"]:
                                                                 #antecedent = ancestor.head.head.left_edge
                                                                 asd=[t for t in tokens[xtoken.head.left_edge.i:xtoken.head.right_edge.i+1] if t.tag_ not in tagrelation and t.pos_ not in ['NUM','CCONJ','PUNCT'] and t.i+1<start and t.lemma_!='who']
                                                                 if(len(asd)>0):
                                                                    subject=''
                                                                    subjectindex=[]
                                                                    for asdword in asd:
                                                                        start=asdword.i
                                                                        end=asdword.i+1
                                                                        subject=appendSOChunk(subject, asdword.text)
                                                                        subjectindex.append((start,end))
                                                                 # antecedent=tokens[ancestor.head.head.left_edge.left_edge.i:ancestor.head.head.left_edge.right_edge.i + 1]
                                                                 # start=ancestor.head.head.left_edge.left_edge.i
                                                                 # end=ancestor.head.head.left_edge.right_edge.i + 1
                                                                 # if('who'in antecedent.text.split()):
                                                                 #     antecedent = ancestor.head.head.left_edge
                                                                 #     start=ancestor.head.head.left_edge.i
                                                                 #     end=ancestor.head.head.left_edge.i + 1                                             
                                                                 break                                            
                                               if antecedent:
                                                             # Extract the full noun phrase referring to the antecedent
                                                             subject_phrase =antecedent
                                                             #start=antecedent.left_edge.i
                                                             #end=antecedent.right_edge.i + 1
                                                             subject=appendSOChunk("", subject_phrase.text)
                                                             subjectindex.append((start,end))
                                                             subjectflageMain=False
                                               start=xtoken.i
                                               end=xtoken.i+1
                                               object = appendChunk(object, xtoken.text)
                                               if xtoken.conjuncts:
                                                  conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                                  for conj in conjuncts:
                                                      if(conj.tag_ not in tagsubject):
                                                          spanconj = conj
                                                          object = appendChunkandtoken(object, spanconj.text)
                                                          end=spanconj.i+1                                              
                                               objectindex=(start,end)
                      elif(((xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.dep_ in SUBJECTS and xtoken.tag_ not in tagsubject)) or ("ROOT" in xtoken.dep_ and xtoken.tag_ not in tagsubject)):
                        if (subject.lower().split(' _ ')[-1]!=xtoken.lower_) and (xtoken.lower_ not in subject.lower().split(' _ ')): 
                            if(object=='' and subject !='' and subjectflage==False):
                                start=xtoken.i
                                end=xtoken.i+1
                                object = appendChunk(object, xtoken.text)
                                objectindex=(start,end)
                                newSubject=object
                                newSubjectIndex=[]
                                newSubjectIndex.append((start,end))
                            elif(subjectflageMain or subjectflage):
                                subjectindex=[]
                                #subject_phrase = tokens[xtoken.left_edge.i:xtoken.right_edge.i + 1]
                                #start=xtoken.left_edge.i
                                #end=xtoken.right_edge.i + 1
                                #tokenEnter=subject_phrase.text
                                subjectflage=False
                                subject=appendSOChunk("", xtoken.text)
                                start=xtoken.i
                                end=xtoken.i+1
                                subjectindex.append((start,end))
                                if xtoken.conjuncts:
                                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                                for conj in conjuncts:
                                                    if(conj.tag_ not in tagsubject):
                                                        spanconj = conj
                                                        subject = appendChunkandtoken(subject, spanconj.text)
                                                        end=spanconj.i+1
                                                        start=spanconj.i
                                                        subjectindex.append((start,end))
                                relation = ''
                                relIndex=[]
                                subjectflageMain=False
                            else:
                                #subject_phrase = tokens[xtoken.left_edge.i:xtoken.right_edge.i + 1]
                                #start=xtoken.left_edge.i
                                #end=xtoken.right_edge.i + 1
                                #tokenEnter=subject_phrase.text
                                
                                subject=appendSOChunk(subject, xtoken.text)
                                start=xtoken.i
                                end=xtoken.i+1
                                subjectindex.append((start,end))
                                if xtoken.conjuncts:
                                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                                for conj in conjuncts:
                                                    if(conj.tag_ not in tagsubject):
                                                        spanconj = conj
                                                        subject = appendChunkandtoken(subject, spanconj.text)
                                                        end=spanconj.i+1
                                                        start=spanconj.i
                                                        subjectindex.append((start,end))
                                relation = ''
                                relIndex=[]
                            lastIndex=end                            
                      if(flageRoot and object!=''):
                          newSubject=object
                          newSubjectIndex=[]
                          newSubjectIndex.append((objectindex[0],objectindex[1]))
                          flageRoot=False
                      if(object=="" and (xtoken.text not in subject)  and relation!="" and relation==xtoken.text):
                          #
                        bflage=True
                        # Find all tokens before punctuation
                        tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
                        if tokenslist:
                        # Get the last token
                          last_token = tokenslist[-1]
                        if(xtoken.text!=last_token):
                            bflage=False
                        if(bflage):            
                            objecttoken=tokens[xtoken.head.left_edge.i:xtoken.head.left_edge.i+1]
                            if(objecttoken.text!=subject):
                                object = appendSOChunk(object, objecttoken.text)
                                start=xtoken.head.left_edge.i
                                end=xtoken.head.left_edge.i+1
                                objectindex=(start,end)  
                      if(object=="" and (xtoken.text not in subject)  and relation!="" and relation!=xtoken.text):
                        #
                        bflage=True
                        # Find all tokens before punctuation
                        tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
                        if tokenslist:
                        # Get the last token
                          last_token = tokenslist[-1]
                        if(xtoken.text!=last_token):                            
                            tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text.split('who')[0])
                            if tokenslist:
                            # Get the last token
                              last_token = tokenslist[-1]
                            if(xtoken.text!=last_token):
                                bflage=False                        
                        if(bflage):            
                          relation=relation.replace(xtoken.text, '')
                          if(xtoken.i in relIndex):
                            relIndex.remove(xtoken.i)
                            object = appendSOChunk(object, xtoken.text)
                            start=xtoken.i
                            end=xtoken.i+1
                            objectindex=(start,end)  
                      if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip()) and (subject.strip() not in object.strip().split(' & '))):
                        bflage=True
                        if((any(objectindex in tup for tup in subjectindex))):
                            bflage=False
                        if((any(objectindex in tup and subjectindex in tup for tup in trible))):
                            bflage=False
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2] and len(object.strip()) ==len(x[2])):#
                                  bflage=False
                          if(bflage):
                            for objectcong in object.split(' & '):
                                  print (subject.strip(), "/", relation.strip(), "/", objectcong.strip())
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            #print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            if(newObject!='' and newObject.strip()!=object.strip() and newObject.strip()!=subject.strip()):
                              trible.append((subject.strip(), relation.strip(), newObject.strip(),subjectindex[:],newObjectIndex,relIndex))
                              rtoken1.append(rtoken)
                              print (subject.strip(), "/", relation.strip(), "/", newObject.strip())
                              newObject=''
                              newObjectIndex=[]
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            #relation = ''
                            #relIndex=[]
                            subjectflage=True
                            if(newSubject!=''):
                                subjectindex=[]
                                subject=appendSOChunk("", newSubject)
                                subjectindex=newSubjectIndex
                                newSubject=''
                                newSubjectIndex=[]
                            
                                #relation = ''
                                #relIndex=[]
                            
                            #subjectflageMain=True
                          else:
                              object = ''
                              objectindex=[]
                              #relation = ''
                              #relIndex=[]
                              subjectflage=True
                        else:
                            #subjectflageMain=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            #relation = ''
                            #relIndex=[]
                            subjectflage=True
       if(subjectfromsubtree):
           subject=''
           subjectindex=[]
           subjectfromsubtree=False
       # if(object=="" and (token.text not in subject)  and relation!="" and relation!=token.text):
       #      #
       #      bflage=True
       #      # Find all tokens before punctuation
       #      tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
       #      if tokenslist:
       #      # Get the last token
       #          last_token = tokenslist[-1]
       #      if(token.text!=last_token):
       #          bflage=False
            
       #      if(bflage):
            
       #        relation=relation.replace(token.text, '')
       #        if(token.i in relIndex):
       #          relIndex.remove(token.i)
       #        object = appendSOChunk(object, token.text)
       #        start=token.i
       #        end=token.i+1
       #        if token.conjuncts:
       #                      conjuncts = token.conjuncts             # tuple of conjuncts
       #                      for conj in conjuncts:
       #                          if(conj.tag_ not in tagsubject):
       #                              spanconj = conj
       #                              object = appendChunkandtoken(object, spanconj.text)
       #                              end=spanconj.i+1
       #        objectindex=(start,end)  
       if subject1 and direct_object3 :
                            relation = ''
                            relIndex=[]
                            if(token.lemma_ == 'say'):
                                relation = appendChunk(relation,token.text)
                                relIndex.append(token.i)
                            # else:
                            #   for childrenaux in token.head.children:
                            #             if(childrenaux.tag_ in tagrelation or childrenaux.dep_ in ADJECTIVES):
                            #                if(token.i<childrenaux.i):
                            #                    relation = appendChunk(relation,token.text)
                            #                    relIndex.append(token.i)
                            #                relation = appendChunk(relation,childrenaux.text)
                            #                relIndex.append(childrenaux.i)
                            # # Find all tokens before punctuation
                            # tokenslist = re.findall(r'\b\w+\b(?=\W)', tokens.text)
                            # if tokenslist:
                            # # Get the last token
                            #     last_token = tokenslist[-1]
                            # if(token.text!=last_token and token.i not in relIndex):
                            #     relation = appendChunk(relation,token.text)
                            #     relIndex.append(token.i)
                            subjectindex=[]
                            subject=''
                            for subjectItem in subject1:
                                subject_phrase = tokens[subjectItem.left_edge.i:subjectItem.right_edge.i + 1]
                                start=subjectItem.left_edge.i
                                end=subjectItem.right_edge.i + 1
                                tokenEnter=subject_phrase.text
                                tokenconjuncts=''
                                if subjectItem.conjuncts:
                                    conjuncts = subjectItem.conjuncts             # tuple of conjuncts
                                    for conj in conjuncts:
                                        if(conj.tag_ not in tagsubject):
                                            spanconj = conj
                                            tokenconjuncts +=spanconj.text
                                            end=spanconj.i+1 
                                if(end-start<=5):
                                    subject=appendSOChunk("", tokenEnter)
                                    subjectindex.append((start,end))
                                elif(end-start>=5):
                                    subject_phrase =subjectItem# tokens[subjectItem.left_edge.i:subjectItem.right_edge.i + 1]
                                    start=subjectItem.i
                                    if(subjectItem.i + 1>end):
                                        end=subjectItem.i + 1
                                    tokenEnter=subject_phrase.text+tokenconjuncts
                                    subject=appendSOChunk("", tokenEnter)
                                    subjectindex.append((start,end))
                                else:
                                 for xentites in subject_phrase.ents:
                                   if(xentites.label_!='DATE'):
                                    subject_phrase =xentites# tokens[subjectItem.left_edge.i:subjectItem.right_edge.i + 1]
                                    start=xentites.start
                                    if(xentites.end + 1>end):
                                        end=xentites.end
                                    
                                    tokenEnter=subject_phrase.text+tokenconjuncts
                                    subject=appendSOChunk(subject, tokenEnter)
                                    subjectindex.append((start,end))
                            for direct_object in direct_object3:
                                if(direct_object.dep_ in ['appos','punct']):
                                  continue
                                # if(direct_object.dep_ in ['nsubj']):
                                #     break
                                if(token.lemma_ == 'say'):  
                                    object = appendChunk(object,direct_object.text)
                                    start=direct_object.i
                                    end=direct_object.i+1
                                    objectindex=(start,end) 
                                # elif(direct_object.i not in relIndex and direct_object.i>max(relIndex)):
                                #    if(direct_object.tag_ in tagrelation and direct_object.i>max(relIndex) and direct_object.i not in relIndex and len(objectindex)==0):
                                #         relation = appendChunk(relation,direct_object.text)
                                #         relIndex.append(direct_object.i)
                                #    else:
                                #     object = appendChunk(object,direct_object.text)
                                #     start=direct_object.i
                                #     end=direct_object.i+1
                                #     objectindex=(start,end)  
                            # if(object=='' and token.text==last_token):
                            #     object = appendChunk(object,token.text)
                            #     start=token.i
                            #     end=token.i+1
                            #     objectindex=(start,end) 
       if((subject.strip() !='' and relation.strip() !='' and object.strip() !='') and (subject.strip() != object.strip())):
                        
                        if ((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex) not in trible):
                          bflage=True
                          for x in trible:
                              if(subject.strip() in x[0] and relation.strip()  in x[1] and object.strip()  in x[2] and len(object.strip()) ==len(x[2])):#
                                  bflage=False
                          if(bflage):
                            trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex[:],objectindex,relIndex))
                            print (subject.strip(), "/", relation.strip(), "/", object.strip())
                            start=np.min(relIndex)
                            end=np.max(relIndex)
                            if(subjectindex[0][0]<start):
                                start=subjectindex[0][0]
                            if(objectindex[1]>end):
                                 end=objectindex[1]
                            end=start+100
                            if(end>len(tokens)):
                                end=len(tokens)-1
                            start=0
                            end=len(tokens)
                            rtoken=[token.text for token in tokens[start:end]]
                            rtoken1.append(rtoken)
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
                            # subject=''
                            # subjectindex=[]
                            subjectflageMain=True
                         
                        else:
                            subjectflageMain=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                            relation = ''
                            relIndex=[]
    return trible,rtoken1
def splitMergeSentences(tokens,nlp):
   
    sents_doc = nlp(tokens)
    sentencesAfterpr1=''
    for x in sents_doc.sents:
         if(len(sentencesAfterpr1)>0):
            sentencesAfterpr1+=' '
         for token in x:
            if(len(sentencesAfterpr1)>0):
                    sentencesAfterpr1+=' '
            if(len(token)>25):
                if(bool(re.search(r"\s", str(token)))==False and bool(re.search("[\d+]", str(token)))==False):
                    flat_words=wordninja.split(str(token))
                    sentencesAfterpr1+=' '.join(flat_words)
                else:
                    if(len(sentencesAfterpr1)>0):
                        sentencesAfterpr1+=' '+str(token)
                    else:
                        sentencesAfterpr1+=str(token)
            else:
                if(len(sentencesAfterpr1)>0):
                    sentencesAfterpr1+=' '+str(token)
                else:
                    sentencesAfterpr1+=str(token)
    return sentencesAfterpr1
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
                
                line=re.sub(r'(?<=\d)\s+(?=(?![or,but,and])\w+)', '', line)
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
def chunktext(maxlength,text):
   #, token_estimator=BertTokenEstimator()
   text_chunker = TextChunker(chunk_size=maxlength,overlap_percent=0.3,tokens=True)
   # Generate chunks with overlapping
 
   chunks = text_chunker.chunk(text)
   #Print the resulting chunks
   #for i, chunk in enumerate(chunks):
   #     print(f"Chunk {i + 1}: {chunk}")
   return chunks
class BertTokenEstimator(TokenEstimator):
    def __init__(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def estimate_tokens(self, text):
        return len(self.bert_tokenizer.encode(text))
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
if __name__ == "__main__":
                # from spacy.cli import download

                # download("en_core_web_lg")
                
                nlp = spacy.load("en_core_web_lg")#
                nlp.add_pipe("merge_entities")
                nlp.add_pipe("merge_noun_chunks")
                #import re
                #from spacy.tokenizer import Tokenizer
                #infix_re = re.compile(r'''(?<=\d)-(?=\d)''')

                #nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)
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
                linesen=["32.7 % of all households were made up of individuals and 15.7 % had someone living alone who was 65 years of age or older ."
                          ,"`` Greenfish '' was launched by the Electric Boat Co. , Groton , Conn. , 21 December 1945 ; sponsored by Mrs. Thomas J. Doyle ; and commissioned 7 June 1946 , Comdr. R. M. Metcalf commanding ."
                         ,"Mr. Stoll suspected the intruder was one of those precocious students who has fun breaking into computers ."
                         ,"Nobody told us ; nobody called us , '' says an official close to the case who asked not to be named .",
                         "The ` Charleston Courier , '' founded in 1803 , and  Charleston Daily News , '' founded in 1865 , merged to form the  News and Courier '' in 1873 ."]
                linesen=["Graner handcuffed him to the bars of a cell window and left him there, feet dangling off the floor, for nearly five hours.","Joseph Robinette Biden Jr. is an American politician who is the 46th and current president of the United States. A member of the Democratic Party, he served as the 47th vice president from 2009to2017 under Barack Obama and represented Delaware in the United States Senate from 1973to2009.","23.8% of all households were made up of individuals and 13.0% had someone living alone who was 65 years of age or older.","Several years later the remaining trackage at Charles City was abandoned.","The three existing plants and their land will be sold.","A Democrat, he became the youngest mayor in Pittsburgh's history in September 2006 at the age of 26.","The Greater Tokyo Area is the most populous metropolitan area in the world.","Mr. Stoll suspected the intruder was one of those precocious students who has fun breaking into computers .","The third is about sudden market losses that dry up spending and demand .","The `` Charleston Courier , '' founded in 1803 , and `` Charleston Daily News , '' founded in 1865 , merged to form the `` News and Courier '' in 1873 .","John gave Mary a book","The sun rises in the east","Rep. Fortney Stark (D.,Calif.) said '' To allow this massive level of unfettered federal borrowing without prior congressional approval would be irresponsible'', who has introduced a bill to limit the RTC 's authority to issue debt ."]

                linesen_=["He is playing tennis with his friends.",
                          "She is a student at the university.",
                          "`` Greenfish '' was launched by the Electric Boat Co. , Groton , Conn. , 21 December 1945 ; sponsored by Mrs. Thomas J. Doyle ; and commissioned 7 June 1946 , Comdr. R. M. Metcalf commanding .",
                          "Mr. Stoll suspected the intruder was one of those precocious students who has fun breaking into computers .",
                          "John gave Mary a book.",
                         "The sun rises in the east.",
                          "Rep. Fortney Stark (D.,Calif.) said '' To allow this massive level of unfettered federal borrowing without prior congressional approval would be irresponsible'', who has introduced a bill to limit the RTC 's authority to issue debt .",
                          "`` To allow this massive level of unfettered federal borrowing without prior congressional approval would be irresponsible , '' said Rep. Fortney Stark (D.,Calif.) , who has introduced a bill to limit the RTC 's authority to issue debt .",
                        "32.7 % of all households were made up of individuals and 15.7 % had someone living alone who was 65 years of age or older .","The `` Charleston Courier , '' founded in 1803 , and `` Charleston Daily News , '' founded in 1865 , merged to form the `` News and Courier '' in 1873 .",
                        "In the 1960s and 70s most of Kabul's economy depended on tourism .",
                        "They beat Milligan 1-0 , Grand View 3-0 , Webber International 1-0 and Azusa Pacific 0-0 to win the NAIA National Championships .",
                        "The three existing plants and their land will be sold .",
                        "One major difference between the two models is that the Photographic Model follows more of a step-by-step process in the development of flashbulb accounts , whereas the Comprehensive Model demonstrates an interconnected relationship between the variables .",
                        "Few people in the advertising business have raised as many hackles as Alvin A. Achenbaum .",
                        "Meanwhile , the Mason City Division continued to operate as usual .",
                        "Because of this association , St. Michael was considered to be the patron saint of colonial Maryland , and as such was honored by the river being named for St. Michael .",
                        "It deals with cases of fraud in relation to direct taxes and indirect taxes , tax credits , drug smuggling , and money laundering , cases involving United Nations trade sanctions , conflict diamonds and CITES .",
                        "The show was designed to appear as if the viewer was channel surfing through a multi-channel wasteland , happening upon spoof adverts , short sketches , and recurring show elements .",
                        "Several years later the remaining trackage at Charles City was abandoned .",
                        "He served as the first Prime Minister of Australia and became a founding justice of the High Court of Australia .",
                        "Graner handcuffed him to the bars of a cell window and left him there , feet dangling off the floor , for nearly five hours .",
                        "Sen.Mitchell, who is from maine, is a lawyer."
                        ,"President Obama said: that Mandela's life was remarkable."
                        ,"But wire transfers from a standing account -- including those bigger than $ 10,000 -- aren't reported ."
                        ,"The third is about sudden market losses that dry up spending and demand ."
                        ,"US president Donald Trump gave a speech on Wednesday.",
                        "Other signs of lens subluxation include mild conjunctival redness, vitreous humour degeneration,and an increase or decrease of anterior chamber depth .",
                        "Salomon Brothers says , `` We believe the real estate properties would trade at a discount ... after the realty unit is spun off ... .",
                        "Nobody told us ; nobody called us , '' says an official close to the case who asked not to be named ."]
##                      #line=preprocessMyDataSet(linesen,nlp)
                #text1=line
                


                # from misc import processSubjectObjectPairs2
                if linesen not in [" ", "\n", ""]:
                   #text_chunks=chunktext(500,text1)
                   #for text_chunk in text_chunks:
                   #     x=nlp(text_chunk)
                   #     for x1 in xtoken.sents:
                   cluster = []
                   #predictor=Predictor.from_path("./allenepi/coref-spanbert-large-2021.03.10.tar.gz")#("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")#("./allenepi/coref-spanbert-large-2021.03.10.tar.gz")#("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")#
                
                   #list1,list2= SubjectObjectrelation(nlp(updated_text)) 
                   for index,doc in enumerate(nlp.pipe(linesen)):
                            print("sentence: ")
                            print(doc.text+'\n')   
                            #try:
                            #     prediction = predictor.predict(document= doc.text)  # get prediction
                            #     updated_document = replace_coreferences(prediction['document'], prediction['clusters'])
                            #     updated_text = ' '.join(updated_document)
                            #except :
                            #    pass
                            updated_text=doc.text
                            # print("sentence After replace_coreferences: ")
                            # print(updated_text+'\n') 
                            #SubjectObjectrelationtest(doc)
                            print("test Triples:"+'\n')
                            # Regular expression to convert "Comdr. R. M. Metcalf" to "Comdr.R.M.Metcalf"
                            updated_text = re.sub(r'Comdr\.\s([A-Za-z])\.\s([A-Za-z])\.\s([A-Za-z]+)', r'Comdr \1 \2 \3', updated_text)
                            # Regular expression to convert "Mrs. Thomas J. Doyle" to "Mrs.Thomas J.Doyle"
                            updated_text = re.sub(r'Mrs\.\s([A-Za-z]+)\s([A-Za-z])\.\s([A-Za-z]+)', r'Mrs \1 \2 \3', updated_text)
                            list1,list2= SubjectObjectrelation(nlp(updated_text))
                            # print('\n'+"test")
                            # processSubjectObjectPairs2(doc)
                           


                   #         # with open('dataset/'+'oie.txt', 't+a') as f:
                   #         #     f.write('sent_id:'+(index+1).__str__())
                   #         #     f.write('\t')
                   #         # with open('dataset/'+'oie.txt', 'a') as f:    
                   #         #     f.write(doc.text)
                   #         #     f.write('\n')
                   #         # for indexcluster, x in enumerate(list1):
                   #         #     doc1 = nlp(x[0]+" "+x[1]+" "+x[2])
                   #         #     doc2 = nlp(doc.text)
                   #         #     ratiosim=doc1.similarity(doc2)
                   #         #     if(ratiosim>0.90):
                   #         #         print (doc1.similarity(doc2))
                   #         #     with open('dataset/'+'oie.txt', 'a') as f:
                   #         #         f.write((index+1).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
                   #         #         f.write('\n')
                   #         #     with open('dataset/'+'oie.txt', 'a') as f:
                   #         #         f.write(x[0]+' --> '+x[1]+' --> '+x[2])
                   #         #         f.write('\n')                       
                   #         #     relation=x[0]+' --> '+x[1]+' --> '+x[2]+"\n"  
                   #         #     if(len(x[0].split(' _ ')) >1 or len(x[1].split(', '))>1 or len(x[2].split(' _ '))>1):
                   #         #       for s in x[0].split(' _ '):
                   #         #         for r in x[1].split(', '):
                   #         #             for o in x[2].split(' _ '):
                   #         #                 strrela=s+' --> '+r+' --> '+o+"\n"
                   #         #                 if(strrela not in relation):
                   #         #                     with open('dataset/'+'oie.txt', 'a') as f:
                   #         #                         f.write(s+' --> '+r+' --> '+o)
                   #         #                         f.write('\n')
                   #         #                     relation+=s+' --> '+r+' --> '+o+"\n"
