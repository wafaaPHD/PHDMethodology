# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import spacy
import re
import wordninja
import numpy as np
from chunkipy import TextChunker, TokenEstimator
from transformers import AutoTokenizer
from tqdm import tqdm
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
    return original + ' and ' + chunk
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

def SubjectObjectrelation(tokens):
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
        
        if(((token.dep_ in SUBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.dep_ in SUBJECTS and token.tag_ not in tagsubject)) or ("ROOT" in token.dep_ and token.tag_ not in tagsubject)):
                      if (subject.lower().split(' _ ')[-1]!=token.lower_) and (token.lower_ not in subject.lower().split(' _ ')): 
                        start = token.i
                        end = token.i + 1
                        if(subjectflageMain or subjectflage):
                           subject=appendSOChunk("", token.text)
                           subjectflageMain=False
                        else:                            
                            subject=appendSOChunk(subject, token.text)
                        if token.conjuncts:
                            conjuncts = token.conjuncts             # tuple of conjuncts
                            for conj in conjuncts:
                                if(conj.tag_ not in tagsubject):
                                    spanconj = conj
                                    subject = appendChunkandtoken(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                      else:                          
                          if(subjectflageMain):
                           start = token.i
                           end = token.i + 1
                           subject=appendSOChunk("", token.text)
                           subjectflageMain=False
                           subjectindex.append((start,end))
                        
        if(token.dep_ in OBJECTS and token.tag_ not in tagsubject) or (token.pos_ in ["NOUN","PRON","PROPN"] and token.tag_ not in tagsubject and token.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject=="" or subjectflageMain):
                            start = token.i
                            end = token.i + 1
                            if(subjectflageMain):
                                subject=appendSOChunk("", token.text)
                                subjectflageMain=False
                            else:
                                subject=appendSOChunk(subject, token.text)
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
                        
                         object = appendChunk(object, token.text)
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
          for xtoken in subtree:
              #object                    
                    if(xtoken.dep_ in OBJECTS and xtoken.tag_ not in tagsubject) or (xtoken.pos_ in ["NOUN","PRON","PROPN"] and xtoken.tag_ not in tagsubject and xtoken.dep_ in OBJECTS):#or (xtoken.ent_type_ in ['PERCENT','CARDINAL'])
                        if(subject=='' or subjectflage):
                          
                            start = xtoken.i
                            end = xtoken.i + 1
                            
                            subject=appendSOChunk(subject, xtoken.text)
                            subjectflage=False
                            if xtoken.conjuncts:
                                conjuncts = xtoken.conjuncts             # tuple of conjuncts
                                for conj in conjuncts:
                                    if(conj.tag_ not in tagsubject):
                                        spanconj = conj
                                        subject = appendChunkandtoken(subject, spanconj.text)
                                        end=spanconj.i+1
           
                            subjectindex.append((start,end))
                        if (object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split(' _ ') and xtoken.text not in subject.split(' and ') and xtoken.text not in subject.split(' _ ')):
                         if(xtoken.text in subject.split(' _ ')):
                                subjectlist =subject.split(' _ ')
                                if(len(subjectlist)>0):
                                    objectindex=subjectlist.index(xtoken.text)
                                    subjectindexobject=subjectindex[objectindex]
                                    subjectindex.remove(subjectindexobject)
                                    subject=subject.replace(xtoken.text+' _ ','')
                         if(xtoken.i in relIndex):
                                relationlist =relation.replace(',','').split()
                                if(len(relationlist)>0):
                                    
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
                                    object = appendChunkandtoken(object, spanconj.text)
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
                                    subject = appendChunkandtoken(subject, spanconj.text)
                                    end=spanconj.i+1
           
                        subjectindex.append((start,end))
                        #Relation
                    # if(relation in ['is','are','am'] and xtoken.text in ['is','are','am']):
                    #         relation = ''
                    #         relIndex=[] 
              #relation      
                    if ((xtoken.dep_ in ADJECTIVES or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"] or xtoken.tag_ in tagRelation) and (xtoken.text not in relation.split() or subjectflage)):#and xtoken.ent_type_ not in ['PERCENT','CARDINAL']
                       if'prep' in xtoken.dep_ or xtoken.pos_ in ["VERB","ADP","ADV","PART","AUX"]:
                         if relation == '' or subjectflage:
                            subjectflage=False
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
                           if(xtoken.text not in relation.split(' ')):
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
                            
                          else:  
                            subjectflage=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                           
                        else:
                            subjectflage=True
                            object = ''
                            objectindex=[]
                            rtoken=[]
                           
          subject=mainsubject
          subjectindex=mainsubjectindex
          relation=mainrelation
          relIndex=mainrelationindex
          for xtoken in subtree:
                        if xtoken.dep_ in ["punct","cc"]  or "PUNCT" in xtoken.pos_ or xtoken.is_punct:
                            continue
                        if (object!=xtoken.text and subject!=xtoken.text and xtoken.text not in object.split('_') and xtoken.text not in subject.split(' _ ') and xtoken.text not in relation.split() and xtoken.text not in subject.split(' and ')):
                        
                         object = appendChunk(object, xtoken.text)
                         start=xtoken.i
                         end=xtoken.i+1
                         objectindex=(start,end)

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
                linesen=["The third is about sudden market losses that dry up spending and demand .",
                    "US president Donald Trump gave a speech on Wednesday.",
                    "Other signs of lens subluxation include mild conjunctival redness, vitreous humour degeneration,and an increase or decrease of anterior chamber depth .",
                    "Salomon Brothers says , `` We believe the real estate properties would trade at a discount ... after the realty unit is spun off ... ."
                         ,"Sen.Mitchell, who is from maine, is a lawyer.","The `` Charleston Courier , '' founded in 1803 , and `` Charleston Daily News , '' founded in 1865 , merged to form the `` News and Courier '' in 1873 .",
                    "In the 1960s and 70s most of Kabul's economy depended on tourism .",
                    "32.7 % of all households were made up of individuals and 15.7 % had someone living alone who was 65 years of age or older .",
                    "They beat Milligan 1-0 , Grand View 3-0 , Webber International 1-0 and Azusa Pacific 0-0 to win the NAIA National Championships .",
                    "The three existing plants and their land will be sold .",
                    "Mr. Stoll suspected the intruder was one of those precocious students who has fun breaking into computers .",
                         "One major difference between the two models is that the Photographic Model follows more of a step-by-step process in the development of flashbulb accounts , whereas the Comprehensive Model demonstrates an interconnected relationship between the variables .",
                         "Few people in the advertising business have raised as many hackles as Alvin A. Achenbaum .",
                         "Meanwhile , the Mason City Division continued to operate as usual .",
                         "Because of this association , St. Michael was considered to be the patron saint of colonial Maryland , and as such was honored by the river being named for him .",
                         "It deals with cases of fraud in relation to direct taxes and indirect taxes , tax credits , drug smuggling , and money laundering , cases involving United Nations trade sanctions , conflict diamonds and CITES .",
                         "The show was designed to appear as if the viewer was channel surfing through a multi-channel wasteland , happening upon spoof adverts , short sketches , and recurring show elements .",
                         "Several years later the remaining trackage at Charles City was abandoned .",
                         "They beat Milligan 1-0 , Grand View 3-0 , Webber International 1-0 and Azusa Pacific 0-0 to win the NAIA National Championships .",
                         "The three existing plants and their land will be sold .",
                         "He served as the first Prime Minister of Australia and became a founding justice of the High Court of Australia .",
                         "Graner handcuffed him to the bars of a cell window and left him there , feet dangling off the floor , for nearly five hours .",
                         "Nobody told us ; nobody called us , '' says an official close to the case who asked not to be named ."]
                      #line=preprocessMyDataSet(linesen,nlp)
                #text1=line
                from misc import processSubjectObjectPairs2
                if linesen not in [" ", "\n", ""]:
                   #text_chunks=chunktext(500,text1)
                   #for text_chunk in text_chunks:
                   #     x=nlp(text_chunk)
                   #     for x1 in xtoken.sents:
                   
                   for index,doc in enumerate(nlp.pipe(linesen)):
                            print(doc.text+'\n')     
                            #SubjectObjectrelationtest(doc)
                            print('\n'+"test")
                            list1,list2= SubjectObjectrelation(doc)
                            # print('\n'+"test")
                            # processSubjectObjectPairs2(doc)
                           


                            # with open('dataset/'+'oie.txt', 't+a') as f:
                            #     f.write('sent_id:'+(index+1).__str__())
                            #     f.write('\t')
                            # with open('dataset/'+'oie.txt', 'a') as f:    
                            #     f.write(doc.text)
                            #     f.write('\n')
                            # for indexcluster, x in enumerate(list1):
                            #     doc1 = nlp(x[0]+" "+x[1]+" "+x[2])
                            #     doc2 = nlp(doc.text)
                            #     ratiosim=doc1.similarity(doc2)
                            #     if(ratiosim>0.90):
                            #         print (doc1.similarity(doc2))
                            #     with open('dataset/'+'oie.txt', 'a') as f:
                            #         f.write((index+1).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
                            #         f.write('\n')
                            #     with open('dataset/'+'oie.txt', 'a') as f:
                            #         f.write(x[0]+' --> '+x[1]+' --> '+x[2])
                            #         f.write('\n')                       
                            #     relation=x[0]+' --> '+x[1]+' --> '+x[2]+"\n"  
                            #     if(len(x[0].split(' _ ')) >1 or len(x[1].split(', '))>1 or len(x[2].split(' _ '))>1):
                            #       for s in x[0].split(' _ '):
                            #         for r in x[1].split(', '):
                            #             for o in x[2].split(' _ '):
                            #                 strrela=s+' --> '+r+' --> '+o+"\n"
                            #                 if(strrela not in relation):
                            #                     with open('dataset/'+'oie.txt', 'a') as f:
                            #                         f.write(s+' --> '+r+' --> '+o)
                            #                         f.write('\n')
                            #                     relation+=s+' --> '+r+' --> '+o+"\n"
