# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:46:13 2019

@author: WT
"""
import os
import pickle
import re
from itertools import permutations

#def load_pickle(filename):
#    completeName = os.path.join("./data/",\
#                                filename)
#    with open(completeName, 'rb') as pkl_file:
#        data = pickle.load(pkl_file)
#    return data

#def save_as_pickle(filename, data):
#    completeName = os.path.join("./data/",\
#                                filename)
#    with open(completeName, 'wb') as output:
#        pickle.dump(data, output)


def load_pickle(filename,args):
    completeName = os.path.join(args.PathDataset,\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data,args):
    completeName = os.path.join(args.PathDataset,\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def load_pickleResult(filename,args):
    completeName = os.path.join(args.ResultPathDataset,\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickleResult(filename, data,args):
    completeName = os.path.join(args.ResultPathDataset,\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def get_subject_objects(sent_):
    ### get subject, object entities by dependency tree parsing
    sent_ = next(sent_.sents)
    root = sent_.root
    subject = None; objs = []; pairs = []
    for child in root.children:
        #print(child.dep_)
        if child.dep_ in ["nsubj", "nsubjpass"]:
            if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                subject = child; #print('Subject: ', child)
        elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
            objs.append(child); #print('Object ', child)
    if (subject is not None) and (len(objs) > 0):
        for a, b in permutations([subject] + [obj for obj in objs], 2):
            a_ = [w for w in a.subtree]
            b_ = [w for w in b.subtree]
            pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))
            
    return pairs
def printToken(token):
    print(token, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod","ccomp","advcl","relcl"]
    return any(subs in token.dep_ for subs in deps)
def isConstructionCandidate(token):
    deps = ["compound",  "pobj","dobj","appos","npadvmod","nsubj","advmod"]#"prep","mod","conj",
    return any(subs in token.dep_ for subs in deps)
def processSubjectObjectPairs2(tokens):
    subject = ''
    object = ''
    relation = ''
    objectConstruction = ''
    trible=[]
    rtoken=[]
    subjectindex=[]
    objectindex=[]
    relIndex=[]
    
    x = [token.text for token in tokens]
    rtoken.append(x)
    for token in tokens:
        #printToken(token)       
        if "punct" in token.dep_:
            continue        
        #Relation
        if isRelationCandidate(token):
            if relation == '' and subject != '':
             relation = appendChunk(relation,token.text)
             relIndex=[token.i]
            elif relation!=appendChunk('',token.text) and subject != '':
             relation = appendChunk(relation,token.text)
             relIndex.append(token.i)
        elif 'prep' in token.dep_ or token.pos_ in ["VERB","ADP","ADV","PART"]:
            if relation == '':
                relation = appendChunk(token.head.text,token.text)
                relIndex=[token.head.i,token.i]
                #relation = appendChunk('', token)
                #relIndex=[token.i]
            else:
               relation = appendChunk(relation+', '+token.head.text,token.text) 
               relIndex.append(token.head.i)
               relIndex.append(token.i) 
               #relation = appendChunk(relation,token.text) 
               #relIndex.append(token.i)
        

        #subject
        if token.ent_type_ !='':
         if token.dep_=='ROOT' and token.head.dep_=='ROOT':
            subject=''
            subjectindex=[]
            subject = appendChunk(subject,token.text)
            if(len(subjectindex)==0):
                subjectindex=(token.i,token.i+1)
            else:
                subjectindex=(token.i,token.i+1)
         elif "subj" in token.dep_:
            subject=''
            subjectindex=[]
            subject = appendChunk(subject,token.text)
            if(len(subjectindex)==0):
                subjectindex=(token.i,token.i+1)
            else:
                subjectindex=(token.i,token.i+1)
         elif isConstructionCandidate(token):
           if token.dep_!='pobj':
            if 'nsubj' in token.head.dep_ and subject !='':
                subject = appendChunk(token.head.text,token.text)
                if(len(subjectindex)==0):
                 subjectindex=(token.head.i,token.i+1)
                else:
                    subjectindex=(token.head.i,token.i+1)
                           
            elif 'ROOT' in token.head.dep_ and subject =='':
                if(token.head.i <token.i):
                 subject = appendChunk(token.head.text,token.text)
                 if(len(subjectindex)==0):
                  subjectindex=(token.head.i,token.i+1)
                 else:
                  subjectindex=(token.head.i,token.i+1)
                else:
                    subject = appendChunk(token.text,token.head.text)
                    if(len(subjectindex)==0):
                     subjectindex=(token.i,token.head.i+1)
                    else:
                     subjectindex=(token.i,token.head.i+1)
            if relation == '':
             if isRelationCandidate(token.head):
              if relation == '':
               relation = appendChunk(relation, token.head.text)
               relIndex=[token.head.i]
              else:
               relation = appendChunk(relation, token.head.text)
               relIndex.append(token.head.i)
             elif isRelationCandidate(token.head.head):
              if relation == '':
               relation = appendChunk(relation, token.head.head.text)
               relIndex=[token.head.head.i]
              else:
               relation = appendChunk(relation, token.head.head.text)
               relIndex.append(token.head.head.i)
            
        elif 'nsubjpass' in token.dep_ and token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"]:
            subject=''
            subjectindex=[]
            subject = appendChunk(subject,token.text)
            if(len(subjectindex)==0):
                 subjectindex=(token.i,token.i+1)
            else:
                 subjectindex=(token.i,token.i+1)
            
        elif 'nsubj'in token.dep_ and 'ROOT' in token.head.dep_ and token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"]: 
            subject=''
            subjectindex=[]
            subject = appendChunk(subject,token.text)
            if(len(subjectindex)==0):
                 subjectindex=(token.i,token.i+1)
            else:
                 subjectindex=(token.i,token.i+1)
        elif 'nsubj' in token.dep_ and 'NN' in token.tag_ and subject=='': 
            subject=''
            subjectindex=[]
            subject = appendChunk(subject,token.text)
            if(len(subjectindex)==0):
                 subjectindex=(token.i,token.i+1)
            else:
                 subjectindex=(token.i,token.i+1)            
#object
        if token.ent_type_ !='':
         if "obj" in token.dep_:
            object=""
            objectindex=[]
            object = appendChunk(object,token.text)
            if(len(objectindex)==0):
                 objectindex=(token.i,token.i+1)
            else:
                 objectindex=(token.i,token.i+1)
         elif isConstructionCandidate(token):
           if token.tag_ =='NNP':
              object = appendChunk('',token.text)
              if(len(objectindex)==0):
                  objectindex=(token.i,token.i+1)
              else:
                  objectindex=(token.i,token.i+1)
           if 'pobj' in token.dep_  or 'dobj' in token.dep_ or 'pobj' in token.head.dep_:
            object = appendChunk('',token.text)
            if(len(objectindex)==0):
                  objectindex=(token.i,token.i+1)
            else:
                  objectindex=(token.i,token.i+1)
                 ##object = appendChunk(objectConstruction, object)
            if relation == '':
             if isRelationCandidate(token.head):
              if relation == '':
               relation = appendChunk(relation, token.head.text)
               relIndex=[token.head.i]
              else:
               relation = appendChunk(relation, token.head.text)
               relIndex.append(token.head.i)
             elif isRelationCandidate(token.head.head):
              if relation == '':
               relation = appendChunk(relation, token.head.head.text)
               relIndex=[token.head.head.i]
              else:
               relation = appendChunk(relation, token.head.head.text)
               relIndex.append(token.head.head.i)
            #elif objectConstruction:
            #    objectConstruction = appendChunk(objectConstruction, token)
         if(len(objectindex)!=0):
             start=objectindex[0]
             end=objectindex[1]
             if token.conjuncts:
                       conjuncts = token.conjuncts             # tuple of conjuncts
                       for conj in conjuncts:
                          if(conj.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"]):
                           spanconj = conj
                           object = appendChunk(object, spanconj.text)
                           end=spanconj.i+1
             objectindex=(start,end)
        elif ('pobj' in token.dep_  or 'dobj' in token.dep_) and token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"]:
                 object = appendChunk('',token.text)
                 if(len(objectindex)==0):
                  objectindex=(token.i,token.i+1)
                 else:
                  objectindex=(token.i,token.i+1)

        elif 'appos' in token.dep_ and token.tag_ !='CC':
            object=""
            objectindex=[]
            object = appendChunk(object,token.text)
            if(len(objectindex)==0):
                  objectindex=(token.i,token.i+1)
            else:
                  objectindex=(token.i,token.i+1)
        			
    
        elif 'nsubj' in token.dep_ and token.ent_type_ =='' and subject !='' and object =='' and  token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"] and token.head.dep_ !='ROOT':
            object = appendChunk(object,token.text)
            if(len(objectindex)==0):
                  objectindex=(token.i,token.i+1)
            else:
                  objectindex=(token.i,token.i+1)
        elif ('pobj' in token.dep_  or 'dobj' in token.dep_) and token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"]:
                 object = appendChunk('',token.text)
                 if(len(objectindex)==0):
                  objectindex=(token.i,token.i+1)
                 else:
                  objectindex=(token.i,token.i+1)

        elif isConstructionCandidate(token):
            if 'nsubj' in token.head.dep_ and subject =='' and token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB"]:
                subject = appendChunk(token.head.text,token.text)
                if(len(subjectindex)==0):
                  subjectindex=(token.head.i,token.i+1)
                else:
                  subjectindex=(token.head.i,token.i+1)
                
            elif 'ROOT' in token.head.dep_ and subject =='' and token.tag_ not in ["WP","VBD","IN","WDT","WP$","WRB","RB"] and token.dep_	!='advmod':
                subject = appendChunk(token.head.text,token.text)
                if(len(subjectindex)==0):
                  subjectindex=(token.head.i,token.i+1)
                else:
                  subjectindex=(token.head.i,token.i+1)
            elif token.head.dep_ ==	'dobj':
                object = appendChunk(token.head.text,token.text)
                if(token.head.i==token.i+1):
                     objectindex=(token.head.i,token.i+2)
                elif(len(objectindex)==0):                   
                  objectindex=(token.head.i,token.i+1)
                else:
                  objectindex=(token.head.i,token.i+1)                
                  
            elif objectConstruction:
                objectConstruction = appendChunk(objectConstruction,token.text)
            if relation == '' and subject.strip() !='':
             if isRelationCandidate(token.head):
              if relation == '':
               relation = appendChunk(relation, token.head.text)
               relIndex=[token.head.i]
              else:
               relation = appendChunk(relation, token.head.text)
               relIndex.append(token.head.i)
             elif isRelationCandidate(token.head.head):
              if relation == '':
               relation = appendChunk(relation, token.head.head.text)
               relIndex=[token.head.head.i]
              else:
               relation = appendChunk(relation, token.head.head.text)
               relIndex.append(token.head.head.i)
        
       

        if(subject.strip() !='' and relation.strip() !='' and object.strip() !=''):
            if subject == object:
             object = ''
             objectindex=[]
             objectConstruction = ''
             continue
            else :
             trible.append((subject.strip(), relation.strip(), object.strip(),subjectindex,objectindex,relIndex))
             print (subject.strip(), ",", relation.strip(), ",", object.strip())
             object = ''
             objectindex=[]
             relation = ''
             objectConstruction = ''
            
            
    
    return trible,rtoken