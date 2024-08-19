from sqlite3 import Row
from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from graph4nlp.pytorch.data import GraphData
from stanfordcorenlp import StanfordCoreNLP
from sklearn.metrics import classification_report
import pandas as pd
import numpy

def extract_triples(text):
    # Load pre-trained word embeddings
    nlp= StanfordCoreNLP('http://localhost', port=9000, timeout=300000000)
    props_coref = {
                'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
                "tokenize.options":
                    "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                "tokenize.whitespace": False,
                'ssplit.isOneSentence': False,
                'outputFormat': 'json'
                }

    props_openie = {
    'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
    "tokenize.options":
        "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
    "tokenize.whitespace": False,
    'ssplit.isOneSentence': False,
    'outputFormat': 'json',
    "openie.triple.strict": "true"
                    }

    processor_args = [props_coref, props_openie]
    iegraph=GraphData()
    try:
        iegraph=IEBasedGraphConstruction.static_topology(text,nlp,processor_args,merge_strategy=None,edge_strategy=None)
    except :  
        iegraph={}
        #pass
    
    return iegraph

def evaluate_extraction(reference_triples, extracted_triples,args,sents,nlp,ev1="0"):
    # Calculate precision, recall, and F1 score
    precision, recall, f1_score, entity_coverage, false_positives1, false_negatives1,accuracy,accuracy2GraphNlp=[],[],[],[],[],[],[],[]
    Totalsentes,TotalText,Totalrelation1,TotalGraph4nlprelation1=[],[],[],[]
    true_positivesP1=[]
    true_positivest1=[]
    num_NottrippleGraph4nlp1=[]
    num_NottrippleGraph4nlp1=[]
    num_Correct1=[]
    numberofcorresctTripelA=[]
    similaritydocumymethod=[]
    similaritydocu4nlp=[]
    num_correct=0
    sentsindex=1
    num_NottrippleGraph4nlp=0
    numberofcorresctTripel=0
    stringx1=''
    stringy1=''
    for y_test,y_pred,y_sents in zip(reference_triples,extracted_triples,sents):
        num_correct=0
        num_NottrippleGraph4nlp=0
        stringx=''
        stringy=''
        numberofcorresctTripel=0
        stringx1=''
        stringy1=''
        for x in y_test:
            for y in y_pred:
                stringy1+=y[0]+" "+y[1]+" "+y[2]+", "
                if(x[0]!=0 and x[2]!=0):
                    stringx1+=str(x[0])+" "+str(x[1])+" "+str(x[2])+", "                
                if(x[0]==0 and x[2]==0):
                    num_NottrippleGraph4nlp+=1  
                    break
                if(len(set(x) & set(y))==3):
                    num_correct += (len(set(x) & set(y))/3)
                    numberofcorresctTripel+=(len(set(x) & set(y))/3)
                    if (y) not in true_positivesP1:
                         true_positivesP1.append(y)
                    if (x) not in true_positivest1:
                         true_positivest1.append(x)
                    break
                else:
                    stringx=x[0]+" "+x[1]+" "+x[2]
                    stringy=y[0]+" "+y[1]+" "+y[2]
                    
                    doc1 = nlp(stringy)
                    doc2 = nlp(stringx)
                    if(doc1.similarity(doc2)>=0.8):
                        num_correct +=1
                        if (y) not in true_positivesP1:
                            true_positivesP1.append(y)
                        if (x) not in true_positivest1:
                            true_positivest1.append(x)
                        break
                #elif(len(set(x) & set(y))==2):
                #    #if x[1] in y[1]:
                #        num_correct +=1
                #        if (y) not in true_positivesP1:
                #            true_positivesP1.append(y)
                #        if (x) not in true_positivest1:
                #            true_positivest1.append(x)
                #        break
                #elif(((x[0] in y[0].split()) and (x[2] in y[2].split())) or ((y[0] in x[0].split()) and (y[2] in x[2].split()))):
                #    num_correct +=1
                #    if (y) not in true_positivesP1:
                #         true_positivesP1.append(y)
                #    if (x) not in true_positivest1:
                #         true_positivest1.append(x)
                #    break             

        stringy1=stringy1.replace(' _ ',' ')
        stringx1=stringx1.replace(' _ ',' ')
        doc1my = nlp(stringy1)
        doc24nlp = nlp(stringx1)
        docsen=nlp(y_sents[0])
        similaritydocumymethod.append(docsen.similarity(doc1my))
        similaritydocu4nlp.append(docsen.similarity(doc24nlp))
        numberofcorresctTripelA.append(numberofcorresctTripel)
        Totalsentes.append(sentsindex)
        sentsindex+=1
        TotalText.append(len(y_sents))
        Totalrelation1.append(len(y_pred)) 
        TotalGraph4nlprelation1.append(len(y_test)-num_NottrippleGraph4nlp)
        num_NottrippleGraph4nlp1.append(num_NottrippleGraph4nlp)
        num_Correct1.append(num_correct)
        num_extracted = len(y_pred)
        num_relevant = len(y_test)
        if(num_extracted==0 or num_relevant==0):
            accuracy.append(0)
            accuracy2GraphNlp.append(0)
        else:
                accuracy.append(num_correct/num_extracted)
                accuracy2GraphNlp.append(num_correct/num_relevant)
        
        # Error analysis
        false_positives1.append( (set(y_pred) - set(y_test))-set(true_positivesP1))
        false_negatives1.append( (set(y_test) - set(y_pred))-set(true_positivest1)-set([(0, 0, 0)]))

        true_positives = num_correct
        false_positives =len(y_pred) - true_positives #len(false_positives1[-1])  
        false_negatives =len(y_test) - true_positives - num_NottrippleGraph4nlp #len(false_negatives1[-1])
        
        try:
            precision.append( true_positives / (true_positives + false_positives))
        except :
            precision.append(0)  

        try:
            recall.append(true_positives / (true_positives + false_negatives))
        except :
            recall.append(0)  
        
        try:
            f1_score.append( 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
        except :
            f1_score.append(0)
        
        

    # Calculate entity types coverage
        try:
            entity_types = set([triple[1] for triple in y_pred])
            entity_coverage.append( len(entity_types) / len(y_test)) 
        except :
            entity_coverage.append(0)
        
    # Create a Pandas dataframe from some data.
    
    data_1 = {'Precision': precision, 'Recall': recall,'F1 Score':f1_score,'Accuracy':accuracy,'Accuracy2GraphNlp':accuracy2GraphNlp,'Totalsentes':Totalsentes,'TotalText':TotalText,'Totalrelation1':Totalrelation1,'TotalGraph4nlprelation1':TotalGraph4nlprelation1,'num_Not_trippleGraph4nlp':num_NottrippleGraph4nlp1,'num_Correct':num_Correct1,'Entity Types Coverage':entity_coverage,'False Positives':false_positives1,'False Negatives':false_negatives1}
    df = pd.DataFrame(data_1)

    data_2={'Precision':[numpy.average(precision)],'Recall':[numpy.average(recall)],'F1 Score':[numpy.average(f1_score)],'Accuracy':[numpy.average(accuracy)],'Accuracy2GraphNlp':[numpy.average(accuracy2GraphNlp)],'Totalsentes':[numpy.sum(Totalsentes)],'TotalText':[numpy.sum(TotalText)],'Totalrelation1':[numpy.sum(Totalrelation1)],'TotalGraph4nlprelation1':[numpy.sum(TotalGraph4nlprelation1)],'num_Not_trippleGraph4nlp':[numpy.sum(num_NottrippleGraph4nlp1)],'num_Correct':[numpy.sum(num_Correct1)]}
    df2 = pd.DataFrame(data_2)
    
    data_3={'similarity docu mymethod':similaritydocumymethod,'similarity docu 4nlp':similaritydocu4nlp,'number of corresct Tripel':numberofcorresctTripelA}
    df3 = pd.DataFrame(data_3)
 
    # Create a Pandas Excel writer
    # object1 using XlsxWriter as the engine.
    writer = pd.ExcelWriter('XL_File_eval.xlsx',
                        engine='xlsxwriter')
 
    # Write a dataframe to the worksheet.
    df.to_excel(writer, sheet_name=args.pretrain_data+ev1)
    df2.to_excel(writer, sheet_name=args.pretrain_data+ev1+"1")
    df3.to_excel(writer, sheet_name=args.pretrain_data+ev1+"2")
 
    # Close the Pandas Excel writer
    # object1 and output the Excel file.
    writer.close()    

#if __name__ == "__main__":
##    precision, recall, f1_score, entity_coverage, false_positives1, false_negatives1,accuracy,accuracy2GraphNlp=[],[],[],[],[],[],[],[]
##    Totalsentes,TotalText,Totalrelation1,TotalGraph4nlprelation1=[],[],[],[]
#    reference_triples=[[('U.S. President Barack Obama','wants','lawmakers')],[('U.S. President Barack Obama','wants','lawmakers')]]
#    extracted_triples=[[('U.S. President Barack Obama','wants','lawmakers')],[('U.S. President Barack Obama','wants,weigh to','lawmakers')]]
#    sents=[['U.S. President Barack Obama wants lawmakers'],['jhjjkkhhkhkhkjhjhjkhkhk']]
##    # Calculate precision, recall, and F1 score
##    # Generating workbook and writer engine
##    num_correct=0
##    sentsindex=1
#    for y_test,y_pred,y_sents in zip(reference_triples,extracted_triples,sents):
#        num_correct=0
#        for x in y_test:
#            for y in y_pred:
#                 # Calculate true positives
#                true_positives = len(reference_triples.intersection(extracted_triples))

#    # Calculate false positives
#                false_positives = len(extracted_triples.difference(reference_triples))

#    # Calculate false negatives
#                false_negatives = len(reference_triples.difference(extracted_triples))

#    # Calculate accuracy
#                accuracy = true_positives / len(reference_triples)

#    # Calculate precision
#                precision = true_positives / (true_positives + false_positives)

#    # Calculate recall
#                recall = true_positives / (true_positives + false_negatives)

#    # Calculate F1-score
            
#                if(len(set(x) & set(y))==3):
#                    num_correct += (len(set(x) & set(y))/3)
#                    break
#                elif(len(set(x) & set(y))==2):
#                    if x[1] in y[1]:
#                        num_correct +=1
#                        break
#        Totalsentes.append(sentsindex)
#        sentsindex+=1
#        TotalText.append(len(y_sents))
#        Totalrelation1.append(len(y_pred)) 
#        TotalGraph4nlprelation1.append(len(y_test))
#        num_extracted = len(y_pred)
#        num_relevant = len(y_test)
#        accuracy.append(num_correct/num_extracted)
#        accuracy2GraphNlp.append(num_correct/num_relevant)
        
#        # Error analysis
#        false_positives1.append( set(y_pred) - set(y_test))
#        false_negatives1.append( set(y_test) - set(y_pred))

#        true_positives = num_correct
#        false_positives = len(y_pred) - true_positives
#        false_negatives = len(y_test) - true_positives



#        precision.append( true_positives / (true_positives + false_positives))
#        recall .append(true_positives / (true_positives + false_negatives))
#        f1_score.append( 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
        

#    # Calculate entity types coverage
#        entity_types = set([triple[1] for triple in y_pred])
#        entity_coverage.append( len(entity_types) / len(y_test)) 
        
#    # Create a Pandas dataframe from some data.
    
#    data_1 = {'Precision': precision, 'Recall': recall,'F1 Score':f1_score,'Entity Types Coverage':entity_coverage,'False Positives':false_positives1,'False Negatives':false_negatives1,'accuracy':accuracy,'Totalsentes':Totalsentes,'TotalText':TotalText,'Totalrelation1':Totalrelation1,'TotalGraph4nlprelation1':TotalGraph4nlprelation1}
#    #,'Totalsentes':[len(sents)],'TotalText':[len(y_sents)],'Totalrelation1':[len(y_pred)],'TotalGraph4nlprelation1':[len(y_test)]
#    df = pd.DataFrame(data_1)
 
#    # Create a Pandas Excel writer
#    # object1 using XlsxWriter as the engine.
#    writer = pd.ExcelWriter('XL_File_eval.xlsx',
#                        engine='xlsxwriter')
 
#    # Write a dataframe to the worksheet.
#    df.to_excel(writer, sheet_name='Sheet2')
 
#    # Close the Pandas Excel writer
#    # object1 and output the Excel file.
#    writer.save()



