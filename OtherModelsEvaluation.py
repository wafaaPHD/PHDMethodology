import pandas as pd
from SubjectObjectrelation import SubjectObjectrelation
from rouge_score import rouge_scorer
import spacy 
def evaluate_extractionOtherModel():
    nlp = spacy.load("en_core_web_lg")       
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    similaritydocuclausie_benchie_form=[]
    similaritydocucompactie_benchie_form=[]
    similaritydocuimojie_benchie_form=[]
    similaritydocum2oie_benchie_form=[]
    similaritydocuminie_benchie_form=[]
    similaritydocuopenie6_benchie_form=[]
    similaritydocureverb_benchie_form=[]
    similaritydocuroi_t_explicit=[]
    similaritydocunaive_oie_explicit=[]
    stringphath="D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie-main\\benchie-main\\data\\oie_systems_explicit_extractions\\"

    with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\cnn.txt", "r", encoding="utf8") as f:
    #with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire57_sentences.txt", "r", encoding="utf8") as f:
    
            text = f.readlines()
    
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\clausie_benchie_form.txt', "r", encoding="utf8") as f:
    #        clausie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\compactie_benchie_form.txt', "r", encoding="utf8") as f:
    #        compactie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\imojie_benchie_form.txt', "r", encoding="utf8") as f:
    #        imojie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\m2oie_benchie_form.txt', "r", encoding="utf8") as f:
    #        m2oie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\minie_benchie_form.txt', "r", encoding="utf8") as f:
    #        minie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\openie6_benchie_form.txt', "r", encoding="utf8") as f:
    #        openie6_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie\\reverb_benchie_form.txt', "r", encoding="utf8") as f:
    #        reverb_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\clausie_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        clausie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\compactie_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        compactie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\imojie_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        imojie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\m2oie_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        m2oie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\minie_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        minie_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\openie6_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        openie6_benchie_form = f.readlines()
    #with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\wire\\reverb_wire_benchie_form.txt', "r", encoding="utf8") as f:
    #        reverb_benchie_form = f.readlines()
    with open(stringphath+'clausie_explicit.txt', "r", encoding="utf8") as f:
            clausie_benchie_form = f.readlines()
    with open(stringphath+'graphene_explicit.txt', "r", encoding="utf8") as f:
            compactie_benchie_form = f.readlines()
    with open(stringphath+'stanford_explicit.txt', "r", encoding="utf8") as f:
            imojie_benchie_form = f.readlines()
    with open(stringphath+'m2oie_en_explicit.txt', "r", encoding="utf8") as f:
            m2oie_benchie_form = f.readlines()
    with open(stringphath+'minie_explicit.txt', "r", encoding="utf8") as f:
            minie_benchie_form = f.readlines()
    with open(stringphath+'openie6_explicit.txt', "r", encoding="utf8") as f:
            openie6_benchie_form = f.readlines()
    with open(stringphath+'roi_n_explicit.txt', "r", encoding="utf8") as f:
            reverb_benchie_form = f.readlines()
    with open(stringphath+'roi_t_explicit.txt', "r", encoding="utf8") as f:
            roi_t_explicit = f.readlines()
    with open(stringphath+'naive_oie_explicit.txt', "r", encoding="utf8") as f:
            naive_oie_explicit = f.readlines()
    
    for idx,x in enumerate(text):
        stringmodel=''
        for y in clausie_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocuclausie_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in compactie_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocucompactie_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in imojie_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocuimojie_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in m2oie_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocum2oie_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in minie_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocuminie_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in openie6_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocuopenie6_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in reverb_benchie_form:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocureverb_benchie_form.append(docsen.similarity(doc1model))

        stringmodel=''
        for y in roi_t_explicit:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocuroi_t_explicit.append(docsen.similarity(doc1model))
        
        stringmodel=''
        for y in naive_oie_explicit:
            triplesLines=y.replace('\n','').split('\t')
            if((idx+1)==int(triplesLines[0])):
                stringmodel+=triplesLines[1]+" "+triplesLines[2]+" "+triplesLines[3]+" "
            elif((idx+1)>int(triplesLines[0])):
                continue
            else:
                break
        doc1model=nlp(stringmodel)
        docsen=nlp(x.replace('\n',''))
        similaritydocunaive_oie_explicit.append(docsen.similarity(doc1model))

        

    # Create a Pandas dataframe from some data.
    
    data_3={'similarity docu clausie_benchie_form':similaritydocuclausie_benchie_form,'similarity docu compactie_benchie_form':similaritydocucompactie_benchie_form,
            'similarity docu imojie_benchie_form':similaritydocuimojie_benchie_form,'similarity docu openie6_benchie_form':similaritydocuopenie6_benchie_form,
            'similarity docu m2oie_benchie_form':similaritydocum2oie_benchie_form,'similarity docu reverb_benchie_form':similaritydocureverb_benchie_form,
            'similarity docu minie_benchie_form':similaritydocuminie_benchie_form,'similarity docu roi_t_explicit':similaritydocuroi_t_explicit,'similarity docu naive_oie_explicit':similaritydocunaive_oie_explicit}
    df3 = pd.DataFrame(data_3)
 
    # Create a Pandas Excel writer
    # object1 using XlsxWriter as the engine.
    writer = pd.ExcelWriter("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\XL_File_eval_1.xlsx",
                        engine='xlsxwriter')
 
    # Write a dataframe to the worksheet.
    df3.to_excel(writer, sheet_name="Models_benchie")
 
    # Close the Pandas Excel writer
    # object1 and output the Excel file.
    writer.close()
def evaluate_extractionOtherModelOtherDataset():
    nlp = spacy.load("en_core_web_lg")       
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    similaritydocuclausie_benchie_form=[]
    with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\benchie_gold_annotations_en_org.txt", "r", encoding="utf8") as f:
            text = f.readlines()   
    sent=''
    stringmodel=''
    for idx,x in enumerate(text):      
        triplesLines=x.replace('\n','').split('\t')
        extractline=x.split(' --> ')
        if(triplesLines[0].split(':')[0]=='sent_id'):
           if(sent!=''):
            doc1model=nlp(stringmodel)
            docsen=nlp(sent.replace('\n',''))
            similaritydocuclausie_benchie_form.append(docsen.similarity(doc1model))
            stringmodel=''
           sent=triplesLines[1]
        if (len(extractline)>1):            
            stringmodel+=extractline[0]+" "+extractline[1]+" "+extractline[2].replace('\n','')+" "
        
    # Create a Pandas dataframe from some data.
    
    data_3={'similarity docu clausie_benchie_form':similaritydocuclausie_benchie_form}
    df3 = pd.DataFrame(data_3)
 
    # Create a Pandas Excel writer
    # object1 using XlsxWriter as the engine.
    writer = pd.ExcelWriter("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\benchie_fl-main\\benchie_fl-main\\extractions\\XL_File_eval_carb.xlsx",
                        engine='xlsxwriter')
 
    # Write a dataframe to the worksheet.
    df3.to_excel(writer, sheet_name="Models_benchie")
 
    # Close the Pandas Excel writer
    # object1 and output the Excel file.
    writer.close()
def evaluate_rouge(triples, sentence):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = {}
    # Convert the sentence and triples into strings for ROUGE evaluation
    sentence_str = sentence
    triples_str = [''.join(elem) for triple in triples for elem in triple if isinstance(elem, str)]
    triples_str = [''.join(elem) for triple in triples for elem in triple if isinstance(elem, str)]
    triple_str1=''
    for triple_str in triples_str:
        triple_str1+=triple_str+' '
    scores = scorer.score(sentence_str, triple_str1)
    rouge_scores[triple_str1] = scores 
            
    #for triple_str in triples_str:
    #        scores = scorer.score(sentence_str, triple_str)
    #        rouge_scores[triple_str] = scores
    return rouge_scores
# Example sentences with expected triples

sentences = [
    ("John gave Mary a book.", [("John", "gave", "Mary"), ("John", "gave", "a book")]),
    ("She is a student at the university.", [("She", "is", "a student"), ("She", "is", "at the university")]),
    ("The sun rises in the east.", [("The sun", "rises", "the east")]),
    ("He is playing tennis with his friends.", [("He", "is playing", "tennis"), ("He", "is playing", "with his friends")])
]
nlp = spacy.load("en_core_web_lg")#
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")
# Evaluate each sentence
for sentence, ground_truth in sentences:
    print(f"Evaluating sentence: {sentence}")
    sents_doc = nlp(sentence)
    # Extract triples from the sentence
    extracted_triples,tokensarr=SubjectObjectrelation(sents_doc)
    
    # Evaluate using ROUGE
    rouge_scores = evaluate_rouge(extracted_triples, sentence)
    
    # Print ROUGE scores
    print("ROUGE Scores:")
    print(f"ROUGE: {rouge_scores}")    
    print("\n")
#if __name__ == "__main__":
#     #evaluate_extractionOtherModelOtherDataset()
#   #evaluate_extractionOtherModel()

