import spacy
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel
import dgl
from dgl.nn import GraphConv

# 1. Parse the sentence to get dependencies
nlp = spacy.load("en_core_web_lg")#
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")
doc = nlp("US president Donald Trump gave a speech on Wednesday.")

# 2. Extract nodes (words) and edges (dependencies)
SUBJECTS = ["subj","nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]#
OBJECTS = ["dobj", "dative", "attr", "oprd","pobj","appos","nummod","compound","npadvmod","advmod","mod"]
tagsubject=["WP","VBD","IN","WDT","WP$","WRB","VBD","VB","VBG","VBN","VBP","VBZ","DT","RB","JJ","JJR","JJS","RBR","RBS","RP"]
allwords=[token.text for token in doc]
nodesSubject = [token.text for token in doc if token.dep_ in SUBJECTS and token.tag_ not in tagsubject]
nodesObject = [token.text for token in doc if token.dep_ in OBJECTS and token.tag_ not in tagsubject]

# Initialize a list to store pairs of tokens
edges = []
# Check if token.i and token1.i are found in the list of indices
for token in doc:
    if token.text in nodesSubject:
        for token1 in doc:
            if token1.text in nodesObject and token.i != token1.i:
                edges.append((token.i, token1.i))
#edges = [(token.head.i, token.i) for token in doc if token.dep_ in SUBJECTS and token.tag_ not in tagsubject]
#edges = [(token.head.i, token.i) for token in doc if token.dep_ != "ROOT"]
dep_tags = [token.dep_ for token in doc]  # Get POS tags for each token
pos_tags = [token.pos_ for token in doc]
tag_tags = [token.tag_ for token in doc]
ent_iob_tags = [token.ent_iob_ for token in doc] 
ent_type_tags = [token.ent_type_ for token in doc] 

# Define a mapping of POS tags to numeric values
pos_to_idx = {pos: idx for idx, pos in enumerate(set(pos_tags))}
pos_encoded = torch.tensor([pos_to_idx[tag] for tag in pos_tags], dtype=torch.long)

# Convert POS tags to embeddings using an embedding layer
pos_embedding_dim = 8  # Set POS embedding dimension
pos_embedding = torch.nn.Embedding(len(pos_to_idx), pos_embedding_dim)
pos_embeddings = pos_embedding(pos_encoded)
#-----------------------------------------
# Define a mapping of dep tags to numeric values
dep_to_idx = {dep: idx for idx, dep in enumerate(set(dep_tags))}
dep_encoded = torch.tensor([dep_to_idx[tag] for tag in dep_tags], dtype=torch.long)

# Convert dep tags to embeddings using an embedding layer
dep_embedding_dim = 8  # Set dep embedding dimension
dep_embedding = torch.nn.Embedding(len(dep_to_idx), dep_embedding_dim)
dep_embeddings = dep_embedding(dep_encoded)
#-----------------------------------------
# Define a mapping of tag tags to numeric values
tag_to_idx = {tag: idx for idx, tag in enumerate(set(tag_tags))}
tag_encoded = torch.tensor([tag_to_idx[tag] for tag in tag_tags], dtype=torch.long)

# Convert tag tags to embeddings using an embedding layer
tag_embedding_dim = 8  # Set tag embedding dimension
tag_embedding = torch.nn.Embedding(len(tag_to_idx), tag_embedding_dim)
tag_embeddings = tag_embedding(tag_encoded)
#-----------------------------------------
# Define a mapping of ent_iob tags to numeric values
ent_iob_to_idx = {ent_iob: idx for idx, ent_iob in enumerate(set(ent_iob_tags))}
ent_iob_encoded = torch.tensor([ent_iob_to_idx[tag] for tag in ent_iob_tags], dtype=torch.long)

# Convert ent_iob tags to embeddings using an embedding layer
ent_iob_embedding_dim = 8  # Set ent_iob embedding dimension
ent_iob_embedding = torch.nn.Embedding(len(ent_iob_to_idx), ent_iob_embedding_dim)
ent_iob_embeddings = ent_iob_embedding(ent_iob_encoded)
#-----------------------------------------
# Define a mapping of ent_type tags to numeric values
ent_type_to_idx = {ent_type: idx for idx, ent_type in enumerate(set(ent_type_tags))}
ent_type_encoded = torch.tensor([ent_type_to_idx[tag] for tag in ent_type_tags], dtype=torch.long)

# Convert ent_type tags to embeddings using an embedding layer
ent_type_embedding_dim = 8  # Set ent_type embedding dimension
ent_type_embedding = torch.nn.Embedding(len(ent_type_to_idx), ent_type_embedding_dim)
ent_type_embeddings = ent_type_embedding(ent_type_encoded)

#-----------------------------------------
# Define a mapping of node tags to numeric values
node_to_idx = {node: idx for idx, node in enumerate(set(nodesSubject))}
node_encoded = torch.tensor([node_to_idx[tag] for tag in nodesSubject], dtype=torch.long)

# Convert node tags to embeddings using an embedding layer
node_embedding_dim = 8  # Set node embedding dimension
node_embedding = torch.nn.Embedding(len(node_to_idx), node_embedding_dim)
node_embeddings = node_embedding(node_encoded)

# Define a mapping of node tags to numeric values
nodeO_to_idx = {node: idx for idx, node in enumerate(set(nodesObject))}
nodeO_encoded = torch.tensor([nodeO_to_idx[tag] for tag in nodesObject], dtype=torch.long)

# Convert node tags to embeddings using an embedding layer
nodeO_embedding_dim = 8  # Set node embedding dimension
nodeO_embedding = torch.nn.Embedding(len(nodeO_to_idx), nodeO_embedding_dim)
nodeObject_embeddings = nodeO_embedding(nodeO_encoded)

# 3. Get embeddings for each node using BERT
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize with `is_split_into_words=True` to match tokenization to original words
inputs = tokenizer(allwords, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
outputs = model(**inputs)

# Aggregate subword embeddings to get a single embedding per word
word_embeddings = []
for i in range(len(allwords)):
    word_indices = [j for j, word_id in enumerate(inputs.word_ids()) if word_id == i]
    word_embedding = outputs.last_hidden_state[0, word_indices].mean(dim=0)
    word_embeddings.append(word_embedding)

# Convert to tensor with shape [num_nodes, bert_embedding_dim]
node_embeddings = torch.stack(word_embeddings)

# Concatenate BERT embeddings with POS embeddings
node_features = torch.cat((node_embeddings,dep_embeddings, pos_embeddings,tag_embeddings,ent_iob_embeddings,ent_type_embeddings), dim=1)
# 4. Create a DGL graph
# Convert edges to a DGL graph format
src, dst = zip(*edges)
g = dgl.graph((src, dst),num_nodes=len(allwords))
g.ndata['x'] = node_features  # Assign embeddings as node features
# 5. Define the GNN model with DGL's GraphConv
class GNNModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = F.relu(self.conv2(g, x))
        x = self.fc(x)
        return x

# Initialize model with input, hidden, and output dimensions
in_feats = node_features.size(1)


# Instantiate your GNN model with appropriate input features, hidden size, and number of classes
model = GNNModel(in_feats=in_feats, hidden_size=64, num_classes=3)

# Assuming you have the DGL graph `g` and node features `features` ready for input
out = model(g, g.ndata['x'])

# Prediction output (subject, relation, object)
#predicted_classes = out.argmax(dim=1)
# Assuming 'out' is the output tensor from your model with shape [batch_size, num_nodes, num_classes]
out_shape = out.size()

# Check the number of dimensions in the output tensor and unpack accordingly
if len(out_shape) == 3:  # If the tensor has 3 dimensions
    batch_size, num_nodes, num_classes = out_shape
    reshaped_out = out.view(batch_size * num_nodes, num_classes)
elif len(out_shape) == 2:  # If the tensor has 2 dimensions
    batch_size, num_classes = out_shape
    reshaped_out = out.view(batch_size, num_classes)
else:
    raise ValueError("Unexpected number of dimensions in the output tensor")

## Reshape the output tensor to [batch_size * num_nodes, num_classes]
#reshaped_out = out.view(batch_size * num_nodes, num_classes)

# Further processing of the reshaped output tensor
# For example, if you want to get the predicted classes for each node in each graph
predicted_classes = reshaped_out.argmax(dim=1)


# Define labels for each class
labels = ["subject", "relation", "object"]

# Print the prediction for each node
for i, prediction in enumerate(predicted_classes):
    print(f"Node {i} ({allwords[i]}): {labels[prediction]}")


# import os
# from tqdm import tqdm
# import re
# import pickle

# if __name__ == "__main__":
#     #-------------------------------------neural_oie DATASET
#     with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\NeuralOpenIE\\neural_oie.sent',mode="r",encoding="utf8") as f3:
#             Lines=f3.readlines()
#     with open('D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\NeuralOpenIE\\neural_oie.triple',mode="r",encoding="utf8") as f3:
#             Lines2 = f3.readlines()
#     index=0
#     for line,line2 in zip(Lines, Lines2):
#                 t=line.rstrip().replace('\n','')
                
#                 index+=1
#                 indexcluster=0
#                 relation=''
#                 if(index>1):
#                     with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\NeuralOpenIE\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#                                f3.write('\n')
#                 with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\NeuralOpenIE\\benchie_gold_annotations_en_org.txt", 'a') as f2:
#                              f2.write('sent_id:'+(index).__str__())
#                              f2.write('\t')
#                 with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\NeuralOpenIE\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#                              f3.write(t)
#                              f3.write('\n')
#                              f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#                              f3.write('\n')                 
                
#                 t2=line2.rstrip().split('\t')
#                 t3=t2[0].rstrip().split(' </arg1> <rel> ')
#                 t4=t3[1].rstrip().split(' </rel> <arg2> ')
#                 if(t4[0].replace(' </arg2>','')=='' or t3[0].replace('<arg1> ','')=='' or t4[0]==''):
#                         continue
                
#                 with open("D:\\Private\\PHD\\20-7-2024ResultPDF\\20-7-2024MyWorkProposal\\UpdateBertRelation\\dataset\\NeuralOpenIE\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#                                #f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#                                #f3.write('\n')
#                                strrela=t3[0].replace('<arg1> ','')+' --> '+t4[0]+' --> '+t4[1].replace(' </arg2>','')+"\n"
#                                if(strrela not in relation):
#                                 relation+=t3[0].replace('<arg1> ','')+' --> '+t4[0]+' --> '+t4[1].replace(' </arg2>','')+"\n"                           
#                                 f3.write(t3[0].replace('<arg1> ','')+' --> '+t4[0]+' --> '+t4[1].replace(' </arg2>',''))
#                                 f3.write('\n')
#                    # index1_0=sent.split().index(arg0.split()[0])
#                    # index1_1=index1_0+len(arg0.split())
#                    # index2_0=sent.split().index(arg1.split()[0])
#                    # index2_1=index2_0+len(arg1.split())
#                    # rangeindex=list(range(sent.split().index(pred.split()[0]),sent.split().index(pred.split()[0])+len(pred.split())))
#                    # reindedx=[w for w in rangeindex]
#                    # r = ([w for w in sent.split()],(index1_0,index1_1),(index2_0,index2_1),pred,reindedx)
#                    # D.append((r, arg0, arg1))
#                 indexcluster+=1
                
#     #completeName = os.path.join("D:\\Private\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\NeuralOpenIE",\
#     #                             "DORg.pkl")
#     #with open(completeName, 'wb') as output:
#     #     pickle.dump(D, output)  
         
      
#     #-------------------------------------Re-OIE2016.json DATASET
#     # with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\Re-OIE2016.json", "r") as f1:#, encoding="utf8"
#     #         text2 = f1.readlines()    
#     # index=0
#     # sflage=False
#     # pflage=False
#     # a0flage=False
#     # a1flage=False
#     # a2flage=False
#     # a3flage=False
#     # D=[]
#     # for line in tqdm(text2, total=len(text2)):
        
#     #     if(line.find('.": [')>=0 or line.find('. \'\'": [\n')>0):
#     #         index+=1
#     #         indexcluster=0
#     #         sent=line.replace(': [\n','').replace('"','').replace('  ','')
#     #         if(index>1):
#     #             with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                            f3.write('\n')
#     #         with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\benchie_gold_annotations_en_org.txt", 'a') as f2:
#     #                           f2.write('sent_id:'+(index).__str__())
#     #                           f2.write('\t')
#     #         with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                           f3.write(sent)
#     #                           f3.write('\n')                              
#     #         sflage=True
#     #         pflage=False
#     #         a0flage=False 
#     #         a1flage=False 
#     #         a2flage=False 
#     #         a3flage=False       
#     #     if(line.find('\"arg0\": ')>=0):
#     #         arg0=line.replace('\"arg0\": ','')
#     #         arg0=arg0.replace('      "','')
#     #         arg0=arg0.replace(',\n','').replace('"','')
#     #         a0flage=True
#     #         if(arg0==''):
#     #             pflage=False
#     #             a0flage=False
#     #             a1flage=False
#     #             a2flage=False
#     #             a3flage=False

#     #     if(line.find('\"pred\": ')>=0):
#     #         pred=line.replace('\"pred\": ','')
#     #         pred=pred.replace('      "','')
#     #         pred=pred.replace(',\n','').replace('"','')
#     #         pflage=True
#     #         if(arg0=='' or sent.split().__contains__(pred.split()[0])==False):
#     #             pflage=False
#     #             a0flage=False
#     #             a1flage=False
#     #             a2flage=False
#     #             a3flage=False
               
#     #     if(line.find('\"arg1\": ')>=0 and (line.replace('\"arg1\": ','')!='      "",\n' and line.replace('\"arg1\": ','')!='"",\n')):
#     #         arg1=line.replace('\"arg1\": ','')
#     #         arg1=arg1.replace('      "','')
#     #         arg1=arg1.replace(',\n','').replace('"','')
#     #         a1flage=True
#     #         if(arg0==''):
#     #             pflage=False
#     #             a0flage=False
#     #             a1flage=False
#     #             a2flage=False
#     #             a3flage=False
#     #     if(line.find('\"arg2\": ')>=0 and (line.replace('\"arg2\": ','')!='      "",\n' and line.replace('\"arg2\": ','')!='"",\n')):
#     #         arg2=line.replace('\"arg2\": ','')
#     #         arg2=arg2.replace('      "','')
#     #         arg2=arg2.replace(',\n','').replace('"','')
#     #         a2flage=True
#     #         if(arg0==''):
#     #             pflage=False
#     #             a0flage=False
#     #             a1flage=False
#     #             a2flage=False
#     #             a3flage=False
#     #     if(line.find('\"arg3\": ')>=0 and (line.replace('\"arg3\": ','')!='      "",\n' and line.replace('\"arg3\": ','')!='"",\n')):
#     #         arg3=line.replace('\"arg3\": ','')
#     #         arg3=arg3.replace('      "','')
#     #         arg3=arg3.replace(',\n','').replace('"','')
#     #         a3flage=True
#     #         if(arg0==''):
#     #             pflage=False
#     #             a0flage=False
#     #             a1flage=False
#     #             a2flage=False
#     #             a3flage=False
        

#     #     #if(line.find('\"context\": ')>=0 and line.replace('\"context\": ','')!='      "",\n'):
#     #     #    if(a3flage==False):
#     #     #        arg3=line.replace('\"context\": ','').replace('"','').replace('      ','').replace(',\n','')
#     #     #    else:
#     #     #        arg3+=line.replace('\"context\": ','').replace('"','').replace('      ','').replace(',\n','')
#     #     #    a3flage=True
#     #     if(sflage and a0flage and pflage and a1flage):
#     #         relation=''
            
#     #         with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                             strrela=arg0+' --> '+pred+' --> '+arg1+"\n"
#     #                             if(strrela not in relation):
#     #                              relation+=arg0+' --> '+pred+' --> '+arg1+"\n"
#     #                              f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#     #                              f3.write('\n')
#     #                              f3.write(arg0+' --> '+pred+' --> '+arg1)
#     #                              f3.write('\n')
            
#     #         index1_0=sent.split().index(arg0.split()[0])
#     #         index1_1=index1_0+len(arg0.split())
#     #         index2_0=sent.split().index(arg1.split()[0])
#     #         index2_1=index2_0+len(arg1.split())
#     #         rangeindex=list(range(sent.split().index(pred.split()[0]),sent.split().index(pred.split()[0])+len(pred.split())))
#     #         reindedx=[w for w in rangeindex]
#     #         r = ([w for w in sent.split()],(index1_0,index1_1),(index2_0,index2_1),pred,reindedx)
#     #         D.append((r, arg0, arg1))
#     #         a1flage=False
#     #         indexcluster+=1 
#     #     if(sflage and a0flage and pflage and a2flage):
#     #         relation=''
#     #         with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                              f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#     #                              f3.write('\n')
#     #                              f3.write(arg0+' --> '+pred+' --> '+arg2)
#     #                              f3.write('\n')
            
#     #         index1_0=sent.split().index(arg0.split()[0])
#     #         index1_1=index1_0+len(arg0.split())
#     #         index2_0=sent.split().index(arg2.split()[0])
#     #         index2_1=index2_0+len(arg2.split())
#     #         rangeindex=list(range(sent.split().index(pred.split()[0]),sent.split().index(pred.split()[0])+len(pred.split())))
#     #         reindedx=[w for w in rangeindex]
#     #         r = ([w for w in sent.split()],(index1_0,index1_1),(index2_0,index2_1),pred,reindedx)
#     #         D.append((r, arg0, arg2))
#     #         a2flage=False
#     #         indexcluster+=1
#     #     if(sflage and a0flage and pflage and a3flage):
#     #         relation=''
#     #         with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                              f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#     #                              f3.write('\n')
#     #                              f3.write(arg0+' --> '+pred+' --> '+arg3)
#     #                              f3.write('\n')
#     #         index1_0=sent.split().index(arg0.split()[0])
#     #         index1_1=index1_0+len(arg0.split())
#     #         index2_0=sent.split().index(arg3.split()[0])
#     #         index2_1=index2_0+len(arg3.split())
#     #         rangeindex=list(range(sent.split().index(pred.split()[0]),sent.split().index(pred.split()[0])+len(pred.split())))
#     #         reindedx=[w for w in rangeindex]
#     #         r = ([w for w in sent.split()],(index1_0,index1_1),(index2_0,index2_1),pred,reindedx)
#     #         D.append((r, arg0, arg3))
#     #         a3flage=False
#     #         #sflage=False
#     #         pflage=False
#     #         a0flage=False
#     #         indexcluster+=1
        
    
#     # completeName = os.path.join("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\Re-OIE2016",\
#     #                             "DORg.pkl")
#     # with open(completeName, 'wb') as output:
#     #     pickle.dump(D, output)
    
#     #-------------------------------------CARB DATASET
#     # with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\carb_sentences\\carb_sentences.txt", "r") as f:#, encoding="utf8"
#     #        text = f.readlines()
#     # with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\carb_sentences\\test_gold_allennlp_format.txt", "r") as f1:#, encoding="utf8"
#     #        text2 = f1.readlines()    
#     # index=0
#     # for line in tqdm(text, total=len(text)):
#     #         t=line.rstrip().replace('\n','')
#     #         index+=1
#     #         if any(t in s for s in text2):
#     #             indexcluster=0
#     #             relation=''
#     #             if(index>1):
#     #                 with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\carb_sentences\\benchie_gold_annotations_en_org.txt", 'a') as f2:
#     #                          f2.write('\n')
#     #             with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\carb_sentences\\benchie_gold_annotations_en_org.txt", 'a') as f2:
#     #                          f2.write('sent_id:'+(index).__str__())
#     #                          f2.write('\t')
#     #             with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\carb_sentences\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                          f3.write(t)
#     #                          f3.write('\n')
#     #                          f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#     #                          f3.write('\n')
                 
#     #             for line in tqdm(text2, total=len(text2)):
#     #                t2=line.rstrip().split('\t')
#     #                t3=t2[1].rstrip().split(' </arg1> <rel> ')
#     #                t4=t3[1].rstrip().split(' </rel> <arg2> ')
#     #                if(t4[1].replace(' </arg2>','')=='' or t3[0].replace('<arg1> ','')=='' or t4[0]==''):
#     #                     continue
#     #                if(t!=t2[0]):
#     #                     continue
#     #                with open("E:\\PHD\\16-7-2024\\UpdateBertRelation\\BERT-Relation-Extraction-master\\dataset\\carb_sentences\\benchie_gold_annotations_en_org.txt", 'a') as f3:    
#     #                            #f3.write((index).__str__()+'--> Cluster '+(indexcluster+1).__str__()+':')
#     #                            #f3.write('\n')
#     #                            strrela=t3[0].replace('<arg1> ','')+' --> '+t4[0]+' --> '+t4[1].replace(' </arg2>','')+"\n"
#     #                            if(strrela not in relation):
#     #                             relation+=t3[0].replace('<arg1> ','')+' --> '+t4[0]+' --> '+t4[1].replace(' </arg2>','')+"\n"                           
#     #                             f3.write(t3[0].replace('<arg1> ','')+' --> '+t4[0]+' --> '+t4[1].replace(' </arg2>',''))
#     #                             f3.write('\n')
#     #             indexcluster+=1

               