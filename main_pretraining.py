from src.preprocessing_funcs import load_dataloaders
from sentence_transformers import SentenceTransformer
from src.enhanced_trainer import train_enhanced_model
import os
import logging
from argparse import ArgumentParser
import json
from src.AnalysisError import main_analysis
from src.AnalysisError2 import mainanalysis2
from src.AnalysisMemoryTime import main_analysis2
from src.AnalysisMemoryTime2 import mainMomery2
from Evaluation import evaluate_openie
from Evaluation import evaluate_openie_no_gold
'''
This trains the BERT model on matching the blanks 
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--PathDataset", type=str, default="./dataset/CDR/1/", \
                        help="pre-training data .txt file path")
    parser.add_argument("--ResultPathDataset", type=str, default="./results/CDR/1/", \
                        help="pre-training data .txt file path")
    parser.add_argument("--pretrain_data", type=str, default="cnn.txt", \
                        help="pre-training data .txt file path")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--freeze", type=int, default=0, help='''1: Freeze most layers until classifier layers\
                                                                \n0: Don\'t freeze \
                                                                (Probably best not to freeze if GPU memory is sufficient)''')
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=12, help="No of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")#0.0001
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT\n
                                                                            2 - BioBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2',\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
    #parser.add_argument("--cosine_similarity", type=float, default=0, help="cosine_similarity")
    #parser.add_argument("--Z_scores", type=float, default=1.2, help="Z_scores")
    parser.add_argument("--num_relations", type=int, default=12, help="num_relations")
    args = parser.parse_args()
    #args.PathDataset="./dataset/Re-OIE2016/"
    #args.ResultPathDataset="./dataset/Re-OIE2016/"
    #args.pretrain_data="Re-OIE2016.json"
    modelt=None
    #modelt=SentenceTransformer('all-MiniLM-L6-v2')
    #cosine_similarity=[0]
    #Z_scores=[0]#[0,1,1.5,2,2.5,3,3.5]#
    args.PathDataset="./dataset/sentences_webnlg/"
    args.ResultPathDataset="./dataset/sentences_webnlg/"
    args.pretrain_data="sentences_webnlg.txt"
    args.cosine_similarity=0
    args.Z_scores=0
    pathFile='allall'+str(0)+'_cosine_similarity'+str(args.cosine_similarity)
    os.makedirs(args.ResultPathDataset+pathFile, exist_ok=True)
    args.PathDataset=args.ResultPathDataset+pathFile+'/'    
    train_loader = load_dataloaders(args, modelt) 
    SENTENCES_PATH="./dataset/sentences_webnlg/sentences_webnlg.txt"
    SYS_PATH=args.PathDataset+'AllMyMethod.txt'
    OUT_DIR=args.PathDataset+pathFile+'/'+"Evaluation/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
    SYS_PATH='./dataset/sentences_webnlg/claude3_7sonnet_triples.txt'
    OUT_DIR="./dataset/sentences_webnlg/claude3_7sonnet_triplesEvaluation/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
    SYS_PATH='./dataset/sentences_webnlg/gemini2_5pro_triples.txt'
    OUT_DIR="./dataset/sentences_webnlg/gemini2_5pro_triplesEvaluation/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
    SYS_PATH='./dataset/sentences_webnlg/GPT-4o-mini_triples.txt'
    OUT_DIR="./dataset/sentences_webnlg/GPT-4o-mini_triplesEvaluation/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
    #GOLD_PATH ='./dataset/carb_sentences/carb_test_benchie_format.txt'
    #evaluate_openie.evaluate(GOLD_PATH,SYS_PATH,OUT_DIR) 








    cosine_similarity=[0]
    Z_scores=[0]#[0,1,1.5,2,2.5,3,3.5]#
    args.PathDataset="./dataset/wire57/"
    args.ResultPathDataset="./dataset/wire57/"
    args.pretrain_data="wire57_sentences.txt"
    args.cosine_similarity=0
    args.Z_scores=0
    pathFile='allall'+str(0)+'_cosine_similarity'+str(args.cosine_similarity)
    os.makedirs(args.ResultPathDataset+pathFile, exist_ok=True)
    args.PathDataset=args.ResultPathDataset+pathFile+'/'    
    train_loader = load_dataloaders(args, modelt) 
    SENTENCES_PATH="./dataset/wire57/wire57_sentences.txt"
    SYS_PATH=args.PathDataset+'AllMyMethod.txt'
    OUT_DIR=args.PathDataset+pathFile+'/'+"Evaluation/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
    GOLD_PATH ='./dataset/wire57/wire57-annotated(57).txt'
    evaluate_openie.evaluate(GOLD_PATH,SYS_PATH,OUT_DIR) 


    #args.PathDataset="./dataset/wire57/"
    #args.ResultPathDataset="./dataset/wire57/"
    #args.pretrain_data="wire57_sentences.txt"
    args.PathDataset="./dataset/OIE/"
    args.ResultPathDataset="./dataset/OIE/"
    args.pretrain_data="cnn.txt"
    args.cosine_similarity=0
    args.Z_scores=0
    pathFile='allall'+str(0)+'_cosine_similarity'+str(args.cosine_similarity)
    os.makedirs(args.ResultPathDataset+pathFile, exist_ok=True)
    args.PathDataset=args.ResultPathDataset+pathFile+'/'    
    train_loader = load_dataloaders(args, modelt) 
    #SENTENCES_PATH="./dataset/wire57/wire57_sentences.txt"
    SENTENCES_PATH="./dataset/OIE/cnn.txt"
    SYS_PATH=args.PathDataset+'AllMyMethod.txt'
    OUT_DIR=args.PathDataset+pathFile+'/'+"Evaluation/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
    #GOLD_PATH ='./dataset/wire57/wire57-annotated(57).txt'
    GOLD_PATH ='./dataset/OIE/benchie_gold_annotations_en.txt'

    evaluate_openie.evaluate(GOLD_PATH,SYS_PATH,OUT_DIR) 

  
    args.PathDataset="./dataset/wire57/"
    args.ResultPathDataset="./dataset/wire57/"
    args.pretrain_data="wire57_sentences.txt"
    #args.PathDataset="./dataset/OIE/"
    #args.ResultPathDataset="./dataset/OIE/"
    #args.pretrain_data="cnn.txt"
    for  cosine_similarity_parm in cosine_similarity:
            for  Z_scores_parm in Z_scores:
                #SENTENCES_PATH=args.PathDataset+"cnn.txt"
                SENTENCES_PATH=args.PathDataset+"wire57_sentences.txt"
                pathFile='_attributive_adj_triples'+str(Z_scores_parm)+'_cosine_similarity'+str(cosine_similarity_parm)

                SYS_PATH=args.PathDataset+pathFile+'/'+str(cosine_similarity_parm)+str(Z_scores_parm)+'MyMethod.txt'
                OUT_DIR=args.PathDataset+pathFile+'/'+"Evaluation/"
                evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
                #GOLD_PATH = args.PathDataset+'benchie_gold_annotations_en.txt'
                GOLD_PATH = args.PathDataset+'wire57-annotated(57).txt'
                evaluate_openie.evaluate(GOLD_PATH,SYS_PATH,OUT_DIR) 
                  
    cosine_similarity=[0]
    Z_scores=[1]#
    
    for  cosine_similarity_parm in cosine_similarity:
        for  Z_scores_parm in Z_scores:
                args.cosine_similarity=cosine_similarity_parm
                args.Z_scores=Z_scores_parm
                pathFile='1_Z_scores'+str(Z_scores_parm)+'_cosine_similarity'+str(args.cosine_similarity)
                os.makedirs(args.ResultPathDataset+pathFile, exist_ok=True)
                args.PathDataset=args.ResultPathDataset+pathFile+'/'

                train_loader = load_dataloaders(args, modelt) 

                SENTENCES_PATH=args.PathDataset+'sentenses.txt'
                SYS_PATH=args.PathDataset+str(args.cosine_similarity)+str(args.Z_scores)+'MyMethod.txt'
                OUT_DIR=args.PathDataset+"Evaluation/"
                evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)

                GOLD_PATH = args.PathDataset+"re_oie2016_benchie_format.txt"
                evaluate_openie.evaluate(GOLD_PATH,SYS_PATH,OUT_DIR)                
                try:# Load the data
                    with open(args.PathDataset+'Analysis_stats.json', 'r') as f:
                        data = json.load(f)
                    main_analysis(data=data,args=args)
                    mainanalysis2(data=data,args=args)
                    """Load and parse the processing statistics data"""
                    filename=args.PathDataset+'processing_stats.json'
                    with open(filename, 'r') as f:
                        content = f.read()
                        start_idx = content.find('[')
                        alldatacontent=json.loads(content[start_idx:])
                    main_analysis2(alldatacontent,args=args)
                    mainMomery2(alldatacontent,args=args)
                except FileNotFoundError:
                         pass    
                
                #output = train_enhanced_model(args, modelt)
    
    args.PathDataset="./dataset/OIE/"
    args.ResultPathDataset="./dataset/OIE/"
    args.pretrain_data="cnn.txt"
    SENTENCES_PATH=args.ResultPathDataset+args.pretrain_data
    SYS_PATH=args.PathDataset+'aggressiveAllMyMethod.txt'
    OUT_DIR=args.PathDataset+"EvaluationAll/"
    evaluate_openie_no_gold.evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
           
    GOLD_PATH = "./Evaluation/data/gold/2_annotators/benchie_gold_annotations_en.txt"
    evaluate_openie.evaluate(GOLD_PATH,SYS_PATH,OUT_DIR)  
        
    #output = train_and_fit(args,model)

    #args = parser.parse_args()
    #args.PathDataset="./dataset/carb_sentences/"
    #args.ResultPathDataset="./dataset/carb_sentences/"
    #args.pretrain_data="cnn.txt"
    #model=SentenceTransformer('all-MiniLM-L6-v2')
    #output = train_and_fit(args,model)

    #args = parser.parse_args()
    #args.PathDataset="./dataset/Re-carb_sentences2016/"
    #args.ResultPathDataset="./dataset/Re-carb_sentences2016/"
    #args.pretrain_data="Re-carb_sentences2016.json"
    #model=SentenceTransformer('all-MiniLM-L6-v2')
    #output = train_and_fit(args,model)

    #args = parser.parse_args()
    #args.PathDataset="./dataset/CDR/"
    #args.ResultPathDataset="./results/CDR/"
    #args.pretrain_data="train_filter.data"
    #args.model_no=2
    #output = train_and_fit(args)
    
    '''
    # For testing additional models
    from src.model.BERT.modeling_bert import BertModel, BertConfig
    from src.model.BERT.tokenization_bert import BertTokenizer as Tokenizer
    config = BertConfig.from_pretrained('./additional_models/biobert_v1.1_pubmed/bert_config.json')
    model = BertModel.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed.bin', 
                                      config=config,
                                      force_download=False, \
                                      model_size='bert-base-uncased',
                                      task='classification',\
                                      n_classes_=12)
    tokenizer = Tokenizer(vocab_file='./additional_models/biobert_v1.1_pubmed/vocab.txt',
                          do_lower_case=False)
    '''