from src.preprocessing_funcs import load_dataloaders
from src.trainer import train_and_fit
import logging
from argparse import ArgumentParser
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
                                                                             2 - BioBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
    
    #args = parser.parse_args()
    #args.PathDataset="./dataset/carb_sentences/"
    #args.ResultPathDataset="./results/carb_sentences/"
    #args.pretrain_data="carb_sentences.txt"
    #output = train_and_fit(args)

    args = parser.parse_args()
    args.PathDataset="./dataset/CDR/"
    args.ResultPathDataset="./results/CDR/"
    args.pretrain_data="train_filter.data"
    args.model_no=2
    output = train_and_fit(args)   
   
