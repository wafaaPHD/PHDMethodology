
import os
import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
from .misc import load_pickleResult
import logging
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = args.ResultPathDataset
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results(args):
    """ Loads saved results if exists """
    losses_path =args.ResultPathDataset+"test_losses_per_epoch_%d.pkl" % args.model_no
    f1_path =args.ResultPathDataset+ "test_f1_per_epoch_%d.pkl" % args.model_no
    if os.path.isfile(losses_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickleResult("test_losses_per_epoch_%d.pkl" % args.model_no,args)
        f1_per_epoch = load_pickleResult("test_f1_per_epoch_%d.pkl" % args.model_no,args)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, f1_per_epoch = [], []
    return losses_per_epoch, f1_per_epoch


def evaluate_(
    lm_logits=None, 
    blanks_logits=None, 
    masked_for_pred=None, 
    blank_labels=None, 
    tokenizer=None,
    triple_predictions=None,
    relation_labels=None,
    triple_targets=None  # Binary targets (1 for positive triples, 0 for negative triples)
):
    """
    Evaluates model outputs including MLM, Blank Matching, Relation Classification, and Triple Scoring.
    
    Returns:
        metrics (dict): Precision, Recall, F1, and AUC for relation and triple tasks.
    """
    metrics = {
        'relation_precision': 0.0,
        'relation_recall': 0.0,
        'relation_f1': 0.0,
        'relation_auc': 0.0,
        'triple_precision': 0.0,
        'triple_recall': 0.0,
        'triple_f1': 0.0,
        'triple_auc': 0.0,
    }

    # =========================================================
    # 1. Relation Classification Metrics
    # =========================================================
    if triple_predictions is not None and 'relation_logits' in triple_predictions and relation_labels is not None:
        rel_logits = triple_predictions['relation_logits'].detach().cpu()
        rel_targets = relation_labels.detach().cpu().numpy()

        # Predictions (Argmax over relations)
        rel_preds = torch.argmax(rel_logits, dim=-1).numpy()
        
        # Softmax probabilities for AUC calculation
        rel_probs = torch.softmax(rel_logits, dim=-1).numpy()

        # Compute Precision, Recall, F1 (Macro-averaged for multi-class)
        prec, rec, f1, _ = precision_recall_fscore_support(
            rel_targets, rel_preds, average='macro', zero_division=0
        )
        metrics['relation_precision'] = float(prec)
        metrics['relation_recall'] = float(rec)
        metrics['relation_f1'] = float(f1)

        # Compute Multi-class ROC-AUC (OVR: One-vs-Rest)
        try:
            num_classes = rel_probs.shape[1]
            # Handle edge case where not all classes are present in the batch/eval set
            if len(np.unique(rel_targets)) > 1:
                metrics['relation_auc'] = float(
                    roc_auc_score(rel_targets, rel_probs, multi_class='ovr', average='macro')
                )
            else:
                metrics['relation_auc'] = 0.0
        except ValueError:
            metrics['relation_auc'] = 0.0

    # =========================================================
    # 2. Triple Scoring / Prediction Metrics
    # =========================================================
    if triple_predictions is not None and 'triple_scores' in triple_predictions and triple_targets is not None:
        scores = triple_predictions['triple_scores'].detach().cpu().numpy()
        targets = triple_targets.detach().cpu().numpy()

        # Convert continuous DistMult triple scores to binary predictions using 0.0 threshold
        triple_preds = (scores > 0.0).astype(int)

        # Compute Precision, Recall, F1
        t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(
            targets, triple_preds, average='binary', zero_division=0
        )
        metrics['triple_precision'] = float(t_prec)
        metrics['triple_recall'] = float(t_rec)
        metrics['triple_f1'] = float(t_f1)

        # Compute ROC-AUC score for continuous triple scores
        try:
            if len(np.unique(targets)) > 1:
                metrics['triple_auc'] = float(roc_auc_score(targets, scores))
            else:
                metrics['triple_auc'] = 0.0
        except ValueError:
            metrics['triple_auc'] = 0.0

    return (
        metrics['relation_precision'], 
        metrics['relation_recall'], 
        metrics['relation_f1'], 
        metrics['relation_auc']
    ), (
        metrics['triple_precision'], 
        metrics['triple_recall'], 
        metrics['triple_f1'], 
        metrics['triple_auc']
    )