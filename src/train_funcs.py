
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
def _decode_span(tokenizer, token_ids, start_idx, end_idx):
    """
    Decode the token span [start_idx, end_idx) into text. Stops early
    (before end_idx) if a special token (e.g. [E1], [/E1], [E2], [/E2],
    [SEP], [PAD]) is hit, since that marks the end of real content.
    """
    if end_idx <= start_idx:
        return '', start_idx
    special_ids = set(tokenizer.all_special_ids)
    span_ids = []
    consumed = 0
    for tok_id in token_ids[start_idx:end_idx].tolist():
        if tok_id in special_ids:
            break
        span_ids.append(tok_id)
        consumed += 1
    text = tokenizer.decode(span_ids, skip_special_tokens=True).strip() if span_ids else ''
    return text, start_idx + consumed
 
 
def _decode_entity_span(tokenizer, token_ids, start_idx, max_span_len=8):
    """
    Best-effort decode of the entity text that follows an E1/E2 marker
    token. Starts right after `start_idx` and stops at the next special
    token (e.g. [E1], [/E1], [E2], [/E2], [SEP], [PAD]) or after
    `max_span_len` tokens, whichever comes first.
 
    Returns (entity_text, end_idx), where end_idx is the token position
    right after the decoded span -- callers use it as the start of the
    "relation" text that follows the entity.
    """
    return _decode_span(tokenizer, token_ids, start_idx + 1, start_idx + 1 + max_span_len)
 
 
def load_seen_triples(args, out_filename='HighScoreUniqueTriples.txt'):
    """
    Loads the (entity1, relation, entity2) triples already present in the
    high-score output file, so that uniqueness can be enforced globally
    -- i.e. across the whole file and across separate training runs --
    rather than only within the current process.
    """
    seen = set()
    out_path = os.path.join(args.ResultPathDataset, out_filename)
    if os.path.isfile(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 4:
                    _, e1, rel, e2 = parts
                    seen.add((e1, rel, e2))
    return seen
def get_mask_predictions(lm_logits, lm_labels, tokenizer, lm_ignore_idx, top_k=1):
    """
    Recovers the model's predicted token(s) for every masked ([MASK])
    position in a batch.

    Args:
        lm_logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]
            -- raw LM head output
        lm_labels: [batch, seq_len] or [seq_len] -- true token id at
            masked positions, lm_ignore_idx everywhere else (this is how
            masked positions are located). Must have one fewer dimension
            than lm_logits (i.e. no vocab axis).
        tokenizer: tokenizer used to encode the input
        lm_ignore_idx: the ignore_index value used when building lm_labels
        top_k: how many candidate predictions to return per masked position

    Returns:
        list of dicts, one per masked position:
            {
                'batch_idx': int, 'seq_idx': int,
                'true_token': str,
                'predicted_tokens': [str, ...],   # top_k, best first
                'confidences': [float, ...]        # softmax probs, aligned
            }
        'batch_idx' is always 0 when lm_labels is 1-D (single sequence,
        no batch axis).
    """
    if lm_labels.dim() == lm_logits.dim() - 1 and lm_labels.dim() == 1:
        # No batch axis -- treat as a single sequence (batch_idx = 0).
        lm_logits = lm_logits.unsqueeze(0)
        lm_labels = lm_labels.unsqueeze(0)
    elif lm_labels.dim() != lm_logits.dim() - 1:
        raise ValueError(
            f"lm_labels has {lm_labels.dim()} dims but lm_logits has "
            f"{lm_logits.dim()} dims -- expected lm_labels to have exactly "
            f"one fewer dim (no vocab axis). Check that you're passing the "
            f"full, unsliced lm_logits/lm_labels (before any [(x == mask_id)] "
            f"filtering)."
        )

    def _id_to_str(token_id):
        # tokenizer.decode() gives the real surface string for a single id
        # (handles byte-level BPE / merges correctly), unlike
        # convert_ids_to_tokens() which can surface raw vocab pieces or
        # print [UNK] even for valid ids.
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        ).strip()

    probs = torch.softmax(lm_logits, dim=-1)  # [batch, seq_len, vocab_size]
    mask_positions = (lm_labels != lm_ignore_idx).nonzero(as_tuple=False)  # [N, 2]

    results = []
    for b, s in mask_positions.tolist():
        top_probs, top_ids = probs[b, s].topk(top_k)
        top_ids = top_ids.tolist()

        results.append({
            'batch_idx': b,
            'seq_idx': s,
            'true_token': _id_to_str(lm_labels[b, s].item()),
            'predicted_tokens': [_id_to_str(tid) for tid in top_ids],
            'confidences': top_probs.tolist()
        })
    return results
def get_mask_predictions_old(lm_logits, lm_labels, tokenizer, lm_ignore_idx, top_k=1):
    """
    Recovers the model's predicted token(s) for every masked ([MASK])
    position in a batch.
 
    Args:
        lm_logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]
            -- raw LM head output
        lm_labels: [batch, seq_len] or [seq_len] -- true token id at
            masked positions, lm_ignore_idx everywhere else (this is how
            masked positions are located). Must have one fewer dimension
            than lm_logits (i.e. no vocab axis).
        tokenizer: tokenizer used to encode the input
        lm_ignore_idx: the ignore_index value used when building lm_labels
        top_k: how many candidate predictions to return per masked position
 
    Returns:
        list of dicts, one per masked position:
            {
                'batch_idx': int, 'seq_idx': int,
                'true_token': str,
                'predicted_tokens': [str, ...],   # top_k, best first
                'confidences': [float, ...]        # softmax probs, aligned
            }
        'batch_idx' is always 0 when lm_labels is 1-D (single sequence,
        no batch axis).
    """
    if lm_labels.dim() == lm_logits.dim() - 1 and lm_labels.dim() == 1:
        # No batch axis -- treat as a single sequence (batch_idx = 0).
        lm_logits = lm_logits.unsqueeze(0)
        lm_labels = lm_labels.unsqueeze(0)
    elif lm_labels.dim() != lm_logits.dim() - 1:
        raise ValueError(
            f"lm_labels has {lm_labels.dim()} dims but lm_logits has "
            f"{lm_logits.dim()} dims -- expected lm_labels to have exactly "
            f"one fewer dim (no vocab axis). Check that you're passing the "
            f"full, unsliced lm_logits/lm_labels (before any [(x == mask_id)] "
            f"filtering)."
        )
 
    probs = torch.softmax(lm_logits, dim=-1)  # [batch, seq_len, vocab_size]
    mask_positions = (lm_labels != lm_ignore_idx).nonzero(as_tuple=False)  # [N, 2]
 
    results = []
    for b, s in mask_positions.tolist():
        top_probs, top_ids = probs[b, s].topk(top_k)
        results.append({
            'batch_idx': b,
            'seq_idx': s,
            'true_token': tokenizer.convert_ids_to_tokens([lm_labels[b, s].item()])[0],
            'predicted_tokens': tokenizer.convert_ids_to_tokens(top_ids.tolist()),
            'confidences': top_probs.tolist()
        })
    return results
