# enhanced_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class EnhancedTwoHeadedLoss(nn.Module):
    """
    Combined loss for relation matching and triple prediction
    """
    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False, 
                 triple_loss_weight=0.5, relation_negatives=10):
        super().__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=self.lm_ignore_idx)
        self.use_logits = use_logits
        self.normalize = normalize
        self.triple_loss_weight = triple_loss_weight
        self.relation_negatives = relation_negatives
        
        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction='mean')
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1/(torch.norm(f1_vec) * torch.norm(f2_vec) + 1e-8)
        else:
            factor = 1.0
        
        if not self.use_logits:
            p = 1/(1 + torch.exp(-factor * torch.dot(f1_vec, f2_vec)))
        else:
            p = factor * torch.dot(f1_vec, f2_vec)
        return p
    
    def forward(self, lm_logits, blanks_logits, lm_labels, blank_labels,
                triple_predictions=None, relation_labels=None, args=None, verbose=False):
        """
        Args:
            lm_logits: (batch_size, sequence_length, vocab_size)
            blanks_logits: (batch_size, hidden_size)
            lm_labels: (batch_size, sequence_length)
            blank_labels: (batch_size, 1) - 1 if positive, 0 if negative
            triple_predictions: dict from EnhancedTriplePredictionModel
            relation_labels: (batch_size) - relation IDs for each example
        """
        # Original MTB loss
        pos_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1]
        neg_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0]
        
        # Positives
        if len(pos_idxs) > 1:
            pos_logits = []
            # BUGFIX: was zip(pos_idxs[:-1], pos_idxs[1:]) -- only adjacent
            # pairs. Use all C(n,2) pairs, matching Two_Headed_Loss /
            # compute_pairwise_metrics in train_funcs.py, so the loss is
            # optimizing the same pair set the metrics evaluate.
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(self.p_(blanks_logits[pos1, :], blanks_logits[pos2, :]))
            pos_logits = torch.stack(pos_logits, dim=0) if pos_logits else torch.FloatTensor([])
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])] if pos_logits.numel() > 0 else []
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
            if blanks_logits.is_cuda:
                pos_logits = pos_logits.cuda()
        
        # Negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(self.p_(blanks_logits[pos_idx, :], blanks_logits[neg_idx, :]))
        neg_logits = torch.stack(neg_logits, dim=0) if neg_logits else torch.FloatTensor([])
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])] if neg_logits.numel() > 0 else []
        
        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)
        if blanks_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()
        
        # LM loss
        lm_loss = self.LM_criterion(lm_logits, lm_labels) if lm_logits.numel() > 0 else torch.tensor(0.0)
        
        # Blank matching loss
        all_logits = torch.cat([pos_logits, neg_logits], dim=0) if pos_logits.numel() > 0 or neg_logits.numel() > 0 else torch.FloatTensor([])
        if all_logits.numel() > 0 and blank_labels_.numel() > 0:
            blank_loss = self.BCE_criterion(all_logits, blank_labels_)
        else:
            blank_loss = torch.tensor(0.0)
        if blanks_logits.is_cuda:
            lm_loss = lm_loss.cuda()
            blank_loss = blank_loss.cuda()
        
        # Triple prediction loss
        triple_loss = torch.tensor(0.0)
        triple_acc = 0.0
        if triple_predictions is not None and relation_labels is not None:
            relation_logits = triple_predictions['relation_logits']  # [batch, num_relations]

            # relation_labels: [batch] or [batch, 1], -1 = ignore (padding or
            # examples we couldn't map to a relation id during preprocessing)
            relation_labels_flat = relation_labels.view(-1).long()

            if relation_labels_flat.numel() > 0:
                # BUGFIX: this used to also compute "entity type" losses by
                # running cross_entropy between the vocab-sized e1_logits/
                # e2_logits and `relation_labels` -- i.e. supervising entity
                # type prediction with relation ids, against a vocab-sized
                # output. There is no entity-type ground truth anywhere in
                # the data pipeline, so that term was training the model on
                # a fabricated target. It's removed rather than left silently
                # wrong; only real relation classification is trained here.
                relation_loss = F.cross_entropy(relation_logits, relation_labels_flat, ignore_index=-1)
                triple_loss = relation_loss

                # Calculate relation prediction accuracy (only over non-ignored examples)
                valid_mask = relation_labels_flat != -1
                if valid_mask.any():
                    pred_relations = relation_logits.argmax(dim=-1)
                    triple_acc = (pred_relations[valid_mask] == relation_labels_flat[valid_mask]).float().mean()
        
        total_loss = lm_loss + blank_loss + self.triple_loss_weight * triple_loss
        
        if verbose:
            print(f"LM loss: {lm_loss.item():.5f}, Blank loss: {blank_loss.item():.5f}, Triple loss: {triple_loss.item():.5f}")
            print(f"Triple relation accuracy: {triple_acc:.3f}")
        
        return total_loss