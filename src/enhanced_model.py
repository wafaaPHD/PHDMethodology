import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

class EnhancedTriplePredictionModel(BertPreTrainedModel):
    """
    Enhanced BERT model with joint Cross-Entropy and Margin-Based Loss 
    for multi-task relation extraction and triple prediction.
    """
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

        # Standard HuggingFace BERT encoder
        self.bert = BertModel(config)

        # Heads
        self.lm_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.blanks_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.relation_classifier = nn.Linear(config.hidden_size * 2, args.num_relations)
        self.relation_embedding = nn.Embedding(args.num_relations, config.hidden_size)

        # Optional dimension projection
        self.projection = None

        # Loss Functions & Hyperparameters
        self.ce_loss = nn.CrossEntropyLoss()
        self.margin_loss = nn.MarginRankingLoss(
            margin=getattr(args, 'margin', 1.0)
        )

        # Task loss balancing weights
        self.alpha = getattr(args, 'alpha_rel_loss', 1.0)
        self.beta = getattr(args, 'beta_triple_loss', 1.0)

        # Initialize weights
        self.post_init()

    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        attention_mask=None, 
        e1_e2_start=None,
        neg_e2_start=None,             # Optional entity indices for negative sampling
        labels=None,                   # LM labels [batch_size, seq_len]
        blank_labels=None,             # Blank matching labels [batch_size]
        relation_labels=None,          # True relation targets [batch_size]
        return_triple_predictions=False
    ):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output        # [batch_size, hidden_size]

        # Base predictions
        blanks_logits = self.blanks_linear(pooled_output)  # [batch_size, vocab_size]
        lm_logits = self.lm_linear(sequence_output)        # [batch_size, seq_len, vocab_size]

        total_loss = torch.tensor(0.0, device=input_ids.device)
        losses = {}

        # 1. Base LM and Blank Matching Losses
        if labels is not None:
            lm_loss = self.ce_loss(
                lm_logits.view(-1, self.config.vocab_size), 
                labels.view(-1)
            )
            total_loss += lm_loss
            losses['lm_loss'] = lm_loss

        if blank_labels is not None:
            blank_loss = self.ce_loss(blanks_logits, blank_labels)
            total_loss += blank_loss
            losses['blank_loss'] = blank_loss

        triple_predictions = None

        if return_triple_predictions and e1_e2_start is not None:
            batch_size = sequence_output.size(0)
            device = sequence_output.device

            # Dimension projection safeguard
            actual_hidden = sequence_output.size(-1)
            if actual_hidden != self.config.hidden_size:
                if self.projection is None or self.projection.in_features != actual_hidden:
                    self.projection = nn.Linear(actual_hidden, self.config.hidden_size).to(device)
                sequence_output = self.projection(sequence_output)

            # Extract entity embeddings
            batch_indices = torch.arange(batch_size, device=device)
            e1_start, e2_start = e1_e2_start[:, 0], e1_e2_start[:, 1]

            e1_embeddings = sequence_output[batch_indices, e1_start]  # [batch_size, hidden_size]
            e2_embeddings = sequence_output[batch_indices, e2_start]  # [batch_size, hidden_size]

            # Relation Classification Head
            entity_pair = torch.cat([e1_embeddings, e2_embeddings], dim=-1)
            relation_logits = self.relation_classifier(entity_pair)  # [batch_size, num_relations]

            # 2. Relation Classification Loss
            if relation_labels is not None:
                rel_loss = self.ce_loss(relation_logits, relation_labels)
                total_loss += self.alpha * rel_loss
                losses['relation_loss'] = rel_loss
                rel_ids_for_scoring = relation_labels
            else:
                rel_ids_for_scoring = relation_logits.argmax(dim=-1)

            # Compute DistMult Triple Scores (Positive Triples)
            rel_embeddings = self.relation_embedding(rel_ids_for_scoring)  # [batch_size, hidden_size]
            pos_scores = torch.sum(e1_embeddings * rel_embeddings * e2_embeddings, dim=-1)  # [batch_size]

            # 3. Margin-Based Triple Scoring Loss (with Negative Sampling)
            if neg_e2_start is not None:
                neg_e2_embeddings = sequence_output[batch_indices, neg_e2_start]
                neg_scores = torch.sum(e1_embeddings * rel_embeddings * neg_e2_embeddings, dim=-1)

                # Target = 1 forces pos_scores to be higher than neg_scores by at least margin
                y = torch.ones_like(pos_scores)
                triple_loss = self.margin_loss(pos_scores, neg_scores, y)
                total_loss += self.beta * triple_loss
                losses['triple_margin_loss'] = triple_loss

            # Construct predictions payload
            triple_predictions = {
                'e1_logits': lm_logits[batch_indices, e1_start],
                'e2_logits': lm_logits[batch_indices, e2_start],
                'relation_logits': relation_logits,
                'triple_scores': pos_scores,
            }

        losses['total_loss'] = total_loss if isinstance(total_loss, torch.Tensor) else None

        return {
            'loss': losses['total_loss'],
            'loss_components': losses,
            'blanks_logits': blanks_logits,
            'lm_logits': lm_logits,
            'triple_predictions': triple_predictions
        }
   