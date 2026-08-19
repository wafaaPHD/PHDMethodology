import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
from collections import Counter
import re

class AdaptiveTripleFilter:
    """
    Filters low-precision triples using:
    - Contextual relevance (cosine similarity to sentence)
    - Internal consistency across triples
    - Statistical outlier detection (Z-score on relevance)
    """
    
    def __init__(self,model, z_threshold=1.5, min_similarity=0.5):
        self.model = model
        self.z_threshold = z_threshold
        self.min_similarity = min_similarity

    def _encode(self, text: str):
        return self.model.encode(text, convert_to_tensor=True)

    def _triple_to_text(self, triple: Tuple[str, str, str]) -> str:
        subj, pred, obj = triple[:3] 
        return f"{subj} {pred} {obj}"

    def _context_relevance(self, sentence: str, triple: Tuple[str, str, str]) -> float:
        """How relevant is the triple to the sentence context?"""
        sent_emb = self._encode(sentence)
        triple_emb = self._encode(self._triple_to_text(triple))
        return util.cos_sim(sent_emb, triple_emb).item()

    def _triple_consistency(self, triples: List[Tuple]) -> Dict[Tuple, float]:
        """
        Measure consistency of each triple with the majority of others.
        Consistent triples share high semantic similarity with others.
        """
        if len(triples) <= 1:
            return {t: 1.0 for t in triples}
        
        embeds = [self._encode(self._triple_to_text(t)) for t in triples]
        consistency_scores = {}
        
        for i, t in enumerate(triples):
            sims = [util.cos_sim(embeds[i], embeds[j]).item() for j in range(len(triples)) if j != i]
            consistency_scores[t] = np.mean(sims) if sims else 0.0
        
        return consistency_scores

    def filter_and_analyze(self, sentence: str, triples: List[Tuple]) -> Tuple[List[Tuple], Dict]:
        """
        Returns:
            - filtered_triples: high-quality triples
            - analysis: error breakdown and scores
        """
        if not triples:
            return [], {"error": "No triples provided"}

        # 1. Context relevance scores
        relevance_scores = {t: self._context_relevance(sentence, t) for t in triples}
        
        # 2. Internal consistency scores
        consistency_scores = self._triple_consistency(triples)
        
        # 3. Combined score (geometric mean of relevance and consistency)
        combined_scores = {}
        for t in triples:
            rel = relevance_scores[t]
            con = consistency_scores[t]
            combined_scores[t] = np.sqrt(rel * con) if rel > 0 and con > 0 else 0.0
        
        # 4. Z-score based outlier detection
        score_values = np.array(list(combined_scores.values()))
        if len(score_values) > 1:
            mean_score = np.mean(score_values)
            std_score = np.std(score_values) + 1e-8
            z_scores = {t: (combined_scores[t] - mean_score) / std_score for t in triples}
        else:
            z_scores = {t: 0.0 for t in triples}
        
        # 5. Filtering decision
        filtered = []
        error_types = []
        
        for t in triples:
            keep = True
            error_reason = None
            
            # Rule 1: Minimum relevance
            if relevance_scores[t] < self.min_similarity:
                keep = False
                error_reason = "low_context_relevance"
            
            # Rule 2: Outlier (Z-score too low)
            if z_scores[t] < -self.z_threshold:
                keep = False
                error_reason = "outlier_inconsistent"
            
            # Rule 3: Lexical hallucination check (optional)
            triple_tokens = set(re.findall(r'\b\w+\b', self._triple_to_text(t).lower()))
            sent_tokens = set(re.findall(r'\b\w+\b', sentence.lower()))
            if len(triple_tokens - sent_tokens) / (len(triple_tokens) + 1e-5) > 0.8:
                keep = False
                error_reason = "lexical_hallucination"
            
            if keep:
                filtered.append(t)
                error_types.append("good")
            else:
                error_types.append(error_reason)
        
        # 6. Error analysis dictionary
        error_counts = Counter([e for e in error_types if e != "good"])
        
        analysis = {
            "total_triples": len(triples),
            "kept_triples": len(filtered),
            "filtered_out": len(triples) - len(filtered),
            "precision": len(filtered) / len(triples) if triples else 0,
            "error_breakdown": dict(error_counts),
            "per_triple_scores": [
                {
                    "triple": t,
                    "relevance": relevance_scores[t],
                    "consistency": consistency_scores[t],
                    "combined": combined_scores[t],
                    "z_score": z_scores[t],
                    "kept": t in filtered,
                    "error_type": error_types[i] if error_types[i] != "good" else None
                }
                for i, t in enumerate(triples)
            ]
        }
        
        return filtered, analysis
