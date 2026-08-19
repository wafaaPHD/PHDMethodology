#        <NAME OF THE PROGRAM THIS FILE BELONGS TO>
# 	  
#   File:     scores.py
#   Authors:  Kiril Gashteovski (kiril.gashteovski@neclab.eu) 
#             Mingying Yu (mingying.yu@neclab.eu)
#             Bhushan Kotnis (bhushan.kotnis@neclab.eu)
#             Carolin Lawrence (carolin.lawrence@neclab.eu)
#             Goran Glavaš (goran@informatik.uni-mannheim.de)
#             Mathias Niepert (mathias.niepert@neclab.eu)
# 
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.  
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
# 
#        PROPRIETARY INFORMATION ---  
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.  
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

import numpy as np

class Scores():

    # Shared across all Scores instances (one instance is created per OIE system in
    # benchie.py). The golden annotations are identical for every system being
    # evaluated in a run, so caching spaCy Docs by (nlp id, text) here means each
    # unique gold/extraction string is only ever embedded once for the whole run,
    # instead of once per system x per comparison x per precision/recall call.
    _DOC_CACHE = {}

    def __init__(self) -> None:
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

        # Cache of (true_pos, false_pos, false_negs) keyed by match_type. compute_precision
        # and compute_recall used to each independently redo the full TP/FP/FN scan (and,
        # for "Similarty", re-embed every string with spaCy) even though they need the exact
        # same confusion counts. We compute them once per match_type and reuse.
        self.__confusion_cache = {}

    def compute_precision(self,nlp, extractions: list, golden_annotations: dict, match_type: str):
        """
            Args
            ----
                extractions: list
                    List of triples, each of them written in the following format: [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
                match_type: str
                    Matching type for true positive. Can be either "slot" or "lexical" for per-slot matching and lexical matching respectively.
        """
        true_pos, false_pos, _ = self.__get_confusion_counts(nlp, extractions, golden_annotations, match_type)

        if true_pos == 0 and false_pos == 0:
            self.precision = 0.0
            return 0.0

        self.precision = true_pos / (true_pos + false_pos)

    def compute_recall(self,nlp, extractions: list, golden_annotations: dict, match_type: str):
        """
            Computes the recall of the golden extractions w.r.t. the golden annotations

            Args
            ----
                extractions: list
                    List of triples, each of them written in the following format: [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
                match_type: str
                    Matching type for true positive. Can be either "slot" or "lexical" for per-slot matching and lexical matching respectively.
        """
        true_pos, _, false_negs = self.__get_confusion_counts(nlp, extractions, golden_annotations, match_type)

        denom = true_pos + false_negs
        self.recall = (true_pos / denom) if denom else 0.0

    def compute_f1(self):
        """
            Computes the F1 score. Note that precision and recall should be already computed; otherwise it will compute f1 = 0.0
        """
        if (self.precision + self.recall) == 0:
            self.f1 = 0.0
        else:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __get_confusion_counts(self, nlp, extractions: list, golden_annotations: dict, match_type: str):
        """
            Returns (true_pos, false_pos, false_negs) for the given match_type, computing them only once
            (cached) no matter how many times compute_precision/compute_recall are called for this match_type.
        """
        if match_type in self.__confusion_cache:
            return self.__confusion_cache[match_type]

        result = self.__compute_confusion_counts(nlp, extractions, golden_annotations, match_type)
        self.__confusion_cache[match_type] = result
        return result

    def __compute_confusion_counts(self, nlp, extractions: list, golden_annotations: dict, match_type: str):
        scores = self.__get_empty_synset_scores(golden_annotations)
        false_pos = 0

        if match_type == "slot":
            lookup = self.__build_slot_lookup(golden_annotations)
            for triple in extractions:
                sent_id = triple[0]
                if sent_id not in golden_annotations:
                    continue
                sent_lookup = lookup[sent_id]
                stripped_key = (triple[1].strip(), triple[2].strip(), triple[3].strip())
                idx = sent_lookup['stripped'].get(stripped_key)
                if idx is None:
                    false_pos += 1
                    continue
                # Preserve original semantics: the synset index used for scoring is looked
                # up via the *unstripped* triple (matching the legacy __get_true_positive_synset_ind
                # behaviour), it just no longer needs a linear scan to find it.
                raw_key = (triple[1], triple[2], triple[3])
                raw_idx = sent_lookup['raw'].get(raw_key, -1)
                scores[sent_id][raw_idx] = 1

        elif match_type == "lexical":
            lookup = self.__build_lexical_lookup(golden_annotations)
            for triple in extractions:
                sent_id = triple[0]
                if sent_id not in golden_annotations:
                    continue
                sent_lookup = lookup[sent_id]
                stripped_key = triple[1].strip() + " " + triple[2].strip() + " " + triple[3].strip()
                if stripped_key not in sent_lookup:
                    false_pos += 1
                    continue
                raw_key = triple[1] + " " + triple[2] + " " + triple[3]
                idx = sent_lookup.get(raw_key, -1)
                scores[sent_id][idx] = 1

        elif match_type == "Similarty":
            self.__warm_doc_cache_for_similarity(nlp, extractions, golden_annotations)
            for triple in extractions:
                sent_id = triple[0]
                if sent_id not in golden_annotations:
                    continue
                try:
                    tp_synset_indices = self.__get_Similarty_true_positive_synset_ind(nlp, triple, golden_annotations)
                except Exception:
                    tp_synset_indices = []

                if not tp_synset_indices:
                    false_pos += 1
                else:
                    for idx in tp_synset_indices:
                        scores[sent_id][idx] = 1
        else:
            raise Exception

        true_pos = 0
        false_negs = 0
        for sent_id in scores:
            nz = np.count_nonzero(scores[sent_id])
            true_pos += nz
            false_negs += scores[sent_id].shape[0] - nz

        return true_pos, false_pos, false_negs

    def __build_slot_lookup(self, golden_annotations: dict) -> dict:
        """
            Builds, once per call, an O(1) lookup from (subj, rel, obj) -> first matching synset index,
            per sentence id, for both raw and stripped comparisons. Replaces the O(#gold_triples) linear
            scan that used to run for every single extracted triple.
        """
        lookup = {}
        for sent_id, synsets in golden_annotations.items():
            raw_map = {}
            stripped_map = {}
            for i, triple_synset in enumerate(synsets):
                for tr in triple_synset:
                    raw_key = (tr[0], tr[1], tr[2])
                    if raw_key not in raw_map:
                        raw_map[raw_key] = i
                    stripped_key = (tr[0].strip(), tr[1].strip(), tr[2].strip())
                    if stripped_key not in stripped_map:
                        stripped_map[stripped_key] = i
            lookup[sent_id] = {'raw': raw_map, 'stripped': stripped_map}
        return lookup

    def __build_lexical_lookup(self, golden_annotations: dict) -> dict:
        """
            Builds, once per call, an O(1) lookup from the concatenated (stripped) gold triple string
            to its first matching synset index, per sentence id.
        """
        lookup = {}
        for sent_id, synsets in golden_annotations.items():
            str_map = {}
            for i, triple_synset in enumerate(synsets):
                for tr in triple_synset:
                    tr_str = tr[0].strip() + " " + tr[1].strip() + " " + tr[2].strip()
                    if tr_str not in str_map:
                        str_map[tr_str] = i
            lookup[sent_id] = str_map
        return lookup

    def __warm_doc_cache_for_similarity(self, nlp, extractions: list, golden_annotations: dict):
        """
            Collects every unique string that will need a spaCy Doc for similarity matching (both the
            gold triple strings and the extraction triple strings) and embeds them in a single batched
            nlp.pipe() call, instead of calling nlp() one string at a time inside the matching loop.
            Already-cached strings (e.g. gold strings reused across other OIE systems' Scores instances)
            are skipped.
        """
        needed = set()
        for sent_id, synsets in golden_annotations.items():
            for triple_synset in synsets:
                for tr in triple_synset:
                    needed.add(tr[0].strip() + " " + tr[1].strip() + " " + tr[2].strip())

        for triple in extractions:
            if triple[0] in golden_annotations:
                needed.add(triple[1] + " " + triple[2] + " " + triple[3])

        to_process = [text for text in needed if (id(nlp), text) not in Scores._DOC_CACHE]
        if not to_process:
            return

        for text, doc in zip(to_process, nlp.pipe(to_process)):
            Scores._DOC_CACHE[(id(nlp), text)] = doc

    def __get_doc(self, nlp, text: str):
        key = (id(nlp), text)
        doc = Scores._DOC_CACHE.get(key)
        if doc is None:
            doc = nlp(text)
            Scores._DOC_CACHE[key] = doc
        return doc

    def __get_empty_synset_scores(self, golden_annotations: dict) -> dict:
        synset_scores = {}
        for sent_id in golden_annotations:
            synset_scores[sent_id] = np.zeros(len(golden_annotations[sent_id]))
        return synset_scores

    def __get_true_positive_synset_ind(self, triple: list, golden_annotations: dict) -> int:
        """
            Args
            ----
                triple: list 
                    The object 'triple' is in the following format [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
            
            Returns:
                int: index of the triple synset, -1 if it is not TP
        """
        subj = triple[1]
        rel = triple[2]
        obj = triple[3]
        
        sent_id = triple[0]
        for i in range(0, len(golden_annotations[sent_id])):
            triple_synset = golden_annotations[sent_id][i]
            for tr in triple_synset:
                if subj == tr[0] and rel == tr[1] and obj == tr[2]:
                    return i
        return -1

    def __get_lexical_true_positive_synset_ind(self, triple: list, golden_annotations: dict) -> int:
        """
            Args
            ----
                triple: list 
                    The object 'triple' is in the following format [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
            
            Returns:
                int: index of the triple synset, -1 if it is not TP
        """
        subj = triple[1]
        rel = triple[2]
        obj = triple[3]
        
        sent_id = triple[0]
        for i in range(0, len(golden_annotations[sent_id])):
            triple_synset = golden_annotations[sent_id][i]
            for tr in triple_synset:
                triple_str = subj + " " + rel + " " + obj
                tr_str = tr[0].strip() + " " + tr[1].strip() + " " + tr[2].strip()
                if triple_str == tr_str:
                    return i
        
        return -1
    
    def __get_Similarty_true_positive_synset_ind(self,nlp, triple: list, golden_annotations: dict) -> int:
        """
            Args
            ----
                triple: list 
                    The object 'triple' is in the following format [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
            
            Returns:
                int: index of the triple synset, -1 if it is not TP
        """
        subj = triple[1]
        rel = triple[2]
        obj = triple[3]
        
        sent_id = triple[0]
        triple_str = subj + " " + rel + " " + obj
        doc1golden = self.__get_doc(nlp, triple_str)

        indexarray=[]
        for i in range(0, len(golden_annotations[sent_id])):
            triple_synset = golden_annotations[sent_id][i]
            for tr in triple_synset:
                tr_str = tr[0].strip() + " " + tr[1].strip() + " " + tr[2].strip()
                doc2 = self.__get_doc(nlp, tr_str)
                if doc1golden.similarity(doc2)>.90:  
                    if(i not in indexarray):
                        indexarray.append(i)
        
        return indexarray

    def is_true_positive(self, triple: list , golden_annotations: dict) -> bool:
        """
            Checks if an extracted triple is considered as true positive w.r.t. the golden annotations. Matching is performed on per-slot level.

            Args
            ----
                triple: list 
                    The object 'triple' is in the following format [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
            
            Returns
            -------
                boolean:
                    True: if the triple is found in the golden annotations
                    False: otherwise
        """
        subj = triple[1].strip()
        rel = triple[2].strip()
        obj = triple[3].strip()
        
        sent_id = triple[0]
        for triple_synset in golden_annotations[sent_id]:
            for tr in triple_synset:
                if subj == tr[0].strip() and rel == tr[1].strip() and obj == tr[2].strip():
                    return True
        
        return False

    def is_true_positive_lexical(self, triple: list, golden_annotations: dict) -> bool:
        """
            Checks if an extracted triple is considered as true positive w.r.t. the golden annotations. Matching is performed on lexical level.

            Args
            ----
                triple: list 
                    The object 'triple' is in the following format [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
            
            Returns
            -------
                boolean:
                    True: if the triple is found in the golden annotations
                    False: otherwise
        """
        subj = triple[1].strip()
        rel = triple[2].strip()
        obj = triple[3].strip()
        
        sent_id = triple[0]
        for triple_synset in golden_annotations[sent_id]:
            for tr in triple_synset:
                triple_str = subj + " " + rel + " " + obj
                tr_str = tr[0].strip() + " " + tr[1].strip() + " " + tr[2].strip()
                if triple_str == tr_str:
                    return True
        return False
    def is_true_positive_Similarty(self,nlp, triple: list, golden_annotations: dict) -> bool:
        """
            Checks if an extracted triple is considered as true positive w.r.t. the golden annotations. Matching is performed on lexical level.

            Args
            ----
                triple: list 
                    The object 'triple' is in the following format [sent_id, subj, rel, obj]
                golden_annotations: dict
                    key: sentence id (sent_id from triple)
                    value: list of lists. Each list represents a triple synset (i.e., list of triples having the same meaning).
            
            Returns
            -------
                boolean:
                    True: if the triple is found in the golden annotations
                    False: otherwise
        """
        subj = triple[1].strip()
        rel = triple[2].strip()
        obj = triple[3].strip()
        
        sent_id = triple[0]
        triple_str = subj + " " + rel + " " + obj
        doc1golden = self.__get_doc(nlp, triple_str)

        for triple_synset in golden_annotations[sent_id]:
            for tr in triple_synset:
                tr_str = tr[0].strip() + " " + tr[1].strip() + " " + tr[2].strip()
                doc2 = self.__get_doc(nlp, tr_str)
                if doc1golden.similarity(doc2)>.90:
                    return True
        return False

    def print_scores(self, oie_system: str):
        """
            Print all scores (precision, recall, f1 score).

            Args
            ----
                oie_system: str
                    The name of the OIE system
        """
        
        print(oie_system + " precision: " + str(self.precision))
        print(oie_system + " recall: " + str(self.recall))
        print(oie_system + " f1: " + str(self.f1))
        print("===============")
