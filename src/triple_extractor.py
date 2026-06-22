"""
TripleExtractor — spaCy-based open information extraction.

Covered sentence patterns
──────────────────────────
1.  Active-verb  (SVO)              John bought a car.
2.  Passive                         The car was sold by David.
3.  Copular / linking-verb          Paris is beautiful.
4.  Apposition                      John, the CEO, announced a product.
5.  Relative clause (relcl)         The engineer who designed the bridge …
6.  Adjectival clause (acl)         The book written by Orwell …
7.  Adverbial clause (advcl)        He left because he was tired.
8.  Prepositional relations         John works in Paris.
9.  xcomp / ccomp                   John wants to buy a house.
10. Possessive (poss)               John's car is red.
11. Noun–noun compound / nmod       Tesla CEO Elon Musk …
12. Conjunction expansion           John and Mary bought a car.
13. Reported speech / say-that      He said that the deal was done.
14. Attributive adjective           The tall man left.
15. Existential there               There is a problem.
16. Negation                        John did not buy a car.
17. Coreference resolution          (requires en_coreference_web_trf)
18. Fallback heuristic              any short sentence that produced nothing else

Install
───────
    pip install spacy spacy-experimental
    pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.4/en_coreference_web_trf-3.7.0-py3-none-any.whl
    python -m spacy download en_core_web_trf   # for pipeline without coref
"""

from __future__ import annotations
import sys
import re
from collections import deque
import spacy
print(f"spaCy version: {spacy.__version__}")

import scispacy
print(f"scispaCy version: {scispacy.__version__}")
from tqdm import tqdm
from AdaptiveTripleFilter  import AdaptiveTripleFilter


# ─── dependency label sets ───────────────────────────────────────────────────

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}

OBJECT_DEPS = {
    "dobj", "obj", "pobj", "iobj", "attr",
    "oprd", "dative", "acomp"
}

RELATION_AUX = {"aux", "auxpass", "neg", "prt", "advmod"}

CLAUSAL_COMPLEMENT = {"xcomp", "ccomp"}

FORBIDDEN_SUBJECT_DEPS = {"appos", "relcl", "acl", "advcl", "punct"}


# ─── helper ──────────────────────────────────────────────────────────────────

Triple = tuple  # (subj_text, rel_text, obj_text, subj_idx, obj_idx, rel_idx)


def _clean(text: str) -> str:
    text = str(text) if text is not None else ""
    return re.sub(r"\s+", " ", text).strip()


# ─── main class ──────────────────────────────────────────────────────────────

class TripleExtractor:
    """Rule-based triple extractor built on spaCy dependency parses."""

    def __init__(self, nlp):
        # Load whichever model is available
        
        self.nlp = nlp

    # ── span helpers ─────────────────────────────────────────────────────────

    def _subtree_span(
        self,
        token,
        exclude_deps: set[str] | None = None,
        max_len: int = 40,
    ) -> tuple[str, tuple[int, int]]:
        """Return the text and (start, end) of *token*'s subtree,
        optionally skipping children whose dep_ is in *exclude_deps*."""
        if exclude_deps is None:
            exclude_deps = set()

        # Named entity → keep the full entity span
        if token.ent_type_:
            s = token.left_edge.i
            e = token.right_edge.i + 1
            return _clean(token.doc[s:e].text), (s, e)

        nodes: list = []
        for t in token.subtree:
            if t.dep_ in exclude_deps:
                continue
            if t.is_punct:
                continue
            nodes.append(t)

        if not nodes:
            return token.text, (token.i, token.i + 1)

        nodes.sort(key=lambda x: x.i)
        if len(nodes) > max_len:
            nodes = nodes[:max_len]

        s, e = nodes[0].i, nodes[-1].i + 1
        return _clean(token.doc[s:e].text), (s, e)

    def _clean_subject_span(self, token) -> tuple[str, tuple[int, int]]:
        """Subject span that strips appositives, relative clauses, etc."""
        nodes: list = []
        for t in token.subtree:
            if t.dep_ in FORBIDDEN_SUBJECT_DEPS:
                continue
            bad = any(
                anc.dep_ in FORBIDDEN_SUBJECT_DEPS
                for anc in t.ancestors
                if anc != token
            )
            if bad:
                continue
            if t.is_punct:
                continue
            nodes.append(t)

        if not nodes:
            return token.text, (token.i, token.i + 1)

        nodes.sort(key=lambda x: x.i)
        s, e = nodes[0].i, nodes[-1].i + 1
        return _clean(token.doc[s:e].text), (s, e)

    def _predicate_span(self, token) -> tuple[str, tuple[int, int]]:
        """Return the predicate span for copular sentences (excludes cop/subj)."""
        nodes = [
            t for t in token.subtree
            if t.dep_ not in SUBJECT_DEPS
            and t.dep_ != "cop"
            and not t.is_punct
        ]
        if not nodes:
            return token.text, (token.i, token.i + 1)

        nodes.sort(key=lambda x: x.i)
        s, e = nodes[0].i, nodes[-1].i + 1
        return _clean(token.doc[s:e].text), (s, e)

    # ── relation span ────────────────────────────────────────────────────────

    def _relation_span(self, verb) -> tuple[str, list[int]]:
        """Build relation string from verb + auxiliaries + negation."""
        tokens = [
            t for t in verb.subtree
            if t.dep_ in {"aux", "auxpass", "neg", "prt", "advmod"}
        ]
        tokens.append(verb)
        tokens = sorted(set(tokens), key=lambda x: x.i)

        rel = " ".join(t.text for t in tokens)
        if any(c.dep_ == "neg" for c in verb.children):
            rel = "not " + verb.text  # normalise negation
        else:
            rel = verb.text

        return _clean(rel), [t.i for t in tokens]

    # ── conjunction expansion ────────────────────────────────────────────────

    def _expand_conj(self, token) -> list:
        """BFS over conjuncts."""
        results, visited, queue = [], set(), deque([token])
        while queue:
            t = queue.popleft()
            if t.i in visited:
                continue
            visited.add(t.i)
            results.append(t)
            for c in t.conjuncts:
                queue.append(c)
        return results

    # ── subject / object getters ─────────────────────────────────────────────

    def _get_subjects(self, verb) -> list:
        subjects = []
        for child in verb.children:
            if child.dep_ in SUBJECT_DEPS:
                subjects.extend(self._expand_conj(child))

        # Inherit subject from ancestor verb (e.g. xcomp, ccomp)
        if not subjects:
            for ancestor in verb.ancestors:
                for child in ancestor.children:
                    if child.dep_ in SUBJECT_DEPS:
                        subjects.extend(self._expand_conj(child))
                if subjects:
                    break
        return subjects

    def _get_objects(self, verb) -> list:
        objects = []
        for child in verb.children:
            if child.dep_ in OBJECT_DEPS:
                objects.extend(self._expand_conj(child))
        return objects
    def _get_objects_com(self, verb) -> list:
        objects = []
        for child in verb.children:
            if child.dep_ in OBJECT_DEPS or child.dep_ in ['nsubjpass']:
                objects.extend(self._expand_conj(child))
        return objects

    # ── passive agent ────────────────────────────────────────────────────────

    def _passive_agent(self, verb):
        for child in verb.children:
            if child.dep_ == "agent":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        return pobj
        return None

    # ── relative-clause subject ──────────────────────────────────────────────

    def _relcl_subject(self, verb):
        if verb.dep_ == "relcl":
            return verb.head
        for child in verb.children:
            if child.dep_ == "relcl":
                return child.head
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Pattern extractors
    # ─────────────────────────────────────────────────────────────────────────

    # 1. SVO / active verbs ───────────────────────────────────────────────────

    def _verb_triples(self, verb) -> list[Triple]:
        triples = []
        rel, relIdx = self._relation_span(verb)
        subjects = self._get_subjects(verb)
        objects = self._get_objects(verb)

        # Relative-clause head as subject
        relcl = self._relcl_subject(verb)
        if relcl:
            subjects = [relcl]

        # Passive: agent → subject, grammatical subject → object
        passive = self._passive_agent(verb)
        if passive:
            agentText, agentIdx = self._clean_subject_span(passive)
            for subj in subjects:
                objText, objIdx = self._subtree_span(subj)
                triples.append((agentText, rel, objText, [agentIdx], objIdx, relIdx))
            return triples

        for subj in subjects:
            sText, sIdx = self._clean_subject_span(subj)
            for obj in objects:
                oText, oIdx = self._subtree_span(obj)
                triples.append((sText, rel, oText, [sIdx], oIdx, relIdx))

        # Prepositional phrases hanging off the verb
        triples.extend(self._object_prep_relations(verb, subjects))
        # xcomp / ccomp
        triples.extend(self._xcomp_relations(verb))

        return triples

    # 2. Prepositional relations ──────────────────────────────────────────────
    def _object_prep_relations(self, verb, subjects):
        triples = []
        for obj in self._get_objects(verb):
            for prep in obj.children:
                if prep.dep_ != "prep":
                    continue

                for pobj in prep.children:
                    if pobj.dep_ == "pobj":
                        pobj_text, pobj_idx = self._subtree_span(pobj)

                        for subj in subjects:
                            subj_text, subj_idx = self._clean_subject_span(subj)

                            rel = f"{verb.text} {prep.head.text} {prep.text}"

                            triples.append(
                                (
                                    subj_text,
                                    rel,
                                    pobj_text,
                                    [subj_idx],
                                    pobj_idx,
                                    [verb.i,prep.head.i, prep.i]
                                )
                            )
        for prep in verb.children:
            if prep.dep_ != "prep":
                continue
            for pobj in prep.children:
                if pobj.dep_ != "pobj":
                    continue
                oText, oIdx = self._subtree_span(pobj)
                for subj in subjects:
                    sText, sIdx = self._clean_subject_span(subj)
                    if verb.head.dep_=='ROOT' and verb.head.text != verb.text:
                        rel = f"{verb.head.text} to {verb.text} {prep.text}"
                        triples.append((sText, rel, oText, [sIdx], oIdx, [verb.head.i,verb.i,prep.i]))
                    else:
                        rel = f"{verb.text} {prep.text}"
                        triples.append((sText, rel, oText, [sIdx], oIdx, [verb.i,prep.i]))


        return triples
    # 3. xcomp / ccomp ────────────────────────────────────────────────────────

    def _xcomp_relations(self, verb) -> list[Triple]:
        triples = []
        subjects = self._get_subjects(verb)
        for child in verb.children:
            if child.dep_ not in CLAUSAL_COMPLEMENT:
                continue
            rel, relIdx = self._relation_span(child)
            objects = self._get_objects(child)
            if not objects:
                objects2 = self._get_objects_com(child)
            for subj in subjects:
                sText, sIdx = self._clean_subject_span(subj)
                if objects:
                    for obj in objects:
                        oText, oIdx = self._subtree_span(obj)
                        triples.append((sText, rel, oText, [sIdx], oIdx, relIdx))
                elif objects2:
                    rel2 = f"{verb.text} to {child.text}"
                    for obj in objects2:
                        oText, oIdx = self._subtree_span(obj)
                        triples.append((sText, rel2, oText, [sIdx], oIdx, [verb.i, child.i]))
                else:
                    for prep in child.children:
                        if prep.dep_ == "prep":

                            for pobj in prep.children:
                                if pobj.dep_ == "pobj":

                                    oText, oIdx = self._subtree_span(pobj)

                                    triples.append(
                                        (
                                            sText,
                                            f"{verb.text} to {child.text} {prep.text}",                                         oText,
                                            [sIdx],
                                            oIdx,
                                            [verb.i,child.i, prep.i]
                                        )
                                    )
        return triples

    # 4. Copular sentences ────────────────────────────────────────────────────

    def _copular_triples(self, sent) -> list[Triple]:
        triples = []
        for token in sent:
            cop = next((c for c in token.children if c.dep_ == "cop"), None)
            if not cop:
                continue
            subjects = []
            for child in token.children:
                if child.dep_ in SUBJECT_DEPS:
                    subjects.extend(self._expand_conj(child))
            for subj in subjects:
                sText, sIdx = self._clean_subject_span(subj)
                oText, oIdx = self._predicate_span(token)
                triples.append((sText, cop.text, oText, [sIdx], oIdx, [cop.i]))
            for prep in token.children:
                if prep.dep_ == "prep":
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":

                            oText, oIdx = self._subtree_span(pobj)

                            for subj in subjects:
                                sText, sIdx = self._clean_subject_span(subj)

                                triples.append(
                                    (
                                        sText,
                                        f"{cop.text}_{prep.text}",
                                        oText,
                                        [sIdx],
                                        oIdx,
                                        [cop.i, prep.i]
                                    )
                                )
        return triples


    # 5. Appositives ──────────────────────────────────────────────────────────

    def _apposition_triples(self, sent) -> list[Triple]:
        triples = []
        for token in sent:
            if token.dep_ != "appos":
                continue
            sText, sIdx = self._clean_subject_span(token.head)
            oText, oIdx = self._subtree_span(token)
            triples.append((sText, "is", oText, [sIdx], oIdx, [token.i]))
        return triples

    # 6. Adjectival clauses (acl) ─────────────────────────────────────────────

    def _acl_triples(self, token) -> list[Triple]:
        triples = []
        if token.dep_ != "acl":
            return triples

        sText, sIdx = self._clean_subject_span(token.head)
        rel, relIdx = self._relation_span(token)

        for child in token.children:
            if child.dep_ in {"obj", "dobj", "attr"}:
                oText, oIdx = self._subtree_span(child)
                triples.append((sText, rel, oText, [sIdx], oIdx, relIdx))
            elif child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        oText, oIdx = self._subtree_span(pobj)
                        prep_rel = f"{rel} {child.text}"
                        triples.append(
                            (sText, prep_rel, oText,
                             [sIdx], oIdx, relIdx + [child.i])
                        )
        return triples

    # 7. Adverbial clauses (advcl) ────────────────────────────────────────────

    def _advcl_triples(self, verb) -> list[Triple]:
        triples = []
        subjects = self._get_subjects(verb)
        for child in verb.children:
            if child.dep_ != "advcl":
                continue
            rel, relIdx = self._relation_span(child)
            # Use the subordinating conjunction (mark) as part of the relation
            #mark = next((c.text for c in child.children if c.dep_ == "mark"), "advcl")
            mark = next((c.text for c in child.children if c.dep_ == "mark"), "")
            combined_rel = f"{mark} {rel}"
            for subj in subjects:
                sText, sIdx = self._clean_subject_span(subj)
                oText, oIdx = self._subtree_span(child)
                triples.append((sText, combined_rel, oText, [sIdx], oIdx, relIdx))
        return triples

    # 8. Possessives ──────────────────────────────────────────────────────────

    def _possessive_triples(self, sent) -> list[Triple]:
        """John's car → (John, has, car)."""
        triples = []
        for token in sent:
            if token.dep_ == "poss":
                owner_text, owner_idx = self._clean_subject_span(token)
                possessed_text, possessed_idx = self._clean_subject_span(token.head)
                triples.append(
                    (owner_text, "has", possessed_text,
                     [owner_idx], possessed_idx, [token.i])
                )
        return triples

    # 9. Noun-modifier / compound ─────────────────────────────────────────────

    def _nmod_compound_triples(self, sent) -> list[Triple]:
        """CEO of Tesla → (Tesla, has_ceo, CEO)  |  compound: Tesla CEO."""
        triples = []
        for token in sent:
            # nmod:poss or nmod with "of"
            if token.dep_ in {"nmod", "nmod:poss"}:
                for prep in token.children:
                    if prep.dep_ == "prep" and prep.text.lower() == "of":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                oText, oIdx = self._subtree_span(pobj)
                                sText, sIdx = self._clean_subject_span(token)
                                triples.append(
                                    (oText, f"has {token}", sText,
                                     [oIdx], sIdx, [prep.i])
                                )
        return triples

    # 10. Reported-speech / say-that (ccomp) ─────────────────────────────────
    # Already handled via _xcomp_relations for ccomp.

    # 11. Attributive adjectives ─────────────────────────────────────────────

    def _attributive_adj_triples(self, sent) -> list[Triple]:
        """The tall man → (man, is, tall)."""
        triples = []
        for token in sent:
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                sText, sIdx = self._clean_subject_span(token.head)
                triples.append(
                    (sText, "is", token.text,
                     [sIdx], (token.i, token.i + 1), [token.i])
                )
        return triples

    # 12. Existential "there" ─────────────────────────────────────────────────

    def _existential_triples(self, sent) -> list[Triple]:
        """There is a problem → (problem, exists, True)."""
        triples = []
        for token in sent:
            if token.dep_ == "expl" and token.text.lower() == "there":
                verb = token.head
                for child in verb.children:
                    if child.dep_ in {"attr", "nsubj"}:
                        oText, oIdx = self._subtree_span(child)
                        triples.append(
                            (oText, "exists", "true",
                             [oIdx], (token.i, token.i + 1), [verb.i])
                        )
        return triples

    # 13. Fallback heuristic ──────────────────────────────────────────────────
    def _fallback(self, sent) -> list[Triple]:
        tokens = [t for t in sent if not t.is_punct]
        if len(tokens) < 3:
            return []
    
        # Find the main verb (the last verb in the sentence)
        verb_indices = [i for i, t in enumerate(tokens) 
                        if t.tag_.startswith('VB') or t.text.lower() in 
                        ['will', 'would', 'be', 'am', 'are', 'is', 'was', 'were']]
    
        if not verb_indices:
            return []
    
        # Use the last verb as the main predicate
        main_verb_idx = verb_indices[-1]
    
        # Find the start of the verb phrase (first auxiliary verb before main verb)
        verb_start = main_verb_idx
        for i in range(main_verb_idx - 1, -1, -1):
            if tokens[i].tag_.startswith('VB') or tokens[i].text.lower() in ['will', 'would', 'be', 'am', 'are', 'is', 'was', 'were', 'have', 'has', 'had']:
                verb_start = i
            else:
                break
    
        # Calculate indices
        subj_start_idx = 0
        subj_end_idx = verb_start - 1
        subj_indices = list(range(subj_start_idx, subj_end_idx + 1)) if subj_end_idx >= subj_start_idx else []
    
        rel_start_idx = verb_start
        rel_end_idx = main_verb_idx
        rel_indices = list(range(rel_start_idx, rel_end_idx + 1))
        obj_start_idx = main_verb_idx + 1
        if obj_start_idx>len(tokens) - 1:
            rel_end_idx = main_verb_idx-1
            rel_indices = list(range(rel_start_idx, rel_end_idx + 1))
            obj_start_idx = main_verb_idx-1 + 1
            obj_end_idx = len(tokens) - 1
            obj_indices = list(range(obj_start_idx, obj_end_idx + 1)) if obj_start_idx <= obj_end_idx else []
        else:
            obj_end_idx = len(tokens) - 1
            obj_indices = list(range(obj_start_idx, obj_end_idx + 1)) if obj_start_idx <= obj_end_idx else []
    
        # Extract texts
        subject_text = " ".join(tokens[i].text for i in subj_indices) if subj_indices else ""
        rel_text = " ".join(tokens[i].text for i in rel_indices)
        object_text = " ".join(tokens[i].text for i in obj_indices) if obj_indices else ""
    
        return [(
            subject_text,
            rel_text,
            object_text,
            subj_indices,
            (obj_start_idx,obj_end_idx),
            rel_indices
        )]
    def _extract_phrase_tokens(self,token):
        """
        Extracts tokens in a phrase including modifiers.
        Returns list of tokens in order.
        """
        phrase_tokens = [token]
    
        # Add adjectives/modifiers before noun
        for child in token.children:
            if child.dep_ in ["amod", "compound", "det", "poss"]:
                phrase_tokens.append(child)
            elif child.dep_ == "prep" and token.pos_ != "VERB":
                # Include prepositional phrases for object phrases
                for subchild in child.children:
                    if subchild.dep_ == "pobj":
                        phrase_tokens.extend([child, subchild])
                        break
    
        # Sort by token index to maintain original order
        phrase_tokens.sort(key=lambda x: x.i)
        return phrase_tokens
    def _extract_subject_complement_tokens(self,token):
        """
        Extracts subject complement (attribute/predicate) tokens.
        """
        for child in token.children:
            if child.dep_ == "attr" or (child.dep_ == "acomp" and child.pos_ == "ADJ"):
                return self._extract_phrase_tokens(child)
        return None
    def _extract_triple_and_handler(self,doc) -> list[Triple]:
        """
        Extracts (subject, relation, object) triplets from text with indices,
        correctly handling prepositions in relations.
    
        Args:
            text: Input sentence/text
               
        Returns:
            List of tuples: (subj_text, rel_text, obj_text, subj_idx, obj_idx, rel_idx)
            where indices refer to word positions in the original text
        """
        triplets = []
    
        # Dependency patterns for relation extraction
        for token in doc:
            # Pattern 1: Subject - Verb - Object
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
                subject_tokens = self._extract_phrase_tokens(token)
                subject_text = " ".join([t.text for t in subject_tokens])
                subject_idx = [(subject_tokens[0].i,subject_tokens[-1].i+1)]  # Use subject's main token index
            
                verb_token = token.head
                verb_text = verb_token.text
            
                # Find object (direct or indirect)
                obj_tokens = None
                relation_text = None
                obj_idx = None
            
                # Look for direct object
                for child in verb_token.children:
                    if child.dep_ in ["dobj", "attr"]:
                        obj_tokens = self._extract_phrase_tokens(child)
                        relation_text = verb_text
                        obj_idx = (child.i,child.i+1)
                        break
                
                    # Handle prepositional objects (e.g., "relies on", "looks at")
                    elif child.dep_ == "prep":
                        prep = child.text
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                obj_tokens = self._extract_phrase_tokens(prep_child)
                                # Include preposition in relation
                                relation_text = f"{verb_text} {prep}"
                                obj_idx = (prep_child.i,prep_child.i+1)
                                break
                        if obj_tokens:
                            break
            
                # Handle passive voice (e.g., "was created by")
                if token.dep_ == "nsubjpass":
                    # Find agent (by X)
                    for child in verb_token.children:
                        if child.dep_ == "agent" and child.text == "by":
                            for agent_child in child.children:
                                if agent_child.dep_ == "pobj":
                                    obj_tokens = self._extract_phrase_tokens(agent_child)
                                    relation_text = f"is {verb_text} by"
                                    obj_idx = (agent_child.i,agent_child.i+1)
                                    break
            
                if obj_tokens and subject_tokens and relation_text:
                    obj_text = " ".join([t.text for t in obj_tokens])
                    rel_idx = [verb_token.i]
                    triplets.append((subject_text, relation_text, obj_text, 
                                   subject_idx, obj_idx, rel_idx))
        
            # Pattern 2: Nominal subject with copula (is/are/was)
            elif token.dep_ == "nsubj" and token.head.pos_ == "ADJ":
                subject_tokens = self._extract_phrase_tokens(token)
                subject_text = " ".join([t.text for t in subject_tokens])
                subject_idx = [(subject_tokens[0].i,subject_tokens[-1].i+1)]
            
                adj_token = token.head
                relation_text = f"is {adj_token.text}"
                rel_idx = [adj_token.i]
            
                obj_tokens = self._extract_subject_complement_tokens(adj_token)
                if obj_tokens:
                    obj_text = " ".join([t.text for t in obj_tokens])
                    obj_idx = (obj_tokens[0].i if obj_tokens else -1,obj_tokens[0].i+1 if obj_tokens else -1)
                    triplets.append((subject_text, relation_text, obj_text,
                                   subject_idx, obj_idx, rel_idx))
                
            elif token.dep_ == "nsubj" and token.head.pos_ == "NOUN":
                subject_tokens = self._extract_phrase_tokens(token)
                subject_text = " ".join([t.text for t in subject_tokens])
                subject_idx = [(subject_tokens[0].i,subject_tokens[-1].i+1)]
            
                noun_token = token.head
                relation_text = "is"
                rel_idx = [noun_token.i]
            
                obj_tokens = self._extract_phrase_tokens(noun_token)
                obj_text = " ".join([t.text for t in obj_tokens])
                obj_idx = (noun_token.i,noun_token.i+1)
            
                triplets.append((subject_text, relation_text, obj_text,
                               subject_idx, obj_idx, rel_idx))
    
        # Remove duplicates based on all fields
        return self._deduplicate(triplets)
    # ── coreference resolution ────────────────────────────────────────────────
    def _resolve_coref(self, doc) -> str:
        """Replace pronouns with their main cluster mention (if coref loaded)."""
        try:
            resolved = []
            for token in doc:
                if token._.in_coref:
                    resolved.append(token._.coref_cluster.main.text)
                else:
                    resolved.append(token.text_with_ws)
            return "".join(resolved)
        except AttributeError:
            return doc.text  # coref extension not loaded — skip silently

    # ── per-sentence pipeline ─────────────────────────────────────────────────

    def _process_sentence(self, sent) -> list[Triple]:
        triples: list[Triple] = []

        # --- verb-centric patterns ---
        seen_verbs: set[int] = set()
        for token in sent:
            if token.pos_ in {"VERB", "AUX"} and token.i not in seen_verbs:
                seen_verbs.add(token.i)
                for conj in self._expand_conj(token):
                    if conj.i not in seen_verbs:
                        seen_verbs.add(conj.i)
                triples.extend(self._verb_triples(token))
                triples.extend(self._advcl_triples(token))
            elif (token.pos_ == "AUX" and token.dep_ == "auxpass") and token.i not in seen_verbs:
                if token.dep_ == "auxpass" or (token.head.pos_ == "VERB" and token.head.dep_ == "ROOT"):
                    seen_verbs.add(token.head.i)
                    for conj in self._expand_conj(token.head):
                        if conj.i not in seen_verbs:
                            seen_verbs.add(conj.i)
                    triples.extend(self._verb_triples(token.head))
                    triples.extend(self._advcl_triples(token.head))
                else:                    
                    seen_verbs.add(token.i)
                    for conj in self._expand_conj(token):
                        if conj.i not in seen_verbs:
                            seen_verbs.add(conj.i)
                    triples.extend(self._verb_triples(token))
                    triples.extend(self._advcl_triples(token))
            elif token.pos_ in {"VERB", "AUX"} and token.i in seen_verbs:
                triples.extend(self._verb_triples(token))
                triples.extend(self._advcl_triples(token))
            if token.dep_ == "acl":
                triples.extend(self._acl_triples(token))

        # --- noun-centric / structural patterns ---
        triples.extend(self._copular_triples(sent))
        triples.extend(self._apposition_triples(sent))
        triples.extend(self._possessive_triples(sent))
        triples.extend(self._nmod_compound_triples(sent))
        triples.extend(self._attributive_adj_triples(sent))
        triples.extend(self._existential_triples(sent))

        if not triples:
            triples.extend(self._fallback(sent))

        return triples

    # ── deduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(triples: list[Triple]) -> list[Triple]:
        seen: set = set()
        results: list[Triple] = []
        for t in triples:
            key = (t[0].lower().strip(), str(t[1]).lower().strip(), t[2].lower().strip())
            if key in seen or not all(key):   # skip empty-component triples
                continue
            seen.add(key)
            results.append(t)
        return results

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self,doc) -> list[Triple]:
        """
        Extract (subject, relation, object) triples from *text*.

        Returns a deduplicated list of 6-tuples:
          (subj_text, rel_text, obj_text, subj_idx, obj_idx, rel_idx)
        """
        # Try coreference resolution (only if model supports it)
        resolved = self._resolve_coref(doc)
        if resolved != doc.text:
            doc = self.nlp(resolved)

        triples: list[Triple] = []
        rtoken1=[]
        #for sent in doc.sents:
        sent=doc
        triples.extend(self._process_sentence(sent))
        triples.extend(self._extract_triple_and_handler(sent))
        rtoken=[token.text for token in sent]        
        rtoken1.append(rtoken)

        return self._deduplicate(triples),rtoken1
    def extract_biomedical_triples(self,doc):
        triples = []
        rtoken1=[]    
        # Iterate through entities identified by SciSpacy
        for ent in doc.ents:
            # Check if the entity is a subject or object
            if ent.root.dep_ in ("nsubj", "dobj", "pobj"):
                # Find the governing verb (the relation)
                head = ent.root.head
                if head.pos_ == "VERB":
                    # Find the corresponding object/subject
                    for child in head.children:
                        if child.dep_ in OBJECT_DEPS and child.text != ent.text:
                            triples.append((ent.text, head.text, child.text,[(ent.start,ent.end)], (head.i, head.i + 1), [child.i]))
        triples.extend(self._process_sentence(doc))
        triples.extend(self._extract_triple_and_handler(doc))
        rtoken=[token.text for token in doc]        
        rtoken1.append(rtoken)
        return self._deduplicate(triples),rtoken1
# ─── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":    
    model = "en_coreference_web_trf" if "--coref" in sys.argv else "en_core_web_trf"
    print(f"Loading '{model}' …")
    nlp = spacy.load(model)
    #nlp = spacy.load("en_core_web_lg")#       
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    extractor = TripleExtractor(nlp)
    analysisList=[]
    with open("./dataset/carb_sentences/carb_sentences.txt", "r", encoding="utf8") as f:
            text = f.readlines()
            
            for line in tqdm(text, total=len(text)):
               print (line)
               linesen=line
               if linesen not in [" ", "\n", ""]:
                    doc = nlp(linesen)
                    triples,_ = extractor.extract(doc)
                    if triples:
                        for t in triples:
                            print(f"  ({t[0]!r:35s}  |  {t[1]!r:25s}  |  {t[2]!r})")
                        converted = [item[:3] for item in triples]
                        filter_engine = AdaptiveTripleFilter(z_threshold=1.2, min_similarity=0)
                        AllSentTriplesAdaptiveTripleFilter, analysis1 = filter_engine.filter_and_analyze(doc.text, converted)
                        triples[:] = [triple for triple in triples if tuple(triple[:3]) in AllSentTriplesAdaptiveTripleFilter]
                        if 'error' not in analysis1:
                            print("\n" + "=" * 60)
                            print("ERROR ANALYSIS:")
                            print(f"Total: {analysis1['total_triples']}")
                            print(f"Kept: {analysis1['kept_triples']}")
                            print(f"Filtered out: {analysis1['filtered_out']}")
                            print(f"Precision: {analysis1['precision']:.2f}")
                            print("\nError breakdown:")
                            for err, count in analysis1['error_breakdown'].items():
                                print(f"   - {err}: {count}")
    
                            print("\n" + "=" * 60)
                            print("DETAILED SCORES (first 4 triples):")
                            for item in analysis1['per_triple_scores'][:4]:
                                print(f"\n{item['triple']}")
                                print(f"  → Kept: {item['kept']}")
                                if item['error_type']:
                                    print(f"  → Error: {item['error_type']}")
                                print(f"  → Relevance: {item['relevance']:.2f}")
                                print(f"  → Consistency: {item['consistency']:.2f}")
                                print(f"  → Combined: {item['combined']:.2f}")
                                analysisList.append((doc.text,analysis1))
               else:
                   print("  (no triples extracted)")
    # Load the medical-specific model
    nlp = spacy.load("en_core_sci_sm")
    extractor = TripleExtractor(nlp)
    text_sample = ["Naloxone reverses the antihypertensive effect of clonidine. In unanesthetized , spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine , 5 to 20 micrograms / kg , was inhibited or reversed by nalozone , 0 . 2 to 2 mg / kg. The hypotensive effect of 100 mg / kg alpha - methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine , 10 ( - 8 ) to 10 ( - 5 ) M , did not influence stereoselective binding of [ 3H ] - naloxone ( 8 nM ) , and naloxone , 10 ( - 8 ) to 10 ( - 4 ) M , did not influence clonidine - suppressible binding of [ 3H ] - dihydroergocryptine ( 1 nM ). These findings indicate that in spontaneously hypertensive rats the effects of central alpha - adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site , the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha - methyldopa and the possible role of the opiate in the central control of sympathetic tone .",
"Lidocaine - induced cardiac asystole. Intravenous administration of a single 50 - mg bolus of lidocaine in a 67 - year - old man resulted in profound depression of the activity of the sinoatrial and atrioventricular nodal pacemakers. The patient had no apparent associated conditions which might have predisposed him to the development of bradyarrhythmias ; and , thus , this probably represented a true idiosyncrasy to lidocaine .",
"Suxamethonium infusion rate and observed fasciculations. A dose - response study. Suxamethonium chloride ( Sch ) was administered i . v. to 36 adult males at six rates : 0 . 25 mg s - 1 to 20 mg s - 1. The infusion was discontinued either when there was no muscular response to tetanic stimulation of the ulnar nerve or when Sch 120 mg was exceeded. Six additional patients received a 30 - mg i . v. bolus dose. Fasciculations in six areas of the body were scored from 0 to 3 and summated as a total fasciculation score. The times to first fasciculation , twitch suppression and tetanus suppression were inversely related to the infusion rates. Fasciculations in the six areas and the total fasciculation score were related directly to the rate of infusion. Total fasciculation scores in the 30 - mg bolus group and the 5 - mg s - 1 and 20 - mg s - 1 infusion groups were not significantly different .",
"Galanthamine hydrobromide , a longer acting anticholinesterase drug , in the treatment of the central effects of scopolamine ( Hyoscine ). Galanthamine hydrobromide , an anticholinesterase drug capable of penetrating the blood - brain barrier , was used in a patient demonstrating central effects of scopolamine ( hyoscine ) overdosage. It is longer acting than physostigmine and is used in anaesthesia to reverse the non - depolarizing neuromuscular block. However , studies into the dose necessary to combating scopolamine intoxication are indicated .",
"Effects of uninephrectomy and high protein feeding on lithium - induced chronic renal failure in rats. Rats with lithium - induced nephropathy were subjected to high protein ( HP ) feeding , uninephrectomy ( NX ) or a combination of these , in an attempt to induce glomerular hyperfiltration and further progression of renal failure. Newborn female Wistar rats were fed a lithium - containing diet ( 50 mmol / kg ) for 8 weeks and then randomized to normal diet , HP diet ( 40 vs . 19 % ) , NX or HP + NX for another 8 weeks. Corresponding non - lithium pretreated groups were generated. When comparing all lithium treated versus non - lithium - treated groups , lithium caused a reduction in glomerular filtration rate ( GFR ) without significant changes in effective renal plasma flow ( as determined by a marker secreted into the proximal tubules ) or lithium clearance. Consequently , lithium pretreatment caused a fall in filtration fraction and an increase in fractional Li excretion. Lithium also caused proteinuria and systolic hypertension in absence of glomerulosclerosis. HP failed to accentuante progression of renal failure and in fact tended to increase GFR and decrease plasma creatinine levels in lithium pretreated rats. NX caused an additive deterioration in GFR which , however , was ameliorated by HP. NX + HP caused a further rise in blood pressure in Li - pretreated rats. The results indicate that Li - induced nephropathy , even when the GFR is only modestly reduced , is associated with proteinuria and arterial systolic hypertension. In this model of chronic renal failure the decline in GFR is not accompanied by a corresponding fall in effective renal plasma flow , which may be the functional expression of the formation of nonfiltrating atubular glomeruli. The fractional reabsorption of tubular fluid by the proximal tubules is reduced , leaving the distal delivery unchanged .",
"Treatment of Crohn 's disease with fusidic acid : an antibiotic with immunosuppressive properties similar to cyclosporin. Fusidic acid is an antibiotic with T - cell specific immunosuppressive effects similar to those of cyclosporin. Because of the need for the development of new treatments for Crohn 's disease , a pilot study was undertaken to estimate the pharmacodynamics and tolerability of fusidic acid treatment in chronic active , therapy - resistant patients. Eight Crohn 's disease patients were included. Fusidic acid was administered orally in a dose of 500 mg t . d . s. and the treatment was planned to last 8 weeks. The disease activity was primarily measured by a modified individual grading score. Five of 8 patients ( 63 % ) improved during fusidic acid treatment : 3 at two weeks and 2 after four weeks. There were no serious clinical side effects , but dose reduction was required in two patients because of nausea. Biochemically , an increase in alkaline phosphatases was noted in 5 of 8 cases ( 63 % ) , and the greatest increases were seen in those who had elevated levels prior to treatment. All reversed to pre - treatment levels after cessation of treatment. The results of this pilot study suggest that fusidic acid may be of benefit in selected chronic active Crohn 's disease patients in whom conventional treatment is ineffective. Because there seems to exist a scientific rationale for the use of fusidic acid at the cytokine level in inflammatory bowel disease , we suggest that the role of this treatment should be further investigated .",
"The same molecular defects of the gonadotropin - releasing hormone receptor determine a variable degree of hypogonadism in affected kindred. Detailed endocrinological studies were performed in the three affected kindred of a family carrying mutations of the GnRH receptor gene. All three were compound heterozygotes carrying on one allele the Arg262Gln mutation and on the other allele two mutations ( Gln106Arg and Ser217Arg ). When expressed in heterologous cells , both Gln106Arg and Ser217Arg mutations altered hormone binding , whereas the Arg262Gln mutation altered activation of phospholipase C. The propositus , a 30 - yr - old man , displayed complete idiopathic hypogonadotropic hypogonadism with extremely low plasma levels of gonadotropins , absence of pulsatility of endogenous LH and alpha - subunit , absence of response to GnRH and GnRH agonist ( triptorelin ) , and absence of effect of pulsatile administration of GnRH. The two sisters , 24 and 18 yr old , of the propositus displayed , on the contrary , only partial idiopathic hypogonadotropic hypogonadism. They both had primary amenorrhea , and the younger sister displayed retarded bone maturation and uterus development , but both sisters had normal breast development. Gonadotropin concentrations were normal or low , but in both cases were restored to normal levels by a single injection of GnRH. In the two sisters , there were no spontaneous pulses of LH , but pulsatile administration of GnRH provoked a pulsatile secretion of LH in the younger sister. The same mutations of the GnRH receptor gene may thus determine different degrees of alteration of gonadotropin function in affected kindred of the same family .",
"Discordant measures of androgen - binding kinetics in two mutant androgen receptors causing mild or partial androgen insensitivity , respectively. We have characterized two different mutations of the human androgen receptor ( hAR ) found in two unrelated subjects with androgen insensitivity syndrome ( AIS ) : in one , the external genitalia were ambiguous ( partial , PAIS ) ; in the other , they were male , but small ( mild , MAIS ). Single base substitutions have been found in both individuals : E772A in the PAIS subject , and R871G in the MAIS patient. In COS - 1 cells transfected with the E772A and R871G hARs , the apparent equilibrium dissociation constants ( Kd ) for mibolerone ( MB ) and methyltrienolone are normal. Nonetheless , the mutant hAR from the PAIS subject ( E772A ) has elevated nonequilibrium dissociation rate constants ( k ( diss ) ) for both androgens. In contrast , the MAIS subject 's hAR ( R871G ) has k ( diss ) values that are apparently normal for MB and methyltrienolone ; in addition , the R871G hAR 's ability to bind MB resists thermal stress better than the hAR from the PAIS subject. The E772A and R871G hARs , therefore , confer the same pattern of discordant androgen - binding parameters in transfected COS - 1 cells as observed previously in the subjects ' genital skin fibroblasts. This proves their pathogenicity and correlates with the relative severity of the clinical phenotype. In COS - 1 cells transfected with an androgen - responsive reporter gene , trans - activation was 50 % of normal in cells containing either mutant hAR. However , mutant hAR - MB binding is unstable during prolonged incubation with MB , whereas normal hAR - MB binding increases. Thus , normal equilibrium dissociation constants alone , as determined by Scatchard analysis , may not be indicative of normal hAR function. An increased k ( diss ) despite a normal Kd for a given androgen suggests that it not only has increased egress from a mutant ligand - binding pocket , but also increased access to it. This hypothesis has certain implications in terms of the three - dimensional model of the ligand - binding domain of the nuclear receptor superfamily .",
"Genomic organization of the KCNQ1 K + channel gene and identification of C - terminal mutations in the long - QT syndrome. The voltage - gated K + channel KVLQT1 is essential for the repolarization phase of the cardiac action potential and for K + homeostasis in the inner ear. Mutations in the human KCNQ1 gene encoding the alpha subunit of the KVLQT1 channel cause the long - QT syndrome ( LQTS ). The autosomal dominant form of this cardiac disease , the Romano - Ward syndrome , is characterized by a prolongation of the QT interval , ventricular arrhythmias , and sudden death. The autosomal recessive form , the Jervell and Lange - Nielsen syndrome , also includes bilateral deafness. In the present study , we report the entire genomic structure of KCNQ1 , which consists of 19 exons spanning 400 kb on chromosome 11p15 . 5. We describe the sequences of exon - intron boundaries and oligonucleotide primers that allow polymerase chain reaction ( PCR ) amplification of exons from genomic DNA. Two new ( CA ) n repeat microsatellites were found in introns 10 and 14. The present study provides helpful tools for the linkage analysis and mutation screening of the complete KCNQ1 gene. By use of these tools , five novel mutations were identified in LQTS patients by PCR - single - strand conformational polymorphism ( SSCP ) analysis in the C - terminal part of KCNQ1 : two missense mutations , a 20 - bp and 1 - bp deletions , and a 1 - bp insertion. Such mutations in the C - terminal domain of the gene may be more frequent than previously expected , because this region has not been analyzed so far. This could explain the low percentage of mutations found in large LQTS cohorts."]
    for sent in text_sample:
        doc = nlp(sent)
        triples,_=extractor.extract_biomedical_triples(doc)
        if triples:
            for t in triples:
                print(f"  ({t[0]!r:35s}  |  {t[1]!r:25s}  |  {t[2]!r})")        
            converted = [item[:3] for item in triples]
            filter_engine = AdaptiveTripleFilter(z_threshold=1.2, min_similarity=0)
            AllSentTriplesAdaptiveTripleFilter, analysis1 = filter_engine.filter_and_analyze(doc.text, converted)
            triples[:] = [triple for triple in triples if tuple(triple[:3]) in AllSentTriplesAdaptiveTripleFilter]
            if 'error' not in analysis1:
                print("\n" + "=" * 60)
                print("ERROR ANALYSIS:")
                print(f"Total: {analysis1['total_triples']}")
                print(f"Kept: {analysis1['kept_triples']}")
                print(f"Filtered out: {analysis1['filtered_out']}")
                print(f"Precision: {analysis1['precision']:.2f}")
                print("\nError breakdown:")
                for err, count in analysis1['error_breakdown'].items():
                    print(f"   - {err}: {count}")
    
                print("\n" + "=" * 60)
                print("DETAILED SCORES (first 4 triples):")
                for item in analysis1['per_triple_scores'][:4]:
                    print(f"\n{item['triple']}")
                    print(f"  → Kept: {item['kept']}")
                    if item['error_type']:
                        print(f"  → Error: {item['error_type']}")
                    print(f"  → Relevance: {item['relevance']:.2f}")
                    print(f"  → Consistency: {item['consistency']:.2f}")
                    print(f"  → Combined: {item['combined']:.2f}")
                    analysisList.append((doc.text,analysis1))
        else:
            print("  (no triples extracted)")
    with open('./dataset/carb_sentences/analysislist1.txt', 'a', encoding="utf-8") as f:
                                    f.write(str(analysisList))
                                    f.write('\n')  

    #TEST_SENTENCES = [
    #    "Lidocaine-induced cardiac asystole. Intravenous administration of a single 50-mg bolus of lidocaine in a 67-year-old man resulted in profound depression of the activity of the sinoatrial and atrioventricular nodal pacemakers. The patient had no apparent associated conditions which might have predisposed him to the development of bradyarrhythmias ; and, thus, this probably represented a true idiosyncrasy to lidocaine. ",
    #    "John wants to live in Paris.",
    #    "Mary is interested in machine learning.",        
    #    "John relies on his team.",
    #    "The cat looks at the mouse.",       
    #    "John lives in Paris.",
    #    "The book is on the table.",        
    #    "The book was written by the author.",
    #    "The company specializes in artificial intelligence.",
    #    "23.8% of all households were made up of individuals and 13.0% had someone living alone who was 65 years of age or older.",
    #    "The three existing plants and their land will be sold.",
    #    # Reported speech
    #    "She said that the deal was done.",
    #    "A Democrat, he became the youngest mayor in Pittsburgh's history in September 2006 at the age of 26.",
    #    "The `` Charleston Courier, '' founded in 1803, and `` Charleston Daily News, 'founded in 1865, merged to form the `` News and Courier '' in 1873.",
    #    "Several years later the remaining trackage at Charles City was abandoned.",
    #    "23.8\% of all households were made up of individuals and 13.0\% had someone living alone who was 65 years of age or older.",
    #    # SVO
    #    "John and Mary bought a car.",
    #    # Passive
    #    "The car was sold by David.",
    #    # Copular
    #    "Paris is beautiful.",
    #    # Apposition
    #    "John, the CEO of Tesla, announced a new product.",
    #    # Relative clause
    #    "The engineer who designed the bridge won an award.",
    #    # Adverbial clause
    #    "He left early because he was tired.",
    #    # Prepositional relation
    #    "John works in Paris.",
    #    # xcomp
    #    "John wants to buy a house.",
    #    # Possessive
    #    "John's car is red.",
    #    # Attributive adjective
    #    "The tall man walked away.",
    #    # Existential
    #    "There is a serious problem.",
    #    # Negation
    #    "John did not buy a car.",
    #    # Compound / nmod
    #    "The CEO of Tesla announced results.",        
    #    # Newspaper-style compound sentence
    #    "The Charleston Courier, founded in 1803, and Charleston Daily News founded in 1865, merged to form the News and Courier in 1873."
    #]


    #for sent in TEST_SENTENCES:
    #    print(f"\n{'─'*60}")
    #    print(f"INPUT : {sent}")
    #    doc = nlp(sent)
    #    triples,_ = extractor.extract(doc)
    #    if triples:
    #        for t in triples:
    #            print(f"  ({t[0]!r:35s}  |  {t[1]!r:25s}  |  {t[2]!r})")
    #    else:
    #        print("  (no triples extracted)")
